"""RepViT backbone with structural reparameterization.

References:
- Wang et al., "RepViT: Revisiting Mobile CNN From ViT Perspective" (CVPR 2024).
- Ding et al., "RepVGG: Making VGG-style ConvNets Great Again" (CVPR 2021).

RepDWConv combines three depthwise branches during training (3x3+BN, 1x1+BN, identity). 
Calling switch_to_deploy() collapses them into one Conv2d with bias. 
Equivalence tested in tests/test_repvit_reparam.py.

Stage layout for RepViT-M1: channels (48, 96, 192, 384), depths (2, 2, 14, 2).
Input 384x384 produces stride sequence stem->stage4 of 4->8->16->32. 
Detector consumes C3 (s=8), C4 (s=16), C5 (s=32).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepDWConv(nn.Module):
    """Depthwise conv with reparameterizable training-time branches.

    Training-time forward (stride 1):
        y = BN3(DW3x3(x)) + BN1(DW1x1(x)) + BN_skip(x)

    Training-time forward (stride 2):
        y = BN3(DW3x3(x)) + BN1(DW1x1(x))     # no identity skip when downsampling

    Deploy-time forward (after `switch_to_deploy`):
        y = DW3x3_fused(x)                     # single conv with bias
    """

    def __init__(self, channels: int, stride: int = 1) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        self.channels = channels
        self.stride = stride
        self.deploy = False

        self.dw3 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=stride,
            padding=1, groups=channels, bias=False,
        )
        self.bn3 = nn.BatchNorm2d(channels)

        self.dw1 = nn.Conv2d(
            channels, channels, kernel_size=1, stride=stride,
            padding=0, groups=channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # Identity skip is only valid when spatial dims are preserved.
        self.bn_skip: nn.BatchNorm2d | None = (
            nn.BatchNorm2d(channels) if stride == 1 else None
        )

        self.fused: nn.Conv2d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            assert self.fused is not None
            return self.fused(x)

        out = self.bn3(self.dw3(x)) + self.bn1(self.dw1(x))
        if self.bn_skip is not None:
            out = out + self.bn_skip(x)
        return out

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (weight, bias) of an equivalent conv with BN folded in."""
        std = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std                          # (C,)
        weight = conv.weight * scale.view(-1, 1, 1, 1)   # (C, 1, kH, kW)
        bias = bn.bias - bn.running_mean * scale         # (C,)
        return weight, bias

    def _bn_only_identity_kernel(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Treat `bn_skip(x)` as a virtual depthwise conv with identity kernel
        (1 at center, 0 elsewhere) followed by BN, then fold the BN."""
        assert self.bn_skip is not None
        device = self.bn_skip.weight.device
        dtype = self.bn_skip.weight.dtype
        identity = torch.zeros(self.channels, 1, 3, 3, device=device, dtype=dtype)
        identity[:, 0, 1, 1] = 1.0
        std = (self.bn_skip.running_var + self.bn_skip.eps).sqrt()
        scale = self.bn_skip.weight / std
        weight = identity * scale.view(-1, 1, 1, 1)
        bias = self.bn_skip.bias - self.bn_skip.running_mean * scale
        return weight, bias

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return

        w3, b3 = self._fuse_conv_bn(self.dw3, self.bn3)
        w1, b1 = self._fuse_conv_bn(self.dw1, self.bn1)
        # Pad the 1x1 kernel to 3x3 by zero-padding the borders. Applying with
        # the same stride yields an identical computation to the original 1x1.
        w1_padded = F.pad(w1, [1, 1, 1, 1])

        weight = w3 + w1_padded
        bias = b3 + b1

        if self.bn_skip is not None:
            w_id, b_id = self._bn_only_identity_kernel()
            weight = weight + w_id
            bias = bias + b_id

        fused = nn.Conv2d(
            self.channels, self.channels, kernel_size=3, stride=self.stride,
            padding=1, groups=self.channels, bias=True,
        )
        fused.weight.data.copy_(weight)
        fused.bias.data.copy_(bias)

        # Move fused conv to the same device/dtype as the original branch.
        fused = fused.to(self.dw3.weight.device, self.dw3.weight.dtype)
        self.fused = fused

        # Drop the training-time branches so deploy-mode params are clean.
        del self.dw3, self.bn3, self.dw1, self.bn1
        if self.bn_skip is not None:
            del self.bn_skip
        self.deploy = True


# ----- supporting building blocks ------------------------------------------- #


class SqueezeExcite(nn.Module):
    """Standard SE block (Hu et al., 2018), depthwise-friendly default rate."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ChannelMixer(nn.Module):
    """Inverted-residual FFN: 1x1 expand -> GELU -> 1x1 compress, with residual."""

    def __init__(self, channels: int, expand_ratio: int = 2) -> None:
        super().__init__()
        hidden = channels * expand_ratio
        self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.GELU()
        self.compress = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn1(self.expand(x))
        h = self.act(h)
        h = self.bn2(self.compress(h))
        return x + h


class ChannelTransition(nn.Module):
    """1x1 conv + BN to change channel count between stages, no spatial change."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


# ----- RepViT block --------------------------------------------------------- #


class RepViTBlock(nn.Module):
    """Token mixer (RepDWConv with implicit residual) + SE + channel mixer.

    For stride-2 (downsampling) blocks the token mixer has no identity branch
    (no residual is possible), and a `ChannelTransition` follows to widen the
    channels for the next stage.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True,
        expand_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.token_mixer = RepDWConv(in_channels, stride=stride)
        self.se = SqueezeExcite(in_channels) if use_se else nn.Identity()
        self.transition = (
            ChannelTransition(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.channel_mixer = ChannelMixer(out_channels, expand_ratio=expand_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mixer(x)
        x = self.se(x)
        x = self.transition(x)
        x = self.channel_mixer(x)
        return x


# ----- stem ----------------------------------------------------------------- #


class Stem(nn.Module):
    """Two stride-2 3x3 convs with BN+GELU, taking RGB to (out_channels, H/4, W/4)."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(3, mid, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


# ----- full backbone -------------------------------------------------------- #


class RepViT(nn.Module):
    """4-stage RepViT backbone returning C3, C4, C5 feature maps."""

    def __init__(
        self,
        channels: Sequence[int] = (48, 96, 192, 384),
        depths: Sequence[int] = (2, 2, 14, 2),
        expand_ratio: int = 2,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or len(depths) != 4:
            raise ValueError("channels and depths must each have length 4")

        self.out_channels = (channels[1], channels[2], channels[3])

        self.stem = Stem(channels[0])

        # Stage 1 lives at the stem stride (no further downsample inside).
        self.stage1 = self._build_stage(
            in_ch=channels[0], out_ch=channels[0], depth=depths[0],
            downsample=False, expand_ratio=expand_ratio,
        )
        # Stages 2-4: first block downsamples and changes channels.
        self.stage2 = self._build_stage(
            in_ch=channels[0], out_ch=channels[1], depth=depths[1],
            downsample=True, expand_ratio=expand_ratio,
        )
        self.stage3 = self._build_stage(
            in_ch=channels[1], out_ch=channels[2], depth=depths[2],
            downsample=True, expand_ratio=expand_ratio,
        )
        self.stage4 = self._build_stage(
            in_ch=channels[2], out_ch=channels[3], depth=depths[3],
            downsample=True, expand_ratio=expand_ratio,
        )

        self._init_weights()

    @staticmethod
    def _build_stage(
        in_ch: int, out_ch: int, depth: int, downsample: bool, expand_ratio: int,
    ) -> nn.Sequential:
        blocks: list[nn.Module] = []
        if downsample:
            # First block: stride-2 token mixer. Channel widening happens via
            # the transition INSIDE the block (in_ch -> out_ch).
            blocks.append(RepViTBlock(in_ch, out_ch, stride=2, expand_ratio=expand_ratio))
            remaining = depth - 1
        else:
            remaining = depth
        for _ in range(remaining):
            blocks.append(RepViTBlock(out_ch, out_ch, stride=1, expand_ratio=expand_ratio))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)         # stride 4
        c3 = self.stage2(x)        # stride 8
        c4 = self.stage3(c3)       # stride 16
        c5 = self.stage4(c4)       # stride 32
        return c3, c4, c5

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        """Fuse every `RepDWConv` into a single deployable conv. Idempotent."""
        for m in self.modules():
            if isinstance(m, RepDWConv):
                m.switch_to_deploy()

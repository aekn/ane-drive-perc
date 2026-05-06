"""FastViT-T8 backbone for the ANE detector family.

Thin wrapper over the timm implementation of Apple's FastViT, which already
supports loading Apple's ImageNet pretrained weights and the reparameterization
fold used at export time. We expose the three detection scales (strides 8/16/32)
in ``(B, C, H, W)`` layout — the encoder converts to the ANE channels-second
layout where attention happens.

Channels per stage (FastViT-T8): ``[96, 192, 384]`` at strides ``[8, 16, 32]``.
Total backbone params at this configuration: ~3.17M.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from loguru import logger


__all__ = ["FastViTBackbone", "build_backbone"]


_VARIANT_TO_TIMM = {
    "t8": "fastvit_t8",
    "t12": "fastvit_t12",
    "s12": "fastvit_s12",
    "sa12": "fastvit_sa12",
}


class FastViTBackbone(nn.Module):
    """FastViT backbone returning multi-scale BCHW feature maps."""

    def __init__(
        self,
        variant: str = "t8",
        pretrained: bool = True,
        out_indices: Sequence[int] = (1, 2, 3),
    ) -> None:
        super().__init__()
        if variant not in _VARIANT_TO_TIMM:
            raise ValueError(
                f"unknown FastViT variant {variant!r}; "
                f"supported: {list(_VARIANT_TO_TIMM)}"
            )

        import timm

        timm_name = _VARIANT_TO_TIMM[variant]
        self.variant = variant
        self.model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )

        self._channels: tuple[int, ...] = tuple(self.model.feature_info.channels())
        self._strides: tuple[int, ...] = tuple(self.model.feature_info.reduction())

        if pretrained:
            logger.info(
                f"FastViT-{variant} loaded with ImageNet pretrained weights "
                f"(channels={self._channels}, strides={self._strides})"
            )

    @property
    def out_channels(self) -> tuple[int, ...]:
        return self._channels

    @property
    def out_strides(self) -> tuple[int, ...]:
        return self._strides

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model(x)

    def deploy(self) -> "FastViTBackbone":
        """Reparameterize every supporting submodule (RepMixer, fused convs).

        Call this before tracing for export; it's idempotent if no module has a
        ``reparameterize``/``fuse``/``switch_to_deploy`` method.
        """
        self.eval()
        for module in self.model.modules():
            for fn_name in ("reparameterize", "switch_to_deploy", "fuse"):
                fn = getattr(module, fn_name, None)
                if callable(fn):
                    fn()
                    break
        return self


def build_backbone(
    *,
    variant: str = "t8",
    pretrained: bool = True,
) -> FastViTBackbone:
    return FastViTBackbone(variant=variant, pretrained=pretrained)

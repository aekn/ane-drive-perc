import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adp.model.dfine.arch.utils import distance2bbox


__all__ = [
    "ANEMLP",
    "ANEClsHead",
    "ANEFGLBoxHead",
    "fgl_decode_boxes",
    "fgl_weighting_function",
]


def fgl_weighting_function(
    reg_max: int, up: Tensor, reg_scale: Tensor | float
) -> Tensor:
    if reg_max < 4 or reg_max % 2 != 0:
        raise ValueError(f"reg_max must be even and >= 4, got {reg_max}")

    u = torch.as_tensor(up).reshape(-1)[0].abs()
    s = torch.as_tensor(reg_scale, dtype=u.dtype, device=u.device).reshape(-1)[0].abs()

    upper1 = u * s
    upper2 = u * s * 2.0
    step = (upper1 + 1.0) ** (2.0 / (reg_max - 2))

    half = reg_max // 2
    values = [-upper2.reshape(1)]
    for i in range(half - 1, 0, -1):
        values.append((-(step**i) + 1.0).reshape(1))
    values.append(torch.zeros(1, dtype=u.dtype, device=u.device))
    for i in range(1, half):
        values.append((step**i - 1.0).reshape(1))
    values.append(upper2.reshape(1))

    return torch.cat(values, dim=0)


class ANEMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        act: str = "relu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Conv2d(dims[i], dims[i + 1], kernel_size=1) for i in range(num_layers)
        )
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"unsupported activation {act!r}")

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x


class ANEClsHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def forward(self, x_ane: Tensor) -> Tensor:
        # (B, C, 1, Q) -> (B, K, 1, Q) -> (B, Q, K)
        logits = self.proj(x_ane).squeeze(2).transpose(1, 2).contiguous()
        return logits


class ANEFGLBoxHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.mlp = ANEMLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=4 * (reg_max + 1),
            num_layers=3,
            act="relu",
        )

    def forward(self, x_ane: Tensor) -> Tensor:
        # (B, C, 1, Q) -> (B, 4*(reg_max+1), 1, Q) -> (B, Q, 4*(reg_max+1))
        out = self.mlp(x_ane).squeeze(2).transpose(1, 2).contiguous()
        return out


def fgl_decode_boxes(
    pred_corners: Tensor,
    ref_points: Tensor,
    *,
    reg_max: int,
    project: Tensor,
    reg_scale: float | Tensor,
) -> Tensor:
    B, Q, _ = pred_corners.shape
    prob = F.softmax(pred_corners.reshape(B, Q, 4, reg_max + 1), dim=-1)
    distance = (prob * project.to(prob.device).view(1, 1, 1, -1)).sum(
        dim=-1
    )  # (B, Q, 4)
    return distance2bbox(ref_points, distance, reg_scale=reg_scale)

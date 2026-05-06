"""ANE-optimized detector family.

Architecture: FastViT-T8/T12 backbone (Apple ImageNet weights) + ANE-optimized
hybrid encoder (Conv2d 1x1, ANE LayerNorm, split-softmax) + DFINE-shape decoder
with vanilla content-based cross-attention for full ANE residency.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from adp.model.ane.backbone import FastViTBackbone
from adp.model.ane.decoder_full import ANEDecoder
from adp.model.ane.detector import ANEDetector
from adp.model.ane.encoder import ANEHybridEncoder
from adp.model.dfine import build_optimizer as build_optimizer_dfine
from adp.model.dfine.criterion import DFINECriterion
from adp.model.dfine.matcher import HungarianMatcher
from adp.model.dfine.utils import ensure_pretrained, load_tuning_state


__all__ = ["build_model", "build_loss", "build_optimizer", "ANE_CONFIGS"]


ANE_CONFIGS: dict[str, dict] = {
    "n": {
        "backbone_variant": "t8",
        "embed_dim": 128,
        "num_queries": 200,
        "num_layers": 3,
        "num_heads": 8,
        "ffn_ratio": 4.0,
        "reg_max": 32,
        "reg_scale": 4.0,
    },
    "s": {
        "backbone_variant": "t8",
        "embed_dim": 192,
        "num_queries": 300,
        "num_layers": 3,
        "num_heads": 8,
        "ffn_ratio": 4.0,
        "reg_max": 32,
        "reg_scale": 4.0,
    },
    "m": {
        "backbone_variant": "t12",
        "embed_dim": 256,
        "num_queries": 300,
        "num_layers": 4,
        "num_heads": 8,
        "ffn_ratio": 4.0,
        "reg_max": 32,
        "reg_scale": 4.0,
    },
}


ANE_LOSS_CFG: dict = {
    "matcher": {
        "weight_dict": {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        "alpha": 0.25,
        "gamma": 2.0,
        "use_focal_loss": True,
    },
    "criterion": {
        "weight_dict": {
            "loss_vfl": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        },
        "losses": ["vfl", "boxes", "local"],
        "alpha": 0.75,
        "gamma": 2.0,
    },
}


def build_model(
    model_name: str,
    num_classes: int,
    enable_mask_head: bool = False,
    device: str | torch.device = "cpu",
    img_size: list[int] | tuple[int, int] | None = None,
    pretrained_model_path: str | Path | None = None,
    *,
    backbone_pretrained: bool = True,
) -> nn.Module:
    if enable_mask_head:
        raise NotImplementedError("ANE detector does not support mask head yet.")
    if model_name not in ANE_CONFIGS:
        raise KeyError(
            f"unknown ANE size {model_name!r}; available: {list(ANE_CONFIGS)}"
        )

    cfg = deepcopy(ANE_CONFIGS[model_name])

    backbone = FastViTBackbone(
        variant=cfg["backbone_variant"],
        pretrained=backbone_pretrained,
    )
    encoder = ANEHybridEncoder(
        in_channels=backbone.out_channels,
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        ffn_ratio=cfg["ffn_ratio"],
        eval_spatial_size=tuple(img_size) if img_size is not None else None,
    )
    decoder = ANEDecoder(
        num_classes=num_classes,
        embed_dim=cfg["embed_dim"],
        num_queries=cfg["num_queries"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        ffn_ratio=cfg["ffn_ratio"],
        reg_max=cfg["reg_max"],
        reg_scale=cfg["reg_scale"],
        num_levels=len(backbone.out_channels),
        eval_spatial_size=tuple(img_size) if img_size is not None else None,
    )

    model = ANEDetector(backbone=backbone, encoder=encoder, decoder=decoder)

    if pretrained_model_path is not None:
        resolved = ensure_pretrained(str(pretrained_model_path))
        if not Path(resolved).exists():
            raise FileNotFoundError(f"{pretrained_model_path} does not exist")
        model = load_tuning_state(model, resolved)

    return model.to(device)


def build_loss(
    model_name: str,
    num_classes: int,
    label_smoothing: float = 0.0,
    enable_mask_head: bool = False,
) -> nn.Module:
    if enable_mask_head:
        raise NotImplementedError("ANE criterion does not include mask losses in v1")
    if model_name not in ANE_CONFIGS:
        raise KeyError(f"unknown ANE size {model_name!r}")

    reg_max = ANE_CONFIGS[model_name]["reg_max"]

    matcher = HungarianMatcher(**ANE_LOSS_CFG["matcher"])
    return DFINECriterion(
        matcher,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
        reg_max=reg_max,
        **ANE_LOSS_CFG["criterion"],
    )


def build_optimizer(*args, **kwargs):
    """Re-export D-FINE's parameter-grouped AdamW."""
    return build_optimizer_dfine(*args, **kwargs)

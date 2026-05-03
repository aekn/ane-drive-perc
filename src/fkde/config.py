"""Plain-YAML config loader.

Loader returns Config dataclass with type-checked sections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataCfg:
    manifest_path: str
    images_dir: str
    label_json: str
    val_manifest_path: str | None = None
    val_images_dir: str | None = None
    val_label_json: str | None = None
    image_size: int = 384


@dataclass
class ModelCfg:
    num_classes: int = 10
    fpn_channels: int = 256
    backbone_channels: tuple[int, int, int, int] = (48, 96, 192, 384)
    backbone_depths: tuple[int, int, int, int] = (2, 2, 14, 2)


@dataclass
class OptimCfg:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    warmup_epochs: int = 1
    optimizer: str = "adamw"  # "adamw" | "sgd"
    grad_clip: float = 10.0


@dataclass
class TrainCfg:
    epochs: int = 50
    batch_size: int = 8
    num_workers: int = 4
    device: str = "auto"  # "auto" | "cpu" | "mps" | "cuda"
    seed: int = 0
    log_interval: int = 50
    out_dir: str = "runs/cell_a"
    eval_interval: int = 1  # epochs; 0 disables validation
    val_batch_size: int = 16
    save_best: bool = True
    resume: bool = True  # auto-resume from `last.pt` if present
    score_threshold: float = 0.05
    nms_iou: float = 0.6
    amp: bool = False


@dataclass
class LossYAML:
    center_sampling_radius: float = 1.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    area_a_ref: float = 1024.0
    area_beta: float = 0.5
    area_w_max: float = 4.0
    lambda_cls_match: float = 1.0
    lambda_box_match: float = 2.5


@dataclass
class Config:
    name: str
    data: DataCfg
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    loss: LossYAML = field(default_factory=LossYAML)


def _coerce(dc, raw: dict[str, Any]):
    return dc(**raw)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if "name" not in raw or "data" not in raw:
        raise ValueError("config must include `name` and `data` sections")
    return Config(
        name=raw["name"],
        data=_coerce(DataCfg, raw["data"]),
        model=_coerce(ModelCfg, raw.get("model", {})),
        optim=_coerce(OptimCfg, raw.get("optim", {})),
        train=_coerce(TrainCfg, raw.get("train", {})),
        loss=_coerce(LossYAML, raw.get("loss", {})),
    )

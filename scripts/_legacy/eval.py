"""Standalone evaluation of a checkpoint against the val set.

Usage:
    uv run python scripts/eval.py configs/cell_a.yaml runs/cell_a/best.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fkde.config import load_config
from fkde.data.dataset import BDDDetection, detection_collate
from fkde.engine.eval import evaluate
from fkde.engine.train import _resolve_device
from fkde.models.detector import Detector, DetectorConfig


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <config.yaml> <ckpt.pt>", file=sys.stderr)
        sys.exit(2)
    cfg = load_config(Path(sys.argv[1]))
    ckpt_path = Path(sys.argv[2])

    device = _resolve_device(cfg.train.device)
    print(f"[init] device={device}  ckpt={ckpt_path}")

    val_ds = BDDDetection(
        manifest_path=cfg.data.val_manifest_path,
        images_dir=cfg.data.val_images_dir,
        label_json=cfg.data.val_label_json,
        image_size=cfg.data.image_size,
        train=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.val_batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    print(f"[data] val images: {len(val_ds)}")

    det_cfg = DetectorConfig(
        num_classes=cfg.model.num_classes,
        fpn_channels=cfg.model.fpn_channels,
        backbone_channels=tuple(cfg.model.backbone_channels),
        backbone_depths=tuple(cfg.model.backbone_depths),
    )
    model = Detector(det_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    metrics = evaluate(
        model,
        val_loader,
        device,
        image_size=cfg.data.image_size,
        score_threshold=cfg.train.score_threshold,
        nms_iou=cfg.train.nms_iou,
    )
    print(
        f"mAP={metrics['map']:.4f}  mAP50={metrics['map_50']:.4f}  mAP75={metrics['map_75']:.4f}"
    )


if __name__ == "__main__":
    main()

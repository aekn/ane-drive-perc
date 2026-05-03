"""Single-process training loop.

last.pt: saved every epoch.
best.pt: saved when val mAP improves.

If resume is true and last.pt exists, picks up from saved epoch.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import BDDDetection, detection_collate
from ..models.detector import Detector, DetectorConfig
from .anchors import FPN_STRIDES, all_level_points
from .eval import evaluate
from .losses import LossCfg, detection_loss, progloss_weights


def _resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_optimizer(model: torch.nn.Module, opt: dict) -> torch.optim.Optimizer:
    if opt["optimizer"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=opt["lr"], weight_decay=opt["weight_decay"]
        )
    if opt["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=opt["lr"],
            momentum=opt["momentum"],
            weight_decay=opt["weight_decay"],
        )
    raise ValueError(f"unknown optimizer {opt['optimizer']}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
) -> LambdaLR:
    """Linear warmup -> cosine decay to 1% of base LR."""
    total = steps_per_epoch * epochs
    warmup = steps_per_epoch * max(warmup_epochs, 0)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        if total <= warmup:
            return 1.0
        progress = (step - warmup) / (total - warmup)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _save_ckpt(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    best_map: float,
    cfg_name: str,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "best_map": best_map,
        "config": cfg_name,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train_cell(cfg: Config) -> None:
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "name": cfg.name,
                "data": asdict(cfg.data),
                "model": asdict(cfg.model),
                "optim": asdict(cfg.optim),
                "train": asdict(cfg.train),
                "loss": asdict(cfg.loss),
            },
            f,
            indent=2,
        )

    torch.manual_seed(cfg.train.seed)
    device = _resolve_device(cfg.train.device)
    print(f"[init] device={device}  out_dir={out_dir}")

    train_ds = BDDDetection(
        manifest_path=cfg.data.manifest_path,
        images_dir=cfg.data.images_dir,
        label_json=cfg.data.label_json,
        image_size=cfg.data.image_size,
        train=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=detection_collate,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
    print(f"[data] train images: {len(train_ds)}  steps/epoch: {len(train_loader)}")

    val_loader: DataLoader | None = None
    if (
        cfg.train.eval_interval > 0
        and cfg.data.val_manifest_path
        and cfg.data.val_images_dir
        and cfg.data.val_label_json
    ):
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
            persistent_workers=cfg.train.num_workers > 0,
        )
        print(f"[data] val images: {len(val_ds)}")
    else:
        print("[data] no validation set configured")

    det_cfg = DetectorConfig(
        num_classes=cfg.model.num_classes,
        fpn_channels=cfg.model.fpn_channels,
        backbone_channels=tuple(cfg.model.backbone_channels),  # type: ignore[arg-type]
        backbone_depths=tuple(cfg.model.backbone_depths),  # type: ignore[arg-type]
    )
    model = Detector(det_cfg).to(device)

    optim_dict = asdict(cfg.optim)
    optimizer = _build_optimizer(model, optim_dict)
    scheduler = _build_scheduler(
        optimizer,
        steps_per_epoch=len(train_loader),
        epochs=cfg.train.epochs,
        warmup_epochs=cfg.optim.warmup_epochs,
    )

    loss_cfg = LossCfg(
        num_classes=cfg.model.num_classes,
        center_sampling_radius=cfg.loss.center_sampling_radius,
        focal_alpha=cfg.loss.focal_alpha,
        focal_gamma=cfg.loss.focal_gamma,
        area_a_ref=cfg.loss.area_a_ref,
        area_beta=cfg.loss.area_beta,
        area_w_max=cfg.loss.area_w_max,
        lambda_cls_match=cfg.loss.lambda_cls_match,
        lambda_box_match=cfg.loss.lambda_box_match,
        strides=FPN_STRIDES,
    )

    start_epoch = 0
    best_map = -1.0
    last_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"
    if cfg.train.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
        if state.get("rng_cpu") is not None:
            torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        start_epoch = int(state["epoch"]) + 1
        best_map = float(state.get("best_map", -1.0))
        print(f"[resume] last.pt -> start epoch {start_epoch}, best mAP {best_map:.4f}")

    global_step = start_epoch * len(train_loader)
    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "a")

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        epoch_t0 = time.time()
        pl_w = progloss_weights(epoch, cfg.train.epochs)

        for step, (images, boxes, labels, _names) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=cfg.train.amp and device.type == "cuda",
            ):
                cls_logits, bbox_preds, ctr = model(images)
                shapes = [(l.shape[-2], l.shape[-1]) for l in cls_logits]
                points = all_level_points(shapes, FPN_STRIDES, device)

                out = detection_loss(
                    cls_logits,
                    bbox_preds,
                    ctr,
                    points,
                    boxes,
                    labels,
                    loss_cfg,
                    pl_w,
                )
                loss = out.total

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at step {global_step}: {loss.item()}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            scheduler.step()

            if global_step % cfg.train.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                rec = {
                    "epoch": epoch,
                    "step": global_step,
                    "lr": lr,
                    "loss": loss.item(),
                    "o2m_cls": out.o2m_cls.item(),
                    "o2m_reg": out.o2m_reg.item(),
                    "o2m_ctr": out.o2m_ctr.item(),
                    "o2o_cls": out.o2o_cls.item(),
                    "o2o_reg": out.o2o_reg.item(),
                    "o2o_ctr": out.o2o_ctr.item(),
                    "num_gt": out.num_gt,
                    "npos_o2m": out.num_pos_o2m,
                    "npos_o2o": out.num_pos_o2o,
                    "w_o2m": pl_w[0],
                    "w_o2o": pl_w[1],
                }
                print(
                    f"[ep {epoch:>3} st {step:>5}/{len(train_loader)}] "
                    f"loss={loss.item():.3f} "
                    f"o2m(c/r/ctr)={out.o2m_cls.item():.3f}/"
                    f"{out.o2m_reg.item():.3f}/{out.o2m_ctr.item():.3f} "
                    f"o2o(c/r/ctr)={out.o2o_cls.item():.3f}/"
                    f"{out.o2o_reg.item():.3f}/{out.o2o_ctr.item():.3f} "
                    f"npos={out.num_pos_o2m}/{out.num_pos_o2o} "
                    f"lr={lr:.2e}"
                )
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()

            global_step += 1

        epoch_dt = time.time() - epoch_t0

        val_metrics: dict[str, float] = {}
        if val_loader is not None and (
            (epoch + 1) % cfg.train.eval_interval == 0 or epoch + 1 == cfg.train.epochs
        ):
            v_t0 = time.time()
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                image_size=cfg.data.image_size,
                score_threshold=cfg.train.score_threshold,
                nms_iou=cfg.train.nms_iou,
                amp=cfg.train.val_amp,
            )
            v_dt = time.time() - v_t0
            print(
                f"[ep {epoch}] val mAP={val_metrics['map']:.4f} "
                f"mAP50={val_metrics['map_50']:.4f} mAP75={val_metrics['map_75']:.4f} "
                f"({v_dt:.1f}s)"
            )
            log_f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "val": val_metrics,
                    }
                )
                + "\n"
            )
            log_f.flush()

        _save_ckpt(last_path, epoch, model, optimizer, scheduler, best_map, cfg.name)
        if cfg.train.save_best and val_metrics:
            cur_map = val_metrics["map"]
            if cur_map > best_map:
                best_map = cur_map
                _save_ckpt(
                    best_path, epoch, model, optimizer, scheduler, best_map, cfg.name
                )
                print(f"[ep {epoch}] new best mAP={best_map:.4f} -> best.pt")

        print(f"[ep {epoch}] {epoch_dt:.1f}s  ckpt -> last.pt")

    log_f.close()
    print(f"[done] best mAP={best_map:.4f}")

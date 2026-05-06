import random as _random
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from adp.config import write_resolved_config
from adp.eval.coco_eval import CocoEvaluator
from adp.model.dfine.postprocess import DFINEPostProcessor
from adp.model.dfine.utils import ensure_pretrained
from adp.model.registry import get as get_model_spec
from adp.train.augment import build_train_transforms, build_val_transforms
from adp.train.dataset import CocoDetectionDataset, collate_fn
from adp.train.ema import ModelEMA
from adp.train.trainer import Trainer
from adp.utils.paths import ensure_run_paths


_AMP_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _set_seeds(seed: int) -> None:
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        logger.warning("mps requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(name)


def _resolve_amp_dtype(name: str) -> torch.dtype:
    try:
        return _AMP_DTYPES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown train.amp_dtype={name!r}. Expected one of {list(_AMP_DTYPES)}"
        ) from exc


def _build_train_transforms(cfg: DictConfig, img_size: tuple[int, int]):
    if bool(cfg.train.augment):
        return build_train_transforms(img_size)
    logger.info("train.augment=false; using validation transforms for train dataset")
    return build_val_transforms(img_size)


def _build_scheduler(
    *,
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler_name = str(cfg.train.scheduler)
    total_steps = steps_per_epoch * int(cfg.train.epochs)
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    if scheduler_name == "constant":
        logger.info("using constant LR scheduler")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    if scheduler_name == "onecycle":
        max_lrs = [
            float(group.get("initial_lr", group["lr"]))
            for group in optimizer.param_groups
        ]
        logger.info(f"using OneCycleLR scheduler with max_lrs={max_lrs}")
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=float(cfg.train.warmup_pct),
            anneal_strategy="cos",
        )

    raise ValueError(
        f"Unknown train.scheduler={scheduler_name!r}. Expected 'onecycle' or 'constant'."
    )


def _resolve_model_key(cfg: DictConfig) -> str:
    model_key = OmegaConf.select(cfg, "train.model", default=None)
    if model_key is not None:
        return str(model_key)

    legacy_size = OmegaConf.select(cfg, "train.model_name", default=None)
    if legacy_size is not None:
        return f"dfine_{legacy_size}"

    raise ValueError("config must set either train.model or train.model_name")


def _resolve_pretrained_path(cfg: DictConfig) -> str | None:
    raw = OmegaConf.select(cfg, "train.pretrained_path", default=None)
    if raw is None or str(raw).strip() == "" or str(raw).lower() == "null":
        return None
    return ensure_pretrained(str(raw))


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_paths = ensure_run_paths(cfg)
    write_resolved_config(cfg, run_paths.root / "config.resolved.yaml")

    _set_seeds(int(cfg.train.seed))

    device = _resolve_device(str(cfg.train.device))
    amp_dtype = _resolve_amp_dtype(str(cfg.train.amp_dtype))
    img_size = (int(cfg.train.img_size[0]), int(cfg.train.img_size[1]))

    logger.info(f"device: {device}")
    logger.info(f"amp dtype: {amp_dtype}")
    logger.info(f"image size: {img_size}")

    if amp_dtype is torch.float16:
        raise NotImplementedError(
            "fp16 AMP requires GradScaler; this trainer currently supports "
            "bfloat16 AMP or float32. Use train.amp_dtype=bfloat16 on CUDA "
            "or train.amp_dtype=float32."
        )

    model_key = _resolve_model_key(cfg)
    spec = get_model_spec(model_key)
    logger.info(f"model: {model_key}  (family={spec.family} size={spec.size})")

    pretrained_path = _resolve_pretrained_path(cfg)
    if pretrained_path is not None:
        logger.info(f"detector pretrained: {pretrained_path}")

    model = spec.build_model(
        num_classes=int(cfg.train.num_classes),
        enable_mask_head=False,
        device=device,
        img_size=list(img_size),
        pretrained_model_path=pretrained_path,
    )

    criterion = spec.build_loss(
        num_classes=int(cfg.train.num_classes),
        label_smoothing=float(cfg.train.label_smoothing),
        enable_mask_head=False,
    ).to(device)

    coco_root = Path(str(cfg.train.coco_root))
    if not coco_root.exists():
        raise FileNotFoundError(
            f"COCO export not found at {coco_root}. "
            "Run `uv run python -m adp.data.export_coco ...` first."
        )

    train_ds = CocoDetectionDataset(
        coco_root=coco_root,
        annotations_file=str(cfg.train.train_annotations),
        transforms=_build_train_transforms(cfg, img_size),
        category_id_base=int(cfg.coco.category_id_base),
    )
    val_ds = CocoDetectionDataset(
        coco_root=coco_root,
        annotations_file=str(cfg.train.val_annotations),
        transforms=build_val_transforms(img_size),
        category_id_base=int(cfg.coco.category_id_base),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=int(cfg.train.num_workers) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=int(cfg.train.num_workers) > 0,
    )

    if len(train_loader) == 0:
        raise ValueError(
            "Train loader has zero batches. Lower train.batch_size or disable drop_last."
        )

    logger.info(f"train images: {len(train_ds)}; val images: {len(val_ds)}")

    optimizer = spec.build_optimizer(
        model=model,
        lr=float(cfg.train.base_lr),
        backbone_lr=float(cfg.train.backbone_lr),
        betas=tuple(float(b) for b in cfg.train.betas),
        weight_decay=float(cfg.train.weight_decay),
        base_lr=float(cfg.train.base_lr),
    )

    scheduler = _build_scheduler(
        cfg=cfg, optimizer=optimizer, steps_per_epoch=len(train_loader)
    )

    ema = (
        ModelEMA(model, momentum=float(cfg.train.ema_momentum))
        if bool(cfg.train.use_ema)
        else None
    )

    postprocessor = DFINEPostProcessor(
        num_classes=int(cfg.train.num_classes),
        num_top_queries=int(cfg.train.num_top_queries),
    ).to(device)

    evaluator = CocoEvaluator(
        annotations_path=coco_root / "annotations" / str(cfg.train.val_annotations),
        category_id_base=int(cfg.coco.category_id_base),
    )

    log_path = run_paths.root / "train.log"
    logger.add(str(log_path), enqueue=True)
    logger.info(f"run dir: {run_paths.root}")

    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        postprocessor=postprocessor,
        evaluator=evaluator,
        device=device,
        amp_dtype=amp_dtype,
        epochs=int(cfg.train.epochs),
        clip_grad_norm=float(cfg.train.clip_grad_norm),
        eval_every=int(cfg.train.eval_every),
        nan_recovery=bool(cfg.train.nan_recovery),
        nan_max_consecutive=int(cfg.train.nan_max_consecutive),
        checkpoints_dir=Path(str(cfg.train.checkpoints_dir)),
        metrics_path=Path(str(cfg.train.metrics_path)),
        freeze_backbone_epochs=int(cfg.train.freeze_backbone_epochs),
        score_thresholds=tuple(float(x) for x in cfg.train.score_thresholds),
    )

    trainer.fit()
    logger.info("training complete")


if __name__ == "__main__":
    main()

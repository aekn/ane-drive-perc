import random as _random
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from adp.config import write_resolved_config
from adp.distill.losses import (
    AIFIFeatureDistillLoss,
    BoxDistillLoss,
    ClsDistillLoss,
    FDRDistributionDistillLoss,
)
from adp.distill.trainer import DistillTrainer
from adp.eval.coco_eval import CocoEvaluator
from adp.model.dfine.postprocess import DFINEPostProcessor
from adp.model.dfine.utils import ensure_pretrained
from adp.model.registry import get as get_model_spec
from adp.train.augment import build_train_transforms, build_val_transforms
from adp.train.dataset import CocoDetectionDataset, collate_fn
from adp.train.ema import ModelEMA
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


def _build_train_transforms(cfg: DictConfig, img_size: tuple[int, int]):
    if bool(cfg.train.augment):
        return build_train_transforms(img_size)
    return build_val_transforms(img_size)


def _build_scheduler(cfg, optimizer, steps_per_epoch):
    total_steps = steps_per_epoch * int(cfg.train.epochs)
    if str(cfg.train.scheduler) == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    max_lrs = [float(g.get("initial_lr", g["lr"])) for g in optimizer.param_groups]
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=float(cfg.train.warmup_pct),
        anneal_strategy="cos",
    )


def _load_teacher(
    *,
    teacher_key: str,
    checkpoint_path: Path,
    use_ema: bool,
    num_classes: int,
    img_size: list[int],
    device: torch.device,
) -> torch.nn.Module:
    spec = get_model_spec(teacher_key)
    teacher = spec.build_model(
        num_classes=num_classes,
        enable_mask_head=False,
        device=device,
        img_size=img_size,
        pretrained_model_path=None,
    )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if use_ema and "ema" in payload:
        state = payload["ema"]
        logger.info(f"loading teacher EMA weights from {checkpoint_path}")
    else:
        state = payload["model"] if "model" in payload else payload
        logger.info(f"loading teacher non-EMA weights from {checkpoint_path}")

    teacher_state = teacher.state_dict()
    filtered = {}
    dropped = []
    for k, v in state.items():
        if k in teacher_state and teacher_state[k].shape != v.shape:
            dropped.append(k)
            continue
        filtered[k] = v
    if dropped:
        logger.warning(
            f"dropping {len(dropped)} resolution-dependent teacher buffers: {dropped[:4]}..."
        )
    teacher.load_state_dict(filtered, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _peek_aifi_channels(
    model: torch.nn.Module, img_size: list[int], device: torch.device
) -> int:
    """Probe the deepest-scale (1/32) encoder output channel count."""
    h, w = int(img_size[0]), int(img_size[1])
    dummy = torch.zeros(1, 3, h, w, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        feats = model.backbone(dummy)
        enc_in_channels = getattr(model.encoder, "in_channels", None)
        if enc_in_channels is not None and len(feats) > len(enc_in_channels):
            feats = feats[1:]
        enc_feats = model.encoder(feats)
    if was_training:
        model.train()
    return int(enc_feats[-1].shape[1])


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_paths = ensure_run_paths(cfg)
    write_resolved_config(cfg, run_paths.root / "config.resolved.yaml")

    if "distill" not in cfg:
        raise ValueError(
            "distill config block is required; compose with +experiment=ane_s_bdd_distill"
        )

    _set_seeds(int(cfg.train.seed))
    device = _resolve_device(str(cfg.train.device))
    amp_dtype = _AMP_DTYPES[str(cfg.train.amp_dtype)]
    img_size = (int(cfg.train.img_size[0]), int(cfg.train.img_size[1]))

    student_key = str(cfg.train.model)
    teacher_key = str(cfg.distill.teacher_model)
    logger.info(f"distill: teacher={teacher_key} -> student={student_key}")

    # build student.
    student_spec = get_model_spec(student_key)
    pretrained_raw = OmegaConf.select(cfg, "train.pretrained_path", default=None)
    pretrained_path = (
        ensure_pretrained(str(pretrained_raw))
        if pretrained_raw not in (None, "", "null")
        else None
    )
    model = student_spec.build_model(
        num_classes=int(cfg.train.num_classes),
        enable_mask_head=False,
        device=device,
        img_size=list(img_size),
        pretrained_model_path=pretrained_path,
    )
    criterion = student_spec.build_loss(
        num_classes=int(cfg.train.num_classes),
        label_smoothing=float(cfg.train.label_smoothing),
        enable_mask_head=False,
    ).to(device)

    # build the frozen teacher from ckpt
    teacher_ckpt = Path(str(cfg.distill.teacher_checkpoint))
    if not teacher_ckpt.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {teacher_ckpt}")
    teacher = _load_teacher(
        teacher_key=teacher_key,
        checkpoint_path=teacher_ckpt,
        use_ema=bool(OmegaConf.select(cfg, "distill.use_teacher_ema", default=True)),
        num_classes=int(cfg.train.num_classes),
        img_size=list(img_size),
        device=device,
    )

    # check AIFI channel counts on both networks.
    student_aifi_ch = _peek_aifi_channels(model, list(img_size), device)
    teacher_aifi_ch = _peek_aifi_channels(teacher, list(img_size), device)
    logger.info(f"AIFI channels: student={student_aifi_ch} teacher={teacher_aifi_ch}")

    # build distillation loss modules (parameters live in AIFI projector).
    box_distill_loss = BoxDistillLoss().to(device)
    cls_distill_loss = ClsDistillLoss(
        temperature=float(OmegaConf.select(cfg, "distill.cls_temperature", default=2.0))
    ).to(device)
    fdr_distill_loss = FDRDistributionDistillLoss(
        reg_max=int(OmegaConf.select(cfg, "distill.reg_max", default=32)),
        temperature=float(OmegaConf.select(cfg, "distill.fdr_temperature", default=1.0)),
    ).to(device)
    aifi_feat_distill_loss = AIFIFeatureDistillLoss(
        student_channels=student_aifi_ch,
        teacher_channels=teacher_aifi_ch,
    ).to(device)

    coco_root = Path(str(cfg.train.coco_root))
    if not coco_root.exists():
        raise FileNotFoundError(f"COCO export not found at {coco_root}")
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
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.train.batch_size), shuffle=True,
        num_workers=int(cfg.train.num_workers), collate_fn=collate_fn,
        pin_memory=pin, drop_last=True,
        persistent_workers=int(cfg.train.num_workers) > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.train.batch_size), shuffle=False,
        num_workers=int(cfg.train.num_workers), collate_fn=collate_fn,
        pin_memory=pin,
        persistent_workers=int(cfg.train.num_workers) > 0,
    )
    logger.info(f"train images: {len(train_ds)}; val images: {len(val_ds)}")

    optimizer = student_spec.build_optimizer(
        model=model,
        lr=float(cfg.train.base_lr),
        backbone_lr=float(cfg.train.backbone_lr),
        betas=tuple(float(b) for b in cfg.train.betas),
        weight_decay=float(cfg.train.weight_decay),
        base_lr=float(cfg.train.base_lr),
    )
    distill_params = (
        list(box_distill_loss.parameters())
        + list(cls_distill_loss.parameters())
        + list(fdr_distill_loss.parameters())
        + list(aifi_feat_distill_loss.parameters())
    )
    distill_params = [p for p in distill_params if p.requires_grad]
    if distill_params:
        base_lr = float(cfg.train.base_lr)
        optimizer.add_param_group(
            {
                "params": distill_params,
                "lr": base_lr,
                "initial_lr": base_lr,
                "weight_decay": float(cfg.train.weight_decay),
            }
        )

    scheduler = _build_scheduler(cfg, optimizer, len(train_loader))

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

    trainer = DistillTrainer(
        teacher=teacher,
        box_distill_loss=box_distill_loss,
        cls_distill_loss=cls_distill_loss,
        fdr_distill_loss=fdr_distill_loss,
        aifi_feat_distill_loss=aifi_feat_distill_loss,
        gt_loss_weight=float(OmegaConf.select(cfg, "distill.gt_weight", default=1.0)),
        box_distill_weight=float(OmegaConf.select(cfg, "distill.box_weight", default=1.5)),
        cls_distill_weight=float(OmegaConf.select(cfg, "distill.cls_weight", default=0.5)),
        fdr_distill_weight=float(OmegaConf.select(cfg, "distill.fdr_weight", default=0.25)),
        aifi_distill_peak_weight=float(OmegaConf.select(cfg, "distill.aifi_peak_weight", default=20.0)),
        aifi_warmup_frac=float(OmegaConf.select(cfg, "distill.aifi_warmup_frac", default=0.10)),
        aifi_ramp_frac=float(OmegaConf.select(cfg, "distill.aifi_ramp_frac", default=0.20)),
        use_quality_weights=bool(OmegaConf.select(cfg, "distill.use_quality_weights", default=True)),
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
    logger.info("distillation training complete")


if __name__ == "__main__":
    main()

import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor, nn

from adp.distill.losses import (
    AIFIFeatureDistillLoss,
    BoxDistillLoss,
    ClsDistillLoss,
    FDRDistributionDistillLoss,
    aifi_ramp_schedule,
    compute_pair_quality_weights,
    gt_mediated_pairs,
)
from adp.train.trainer import Trainer


__all__ = ["DistillTrainer", "forward_with_features"]


def forward_with_features(
    model: nn.Module,
    images: Tensor,
    targets: list[dict[str, Tensor]] | None,
) -> tuple[dict, list[Tensor]]:
    feats = model.backbone(images)

    low_level_feat = None
    enc_in_channels = getattr(model.encoder, "in_channels", None)
    if enc_in_channels is not None and len(feats) > len(enc_in_channels):
        low_level_feat = feats[0]
        feats = feats[1:]

    enc_feats = model.encoder(feats)

    decoder_sig = inspect.signature(model.decoder.forward)
    decoder_kwargs: dict[str, Any] = {}
    if "low_level_feat" in decoder_sig.parameters:
        decoder_kwargs["low_level_feat"] = low_level_feat

    out = model.decoder(enc_feats, targets, **decoder_kwargs)
    return out, list(enc_feats)


class DistillTrainer(Trainer):
    def __init__(
        self,
        *,
        teacher: nn.Module,
        box_distill_loss: BoxDistillLoss,
        cls_distill_loss: ClsDistillLoss,
        fdr_distill_loss: FDRDistributionDistillLoss,
        aifi_feat_distill_loss: AIFIFeatureDistillLoss,
        gt_loss_weight: float = 1.0,
        box_distill_weight: float = 1.5,
        cls_distill_weight: float = 0.5,
        fdr_distill_weight: float = 0.25,
        aifi_distill_peak_weight: float = 20.0,
        aifi_warmup_frac: float = 0.10,
        aifi_ramp_frac: float = 0.20,
        use_quality_weights: bool = True,
        **trainer_kwargs,
    ) -> None:
        super().__init__(**trainer_kwargs)

        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.box_distill_loss = box_distill_loss.to(self.device)
        self.cls_distill_loss = cls_distill_loss.to(self.device)
        self.fdr_distill_loss = fdr_distill_loss.to(self.device)
        self.aifi_feat_distill_loss = aifi_feat_distill_loss.to(self.device)

        self.gt_loss_weight = float(gt_loss_weight)
        self.box_distill_weight = float(box_distill_weight)
        self.cls_distill_weight = float(cls_distill_weight)
        self.fdr_distill_weight = float(fdr_distill_weight)
        self.aifi_distill_peak_weight = float(aifi_distill_peak_weight)
        self.aifi_warmup_frac = float(aifi_warmup_frac)
        self.aifi_ramp_frac = float(aifi_ramp_frac)
        self.use_quality_weights = bool(use_quality_weights)

        self._total_steps = len(self.train_loader) * self.epochs

        adapter_params = sum(
            p.numel() for p in self.aifi_feat_distill_loss.parameters()
        )
        logger.info(
            f"DistillTrainer: gt={self.gt_loss_weight} "
            f"box={self.box_distill_weight} cls={self.cls_distill_weight} "
            f"fdr={self.fdr_distill_weight} "
            f"aifi_peak={self.aifi_distill_peak_weight} "
            f"(warmup={self.aifi_warmup_frac:.0%}, ramp={self.aifi_ramp_frac:.0%}) "
            f"quality_weights={self.use_quality_weights} "
            f"projector={adapter_params:,} params"
        )

    def distill_module_parameters(self) -> list[nn.Parameter]:
        params = (
            list(self.box_distill_loss.parameters())
            + list(self.cls_distill_loss.parameters())
            + list(self.fdr_distill_loss.parameters())
            + list(self.aifi_feat_distill_loss.parameters())
        )
        return [p for p in params if p.requires_grad]

    def _teacher_autocast(self):
        if self.device.type == "cuda" and self.amp_dtype != torch.float32:
            return torch.amp.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()

    def _compute_step_loss(
        self,
        images: Tensor,
        targets: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        # Student forward (with grad), capturing encoder pyramid for feature KD.
        student_out, student_feats = forward_with_features(self.model, images, targets)

        # GT supervised loss (D-FINE criterion: VFL + bbox + GIoU + FGL + DDF).
        gt_losses = self.criterion(student_out, targets)

        with torch.no_grad(), self._teacher_autocast():
            self.teacher.train()
            try:
                teacher_out, teacher_feats = forward_with_features(
                    self.teacher, images, targets
                )
            finally:
                self.teacher.eval()

            teacher_logits = teacher_out["pred_logits"].detach()
            teacher_boxes = teacher_out["pred_boxes"].detach()
            teacher_corners = teacher_out["pred_corners"].detach()
            teacher_aifi = teacher_feats[-1].detach()

        # GT-mediated matching: run matcher independently on student and teacher
        # final outputs; pair (student_q_for_GT, teacher_q_for_GT) per GT box.
        student_main = {
            "pred_logits": student_out["pred_logits"],
            "pred_boxes": student_out["pred_boxes"],
        }
        teacher_main = {"pred_logits": teacher_logits, "pred_boxes": teacher_boxes}
        student_indices = self.criterion.matcher(student_main, targets)["indices"]
        with torch.no_grad():
            teacher_indices = self.criterion.matcher(teacher_main, targets)["indices"]
        pairs = gt_mediated_pairs(student_indices, teacher_indices)
        n_pairs = sum(len(p) for p in pairs)

        pair_weights: list[Tensor] | None = None
        if self.use_quality_weights and n_pairs > 0:
            with torch.no_grad():
                pair_weights = compute_pair_quality_weights(
                    teacher_logits, teacher_boxes, targets, pairs
                )

        # Distillation losses.
        box_d = self.box_distill_loss(
            student_out["pred_boxes"], teacher_boxes, pairs, weights=pair_weights
        )
        cls_d = self.cls_distill_loss(
            student_out["pred_logits"], teacher_logits, pairs, weights=pair_weights
        )
        fdr_d = self.fdr_distill_loss(
            student_out["pred_corners"], teacher_corners, pairs, weights=pair_weights
        )
        aifi_d = self.aifi_feat_distill_loss(student_feats[-1], teacher_aifi)

        # AIFI ramp: 0 during early training increase 2 peak then stable
        aifi_weight_now = aifi_ramp_schedule(
            step=self._global_step,
            total_steps=self._total_steps,
            peak_weight=self.aifi_distill_peak_weight,
            warmup_frac=self.aifi_warmup_frac,
            ramp_frac=self.aifi_ramp_frac,
        )

        out: dict[str, Tensor] = {}
        for k, v in gt_losses.items():
            out[k] = self.gt_loss_weight * v
        out["loss_box_distill"] = self.box_distill_weight * box_d["loss_box_distill"]
        out["loss_cls_distill"] = self.cls_distill_weight * cls_d["loss_cls_distill"]
        out["loss_fdr_distill"] = self.fdr_distill_weight * fdr_d["loss_fdr_distill"]
        out["loss_aifi_cosine_distill"] = (
            aifi_weight_now * aifi_d["loss_aifi_cosine_distill"]
        )
        out["loss_box_l1_distill_telemetry"] = box_d["loss_box_l1_distill"].detach()
        out["loss_box_giou_distill_telemetry"] = box_d["loss_box_giou_distill"].detach()
        out["distill_pairs_count_telemetry"] = torch.tensor(float(n_pairs))
        out["aifi_weight_now_telemetry"] = torch.tensor(float(aifi_weight_now))
        if pair_weights is not None and n_pairs > 0:
            out["pair_quality_weight_mean_telemetry"] = (
                torch.cat(pair_weights).mean().detach()
            )

        return out

    def _save(self, name: str, *, epoch: int, metrics: dict[str, float]) -> None:
        super()._save(name, epoch=epoch, metrics=metrics)
        adapter_path = Path(self.checkpoints_dir) / f"{name}_distill.pt"
        torch.save(
            {
                "aifi_feat_distill_loss": self.aifi_feat_distill_loss.state_dict(),
                "box_distill_loss": self.box_distill_loss.state_dict(),
                "cls_distill_loss": self.cls_distill_loss.state_dict(),
                "fdr_distill_loss": self.fdr_distill_loss.state_dict(),
            },
            adapter_path,
        )

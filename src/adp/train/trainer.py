import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from adp.eval.coco_eval import CocoEvaluator
from adp.train.ema import ModelEMA


def _to_device(
    targets: list[dict[str, Tensor]], device: torch.device
) -> list[dict[str, Tensor]]:
    return [
        {k: v.to(device, non_blocking=True) for k, v in target.items()}
        for target in targets
    ]


class Trainer:
    """Standard supervised trainer for DETR-ilke detectors."""

    def __init__(
        self,
        *,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        ema: ModelEMA | None,
        postprocessor: nn.Module,
        evaluator: CocoEvaluator,
        device: torch.device,
        amp_dtype: torch.dtype,
        epochs: int,
        clip_grad_norm: float,
        eval_every: int,
        nan_recovery: bool,
        nan_max_consecutive: int,
        checkpoints_dir: Path,
        metrics_path: Path,
        freeze_backbone_epochs: int = 0,
        score_thresholds: tuple[float, ...] = (0.01, 0.05, 0.1, 0.3, 0.5),
        decision_metric: str = "mAP",
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.postprocessor = postprocessor
        self.evaluator = evaluator
        self.device = device
        self.amp_dtype = amp_dtype
        self.epochs = epochs
        self.clip_grad_norm = clip_grad_norm
        self.eval_every = eval_every
        self.nan_recovery = nan_recovery
        self.nan_max_consecutive = nan_max_consecutive
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.score_thresholds = score_thresholds

        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = Path(metrics_path)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        self.decision_metric = decision_metric
        self._best_score: float = -float("inf")
        self._global_step: int = 0
        self._start_epoch: int = 0
        self._consecutive_nan: int = 0
        self._backbone_frozen: bool | None = None

    def fit(self) -> None:
        for epoch in range(self._start_epoch, self.epochs):
            self._configure_backbone_for_epoch(epoch)

            t0 = time.time()
            train_stats = self._train_one_epoch(epoch)
            train_secs = time.time() - t0

            val_stats: dict[str, float] = {}
            should_eval = (epoch + 1) % self.eval_every == 0 or (
                epoch + 1
            ) == self.epochs
            if should_eval:
                val_stats = self._evaluate()

            combined_metrics = {**train_stats, **val_stats}

            self._log_epoch(epoch, train_stats, val_stats, train_secs)
            self._save("last", epoch=epoch, metrics=combined_metrics)

            score = val_stats.get(self.decision_metric)
            if score is not None and score > self._best_score + 1e-9:
                self._best_score = score
                self._save("best", epoch=epoch, metrics=combined_metrics)
                logger.info(
                    f"new best {self.decision_metric}={score:.4f} at epoch {epoch}"
                )

    def _configure_backbone_for_epoch(self, epoch: int) -> None:
        should_freeze = epoch < self.freeze_backbone_epochs

        if self._backbone_frozen is should_freeze:
            return

        matched = 0
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                param.requires_grad = not should_freeze
                matched += 1

        if matched == 0 and self.freeze_backbone_epochs > 0:
            logger.warning(
                "freeze_backbone_epochs > 0 but no parameters matched 'backbone'"
            )

        state = "frozen" if should_freeze else "trainable"
        logger.info(f"backbone parameters are now {state} ({matched} tensors matched)")
        self._backbone_frozen = should_freeze

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.criterion.train()

        loss_sum = 0.0
        loss_count = 0
        component_sums: dict[str, float] = {}

        progress = tqdm(
            self.train_loader,
            desc=f"epoch {epoch} train",
            leave=False,
            dynamic_ncols=True,
        )

        for images, targets in progress:
            images = images.to(self.device, non_blocking=True)
            targets = _to_device(targets, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast():
                loss_dict = self._compute_step_loss(images, targets)
                loss = sum(v for k, v in loss_dict.items() if not k.endswith("_telemetry"))

            if not torch.isfinite(loss):
                self._consecutive_nan += 1
                logger.warning(
                    f"non-finite loss at step={self._global_step}; "
                    f"consecutive={self._consecutive_nan}"
                )

                if (
                    self._consecutive_nan >= self.nan_max_consecutive
                    and self.nan_recovery
                ):
                    self._reload_last_for_recovery()
                    self._consecutive_nan = 0

                continue

            self._consecutive_nan = 0
            loss.backward()

            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.clip_grad_norm,
                )

            self.optimizer.step()
            self.scheduler.step()

            if self.ema is not None:
                self.ema.update(self.model, step=self._global_step)

            self._global_step += 1

            loss_value = float(loss.detach())
            loss_sum += loss_value
            loss_count += 1

            for key, value in loss_dict.items():
                component_sums[key] = component_sums.get(key, 0.0) + float(
                    value.detach()
                )

            progress.set_postfix(loss=f"{loss_value:.3f}")

        avg_loss = loss_sum / max(loss_count, 1)
        stats = {
            "loss": avg_loss,
            "lr": float(self.optimizer.param_groups[-1]["lr"]),
        }

        for key, value in sorted(component_sums.items()):
            stats[f"loss/{key}"] = value / max(loss_count, 1)

        return stats

    def _compute_step_loss(
        self,
        images: Tensor,
        targets: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        outputs = self.model(images, targets=targets)
        return self.criterion(outputs, targets)

    @torch.no_grad()
    def _evaluate(self) -> dict[str, float]:
        eval_model = self.ema.model if self.ema is not None else self.model
        eval_model.eval()

        self.evaluator.reset()

        pred_count = 0
        score_sum = 0.0
        score_max = 0.0
        threshold_counts = {threshold: 0 for threshold in self.score_thresholds}

        progress = tqdm(self.val_loader, desc="eval", leave=False, dynamic_ncols=True)

        for images, targets in progress:
            images = images.to(self.device, non_blocking=True)
            orig_sizes = torch.stack([target["orig_size"] for target in targets]).to(
                self.device
            )
            image_ids = [int(target["image_id"].item()) for target in targets]

            with self._autocast():
                outputs = eval_model(images)

            results = self.postprocessor(outputs, orig_sizes)

            for result in results:
                scores = result.get("scores")
                if scores is None:
                    continue

                scores = scores.detach().float().cpu()
                pred_count += int(scores.numel())

                if scores.numel() > 0:
                    score_sum += float(scores.sum())
                    score_max = max(score_max, float(scores.max()))

                for threshold in self.score_thresholds:
                    threshold_counts[threshold] += int((scores >= threshold).sum())

            self.evaluator.update(image_ids=image_ids, outputs=results)

        stats = self.evaluator.summarize()

        stats["pred/count"] = float(pred_count)
        stats["pred/score_mean"] = score_sum / max(pred_count, 1)
        stats["pred/score_max"] = score_max

        for threshold, count in sorted(threshold_counts.items()):
            key = f"pred/count_ge_{str(threshold).replace('.', '_')}"
            stats[key] = float(count)

        return stats

    def _autocast(self):
        if self.device.type == "cuda" and self.amp_dtype != torch.float32:
            return torch.amp.autocast("cuda", dtype=self.amp_dtype)

        return nullcontext()

    def _save(self, name: str, *, epoch: int, metrics: dict[str, float]) -> None:
        payload: dict[str, Any] = {
            "epoch": epoch,
            "global_step": self._global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_score": self._best_score,
            "metrics": metrics,
        }

        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()

        torch.save(payload, self.checkpoints_dir / f"{name}.pt")

    def _reload_last_for_recovery(self) -> None:
        last_path = self.checkpoints_dir / "last.pt"

        if not last_path.exists():
            logger.warning("nan recovery requested but no last.pt; skipping recovery")
            return

        logger.warning(f"reloading {last_path} after non-finite loss sequence")
        payload = torch.load(last_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])

        if self.ema is not None and "ema" in payload:
            self.ema.load_state_dict(payload["ema"])

    def _log_epoch(
        self,
        epoch: int,
        train_stats: dict[str, float],
        val_stats: dict[str, float],
        train_secs: float,
    ) -> None:
        record: dict[str, Any] = {
            "epoch": epoch,
            "step": self._global_step,
            "train_secs": round(train_secs, 1),
            **{f"train/{key}": value for key, value in train_stats.items()},
            **{f"val/{key}": value for key, value in val_stats.items()},
        }

        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        message = (
            f"epoch {epoch} | {train_secs:.0f}s | loss={train_stats.get('loss', 0):.3f}"
        )

        if val_stats:
            message += (
                f" | mAP={val_stats.get('mAP', 0):.4f}"
                f" | mAP50={val_stats.get('mAP_50', 0):.4f}"
                f" | preds={val_stats.get('pred/count', 0):.0f}"
                f" | max_score={val_stats.get('pred/score_max', 0):.3f}"
            )

        logger.info(message)

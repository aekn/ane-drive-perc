import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.optim import Optimizer

MonitorMode = Literal["min", "max"]


@dataclass(frozen=True)
class CheckpointResult:
    path: Path
    is_best: bool
    metric_name: str | None
    metric_value: float | None
    best_value: float | None


class CheckpointManager:
    """Metric aware checkpoint manager shared by all trainers."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        monitor: str | None = None,
        mode: MonitorMode = "min",
        best_name: str = "best.pt",
        last_name: str = "last.pt",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_name = best_name
        self.last_name = last_name
        self.best_value: float | None = None
        self.best_epoch = 0
        self.best_step = 0

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / self.best_name

    @property
    def last_path(self) -> Path:
        return self.checkpoint_dir / self.last_name

    def save_last(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer | None,
        epoch: int,
        step: int,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> CheckpointResult:
        save_checkpoint(
            self.last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            metrics=metrics,
            extra=self._build_extra(extra),
        )
        return CheckpointResult(
            path=self.last_path,
            is_best=False,
            metric_name=None,
            metric_value=None,
            best_value=self.best_value,
        )

    def save_best_if_needed(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer | None,
        epoch: int,
        step: int,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> CheckpointResult:
        if self.monitor is None:
            return CheckpointResult(self.best_path, False, None, None, self.best_value)

        if self.monitor not in metrics:
            raise KeyError(
                f"Cannot monitor metric {self.monitor!r}; available metrics: {sorted(metrics)}"
            )

        metric_value = float(metrics[self.monitor])
        if not math.isfinite(metric_value):
            return CheckpointResult(
                self.best_path,
                False,
                self.monitor,
                metric_value,
                self.best_value,
            )

        is_best = self._is_better(metric_value)
        if is_best:
            self.best_value = metric_value
            self.best_epoch = epoch
            self.best_step = step
            save_checkpoint(
                self.best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                metrics=metrics,
                extra=self._build_extra(extra),
            )

        return CheckpointResult(
            path=self.best_path,
            is_best=is_best,
            metric_name=self.monitor,
            metric_value=metric_value,
            best_value=self.best_value,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "best_name": self.best_name,
            "last_name": self.last_name,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best_value = state.get("best_value")
        self.best_epoch = int(state.get("best_epoch", 0))
        self.best_step = int(state.get("best_step", 0))

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def _build_extra(self, extra: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(extra or {})
        payload["checkpoint_manager"] = self.state_dict()
        return payload


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    epoch: int = 0,
    step: int = 0,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, output)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint

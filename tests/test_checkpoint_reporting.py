from pathlib import Path

import torch
from torch import nn

from ane_drive_perc.engine.checkpoint import CheckpointManager, load_checkpoint
from ane_drive_perc.engine.reporting import (
    MetricColumn,
    format_metric,
    make_progress_bar,
    visible_length,
)


def test_checkpoint_manager_saves_best_and_last(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(tmp_path, monitor="loss", mode="min")

    first = manager.save_best_if_needed(
        model=model,
        optimizer=opt,
        epoch=1,
        step=1,
        metrics={"loss": 2.0},
    )
    second = manager.save_best_if_needed(
        model=model,
        optimizer=opt,
        epoch=1,
        step=2,
        metrics={"loss": 3.0},
    )
    manager.save_last(
        model=model, optimizer=opt, epoch=1, step=2, metrics={"loss": 3.0}
    )

    assert first.is_best is True
    assert second.is_best is False
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()
    checkpoint = load_checkpoint(tmp_path / "last.pt", model=model)
    assert checkpoint["step"] == 2


def test_reporting_format_helpers() -> None:
    assert make_progress_bar(5, 10, width=10) == "[#####.....]"
    assert (
        format_metric(MetricColumn("loss", "loss", precision=2, width=6), 1.234)
        == "  1.23"
    )
    assert visible_length("\x1b[32mabc\x1b[0m") == 3

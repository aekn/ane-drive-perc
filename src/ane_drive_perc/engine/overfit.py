import json
import math
import random
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from ane_drive_perc.data.build import build_detection_dataloader_from_train_config
from ane_drive_perc.engine.checkpoint import CheckpointManager
from ane_drive_perc.engine.reporting import MetricColumn, TrainingReporter
from ane_drive_perc.losses.simple_anchor_free import simple_anchor_free_detection_loss
from ane_drive_perc.models.tiny_detector import TinyAnchorFreeDetector
from ane_drive_perc.utils.device import resolve_device
from ane_drive_perc.utils.jsonl import JsonlWriter
from ane_drive_perc.utils.run import create_run_dir

SplitName = Literal["train", "val"]


def run_tiny_overfit(
    *,
    config_path: str,
    split: str = "train",
    num_images: int = 16,
    steps: int = 300,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
    device_name: str = "auto",
    output_dir: str = "runs/overfit_tiny",
    log_interval: int = 25,
    save_best: bool = True,
    show_download_progress: bool = False,
) -> None:
    split_name = validate_split(split)
    _validate_positive("num_images", num_images)
    _validate_positive("steps", steps)
    _validate_positive("epochs", epochs)
    _validate_positive("batch_size", batch_size)
    _validate_positive("log_interval", log_interval)

    if not show_download_progress:
        disable_hf_progress_bars()

    device = resolve_device(device_name)
    run_paths = create_run_dir(output_dir=output_dir, config_path=config_path)
    reporter = build_overfit_reporter(run_paths.metrics_jsonl)
    reporter.start_run(
        "overfit-tiny",
        {
            "device": device,
            "split": split_name,
            "images": num_images,
            "epochs": epochs,
            "steps": steps,
            "batch": batch_size,
        },
    )

    samples = collect_samples(
        build_detection_dataloader_from_train_config(
            config_path,
            split=split_name,
            batch_size=min(batch_size, num_images),
            num_workers=0,
            shuffle_buffer=0,
        ),
        max_images=num_images,
    )
    if not samples:
        raise RuntimeError("No samples were collected from the dataloader.")

    image_height = int(samples[0]["image"].shape[-2])
    image_width = int(samples[0]["image"].shape[-1])

    model = TinyAnchorFreeDetector(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    checkpoint_manager = CheckpointManager(
        run_paths.checkpoints, monitor="loss", mode="min"
    )

    model.train()

    first_loss: float | None = None
    last_loss: float | None = None
    best_loss = math.inf
    best_step = 0
    best_epoch = 0
    global_step = 0

    base_steps_per_epoch = steps // epochs
    remainder_steps = steps % epochs

    for epoch in range(1, epochs + 1):
        steps_this_epoch = base_steps_per_epoch + (1 if epoch <= remainder_steps else 0)
        if steps_this_epoch <= 0:
            continue

        epoch_report = reporter.begin_train_epoch(
            epoch=epoch,
            total_epochs=epochs,
            steps_in_epoch=steps_this_epoch,
        )

        for _ in range(steps_this_epoch):
            global_step += 1
            batch_samples = random.choices(samples, k=batch_size)
            images = torch.stack(
                [sample["image"] for sample in batch_samples], dim=0
            ).to(device)
            targets = [
                move_target_to_device(sample["target"], device)
                for sample in batch_samples
            ]

            predictions = model(images)
            loss_output = simple_anchor_free_detection_loss(
                predictions,
                targets,
                image_height=image_height,
                image_width=image_width,
                num_classes=10,
            )
            loss = loss_output["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            cls_value = float(loss_output["cls_loss"].detach().cpu().item())
            box_value = float(loss_output["box_loss"].detach().cpu().item())
            num_pos = float(loss_output["num_pos"].detach().cpu().item())

            if first_loss is None:
                first_loss = loss_value
            last_loss = loss_value

            if loss_value < best_loss:
                best_loss = loss_value
                best_step = global_step
                best_epoch = epoch

            metrics = {
                "loss": loss_value,
                "cls_loss": cls_value,
                "box_loss": box_value,
                "num_pos": num_pos,
                "best_loss": best_loss,
            }
            epoch_report.step(
                global_step=global_step,
                total_steps=steps,
                metrics=metrics,
                write_json=global_step == 1
                or global_step % log_interval == 0
                or global_step == steps,
                extra={
                    "best_step": best_step,
                    "best_epoch": best_epoch,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                },
            )

            if save_best:
                checkpoint_manager.save_best_if_needed(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    metrics={"loss": loss_value, "best_loss": best_loss},
                    extra=checkpoint_extra(
                        config_path=config_path,
                        split=split_name,
                        batch_size=batch_size,
                        num_images=num_images,
                        image_height=image_height,
                        image_width=image_width,
                    ),
                )

        epoch_metrics = epoch_report.mean_metrics()
        epoch_metrics["best_loss"] = best_loss
        epoch_report.finish(global_step=global_step, metrics=epoch_metrics)

    first_loss_value = float(first_loss) if first_loss is not None else float("nan")
    last_loss_value = float(last_loss) if last_loss is not None else float("nan")

    checkpoint_manager.save_last(
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        step=global_step,
        metrics={
            "first_loss": first_loss_value,
            "last_loss": last_loss_value,
            "best_loss": best_loss,
            "best_step": float(best_step),
            "best_epoch": float(best_epoch),
        },
        extra=checkpoint_extra(
            config_path=config_path,
            split=split_name,
            batch_size=batch_size,
            num_images=num_images,
            image_height=image_height,
            image_width=image_width,
        ),
    )

    final_metrics = {
        "split": "train",
        "epochs": epochs,
        "steps": global_step,
        "batch_size": batch_size,
        "num_images": num_images,
        "first_loss": first_loss_value,
        "last_loss": last_loss_value,
        "best_loss": best_loss,
        "best_step": best_step,
        "best_epoch": best_epoch,
        "image_height": image_height,
        "image_width": image_width,
    }
    reporter.log_summary(final_metrics)
    update_summary_json(run_paths.summary_json, final_metrics)
    reporter.finish_run(
        first_loss=first_loss_value,
        last_loss=last_loss_value,
        best_loss=best_loss,
        best_epoch=best_epoch,
        best_step=best_step,
        run_dir=run_paths.root,
        checkpoint_path=run_paths.checkpoints / "last.pt",
    )


def build_overfit_reporter(metrics_path: str | Path) -> TrainingReporter:
    return TrainingReporter(
        writer=JsonlWriter(metrics_path),
        columns=[
            MetricColumn("loss", "loss", precision=4),
            MetricColumn("cls_loss", "cls", precision=4),
            MetricColumn("box_loss", "box", precision=4),
            MetricColumn("num_pos", "pos", kind="int", width=5),
            MetricColumn("best_loss", "best", precision=4),
        ],
        progress_width=12,
    )


def validate_split(split: str) -> SplitName:
    if split == "train":
        return "train"
    if split == "val":
        return "val"
    raise ValueError(f"split must be 'train' or 'val', got {split!r}.")


def collect_samples(loader: Any, *, max_images: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for batch in loader:
        for image, target in zip(batch["images"], batch["targets"], strict=True):
            samples.append(
                {"image": image.detach().cpu(), "target": detach_target(target)}
            )
            if len(samples) >= max_images:
                return samples
    return samples


def detach_target(target: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


def move_target_to_device(
    target: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


def checkpoint_extra(
    *,
    config_path: str,
    split: str,
    batch_size: int,
    num_images: int,
    image_height: int,
    image_width: int,
) -> dict[str, Any]:
    return {
        "config_path": config_path,
        "split": split,
        "batch_size": batch_size,
        "num_images": num_images,
        "image_height": image_height,
        "image_width": image_width,
    }


def update_summary_json(path: str | Path, updates: dict[str, Any]) -> None:
    summary_path = Path(path)
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing.update(updates)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, sort_keys=True)


def disable_hf_progress_bars() -> None:
    try:
        from huggingface_hub.utils.tqdm import disable_progress_bars

        disable_progress_bars()
    except Exception:
        return


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")

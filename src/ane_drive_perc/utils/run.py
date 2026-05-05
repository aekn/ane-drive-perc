import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    logs: Path
    visualizations: Path
    exports: Path
    config_copy: Path
    metrics_jsonl: Path
    summary_json: Path


def create_run_dir(
    *,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> RunPaths:
    root = Path(output_dir)
    checkpoints = root / "checkpoints"
    logs = root / "logs"
    visualizations = root / "visualizations"
    exports = root / "exports"
    for path in (root, checkpoints, logs, visualizations, exports):
        path.mkdir(parents=True, exist_ok=True)

    config_copy = root / "config.yaml"
    metrics_jsonl = root / "metrics.jsonl"
    summary_json = root / "summary.json"

    if config_path is not None and Path(config_path).exists():
        shutil.copy2(config_path, config_copy)
    elif config is not None:
        with config_copy.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    write_summary(summary_json, default_run_summary())
    return RunPaths(
        root=root,
        checkpoints=checkpoints,
        logs=logs,
        visualizations=visualizations,
        exports=exports,
        config_copy=config_copy,
        metrics_jsonl=metrics_jsonl,
        summary_json=summary_json,
    )


def default_run_summary() -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "git_commit": get_git_commit(),
    }


def write_summary(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()

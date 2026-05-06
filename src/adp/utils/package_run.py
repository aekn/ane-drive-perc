from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _read_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []

    return [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _best_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    candidates = [row for row in rows if "val/mAP" in row]
    if not candidates:
        return rows[-1]

    return max(candidates, key=lambda row: float(row.get("val/mAP", -1.0)))


def package_run(cfg: DictConfig) -> None:
    run_name = str(cfg.artifacts.run_name)
    run_dir = Path(str(cfg.paths.runs)) / run_name

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_root = Path(str(cfg.artifacts.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    package_dir = output_root / run_name
    if package_dir.exists() and bool(cfg.artifacts.overwrite):
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.jsonl"
    rows = _read_metrics(metrics_path)
    best = _best_metrics(rows)

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "best": best,
        "num_metric_rows": len(rows),
    }

    (package_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    _copy_if_exists(run_dir / "config.resolved.yaml", package_dir / "config.resolved.yaml")
    _copy_if_exists(run_dir / "metrics.jsonl", package_dir / "metrics.jsonl")
    _copy_if_exists(run_dir / "train.log", package_dir / "train.log")
    _copy_if_exists(run_dir / "checkpoints" / "best.pt", package_dir / "checkpoints" / "best.pt")
    _copy_if_exists(run_dir / "checkpoints" / "last.pt", package_dir / "checkpoints" / "last.pt")
    _copy_if_exists(run_dir / "debug", package_dir / "debug")

    tar_path = output_root / f"{run_name}.tar.gz"
    if tar_path.exists() and bool(cfg.artifacts.overwrite):
        tar_path.unlink()

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(package_dir, arcname=run_name)

    print(f"[adp] packaged run directory: {package_dir}")
    print(f"[adp] wrote tarball: {tar_path}")
    print(f"[adp] best metrics: {json.dumps(best, indent=2)}")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    package_run(cfg)


if __name__ == "__main__":
    main()

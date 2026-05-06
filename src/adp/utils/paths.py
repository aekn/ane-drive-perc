from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig


@dataclass(frozen=True)
class RunPaths:
    root: Path
    reports: Path
    debug: Path


def ensure_run_paths(cfg: DictConfig) -> RunPaths:
    paths = RunPaths(
        root=Path(str(cfg.run.dir)),
        reports=Path(str(cfg.run.reports_dir)),
        debug=Path(str(cfg.run.debug_dir)),
    )

    paths.root.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.debug.mkdir(parents=True, exist_ok=True)

    return paths

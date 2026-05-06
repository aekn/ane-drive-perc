from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def write_resolved_config(cfg: DictConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

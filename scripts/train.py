"""Train one ablation cell from a YAML config.

Usage:
    uv run python scripts/train.py configs/cell_a.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

from fkde.config import load_config
from fkde.engine.train import train_cell


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <config.yaml>", file=sys.stderr)
        sys.exit(2)
    cfg = load_config(Path(sys.argv[1]))
    train_cell(cfg)


if __name__ == "__main__":
    main()

from typing import Any
from pathlib import Path
from collections.abc import Iterable
import json


def image_id_from_path(path: Path) -> str:
    return path.stem


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")))
            f.write("\n")

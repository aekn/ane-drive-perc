from collections.abc import Iterable, Iterator
import json
from pathlib import Path

from adp.data.records import ImageRecord


def read_manifest(path: Path) -> Iterator[ImageRecord]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Manifest line must decode to a JSON object.")
                yield ImageRecord.from_json(value)
            except Exception as e:
                raise ValueError(f"Failed to parse {path}:{line_number}") from e


def write_manifest(path: Path, records: Iterable[ImageRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_json(), sort_keys=True))
            f.write("\n")
            count += 1

    return count

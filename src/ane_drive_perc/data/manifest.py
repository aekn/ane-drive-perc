import json
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field, field_validator


class DetectionObject(BaseModel):
    box_xyxy_abs: tuple[float, float, float, float]
    label: int
    area: float | None = None
    iscrowd: bool = False

    @field_validator("box_xyxy_abs")
    @classmethod
    def validate_box(
        cls,
        value: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = value
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid xyxy box: {value}")
        return value


class DetectionManifestRecord(BaseModel):
    image_id: str
    split: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    image_path: str | None = None
    shard_path: str | None = None
    objects: list[DetectionObject]
    attributes: dict[str, str] = Field(default_factory=dict)


def read_detection_manifest(path: str | Path) -> list[DetectionManifestRecord]:
    records: list[DetectionManifestRecord] = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(DetectionManifestRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid strict detection manifest record at {path}:{line_number}"
                ) from exc

    return records


def iter_detection_manifest(path: str | Path) -> Iterator[DetectionManifestRecord]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                yield DetectionManifestRecord.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"Invalid strict detection manifest record at {path}:{line_number}"
                ) from exc


def write_detection_manifest(
    path: str | Path, records: list[DetectionManifestRecord]
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.model_dump(), separators=(",", ":")))
            f.write("\n")


def read_manifest_image_ids(path: str | Path) -> set[str]:
    ids: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")

            image_id = extract_image_id(record)

            if image_id is None:
                available = ", ".join(sorted(str(key) for key in record.keys()))
                raise ValueError(
                    f"Could not find image ID at {path}:{line_number}. "
                    f"Expected one of id/image_id/name/sample_id. "
                    f"Available keys: {available}"
                )

            ids.add(image_id)

    if not ids:
        raise ValueError(f"Manifest contains no image IDs: {path}")

    return ids


def extract_image_id(record: dict[str, Any]) -> str | None:
    for key in ("id", "image_id", "name", "sample_id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value

    return None

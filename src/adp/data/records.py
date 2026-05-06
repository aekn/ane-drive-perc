from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


BBoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class ObjectAnnotation:
    bbox_xyxy: BBoxXYXY
    class_id: int
    class_name: str
    area: float
    iscrowd: bool = False
    source_category: str | None = None
    source_object_id: int | None = None

    def validate(self, *, width: int, height: int, num_classes: int) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy

        if not 0 <= self.class_id < num_classes:
            raise ValueError(
                f"Invalid class_id={self.class_id!r}; num_classes={num_classes}"
            )

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox with nonpositive area: {self.bbox_xyxy}")

        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError(
                f"BBox out of bounds: bbox={self.bbox_xyxy!r}, image_size={(width, height)!r}"
            )

        expected_area = (x2 - x1) * (y2 - y1)
        tolerance = max(1.0, expected_area * 0.05)

        if self.area <= 0 or abs(self.area - expected_area) > tolerance:
            raise ValueError(
                f"Suspicious bbox area: area={self.area}, expected={expected_area}, bbox={self.bbox_xyxy!r}"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.bbox_xyxy),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "area": self.area,
            "iscrowd": self.iscrowd,
            "source_category": self.source_category,
            "source_object_id": self.source_object_id,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> ObjectAnnotation:
        bbox = value["bbox_xyxy"]

        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox_xyxy: {bbox!r}")

        return cls(
            bbox_xyxy=tuple(float(x) for x in bbox),  # type: ignore[arg-type]
            class_id=int(value["class_id"]),
            class_name=str(value["class_name"]),
            area=float(value["area"]),
            iscrowd=bool(value.get("iscrowd", False)),
            source_category=(
                None
                if value.get("source_category") is None
                else str(value["source_category"])
            ),
            source_object_id=(
                None
                if value.get("source_object_id") is None
                else int(value["source_object_id"])
            ),
        )


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    file_name: str
    width: int
    height: int
    split: str
    objects: tuple[ObjectAnnotation, ...]

    @property
    def path(self) -> Path:
        return Path(self.file_name)

    def validate(self, *, num_classes: int) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid image size: {(self.width, self.height)}")

        for obj in self.objects:
            obj.validate(width=self.width, height=self.height, num_classes=num_classes)

    def to_json(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
            "split": self.split,
            "objects": [obj.to_json() for obj in self.objects],
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> ImageRecord:
        return cls(
            image_id=str(value["image_id"]),
            file_name=str(value["file_name"]),
            width=int(value["width"]),
            height=int(value["height"]),
            split=str(value["split"]),
            objects=tuple(
                ObjectAnnotation.from_json(obj) for obj in value.get("objects", [])
            ),
        )

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ane_drive_perc.data.class_map import BDD_SOURCE_TO_TARGET, encode_bdd100k_label

CANONICAL_WIDTH = 1280
CANONICAL_HEIGHT = 720


@dataclass(frozen=True)
class BDDObject:
    xyxy: tuple[float, float, float, float]
    label: int
    category: str
    source_category: str
    occluded: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class BDDMetadata:
    image_id: str
    width: int | None
    height: int | None
    split: str | None
    image_path: str | None
    label_path: str | None
    weather: str | None
    scene: str | None
    timeofday: str | None
    objects: tuple[BDDObject, ...]

    @property
    def attributes(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key in (
            "split",
            "weather",
            "scene",
            "timeofday",
            "image_path",
            "label_path",
        ):
            value = getattr(self, key)
            if isinstance(value, str):
                out[key] = value
        return out


def read_bdd_label(path: str | Path) -> BDDMetadata:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected BDD label to be a JSON object: {path}")
    return parse_bdd_metadata(data, fallback_image_id=Path(path).stem)


def parse_bdd_metadata(
    metadata: dict[str, Any], *, fallback_image_id: str
) -> BDDMetadata:
    if "frames" in metadata:
        return _parse_raw_bdd_metadata(metadata, fallback_image_id=fallback_image_id)

    image_id = extract_image_id(metadata, fallback=fallback_image_id)
    objects = parse_detection_objects(metadata, image_id=image_id)

    return BDDMetadata(
        image_id=image_id,
        width=_optional_positive_int(metadata.get("width")),
        height=_optional_positive_int(metadata.get("height")),
        split=_optional_str(metadata.get("split")),
        image_path=_optional_str(metadata.get("image")),
        label_path=_optional_str(metadata.get("label")),
        weather=_optional_str(metadata.get("weather")),
        scene=_optional_str(metadata.get("scene")),
        timeofday=_optional_str(metadata.get("timeofday")),
        objects=tuple(objects),
    )


def _parse_raw_bdd_metadata(
    metadata: dict[str, Any], *, fallback_image_id: str
) -> BDDMetadata:
    attrs = metadata.get("attributes", {})
    if not isinstance(attrs, dict):
        attrs = {}

    frames = metadata.get("frames", [])
    frame0 = frames[0] if isinstance(frames, list) and frames else {}
    raw_objects = frame0.get("objects", []) if isinstance(frame0, dict) else []

    image_id = extract_image_id(metadata, fallback=fallback_image_id)
    objects = _parse_raw_objects(raw_objects, image_id=image_id)

    return BDDMetadata(
        image_id=image_id,
        width=_optional_positive_int(metadata.get("width")) or CANONICAL_WIDTH,
        height=_optional_positive_int(metadata.get("height")) or CANONICAL_HEIGHT,
        split=_optional_str(metadata.get("split")),
        image_path=_optional_str(metadata.get("image")),
        label_path=_optional_str(metadata.get("label")),
        weather=_optional_str(attrs.get("weather")),
        scene=_optional_str(attrs.get("scene")),
        timeofday=_optional_str(attrs.get("timeofday")),
        objects=tuple(objects),
    )


def extract_image_id(metadata: dict[str, Any], *, fallback: str) -> str:
    for key in ("id", "image_id", "name", "sample_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return Path(value).stem

    image_path = metadata.get("image")
    if isinstance(image_path, str) and image_path:
        return Path(image_path).stem

    return Path(fallback).stem


def parse_detection_objects(
    metadata: dict[str, Any], *, image_id: str
) -> list[BDDObject]:
    if "boxes" in metadata:
        return _parse_project_boxes(metadata["boxes"], image_id=image_id)
    if "objects" in metadata:
        return _parse_internal_objects(metadata["objects"], image_id=image_id)
    if "labels" in metadata:
        return _parse_bdd_labels(metadata["labels"], image_id=image_id)
    return []


def _parse_project_boxes(raw_boxes: Any, *, image_id: str) -> list[BDDObject]:
    if not isinstance(raw_boxes, list):
        raise ValueError(f"Expected 'boxes' to be a list for image_id={image_id!r}.")

    objects: list[BDDObject] = []

    for raw_obj in raw_boxes:
        if not isinstance(raw_obj, dict):
            raise ValueError(
                f"Expected box entry to be a dict for image_id={image_id!r}."
            )

        xyxy = _read_valid_xyxy(raw_obj.get("xyxy"))
        if xyxy is None:
            continue

        category = str(raw_obj.get("category", ""))
        source_category = str(raw_obj.get("source_category", category))

        if "category_id" in raw_obj:
            label = int(raw_obj["category_id"])
            if not category:
                category = source_category
        elif category:
            label = encode_bdd100k_label(category)
        else:
            raise ValueError(
                f"Box is missing category/category_id for image_id={image_id!r}."
            )

        objects.append(
            BDDObject(
                xyxy=xyxy,
                label=label,
                category=BDD_SOURCE_TO_TARGET.get(category, category),
                source_category=source_category,
                occluded=bool(raw_obj.get("occluded", False)),
                truncated=bool(raw_obj.get("truncated", False)),
            )
        )

    return objects


def _parse_internal_objects(raw_objects: Any, *, image_id: str) -> list[BDDObject]:
    if not isinstance(raw_objects, list):
        raise ValueError(f"Expected 'objects' to be a list for image_id={image_id!r}.")

    objects: list[BDDObject] = []

    for raw_obj in raw_objects:
        if not isinstance(raw_obj, dict):
            raise ValueError(
                f"Expected object entry to be a dict for image_id={image_id!r}."
            )

        xyxy = _read_valid_xyxy(raw_obj.get("box_xyxy_abs"))
        if xyxy is None:
            continue

        label = int(raw_obj["label"])
        category = str(raw_obj.get("category", ""))

        objects.append(
            BDDObject(
                xyxy=xyxy,
                label=label,
                category=category,
                source_category=str(raw_obj.get("source_category", category)),
                occluded=bool(raw_obj.get("occluded", False)),
                truncated=bool(raw_obj.get("truncated", False)),
            )
        )

    return objects


def _parse_bdd_labels(raw_labels: Any, *, image_id: str) -> list[BDDObject]:
    if not isinstance(raw_labels, list):
        raise ValueError(f"Expected 'labels' to be a list for image_id={image_id!r}.")

    objects: list[BDDObject] = []

    for raw_obj in raw_labels:
        if not isinstance(raw_obj, dict):
            continue

        category = raw_obj.get("category")
        box2d = raw_obj.get("box2d")

        if category is None or not isinstance(box2d, dict):
            continue

        xyxy = _read_valid_xyxy(
            [
                box2d.get("x1"),
                box2d.get("y1"),
                box2d.get("x2"),
                box2d.get("y2"),
            ]
        )
        if xyxy is None:
            continue

        target_category = BDD_SOURCE_TO_TARGET.get(str(category), str(category))
        attrs = raw_obj.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}

        objects.append(
            BDDObject(
                xyxy=xyxy,
                label=encode_bdd100k_label(target_category),
                category=target_category,
                source_category=str(category),
                occluded=bool(attrs.get("occluded", False)),
                truncated=bool(attrs.get("truncated", False)),
            )
        )

    return objects


def _parse_raw_objects(raw_objects: Any, *, image_id: str) -> list[BDDObject]:
    if not isinstance(raw_objects, list):
        return []

    labels: list[dict[str, Any]] = []
    for obj in raw_objects:
        if isinstance(obj, dict) and "box2d" in obj:
            labels.append(obj)

    return _parse_bdd_labels(labels, image_id=image_id)


def _read_valid_xyxy(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None

    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    return None

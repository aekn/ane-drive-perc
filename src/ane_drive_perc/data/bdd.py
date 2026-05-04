import json
from pathlib import Path
from typing import Any


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

DETECTION_CATEGORIES: tuple[str, ...] = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)

CATEGORY_TO_ID = {name: i for i, name in enumerate(DETECTION_CATEGORIES)}

BDD_SOURCE_TO_TARGET: dict[str, str] = {
    "person": "pedestrian",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motor": "motorcycle",
    "bike": "bicycle",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}


def read_bdd_label(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if "frames" in data:
        return read_raw_bdd_label(data)

    return read_project_bdd_label(data)


def read_raw_bdd_label(data: dict[str, Any]) -> dict[str, Any]:
    attrs = data["attributes"]
    objects = data["frames"][0]["objects"]

    boxes: list[dict[str, Any]] = []

    for obj in objects:
        if "box2d" not in obj:
            continue

        source_category = obj["category"]
        category = BDD_SOURCE_TO_TARGET[source_category]
        category_id = CATEGORY_TO_ID[category]

        b = obj["box2d"]
        a = obj["attributes"]

        boxes.append(
            {
                "category": category,
                "category_id": category_id,
                "source_category": source_category,
                "xyxy": [
                    float(b["x1"]),
                    float(b["y1"]),
                    float(b["x2"]),
                    float(b["y2"]),
                ],
                "occluded": bool(a["occluded"]),
                "truncated": bool(a["truncated"]),
            }
        )

    return {
        "id": data["name"],
        "weather": attrs["weather"],
        "scene": attrs["scene"],
        "timeofday": attrs["timeofday"],
        "boxes": boxes,
    }


def read_project_bdd_label(data: dict[str, Any]) -> dict[str, Any]:
    boxes: list[dict[str, Any]] = []

    for box in data["boxes"]:
        category = box["category"]
        category_id = CATEGORY_TO_ID[category]

        boxes.append(
            {
                "category": category,
                "category_id": category_id,
                "source_category": box.get("source_category", category),
                "xyxy": [float(v) for v in box["xyxy"]],
                "occluded": bool(box["occluded"]),
                "truncated": bool(box["truncated"]),
            }
        )

    return {
        "id": data["id"],
        "weather": data["weather"],
        "scene": data["scene"],
        "timeofday": data["timeofday"],
        "boxes": boxes,
    }

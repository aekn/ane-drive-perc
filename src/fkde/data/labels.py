"""BDD100K label loader."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

UNDEFINED = "undefined"

# 10 detection categories excludig drivabale area and lane
DETECTION_CATEGORIES = (
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck",
)


@dataclass(frozen=True, slots=True)
class ImageRecord:
    name: str
    weather: str
    timeofday: str
    scene: str
    num_boxes: int
    categories: tuple[tuple[str, int], ...]

    @property
    def stratum_weather_timeofday(self) -> tuple[str, str]:
        return (self.weather, self.timeofday)


def load_records(json_path: str | Path) -> list[ImageRecord]:
    with open(json_path) as f:
        raw = json.load(f)

    records: list[ImageRecord] = []
    for r in raw:
        attrs = r.get("attributes") or {}
        cat_counter: Counter[str] = Counter()
        for lab in r.get("labels") or []:
            cat = lab.get("category")
            if cat in DETECTION_CATEGORIES and "box2d" in lab:
                cat_counter[cat] += 1
        records.append(
            ImageRecord(
                name=r["name"],
                weather=attrs.get("weather") or UNDEFINED,
                timeofday=attrs.get("timeofday") or UNDEFINED,
                scene=attrs.get("scene") or UNDEFINED,
                num_boxes=sum(cat_counter.values()),
                categories=tuple(sorted(cat_counter.items())),
            )
        )
    return records


def filter_existing_images(
    records: list[ImageRecord], images_dir: str | Path
) -> list[ImageRecord]:
    images_dir = Path(images_dir)
    return [r for r in records if (images_dir / r.name).exists()]

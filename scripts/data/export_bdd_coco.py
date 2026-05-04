import argparse
import json
from pathlib import Path
from typing import Any

from ane_drive_perc.data.bdd import (
    DETECTION_CATEGORIES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    read_bdd_label,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def coco_categories() -> list[dict[str, Any]]:
    return [
        {
            "id": i + 1,
            "name": name,
            "supercategory": "object",
        }
        for i, name in enumerate(DETECTION_CATEGORIES)
    ]


def xyxy_to_coco_bbox(xyxy: list[float]) -> tuple[list[float], float] | None:
    x1, y1, x2, y2 = xyxy

    x1 = max(0.0, min(float(IMAGE_WIDTH), x1))
    y1 = max(0.0, min(float(IMAGE_HEIGHT), y1))
    x2 = max(0.0, min(float(IMAGE_WIDTH), x2))
    y2 = max(0.0, min(float(IMAGE_HEIGHT), y2))

    w = x2 - x1
    h = y2 - y1

    if w <= 0.0 or h <= 0.0:
        return None

    return [x1, y1, w, h], w * h


def export_split(split_dir: Path) -> dict[str, Any]:
    label_paths = sorted((split_dir / "labels").glob("*.json"))

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    annotation_id = 1

    for image_id, label_path in enumerate(label_paths, start=1):
        label = read_bdd_label(label_path)
        image_name = f"{label['id']}.jpg"

        images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "weather": label["weather"],
                "scene": label["scene"],
                "timeofday": label["timeofday"],
            }
        )

        for box in label["boxes"]:
            converted = xyxy_to_coco_bbox(box["xyxy"])
            if converted is None:
                continue

            bbox, area = converted

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(box["category_id"]) + 1,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "occluded": bool(box["occluded"]),
                    "truncated": bool(box["truncated"]),
                    "source_category": box.get("source_category", box["category"]),
                }
            )

            annotation_id += 1

    return {
        "info": {
            "description": "BDD100K detection split exported by ane-drive-perc",
            "version": "1.0",
        },
        "licenses": [],
        "categories": coco_categories(),
        "images": images,
        "annotations": annotations,
    }


def main() -> None:
    args = parse_args()

    split_dir = Path(args.split_dir)
    out = Path(args.out)

    coco = export_split(split_dir)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")

    print(f"[done] wrote {out}")
    print(f"       images={len(coco['images'])}")
    print(f"       annotations={len(coco['annotations'])}")
    print(f"       categories={len(coco['categories'])}")


if __name__ == "__main__":
    main()

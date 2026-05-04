import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    path = Path(args.path)
    data = json.loads(path.read_text(encoding="utf-8"))

    categories = {int(c["id"]): c["name"] for c in data["categories"]}
    image_ids = {int(img["id"]) for img in data["images"]}

    boxes_per_category: Counter[str] = Counter()
    images_per_category: dict[str, set[int]] = defaultdict(set)
    boxes_per_image: Counter[int] = Counter()

    for ann in data["annotations"]:
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        category = categories[category_id]

        boxes_per_category[category] += 1
        images_per_category[category].add(image_id)
        boxes_per_image[image_id] += 1

    empty_images = len(image_ids - set(boxes_per_image))

    print(f"path: {path}")
    print(f"images: {len(data['images'])}")
    print(f"annotations: {len(data['annotations'])}")
    print(f"categories: {len(data['categories'])}")
    print(f"empty_images: {empty_images}")

    if boxes_per_image:
        values = list(boxes_per_image.values())
        print(f"boxes/image min: {min(values)}")
        print(f"boxes/image max: {max(values)}")
        print(f"boxes/image avg: {sum(values) / len(data['images']):.2f}")

    print("\ncategory counts:")
    for category_id, category in categories.items():
        print(
            f"  {category:14s} "
            f"boxes={boxes_per_category[category]:7d} "
            f"images={len(images_per_category[category]):6d}"
        )


if __name__ == "__main__":
    main()

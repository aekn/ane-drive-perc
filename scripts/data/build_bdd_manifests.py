import argparse
from collections import defaultdict
from pathlib import Path
import random
from typing import Counter
import json

from ane_drive_perc.data.bdd import (
    DETECTION_CATEGORIES,
    read_bdd_label,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

from ane_drive_perc.utils import jsonl_write


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/bdd"))
    p.add_argument("--out-dir", type=Path, default=Path("data/manifests"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset-sizes", type=int, nargs="*", default=[3000, 5000, 10000])
    return p.parse_args()


def rel(path: Path, root: Path):
    return path.relative_to(root).as_posix()


def make_labeled_entry(root: Path, split: str, image_path: Path) -> dict:
    image_id = image_path.stem
    label_path = root / "100k" / "labels" / split / f"{image_id}.json"
    label = read_bdd_label(label_path)

    return {
        "id": image_id,
        "split": split,
        "image": rel(image_path, root),
        "label": rel(label_path, root),
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "weather": label["weather"],
        "scene": label["scene"],
        "timeofday": label["timeofday"],
        "boxes": label["boxes"],
    }


def make_image_only_entry(root: Path, split: str, image_path: Path) -> dict:
    return {
        "id": image_path.stem,
        "split": split,
        "image": rel(image_path, root),
        "label": None,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "weather": None,
        "scene": None,
        "timeofday": None,
        "boxes": [],
    }


def build_split(root: Path, split: str) -> list[dict]:
    image_dir = root / "100k" / "images" / split
    image_paths = sorted(image_dir.glob("*.jpg"))

    if split in {"train", "val"}:
        return [make_labeled_entry(root, split, ip) for ip in image_paths]

    return [make_image_only_entry(root, split, ip) for ip in image_paths]


def subset_name(size: int, seed: int) -> str:
    if size % 1000 == 0:
        return f"train_{size // 1000:03d}k_seed{seed}"
    return f"train_{size:06d}_seed{seed}"


def strat_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row["timeofday"]),
        str(row["weather"]),
        str(row["scene"]),
    )


def stratified_subset(rows: list[dict], size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    groups = defaultdict(list)
    for row in rows:
        groups[strat_key(row)].append(row)

    total = len(rows)
    quota_parts = []

    for k, g in groups.items():
        exact = size * len(g) / total
        base = int(exact)
        quota_parts.append((exact - base, k, base))

    quotas = {key: base for _, key, base in quota_parts}

    remaining = size - sum(quotas.values())
    for _, key, _ in sorted(quota_parts, reverse=True)[:remaining]:
        quotas[key] += 1

    selected = []

    for k, g in groups.items():
        g = g[:]
        rng.shuffle(g)
        selected.extend(g[: quotas[k]])

    # break group ordering
    rng.shuffle(selected)

    return selected[:size]


def split_stats(rows: list[dict]) -> dict:
    category_counts = Counter()
    image_category_counts = Counter()

    for row in rows:
        seen = set()

        for box in row["boxes"]:
            category = box["category"]
            category_counts[category] += 1
            seen.add(category)

        for category in seen:
            image_category_counts[category] += 1

    return {
        "images": len(rows),
        "boxes": sum(category_counts.values()),
        "categories": {c: category_counts[c] for c in DETECTION_CATEGORIES},
        "image_categories": {c: image_category_counts[c] for c in DETECTION_CATEGORIES},
        "weather": dict(Counter(row["weather"] for row in rows)),
        "timeofday": dict(Counter(row["timeofday"] for row in rows)),
        "scene": dict(Counter(row["scene"] for row in rows)),
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = build_split(args.root, "train")
    val = build_split(args.root, "val")
    test = build_split(args.root, "test")

    manifest_sets = {
        "train_full": train,
        "val_full": val,
        "test_images": test,
    }

    for size in args.subset_sizes:
        name = subset_name(size, args.seed)
        manifest_sets[name] = stratified_subset(train, size, args.seed)

    for name, rows in manifest_sets.items():
        jsonl_write(args.out_dir / f"{name}.jsonl", rows)
        print(f"[manifest] {name}: {len(rows)} images")

    stats = {name: split_stats(rows) for name, rows in manifest_sets.items()}
    stats_path = args.out_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[done] wrote manifests to {args.out_dir}")
    print(f"[done] wrote stats to {stats_path}")


if __name__ == "__main__":
    main()

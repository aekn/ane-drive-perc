from collections import Counter
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from adp.config import write_resolved_config
from adp.data.classes import build_class_map, build_source_category_map
from adp.utils.images import is_image_file
from adp.utils.io import read_json, write_json
from adp.utils.paths import ensure_run_paths


def _image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if is_image_file(p))


def _label_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".json"
    )


def _objects(label: dict[str, Any]) -> list[dict[str, Any]]:
    frames = label.get("frames", [])
    if not isinstance(frames, list) or not frames:
        return []

    first_frame = frames[0]
    if not isinstance(first_frame, dict):
        return []

    objects = first_frame.get("objects", [])
    if not isinstance(objects, list):
        return []

    return [obj for obj in objects if isinstance(obj, dict)]


def _inspect_split(
    *,
    split: str,
    images_dir: Path,
    labels_dir: Path,
    canonical_names: set[str],
    source_category_map: dict[str, str],
) -> dict[str, Any]:
    image_paths = _image_files(images_dir)
    label_paths = _label_files(labels_dir)

    image_stems = {p.stem for p in image_paths}
    label_stems = {p.stem for p in label_paths}

    raw_category_counts: Counter[str] = Counter()
    box_category_counts: Counter[str] = Counter()
    poly_category_counts: Counter[str] = Counter()
    kept_box_counts: Counter[str] = Counter()
    skipped_box_counts: Counter[str] = Counter()
    invalid_mapped_category_counts: Counter[str] = Counter()

    parse_errors: list[str] = []
    sample_labels: list[dict[str, Any]] = []

    for label_path in label_paths:
        try:
            label = read_json(label_path)
        except Exception as exc:
            parse_errors.append(f"{label_path}: {exc}")
            continue

        if len(sample_labels) < 5:
            sample_labels.append(
                {
                    "path": str(label_path),
                    "name": label.get("name"),
                    "top_level_keys": sorted(label.keys()),
                }
            )

        for obj in _objects(label):
            category = str(obj.get("category", ""))
            raw_category_counts[category] += 1

            has_box = isinstance(obj.get("box2d"), dict)
            has_poly = isinstance(obj.get("poly2d"), list)

            if has_box:
                box_category_counts[category] += 1
                mapped_category = source_category_map.get(category)

                if mapped_category is None:
                    skipped_box_counts[category] += 1
                elif mapped_category in canonical_names:
                    kept_box_counts[mapped_category] += 1
                else:
                    invalid_mapped_category_counts[category] += 1

            if has_poly:
                poly_category_counts[category] += 1

    missing_images_for_labels = sorted(label_stems - image_stems)
    missing_labels_for_images = sorted(image_stems - label_stems)

    return {
        "split": split,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "image_extensions": dict(
            sorted(Counter(p.suffix.lower() for p in image_paths).items())
        ),
        "missing_images_for_labels_count": len(missing_images_for_labels),
        "missing_labels_for_images_count": len(missing_labels_for_images),
        "missing_images_for_labels_examples": missing_images_for_labels[:20],
        "missing_labels_for_images_examples": missing_labels_for_images[:20],
        "raw_category_counts": dict(sorted(raw_category_counts.items())),
        "box_category_counts": dict(sorted(box_category_counts.items())),
        "poly_category_counts": dict(sorted(poly_category_counts.items())),
        "kept_box_counts": dict(sorted(kept_box_counts.items())),
        "skipped_box_counts": dict(sorted(skipped_box_counts.items())),
        "invalid_mapped_category_counts": dict(
            sorted(invalid_mapped_category_counts.items())
        ),
        "parse_error_count": len(parse_errors),
        "parse_error_examples": parse_errors[:20],
        "sample_labels": sample_labels,
    }


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_paths = ensure_run_paths(cfg)
    write_resolved_config(cfg, run_paths.root / "config.resolved.yaml")

    class_map = build_class_map(cfg)
    source_category_map = build_source_category_map(cfg)

    report = {
        "raw_root": str(cfg.data.raw_root),
        "canonical_classes": class_map.id_to_name,
        "source_category_map": source_category_map,
        "train": _inspect_split(
            split="train",
            images_dir=Path(str(cfg.data.images.train)),
            labels_dir=Path(str(cfg.data.labels.train)),
            canonical_names=set(class_map.names),
            source_category_map=source_category_map,
        ),
        "val": _inspect_split(
            split="val",
            images_dir=Path(str(cfg.data.images.val)),
            labels_dir=Path(str(cfg.data.labels.val)),
            canonical_names=set(class_map.names),
            source_category_map=source_category_map,
        ),
    }

    output_path = run_paths.reports / "raw_dataset_inspection.json"
    write_json(output_path, report)

    print(f"[adp] wrote raw dataset inspection: {output_path}")
    print(f"[adp] train images: {report['train']['image_count']}")
    print(f"[adp] train labels: {report['train']['label_count']}")
    print(f"[adp] val images: {report['val']['image_count']}")
    print(f"[adp] val labels: {report['val']['label_count']}")

    train_skipped = report["train"]["skipped_box_counts"]
    val_skipped = report["val"]["skipped_box_counts"]
    train_invalid = report["train"]["invalid_mapped_category_counts"]
    val_invalid = report["val"]["invalid_mapped_category_counts"]

    if train_skipped or val_skipped:
        print(
            "[adp] some box2d categories are not in bdd.category_map and will be skipped."
        )

    if train_invalid or val_invalid:
        print(
            "[adp] some mapped categories are not canonical ADP classes. Fix bdd.category_map."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

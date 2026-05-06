import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from adp.config import write_resolved_config
from adp.data.classes import ClassMap, build_class_map, build_source_category_map
from adp.data.manifest import write_manifest
from adp.data.records import ImageRecord, ObjectAnnotation
from adp.utils.images import IMAGE_EXTENSIONS, read_image_size
from adp.utils.io import read_json, write_json
from adp.utils.paths import ensure_run_paths


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


def _label_image_stem(label: dict[str, Any], fallback: str) -> str:
    name = label.get("name")
    if isinstance(name, str) and name:
        return Path(name).stem
    return fallback


def _strata_from_label(label: dict[str, Any]) -> str:
    attrs = label.get("attributes") if isinstance(label, dict) else None
    if not isinstance(attrs, dict):
        attrs = {}
    weather = str(attrs.get("weather", "unknown"))
    scene = str(attrs.get("scene", "unknown"))
    timeofday = str(attrs.get("timeofday", "unknown"))
    return f"{weather}|{scene}|{timeofday}"


def _find_image(images_dir: Path, stem: str) -> Path:
    for suffix in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No image found for label stem={stem!r} in {images_dir}")


def _clip_box(
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    clipped_x1 = max(0.0, min(float(width), x1))
    clipped_y1 = max(0.0, min(float(height), y1))
    clipped_x2 = max(0.0, min(float(width), x2))
    clipped_y2 = max(0.0, min(float(height), y2))

    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return None

    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


def _record_from_label(
    *,
    label_path: Path,
    images_dir: Path,
    split: str,
    class_map: ClassMap,
    source_category_map: dict[str, str],
) -> tuple[ImageRecord, str]:
    label = read_json(label_path)
    image_stem = _label_image_stem(label, fallback=label_path.stem)
    image_path = _find_image(images_dir, image_stem)
    width, height = read_image_size(image_path)

    objects: list[ObjectAnnotation] = []

    for obj in _objects(label):
        if not isinstance(obj.get("box2d"), dict):
            continue

        source_category = str(obj.get("category", ""))
        class_name = source_category_map.get(source_category)

        if class_name is None:
            continue

        if not class_map.contains(class_name):
            raise ValueError(
                f"BDD category {source_category!r} maps to unknown ADP class {class_name!r}"
            )

        box = obj["box2d"]
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])

        clipped = _clip_box(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            width=width,
            height=height,
        )

        if clipped is None:
            continue

        cx1, cy1, cx2, cy2 = clipped
        area = (cx2 - cx1) * (cy2 - cy1)

        objects.append(
            ObjectAnnotation(
                bbox_xyxy=clipped,
                class_id=class_map.id_for_name(class_name),
                class_name=class_name,
                area=area,
                iscrowd=False,
                source_category=source_category,
                source_object_id=(None if obj.get("id") is None else int(obj["id"])),
            )
        )

    record = ImageRecord(
        image_id=image_stem,
        file_name=str(image_path),
        width=width,
        height=height,
        split=split,
        objects=tuple(objects),
    )
    record.validate(num_classes=class_map.num_classes)
    return record, _strata_from_label(label)


def _records_from_split(
    *,
    labels_dir: Path,
    images_dir: Path,
    split: str,
    class_map: ClassMap,
    source_category_map: dict[str, str],
) -> list[tuple[ImageRecord, str]]:
    label_paths = sorted(labels_dir.glob("*.json"))

    items: list[tuple[ImageRecord, str]] = []
    for label_path in label_paths:
        items.append(
            _record_from_label(
                label_path=label_path,
                images_dir=images_dir,
                split=split,
                class_map=class_map,
                source_category_map=source_category_map,
            )
        )

    return items


def _select_random(
    items: list[tuple[ImageRecord, str]], *, limit: int, seed: int
) -> list[tuple[ImageRecord, str]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return sorted(shuffled[:limit], key=lambda item: item[0].image_id)


def _select_stratified(
    items: list[tuple[ImageRecord, str]], *, limit: int, seed: int
) -> list[tuple[ImageRecord, str]]:
    """Largest-remainder apportionment across BDD attribute strata."""
    buckets: dict[str, list[tuple[ImageRecord, str]]] = defaultdict(list)
    for item in items:
        buckets[item[1]].append(item)

    total = len(items)
    raw_alloc = {k: limit * len(v) / total for k, v in buckets.items()}
    alloc = {k: int(v) for k, v in raw_alloc.items()}

    remainder = limit - sum(alloc.values())
    if remainder > 0:
        order = sorted(
            buckets.keys(),
            key=lambda k: (raw_alloc[k] - alloc[k], k),
            reverse=True,
        )
        for k in order[:remainder]:
            alloc[k] += 1

    rng = random.Random(seed)
    selected: list[tuple[ImageRecord, str]] = []
    for key, take in alloc.items():
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        selected.extend(bucket[:take])

    return sorted(selected, key=lambda item: item[0].image_id)


def _select_subset(
    items: list[tuple[ImageRecord, str]],
    *,
    limit: int,
    seed: int,
    strategy: str,
) -> list[tuple[ImageRecord, str]]:
    if limit <= 0 or len(items) <= limit:
        return items

    if strategy == "random":
        return _select_random(items, limit=limit, seed=seed)
    if strategy == "stratified":
        return _select_stratified(items, limit=limit, seed=seed)

    raise ValueError(
        f"Unknown subset.strategy={strategy!r}. Expected 'random' or 'stratified'."
    )


def materialize(cfg: DictConfig) -> None:
    class_map = build_class_map(cfg)
    source_category_map = build_source_category_map(cfg)
    strategy = str(cfg.subset.strategy)

    train_items = _records_from_split(
        labels_dir=Path(str(cfg.data.labels.train)),
        images_dir=Path(str(cfg.data.images.train)),
        split="train",
        class_map=class_map,
        source_category_map=source_category_map,
    )

    val_items = _records_from_split(
        labels_dir=Path(str(cfg.data.labels.val)),
        images_dir=Path(str(cfg.data.images.val)),
        split="val",
        class_map=class_map,
        source_category_map=source_category_map,
    )

    train_items = _select_subset(
        train_items,
        limit=int(cfg.subset.train_limit),
        seed=int(cfg.subset.seed),
        strategy=strategy,
    )

    train_records = [item[0] for item in train_items]
    val_records = [item[0] for item in val_items]

    train_count = write_manifest(Path(str(cfg.data.manifests.train)), train_records)
    val_count = write_manifest(Path(str(cfg.data.manifests.val)), val_records)
    overfit_count = write_manifest(
        Path(str(cfg.data.manifests.overfit16)), train_records[:16]
    )

    train_strata = Counter(item[1] for item in train_items)

    metadata_dir = Path(str(cfg.data.metadata_dir))
    write_json(
        metadata_dir / "classes.json",
        {
            "names": list(class_map.names),
            "id_to_name": {str(k): v for k, v in class_map.id_to_name.items()},
        },
    )
    write_json(
        metadata_dir / "source_category_map.json",
        source_category_map,
    )
    write_json(
        metadata_dir / "dataset.json",
        {
            "name": "bdd_det10",
            "source_layout": "bdd_native_per_image_json",
            "train_count": train_count,
            "val_count": val_count,
            "overfit_count": overfit_count,
            "train_limit": int(cfg.subset.train_limit),
            "seed": int(cfg.subset.seed),
            "strategy": strategy,
            "train_strata": dict(sorted(train_strata.items())),
        },
    )

    print(
        f"[adp] wrote train manifest: {cfg.data.manifests.train} ({train_count} images)"
    )
    print(f"[adp] wrote val manifest: {cfg.data.manifests.val} ({val_count} images)")
    print(
        f"[adp] wrote overfit16 manifest: {cfg.data.manifests.overfit16} ({overfit_count} images)"
    )


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_paths = ensure_run_paths(cfg)
    write_resolved_config(cfg, run_paths.root / "config.resolved.yaml")
    materialize(cfg)


if __name__ == "__main__":
    main()

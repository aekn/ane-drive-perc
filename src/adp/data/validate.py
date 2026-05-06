from collections import Counter
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from adp.config import write_resolved_config
from adp.data.classes import build_class_map
from adp.data.manifest import read_manifest
from adp.data.records import ImageRecord
from adp.utils.images import verify_image
from adp.utils.io import write_json
from adp.utils.paths import ensure_run_paths


def _record_image_path(record: ImageRecord, *, data_root: Path) -> Path:
    path = record.path
    if path.is_absolute():
        return path
    return data_root / path


def _validate_manifest(cfg: DictConfig, manifest_path: Path) -> dict[str, Any]:
    class_map = build_class_map(cfg)
    data_root = Path(str(cfg.data.root))

    image_count = 0
    empty_image_count = 0
    object_count = 0
    class_counts: Counter[str] = Counter()
    errors: list[str] = []

    for record in read_manifest(manifest_path):
        image_count += 1
        image_path = _record_image_path(record, data_root=data_root)

        try:
            record.validate(num_classes=class_map.num_classes)
        except Exception as exc:
            errors.append(f"{record.image_id}: {exc}")

        if bool(cfg.validation.require_images) and not image_path.exists():
            errors.append(f"{record.image_id}: missing image: {image_path}")

        if bool(cfg.validation.verify_images) and image_path.exists():
            try:
                verify_image(image_path)
            except Exception as exc:
                errors.append(
                    f"{record.image_id}: invalid image file: {image_path}: {exc}"
                )

        if not record.objects:
            empty_image_count += 1

        for obj in record.objects:
            object_count += 1
            class_counts[obj.class_name] += 1

            expected_name = class_map.id_to_name.get(obj.class_id)
            if expected_name != obj.class_name:
                errors.append(
                    f"{record.image_id}: class mismatch id={obj.class_id} "
                    f"name={obj.class_name!r} expected={expected_name!r}"
                )

    return {
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "image_count": image_count,
        "empty_image_count": empty_image_count,
        "object_count": object_count,
        "class_counts": dict(sorted(class_counts.items())),
        "errors": errors,
    }


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_paths = ensure_run_paths(cfg)
    write_resolved_config(cfg, run_paths.root / "config.resolved.yaml")

    train_report = _validate_manifest(cfg, Path(str(cfg.data.manifests.train)))
    val_report = _validate_manifest(cfg, Path(str(cfg.data.manifests.val)))

    report = {
        "train": train_report,
        "val": val_report,
    }

    output_path = run_paths.reports / "dataset_validation.json"
    write_json(output_path, report)

    errors = list(train_report["errors"]) + list(val_report["errors"])

    if errors:
        print(f"[adp] validation failed: {output_path}")
        for error in errors[: int(cfg.validation.max_errors)]:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"[adp] validation passed: {output_path}")
    print(f"[adp] train images: {train_report['image_count']}")
    print(f"[adp] train objects: {train_report['object_count']}")
    print(f"[adp] val images: {val_report['image_count']}")
    print(f"[adp] val objects: {val_report['object_count']}")


if __name__ == "__main__":
    main()

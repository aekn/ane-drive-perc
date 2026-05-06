import os
import shutil
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from adp.config import write_resolved_config
from adp.data.classes import build_class_map
from adp.data.manifest import read_manifest
from adp.data.records import ImageRecord
from adp.utils.io import write_json
from adp.utils.paths import ensure_run_paths


def _record_image_path(record: ImageRecord, *, data_root: Path) -> Path:
    path = record.path
    if path.is_absolute():
        return path
    return data_root / path


def _link_or_copy(src: Path, dst: Path, *, copy: bool, overwrite: bool) -> None:
    src_resolved = src.resolve()

    if dst.is_symlink() or dst.exists():
        try:
            if dst.resolve() == src_resolved:
                return
        except OSError:
            pass

        if not overwrite:
            raise FileExistsError(
                f"Refusing to replace existing file (export.overwrite=false): {dst}"
            )
        dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)

    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src_resolved, dst)


def _coco_categories(
    class_names: tuple[str, ...], *, base: int
) -> list[dict[str, Any]]:
    return [
        {"id": idx + base, "name": name, "supercategory": "object"}
        for idx, name in enumerate(class_names)
    ]


def _xyxy_to_xywh(bbox: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def _export_split(
    *,
    manifest_name: str,
    manifest_path: Path,
    data_root: Path,
    coco_root: Path,
    images_split_dir: str,
    class_names: tuple[str, ...],
    category_id_base: int,
    copy_images: bool,
    overwrite: bool,
) -> dict[str, Any]:
    images_dir = coco_root / "images" / images_split_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1

    for record in read_manifest(manifest_path):
        source_image = _record_image_path(record, data_root=data_root)
        if not source_image.exists():
            raise FileNotFoundError(
                f"Missing image referenced by manifest {manifest_path}: {source_image}"
            )

        dst_image = images_dir / source_image.name
        _link_or_copy(source_image, dst_image, copy=copy_images, overwrite=overwrite)

        image_id = next_image_id
        next_image_id += 1

        coco_images.append(
            {
                "id": image_id,
                "file_name": f"images/{images_split_dir}/{source_image.name}",
                "width": record.width,
                "height": record.height,
            }
        )

        for obj in record.objects:
            coco_annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": obj.class_id + category_id_base,
                    "bbox": _xyxy_to_xywh(obj.bbox_xyxy),
                    "area": obj.area,
                    "iscrowd": 1 if obj.iscrowd else 0,
                    "segmentation": [],
                }
            )
            next_annotation_id += 1

    coco_doc = {
        "info": {
            "description": f"ADP COCO export ({manifest_name})",
            "source_manifest": str(manifest_path),
            "category_id_base": category_id_base,
        },
        "categories": _coco_categories(class_names, base=category_id_base),
        "images": coco_images,
        "annotations": coco_annotations,
    }

    annotations_path = coco_root / "annotations" / f"instances_{manifest_name}.json"
    write_json(annotations_path, coco_doc)

    return {
        "manifest": str(manifest_path),
        "annotations": str(annotations_path),
        "images_dir": str(images_dir),
        "image_count": len(coco_images),
        "object_count": len(coco_annotations),
    }


def export_coco(cfg: DictConfig) -> dict[str, Any]:
    class_map = build_class_map(cfg)
    data_root = Path(str(cfg.data.root))
    coco_root = Path(str(cfg.coco.output_dir))
    category_id_base = int(cfg.coco.category_id_base)
    copy_images = bool(cfg.export.copy_images)
    overwrite = bool(cfg.export.overwrite)

    coco_root.mkdir(parents=True, exist_ok=True)
    (coco_root / "annotations").mkdir(parents=True, exist_ok=True)
    (coco_root / "metadata").mkdir(parents=True, exist_ok=True)

    splits = (
        ("train", Path(str(cfg.data.manifests.train)), "train"),
        ("val", Path(str(cfg.data.manifests.val)), "val"),
        ("overfit16", Path(str(cfg.data.manifests.overfit16)), "train"),
    )

    split_reports: dict[str, dict[str, Any]] = {}
    for manifest_name, manifest_path, images_split_dir in splits:
        split_reports[manifest_name] = _export_split(
            manifest_name=manifest_name,
            manifest_path=manifest_path,
            data_root=data_root,
            coco_root=coco_root,
            images_split_dir=images_split_dir,
            class_names=class_map.names,
            category_id_base=category_id_base,
            copy_images=copy_images,
            overwrite=overwrite,
        )

    metadata = {
        "coco_root": str(coco_root),
        "data_root": str(data_root),
        "category_id_base": category_id_base,
        "copy_images": copy_images,
        "categories": _coco_categories(class_map.names, base=category_id_base),
        "adp_class_id_to_coco_category_id": {
            class_id: class_id + category_id_base
            for class_id in class_map.id_to_name
        },
        "splits": split_reports,
    }
    write_json(coco_root / "metadata" / "export_coco.json", metadata)
    return metadata


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_paths = ensure_run_paths(cfg)
    write_resolved_config(cfg, run_paths.root / "config.resolved.yaml")

    metadata = export_coco(cfg)
    write_json(run_paths.reports / "export_coco.json", metadata)

    print(f"[adp] wrote COCO export: {metadata['coco_root']}")
    for name, report in metadata["splits"].items():
        print(
            f"[adp] {name}: {report['image_count']} images, "
            f"{report['object_count']} objects -> {report['annotations']}"
        )
    print(f"[adp] wrote metadata: {metadata['coco_root']}/metadata/export_coco.json")


if __name__ == "__main__":
    main()

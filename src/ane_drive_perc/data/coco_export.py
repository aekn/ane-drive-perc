import io
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from PIL import Image
from webdataset.compat import WebDataset

from ane_drive_perc.data.bdd import BDDMetadata, parse_bdd_metadata
from ane_drive_perc.data.class_map import BDD100K_DET_CLASSES
from ane_drive_perc.data.manifest import read_manifest_image_ids


@dataclass(frozen=True)
class CocoMaterializeResult:
    output_dir: Path
    image_dir: Path
    annotation_path: Path
    num_images: int
    num_annotations: int


def materialize_coco_from_local_shards(
    *,
    shards: list[str | Path],
    output_dir: str | Path,
    split: str,
    image_key: str = "jpg",
    metadata_key: str = "json",
    subset_manifest: str | Path | None = None,
    max_images: int | None = None,
    overwrite: bool = False,
) -> CocoMaterializeResult:
    if max_images is not None and max_images <= 0:
        raise ValueError(f"max_images must be positive or None, got {max_images}.")

    output_root = Path(output_dir)
    image_dir = output_root / "images" / split
    annotation_dir = output_root / "annotations"
    annotation_path = annotation_dir / f"instances_{split}.json"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    allowed_ids = (
        read_manifest_image_ids(subset_manifest)
        if subset_manifest is not None
        else None
    )

    dataset = WebDataset([str(path) for path in shards], shardshuffle=False)
    samples = cast(Iterable[dict[str, Any]], dataset)

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    image_index = 0
    annotation_index = 0

    for sample in samples:
        fallback_id = _sample_fallback_id(sample, image_index=image_index)
        image_raw = _expect_bytes(sample, image_key, fallback_id=fallback_id)
        metadata_raw = _expect_bytes(sample, metadata_key, fallback_id=fallback_id)
        metadata = parse_bdd_metadata(
            decode_json(metadata_raw), fallback_image_id=fallback_id
        )

        if allowed_ids is not None and metadata.image_id not in allowed_ids:
            continue

        width, height = extract_size(metadata, image_raw)
        file_name = f"{metadata.image_id}.{extension_for_image_key(image_key)}"
        image_path = image_dir / file_name
        if overwrite or not image_path.exists():
            image_path.write_bytes(image_raw)

        image_index += 1
        coco_image_id = image_index
        coco_images.append(
            {
                "id": coco_image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        for obj in metadata.objects:
            x1, y1, x2, y2 = clip_xyxy(obj.xyxy, width=width, height=height)
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width <= 0 or box_height <= 0:
                continue

            annotation_index += 1
            coco_annotations.append(
                {
                    "id": annotation_index,
                    "image_id": coco_image_id,
                    "category_id": obj.label + 1,
                    "bbox": [x1, y1, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "segmentation": [],
                    "original_category_id": obj.label,
                    "category": obj.category,
                    "source_category": obj.source_category,
                    "occluded": obj.occluded,
                    "truncated": obj.truncated,
                }
            )

        if max_images is not None and image_index >= max_images:
            break

    coco = {
        "info": {
            "description": "ANE Drive Perception BDD100K detection export",
            "version": "0.1.0",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": build_coco_categories(),
    }
    with annotation_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f)

    return CocoMaterializeResult(
        output_dir=output_root,
        image_dir=image_dir,
        annotation_path=annotation_path,
        num_images=len(coco_images),
        num_annotations=len(coco_annotations),
    )


def build_coco_categories() -> list[dict[str, Any]]:
    return [
        {"id": index + 1, "name": name, "supercategory": "driving"}
        for index, name in enumerate(BDD100K_DET_CLASSES)
    ]


def decode_json(raw: bytes) -> dict[str, Any]:
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Expected metadata JSON to decode to an object.")
    return decoded


def extract_size(metadata: BDDMetadata, image_raw: bytes) -> tuple[int, int]:
    if metadata.width is not None and metadata.height is not None:
        return metadata.width, metadata.height
    with Image.open(io.BytesIO(image_raw)) as image:
        return image.width, image.height


def extension_for_image_key(image_key: str) -> str:
    normalized = image_key.lower()
    if normalized in {"jpg", "jpeg"}:
        return "jpg"
    if normalized == "png":
        return "png"
    return normalized


def clip_xyxy(
    box: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )


def _sample_fallback_id(sample: dict[str, Any], *, image_index: int) -> str:
    sample_key = sample.get("__key__")
    if isinstance(sample_key, str) and sample_key:
        return Path(sample_key).stem
    return f"sample_{image_index:08d}"


def _expect_bytes(sample: dict[str, Any], key: str, *, fallback_id: str) -> bytes:
    if key not in sample:
        raise KeyError(f"Sample {fallback_id!r} is missing key {key!r}.")
    value = sample[key]
    if not isinstance(value, bytes):
        raise TypeError(f"Expected sample {fallback_id!r} field {key!r} to be bytes.")
    return value

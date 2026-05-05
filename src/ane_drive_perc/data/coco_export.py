import io
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from PIL import Image
from webdataset.compat import WebDataset

from ane_drive_perc.data.bdd import parse_bdd_metadata
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
    category_id_base: int = 1,
) -> CocoMaterializeResult:
    """Materialize BDD WebDataset shards into COCO detection format.

    category_id_base controls category IDs in the exported COCO annotations.

    - category_id_base=1: conventional COCO-style IDs, 1..N.
    - category_id_base=0: framework class-index IDs, 0..N-1.

    D-FINE needs 0-based category IDs when remap_mscoco_category=false and
    num_classes=10, otherwise class 10 is out of bounds.
    """
    if max_images is not None and max_images <= 0:
        raise ValueError(f"max_images must be positive or None, got {max_images}.")

    if category_id_base not in {0, 1}:
        raise ValueError(f"category_id_base must be 0 or 1, got {category_id_base}.")

    output_root = Path(output_dir)
    image_dir = output_root / "images" / split
    annotation_dir = output_root / "annotations"
    annotation_path = annotation_dir / f"instances_{split}.json"

    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    allowed_ids: set[str] | None = None
    if subset_manifest is not None:
        allowed_ids = read_manifest_image_ids(subset_manifest)

    dataset = WebDataset([str(path) for path in shards], shardshuffle=False)
    samples = cast(Iterable[dict[str, Any]], dataset)

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []

    annotation_index = 0

    for sample in samples:
        sample_key = sample.get("__key__")
        fallback_id = Path(str(sample_key)).stem if sample_key is not None else "sample"

        if image_key not in sample:
            raise KeyError(
                f"Sample {fallback_id!r} is missing image key {image_key!r}."
            )

        if metadata_key not in sample:
            raise KeyError(
                f"Sample {fallback_id!r} is missing metadata key {metadata_key!r}."
            )

        image_raw = sample[image_key]
        metadata_raw = sample[metadata_key]

        if not isinstance(image_raw, bytes):
            raise TypeError(
                f"Expected image field {image_key!r} to be bytes for {fallback_id!r}."
            )

        if not isinstance(metadata_raw, bytes):
            raise TypeError(
                f"Expected metadata field {metadata_key!r} to be bytes for {fallback_id!r}."
            )

        metadata_dict = decode_json(metadata_raw)
        metadata = parse_bdd_metadata(metadata_dict, fallback_image_id=fallback_id)
        image_id_text = metadata.image_id

        if allowed_ids is not None and image_id_text not in allowed_ids:
            continue

        width, height = extract_size(
            metadata_width=metadata.width,
            metadata_height=metadata.height,
            image_raw=image_raw,
        )

        file_name = f"{image_id_text}.{extension_for_image_key(image_key)}"
        image_path = image_dir / file_name

        if overwrite or not image_path.exists():
            image_path.write_bytes(image_raw)

        coco_image_id = len(coco_images) + 1

        coco_images.append(
            {
                "id": coco_image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        for obj in metadata.objects:
            if obj.label < 0 or obj.label >= len(BDD100K_DET_CLASSES):
                raise ValueError(
                    f"Object label out of range for image_id={image_id_text!r}: {obj.label}"
                )

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
                    "category_id": int(obj.label) + category_id_base,
                    "bbox": [x1, y1, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "segmentation": [],
                    "original_category_id": int(obj.label),
                    "category": obj.category,
                    "source_category": obj.source_category,
                    "occluded": obj.occluded,
                    "truncated": obj.truncated,
                }
            )

        if max_images is not None and len(coco_images) >= max_images:
            break

    coco = {
        "info": {
            "description": "ANE Drive Perception BDD100K detection export",
            "version": "0.1.0",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": build_coco_categories(category_id_base=category_id_base),
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


def build_coco_categories(*, category_id_base: int = 1) -> list[dict[str, Any]]:
    return [
        {
            "id": index + category_id_base,
            "name": name,
            "supercategory": "driving",
        }
        for index, name in enumerate(BDD100K_DET_CLASSES)
    ]


def decode_json(raw: bytes) -> dict[str, Any]:
    decoded = json.loads(raw.decode("utf-8"))

    if not isinstance(decoded, dict):
        raise ValueError("Expected metadata JSON to decode to an object.")

    return decoded


def extract_size(
    *,
    metadata_width: int | None,
    metadata_height: int | None,
    image_raw: bytes,
) -> tuple[int, int]:
    if metadata_width is not None and metadata_height is not None:
        if metadata_width > 0 and metadata_height > 0:
            return metadata_width, metadata_height

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

    clipped_x1 = min(max(float(x1), 0.0), float(width))
    clipped_y1 = min(max(float(y1), 0.0), float(height))
    clipped_x2 = min(max(float(x2), 0.0), float(width))
    clipped_y2 = min(max(float(y2), 0.0), float(height))

    return clipped_x1, clipped_y1, clipped_x2, clipped_y2

import io
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from webdataset.compat import WebDataset

from ane_drive_perc.data.class_map import encode_bdd100k_label
from ane_drive_perc.data.manifest import read_manifest_image_ids


RawSample = dict[str, Any]
ParsedSample = dict[str, Any]
DetectionTransform = Callable[[ParsedSample], ParsedSample]


class BDD100KWebDataset(IterableDataset[ParsedSample]):
    def __init__(
        self,
        shards: list[str | Path],
        *,
        subset_manifest: str | Path | None = None,
        image_key: str = "jpg",
        metadata_key: str = "json",
        shuffle_buffer: int = 0,
        transform: DetectionTransform | None = None,
        shardshuffle: int | bool = 0,
    ) -> None:
        super().__init__()

        self.shards = [str(path) for path in shards]
        self.image_key = image_key
        self.metadata_key = metadata_key
        self.shuffle_buffer = shuffle_buffer
        self.transform = transform
        self.shardshuffle = shardshuffle

        self.allowed_image_ids: set[str] | None = None
        if subset_manifest is not None:
            self.allowed_image_ids = read_manifest_image_ids(subset_manifest)

    def __iter__(self) -> Iterator[ParsedSample]:
        pipeline = WebDataset(self.shards, shardshuffle=self.shardshuffle)

        if self.shuffle_buffer > 0:
            pipeline = pipeline.shuffle(self.shuffle_buffer)

        sample_iter = cast(Iterable[RawSample], pipeline)

        for raw_sample in sample_iter:
            parsed = self._parse_sample(raw_sample)
            if parsed is None:
                continue

            if self.transform is not None:
                parsed = self.transform(parsed)

            yield parsed

    def _parse_sample(self, sample: RawSample) -> ParsedSample | None:
        sample_key = sample.get("__key__")
        if not isinstance(sample_key, str):
            raise ValueError("WebDataset sample is missing '__key__'.")

        fallback_image_id = Path(sample_key).stem

        if self.image_key not in sample:
            raise KeyError(
                f"Sample '{fallback_image_id}' is missing image key '{self.image_key}'."
            )

        if self.metadata_key not in sample:
            raise KeyError(
                f"Sample '{fallback_image_id}' is missing metadata key '{self.metadata_key}'."
            )

        image_raw = sample[self.image_key]
        metadata_raw = sample[self.metadata_key]

        if not isinstance(image_raw, bytes):
            raise TypeError(
                f"Expected image field '{self.image_key}' for sample "
                f"'{fallback_image_id}' to be bytes."
            )

        if not isinstance(metadata_raw, bytes):
            raise TypeError(
                f"Expected metadata field '{self.metadata_key}' for sample "
                f"'{fallback_image_id}' to be bytes."
            )

        metadata = self._decode_json(metadata_raw)

        image_id = self._extract_metadata_image_id(metadata, fallback=fallback_image_id)

        if (
            self.allowed_image_ids is not None
            and image_id not in self.allowed_image_ids
        ):
            return None

        image = self._decode_image(image_raw)

        target = self._parse_detection_target(
            image_id=image_id,
            metadata=metadata,
            width=image.width,
            height=image.height,
        )

        image_tensor = self._pil_to_tensor(image)

        return {
            "image": image_tensor,
            "target": target,
        }

    @staticmethod
    def _decode_image(raw: bytes) -> Image.Image:
        return Image.open(io.BytesIO(raw)).convert("RGB")

    @staticmethod
    def _decode_json(raw: bytes) -> dict[str, Any]:
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("Expected decoded metadata JSON to be an object.")
        return decoded

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.array(image, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor.float() / 255.0

    @staticmethod
    def _extract_metadata_image_id(metadata: dict[str, Any], *, fallback: str) -> str:
        for key in ("id", "image_id", "name", "sample_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value

        image_path = metadata.get("image")
        if isinstance(image_path, str) and image_path:
            return Path(image_path).stem

        return fallback

    @staticmethod
    def _parse_detection_target(
        *,
        image_id: str,
        metadata: dict[str, Any],
        width: int,
        height: int,
    ) -> dict[str, Any]:
        boxes: list[list[float]] = []
        labels: list[int] = []
        occluded: list[bool] = []
        truncated: list[bool] = []
        categories: list[str] = []

        if "boxes" in metadata:
            raw_boxes = metadata["boxes"]
            if not isinstance(raw_boxes, list):
                raise ValueError(
                    f"Expected 'boxes' to be a list for image_id='{image_id}'."
                )

            for obj in raw_boxes:
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Expected box entry to be a dict for image_id='{image_id}'."
                    )

                xyxy = obj.get("xyxy")
                if not isinstance(xyxy, list | tuple) or len(xyxy) != 4:
                    raise ValueError(
                        f"Invalid xyxy box for image_id='{image_id}': {xyxy}"
                    )

                if "category_id" in obj:
                    label = int(obj["category_id"])
                elif "category" in obj:
                    label = encode_bdd100k_label(str(obj["category"]))
                else:
                    raise ValueError(
                        f"Box is missing category/category_id for image_id='{image_id}'."
                    )

                boxes.append([float(v) for v in xyxy])
                labels.append(label)
                occluded.append(bool(obj.get("occluded", False)))
                truncated.append(bool(obj.get("truncated", False)))
                categories.append(str(obj.get("category", "")))

        elif "objects" in metadata:
            objects = metadata["objects"]
            if not isinstance(objects, list):
                raise ValueError(
                    f"Expected 'objects' to be a list for image_id='{image_id}'."
                )

            for obj in objects:
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Expected object entry to be a dict for image_id='{image_id}'."
                    )

                box = obj["box_xyxy_abs"]
                label = obj["label"]

                boxes.append([float(v) for v in box])
                labels.append(int(label))
                occluded.append(bool(obj.get("occluded", False)))
                truncated.append(bool(obj.get("truncated", False)))
                categories.append(str(obj.get("category", "")))

        elif "labels" in metadata:
            raw_labels = metadata["labels"]
            if not isinstance(raw_labels, list):
                raise ValueError(
                    f"Expected 'labels' to be a list for image_id='{image_id}'."
                )

            for obj in raw_labels:
                if not isinstance(obj, dict):
                    continue

                category = obj.get("category")
                box2d = obj.get("box2d")

                if category is None or box2d is None:
                    continue

                if not isinstance(box2d, dict):
                    continue

                boxes.append(
                    [
                        float(box2d["x1"]),
                        float(box2d["y1"]),
                        float(box2d["x2"]),
                        float(box2d["y2"]),
                    ]
                )
                labels.append(encode_bdd100k_label(str(category)))
                occluded.append(bool(obj.get("attributes", {}).get("occluded", False)))
                truncated.append(
                    bool(obj.get("attributes", {}).get("truncated", False))
                )
                categories.append(str(category))

        else:
            raise ValueError(
                f"Metadata for image_id='{image_id}' does not contain "
                f"'boxes', 'objects', or 'labels'."
            )

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        attributes = {
            "weather": metadata.get("weather"),
            "scene": metadata.get("scene"),
            "timeofday": metadata.get("timeofday"),
            "split": metadata.get("split"),
            "image": metadata.get("image"),
            "label": metadata.get("label"),
        }

        return {
            "image_id": image_id,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "orig_boxes": boxes_tensor.clone(),
            "orig_size": (height, width),
            "input_size": (height, width),
            "occluded": torch.tensor(occluded, dtype=torch.bool),
            "truncated": torch.tensor(truncated, dtype=torch.bool),
            "categories": categories,
            "attributes": attributes,
            "raw_metadata": metadata,
        }

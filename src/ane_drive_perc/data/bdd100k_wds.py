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

from ane_drive_perc.data.bdd import BDDMetadata, parse_bdd_metadata
from ane_drive_perc.data.manifest import read_manifest_image_ids

RawSample = dict[str, Any]
ParsedSample = dict[str, Any]
DetectionTransform = Callable[[ParsedSample], ParsedSample]


class BDD100KWebDataset(IterableDataset[ParsedSample]):
    """BDD100K detection dataset backed by WebDataset tar shards."""

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
        self.allowed_image_ids = (
            read_manifest_image_ids(subset_manifest)
            if subset_manifest is not None
            else None
        )

    def __iter__(self) -> Iterator[ParsedSample]:
        pipeline = WebDataset(self.shards, shardshuffle=self.shardshuffle)
        if self.shuffle_buffer > 0:
            pipeline = pipeline.shuffle(self.shuffle_buffer)

        for raw_sample in cast(Iterable[RawSample], pipeline):
            parsed = self._parse_sample(raw_sample)
            if parsed is None:
                continue
            if self.transform is not None:
                parsed = self.transform(parsed)
            yield parsed

    def _parse_sample(self, sample: RawSample) -> ParsedSample | None:
        fallback_image_id = self._fallback_image_id(sample)
        image_raw = self._expect_bytes(
            sample, self.image_key, image_id=fallback_image_id
        )
        metadata_raw = self._expect_bytes(
            sample, self.metadata_key, image_id=fallback_image_id
        )

        metadata = parse_bdd_metadata(
            self._decode_json(metadata_raw),
            fallback_image_id=fallback_image_id,
        )

        if (
            self.allowed_image_ids is not None
            and metadata.image_id not in self.allowed_image_ids
        ):
            return None

        image = self._decode_image(image_raw)
        image_tensor = self._pil_to_tensor(image)
        target = self._to_target(metadata, width=image.width, height=image.height)
        return {"image": image_tensor, "target": target}

    @staticmethod
    def _fallback_image_id(sample: RawSample) -> str:
        sample_key = sample.get("__key__")
        if not isinstance(sample_key, str):
            raise ValueError("WebDataset sample is missing '__key__'.")
        return Path(sample_key).stem

    @staticmethod
    def _expect_bytes(sample: RawSample, key: str, *, image_id: str) -> bytes:
        if key not in sample:
            raise KeyError(f"Sample {image_id!r} is missing key {key!r}.")
        value = sample[key]
        if not isinstance(value, bytes):
            raise TypeError(f"Expected sample {image_id!r} field {key!r} to be bytes.")
        return value

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
    def _to_target(metadata: BDDMetadata, *, width: int, height: int) -> dict[str, Any]:
        boxes = torch.tensor(
            [obj.xyxy for obj in metadata.objects], dtype=torch.float32
        ).reshape(-1, 4)
        labels = torch.tensor([obj.label for obj in metadata.objects], dtype=torch.long)
        return {
            "image_id": metadata.image_id,
            "boxes": boxes,
            "labels": labels,
            "orig_boxes": boxes.clone(),
            "orig_size": (height, width),
            "input_size": (height, width),
            "occluded": torch.tensor(
                [obj.occluded for obj in metadata.objects], dtype=torch.bool
            ),
            "truncated": torch.tensor(
                [obj.truncated for obj in metadata.objects], dtype=torch.bool
            ),
            "categories": [obj.category for obj in metadata.objects],
            "attributes": metadata.attributes,
        }

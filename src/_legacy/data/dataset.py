"""Manifest-driven BDD100K detection dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .labels import DETECTION_CATEGORIES


CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(DETECTION_CATEGORIES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class _LabelEntry:
    name: str
    boxes: torch.Tensor  # (M, 4) xyxy in original 1280x720 coords
    labels: torch.Tensor  # (M,) int


def _build_label_index(label_json: Path) -> dict[str, _LabelEntry]:
    with open(label_json) as f:
        raw = json.load(f)
    index: dict[str, _LabelEntry] = {}
    for r in raw:
        boxes_l: list[list[float]] = []
        labels_l: list[int] = []
        for lab in r.get("labels") or []:
            cat = lab.get("category")
            box = lab.get("box2d")
            if cat in CLASS_TO_ID and box is not None:
                boxes_l.append([box["x1"], box["y1"], box["x2"], box["y2"]])
                labels_l.append(CLASS_TO_ID[cat])
        boxes = (
            torch.tensor(boxes_l, dtype=torch.float32)
            if boxes_l
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labs = (
            torch.tensor(labels_l, dtype=torch.long)
            if labels_l
            else torch.zeros((0,), dtype=torch.long)
        )
        index[r["name"]] = _LabelEntry(name=r["name"], boxes=boxes, labels=labs)
    return index


def _normalize_(t: torch.Tensor) -> torch.Tensor:
    """In-place normalize a CHW float tensor in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=t.dtype, device=t.device).view(3, 1, 1)
    t.sub_(mean).div_(std)
    return t


class BDDDetection(Dataset):
    """BDD100K detection dataset filtered by subset manifest.

    Parameters:
        manifest_path: Path to JSON manifest.
        images_dir: Directory with image files.
        label_json: BDD canonical label JSON.
        image_size: Target square size after resize.
        train: Apply training augmentations.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        images_dir: str | Path,
        label_json: str | Path,
        image_size: int = 384,
        train: bool = True,
    ) -> None:
        super().__init__()
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.train = train

        with open(manifest_path) as f:
            manifest = json.load(f)
        self.ids: list[str] = list(manifest["image_ids"])
        self.manifest_name: str = manifest["name"]

        index = _build_label_index(Path(label_json))
        # Skip ids missing a label entry (BDD has a small number of these).
        self.ids = [name for name in self.ids if name in index]
        self._label_index = index

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        name = self.ids[idx]
        entry = self._label_index[name]

        img = Image.open(self.images_dir / name).convert("RGB")
        ow, oh = img.size
        s = self.image_size
        img = img.resize((s, s), Image.BILINEAR)

        # Convert to CHW float tensor, scale to [0, 1], normalize in-place.
        arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).clone()
        arr = arr.view(s, s, 3).permute(2, 0, 1).contiguous().float().div_(255.0)
        _normalize_(arr)

        # Rescale boxes to the resized image's pixel space.
        sx = s / ow
        sy = s / oh
        boxes = entry.boxes.clone()
        if boxes.numel() > 0:
            boxes[:, 0::2] *= sx
            boxes[:, 1::2] *= sy
            boxes[:, 0::2].clamp_(0, s)
            boxes[:, 1::2].clamp_(0, s)
        labels = entry.labels.clone()

        if self.train and torch.rand(()) < 0.5:
            arr = arr.flip(-1)
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = s - x2
                boxes[:, 2] = s - x1

        # Drop degenerate boxes after rescaling/clamping.
        if boxes.numel() > 0:
            wh = boxes[:, 2:] - boxes[:, :2]
            keep = (wh[:, 0] > 1) & (wh[:, 1] > 1)
            boxes = boxes[keep]
            labels = labels[keep]

        return {
            "image": arr,
            "boxes": boxes,
            "labels": labels,
            "name": name,
        }


def detection_collate(
    batch: list[dict[str, torch.Tensor | str]],
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[str]]:
    """Collate variable-length per-image GT into parallel lists."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]
    names = [b["name"] for b in batch]
    return images, boxes, labels, names

import json
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        *,
        coco_root: Path,
        annotations_file: str,
        transforms: A.Compose,
        category_id_base: int = 1,
    ) -> None:
        self.coco_root = Path(coco_root)
        ann_path = self.coco_root / "annotations" / annotations_file
        with ann_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        self.transforms = transforms
        self.category_id_base = category_id_base

        self.images = list(doc["images"])

        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in doc["annotations"]:
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
        self.anns_by_image = anns_by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        info = self.images[idx]
        image_id = int(info["id"])
        img_path = self.coco_root / info["file_name"]

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        anns = self.anns_by_image.get(image_id, [])
        in_bboxes = [ann["bbox"] for ann in anns]  # COCO xywh
        in_labels = [int(ann["category_id"]) - self.category_id_base for ann in anns]

        out = self.transforms(image=rgb, bboxes=in_bboxes, labels=in_labels)
        image = out["image"]  # CHW float in [0,1]
        post_h, post_w = int(image.shape[-2]), int(image.shape[-1])

        if len(out["bboxes"]) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            arr = np.asarray(out["bboxes"], dtype=np.float32)  # [N,4] xywh in pixels
            x, y, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            cx = (x + w / 2.0) / post_w
            cy = (y + h / 2.0) / post_h
            wn = w / post_w
            hn = h / post_h
            boxes = torch.from_numpy(np.stack([cx, cy, wn, hn], axis=1)).float()
            labels = torch.tensor(out["labels"], dtype=torch.long)

        target: dict[str, Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(image_id, dtype=torch.long),
            "orig_size": torch.tensor(
                [int(info["height"]), int(info["width"])], dtype=torch.long
            ),
        }
        return image, target


def collate_fn(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

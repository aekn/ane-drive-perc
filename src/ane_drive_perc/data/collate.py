import torch
from typing import Any


def detection_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    targets = [sample["target"] for sample in batch]

    return {
        "images": images,
        "targets": targets,
    }

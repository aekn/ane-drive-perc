from typing import Any

import torch


def detection_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([sample["image"] for sample in batch], dim=0),
        "targets": [sample["target"] for sample in batch],
    }

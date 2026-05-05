from pathlib import Path
from typing import Any
import torch

import numpy as np

from PIL import ImageDraw, ImageFont, Image

from ane_drive_perc.data.class_map import decode_bdd100k_label


def save_detection_visualization(
    image: torch.Tensor,
    target: dict[str, Any],
    output_path: str | Path,
    *,
    score_key: str | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    pil_image = tensor_to_pil_image(image)
    draw = ImageDraw.Draw(pil_image)

    boxes = target["boxes"]
    labels = target["labels"]

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.long)

    scores = None
    if score_key is not None and score_key in target:
        scores = target[score_key]
        if not isinstance(scores, torch.Tensor):
            scores = torch.as_tensor(scores, dtype=torch.float32)

    font = ImageFont.load_default()

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]

        if x2 <= x1 or y2 <= y1:
            continue

        label_idx = int(labels[index].item())
        label_name = safe_decode_label(label_idx)

        text = label_name
        if scores is not None:
            text = f"{label_name} {float(scores[index].item()):.2f}"

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=(255, 0, 0),
            width=2,
        )

        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_y = max(0.0, y1 - text_height - 2)

        draw.rectangle(
            [(x1, text_y), (x1 + text_width + 4, text_y + text_height + 4)],
            fill=(255, 0, 0),
        )
        draw.text(
            (x1 + 2, text_y + 2),
            text,
            fill=(255, 255, 255),
            font=font,
        )

    pil_image.save(output)


def tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3:
        raise ValueError(f"Expected image shape CxHxW, got {tuple(image.shape)}.")

    image_cpu = image.detach().cpu().clamp(0.0, 1.0)
    array = image_cpu.permute(1, 2, 0).numpy()
    array_uint8 = (array * 255.0).round().astype(np.uint8)

    return Image.fromarray(array_uint8, mode="RGB")


def safe_decode_label(label_idx: int) -> str:
    try:
        return decode_bdd100k_label(label_idx)
    except ValueError:
        return f"class_{label_idx}"

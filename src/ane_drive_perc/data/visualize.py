from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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
    font = ImageFont.load_default()

    boxes = _as_tensor(target["boxes"], dtype=torch.float32)
    labels = _as_tensor(target["labels"], dtype=torch.long)
    scores = None
    if score_key is not None and score_key in target:
        scores = _as_tensor(target[score_key], dtype=torch.float32)

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        if x2 <= x1 or y2 <= y1:
            continue

        label_name = safe_decode_label(int(labels[index].item()))
        text = (
            label_name
            if scores is None
            else f"{label_name} {float(scores[index].item()):.2f}"
        )
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)
        draw_label(draw, text=text, xy=(x1, y1), font=font)

    pil_image.save(output)


def tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3:
        raise ValueError(f"Expected image shape CxHxW, got {tuple(image.shape)}.")
    image_cpu = image.detach().cpu().clamp(0.0, 1.0)
    array = image_cpu.permute(1, 2, 0).numpy()
    array_uint8 = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array_uint8, mode="RGB")


def draw_label(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    xy: tuple[float, float],
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    text_bbox = draw.textbbox((x, y), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_y = max(0.0, y - text_height - 2)
    draw.rectangle(
        [(x, text_y), (x + text_width + 4, text_y + text_height + 4)],
        fill=(255, 0, 0),
    )
    draw.text((x + 2, text_y + 2), text, fill=(255, 255, 255), font=font)


def safe_decode_label(label_idx: int) -> str:
    try:
        return decode_bdd100k_label(label_idx)
    except ValueError:
        return f"class_{label_idx}"


def _as_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)

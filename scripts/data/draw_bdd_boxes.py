from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ane_drive_perc.data.bdd import read_bdd_label


PALETTE = {
    "pedestrian": (255, 80, 80),
    "rider": (255, 160, 80),
    "car": (80, 180, 255),
    "truck": (80, 255, 180),
    "bus": (180, 255, 80),
    "train": (180, 120, 255),
    "motorcycle": (255, 80, 200),
    "bicycle": (255, 220, 80),
    "traffic light": (80, 255, 80),
    "traffic sign": (255, 255, 80),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--thumb-width", type=int, default=320)
    p.add_argument("--max-boxes", type=int, default=80)
    return p.parse_args()


def image_path_for_label(split_dir: Path, label_path: Path) -> Path:
    return split_dir / "images" / f"{label_path.stem}.jpg"


def resize_with_boxes(
    image: Image.Image,
    boxes: list[dict],
    width: int,
) -> tuple[Image.Image, list[dict]]:
    src_w, src_h = image.size
    height = round(width * src_h / src_w)

    sx = width / src_w
    sy = height / src_h

    image = image.resize((width, height), Image.Resampling.BILINEAR)

    scaled = []
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        scaled.append(
            {
                **box,
                "xyxy": [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
            }
        )

    return image, scaled


def draw_boxes(image: Image.Image, boxes: list[dict], title: str) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rectangle([0, 0, image.width, 16], fill=(0, 0, 0))
    draw.text((4, 3), title, fill=(255, 255, 255), font=font)

    for box in boxes:
        category = box["category"]
        color = PALETTE[category]
        x1, y1, x2, y2 = box["xyxy"]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = category
        text_box = draw.textbbox((0, 0), label, font=font)
        tw = text_box[2] - text_box[0]
        th = text_box[3] - text_box[1]

        y_text = max(17, int(y1) - th - 3)
        draw.rectangle([x1, y_text, x1 + tw + 4, y_text + th + 3], fill=color)
        draw.text((x1 + 2, y_text + 1), label, fill=(0, 0, 0), font=font)

    return image


def make_contact_sheet(images: list[Image.Image], cols: int) -> Image.Image:
    rows = math.ceil(len(images) / cols)

    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)

    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(245, 245, 245))

    for i, image in enumerate(images):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(image, (x, y))

    return sheet


def main() -> None:
    args = parse_args()

    split_dir = Path(args.split_dir)
    label_paths = sorted((split_dir / "labels").glob("*.json"))

    samples = []
    for label_path in label_paths:
        label = read_bdd_label(label_path)
        if label["boxes"]:
            samples.append(label_path)

    rng = random.Random(args.seed)
    chosen = rng.sample(samples, min(args.num_samples, len(samples)))

    drawn = []
    for label_path in chosen:
        label = read_bdd_label(label_path)
        image_path = image_path_for_label(split_dir, label_path)

        image = Image.open(image_path)
        boxes = label["boxes"][: args.max_boxes]

        image, boxes = resize_with_boxes(image, boxes, args.thumb_width)
        image = draw_boxes(
            image, boxes, title=f"{label_path.stem} | boxes={len(label['boxes'])}"
        )

        drawn.append(image)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    sheet = make_contact_sheet(drawn, cols=args.cols)
    sheet.save(out, quality=95)

    print(f"[done] wrote {out}")
    print(f"       samples={len(drawn)}")


if __name__ == "__main__":
    main()

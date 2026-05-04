import argparse
import os
import shutil
from pathlib import Path

from ane_drive_perc.data.bdd import (
    DETECTION_CATEGORIES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    read_bdd_label,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-split-dir", required=True)
    p.add_argument("--val-split-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--yaml-name", default="data.yaml")
    p.add_argument("--copy-images", action="store_true")
    return p.parse_args()


def clip_xyxy(xyxy: list[float]) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = xyxy

    x1 = max(0.0, min(float(IMAGE_WIDTH), x1))
    y1 = max(0.0, min(float(IMAGE_HEIGHT), y1))
    x2 = max(0.0, min(float(IMAGE_WIDTH), x2))
    y2 = max(0.0, min(float(IMAGE_HEIGHT), y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def xyxy_to_yolo(
    xyxy: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy

    cx = ((x1 + x2) * 0.5) / IMAGE_WIDTH
    cy = ((y1 + y2) * 0.5) / IMAGE_HEIGHT
    w = (x2 - x1) / IMAGE_WIDTH
    h = (y2 - y1) / IMAGE_HEIGHT

    return cx, cy, w, h


def link_or_copy_image(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if copy_images:
        shutil.copy2(src, dst)
        return

    rel_src = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(rel_src)


def export_split(split_dir: Path, out_dir: Path, copy_images: bool) -> dict[str, int]:
    split_name = split_dir.name

    out_image_dir = out_dir / "images" / split_name
    out_label_dir = out_dir / "labels" / split_name

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    num_images = 0
    num_boxes = 0
    num_dropped = 0

    for label_path in sorted((split_dir / "labels").glob("*.json")):
        label = read_bdd_label(label_path)

        image_name = f"{label['id']}.jpg"
        src_image = split_dir / "images" / image_name
        dst_image = out_image_dir / image_name
        dst_label = out_label_dir / f"{label['id']}.txt"

        link_or_copy_image(src_image, dst_image, copy_images=copy_images)

        lines: list[str] = []

        for box in label["boxes"]:
            clipped = clip_xyxy(box["xyxy"])

            if clipped is None:
                num_dropped += 1
                continue

            cx, cy, w, h = xyxy_to_yolo(clipped)

            lines.append(f"{box['category_id']} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}")

            num_boxes += 1

        dst_label.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        num_images += 1

    return {
        "images": num_images,
        "boxes": num_boxes,
        "dropped": num_dropped,
    }


def write_dataset_yaml(
    out_dir: Path,
    yaml_path: Path,
    train_split_name: str,
    val_split_name: str,
) -> None:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(DETECTION_CATEGORIES))

    text = f"""path: {out_dir.resolve()}
train: images/{train_split_name}
val: images/{val_split_name}

nc: {len(DETECTION_CATEGORIES)}
names:
{names}
"""

    yaml_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    train_split_dir = Path(args.train_split_dir)
    val_split_dir = Path(args.val_split_dir)
    out_dir = Path(args.out_dir)

    train_stats = export_split(
        split_dir=train_split_dir,
        out_dir=out_dir,
        copy_images=args.copy_images,
    )
    val_stats = export_split(
        split_dir=val_split_dir,
        out_dir=out_dir,
        copy_images=args.copy_images,
    )

    yaml_path = out_dir / args.yaml_name
    write_dataset_yaml(
        out_dir=out_dir,
        yaml_path=yaml_path,
        train_split_name=train_split_dir.name,
        val_split_name=val_split_dir.name,
    )

    print(f"[train] {train_split_dir.name}: {train_stats}")
    print(f"[val]   {val_split_dir.name}: {val_stats}")
    print(f"[done]  wrote {yaml_path}")


if __name__ == "__main__":
    main()

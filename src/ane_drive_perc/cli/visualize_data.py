from __future__ import annotations

import argparse
from pathlib import Path

from ane_drive_perc.data.build import build_detection_dataloader_from_train_config
from ane_drive_perc.data.visualize import save_detection_visualization


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--shuffle-buffer", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_detection_dataloader_from_train_config(
        args.config,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
    )

    saved = 0

    for batch in loader:
        images = batch["images"]
        targets = batch["targets"]

        for image, target in zip(images, targets, strict=True):
            image_id = str(target["image_id"])
            safe_image_id = image_id.replace("/", "_")

            output_path = output_dir / f"{saved:06d}_{safe_image_id}.jpg"

            save_detection_visualization(
                image=image,
                target=target,
                output_path=output_path,
            )

            print(f"saved {output_path}")

            saved += 1
            if saved >= args.num_images:
                print(f"Saved {saved} visualizations to {output_dir}")
                return

    print(f"Dataset ended after saving {saved} visualizations to {output_dir}")

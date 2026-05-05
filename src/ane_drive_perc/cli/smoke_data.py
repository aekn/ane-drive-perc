import argparse

from ane_drive_perc.data.build import build_detection_dataloader_from_train_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--shuffle-buffer", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    loader = build_detection_dataloader_from_train_config(
        args.config,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
    )
    batch = next(iter(loader))
    images = batch["images"]
    targets = batch["targets"]

    print("loaded one detection batch")
    print(f"  split: {args.split}")
    print(f"  images: {tuple(images.shape)}")
    print(f"  targets: {len(targets)}")
    for index, target in enumerate(targets):
        print(
            f"  sample {index}: image_id={target['image_id']} "
            f"boxes={tuple(target['boxes'].shape)} labels={tuple(target['labels'].shape)} "
            f"orig_size={target['orig_size']} input_size={target['input_size']}"
        )

import argparse

from huggingface_hub.utils.tqdm import disable_progress_bars

from ane_drive_perc.data.coco_export import materialize_coco_from_local_shards
from ane_drive_perc.data.resolver import resolve_data_split_from_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--show-download-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show_download_progress:
        disable_progress_bars()

    resolved = resolve_data_split_from_config(args.config, split=args.split)
    result = materialize_coco_from_local_shards(
        shards=resolved.local_shards,
        output_dir=args.output_dir,
        split=args.split,
        image_key=resolved.image_key,
        metadata_key=resolved.metadata_key,
        subset_manifest=resolved.local_manifest,
        max_images=args.max_images,
        overwrite=args.overwrite,
    )

    print("materialized COCO dataset")
    print(f"  split: {args.split}")
    print(f"  images: {result.num_images}")
    print(f"  annotations: {result.num_annotations}")
    print(f"  image_dir: {result.image_dir}")
    print(f"  annotation_path: {result.annotation_path}")

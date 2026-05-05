from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from ane_drive_perc.config.access import (
    expect_float,
    expect_int,
    expect_mapping,
    expect_str,
    get_bool,
)
from ane_drive_perc.config.load import load_yaml
from ane_drive_perc.data.bdd100k_wds import BDD100KWebDataset
from ane_drive_perc.data.collate import detection_collate_fn
from ane_drive_perc.data.resolver import SplitName, resolve_data_split_from_config
from ane_drive_perc.data.transforms import DetectionResizeConfig, ResizeDetectionSample


def build_detection_dataloader_from_train_config(
    train_config_path: str | Path,
    *,
    split: SplitName,
    batch_size: int | None = None,
    num_workers: int = 0,
    shuffle_buffer: int = 0,
) -> DataLoader[Any]:
    train_config = load_yaml(train_config_path)
    data_section = expect_mapping(train_config, "data")
    data_config_path = expect_str(data_section, "config")
    resolved = resolve_data_split_from_config(data_config_path, split=split)

    transform = build_resize_transform(train_config, split=split)
    dataset = BDD100KWebDataset(
        resolved.local_shards,
        subset_manifest=resolved.local_manifest,
        image_key=resolved.image_key,
        metadata_key=resolved.metadata_key,
        shuffle_buffer=shuffle_buffer,
        transform=transform,
        shardshuffle=1000 if split == "train" else 0,
    )

    train_section = expect_mapping(train_config, "train")
    resolved_batch_size = (
        batch_size
        if batch_size is not None
        else expect_int(train_section, "batch_size")
    )

    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )


def build_resize_transform(
    train_config: dict[str, Any],
    *,
    split: SplitName,
) -> ResizeDetectionSample:
    input_section = expect_mapping(train_config, "input")
    size_key = "train_size" if split == "train" else "eval_size"
    size_section = expect_mapping(input_section, size_key)

    return ResizeDetectionSample(
        DetectionResizeConfig(
            height=expect_int(size_section, "height"),
            width=expect_int(size_section, "width"),
            preserve_aspect_ratio=get_bool(
                input_section, "preserve_aspect_ratio", default=True
            ),
            letterbox_value=expect_float(input_section, "letterbox_value") / 255.0,
        )
    )

from pathlib import Path
from typing import Any, Literal

from torch.utils.data import DataLoader

from ane_drive_perc.config.load import load_yaml
from ane_drive_perc.data.bdd100k_wds import BDD100KWebDataset
from ane_drive_perc.data.collate import detection_collate_fn
from ane_drive_perc.data.shards import download_repo_patterns, find_local_matches
from ane_drive_perc.data.transforms import DetectionResizeConfig, ResizeDetectionSample


SplitName = Literal["train", "val"]


def build_detection_dataloader_from_train_config(
    train_config_path: str | Path,
    *,
    split: SplitName,
    batch_size: int | None = None,
    num_workers: int = 0,
    shuffle_buffer: int = 0,
) -> DataLoader[Any]:
    train_config = load_yaml(train_config_path)

    data_section = _expect_dict(train_config, "data")
    data_config_path = _expect_str(data_section, "config")

    data_config = load_yaml(data_config_path)

    repo_id = _expect_str(data_config, "hf_repo")
    repo_type = str(data_config.get("hf_repo_type", "dataset"))
    revision = str(data_config.get("revision", "main"))

    cache_section = _expect_dict(data_config, "cache")
    local_cache_dir = _expect_str(cache_section, "local_dir")

    shards_section = _expect_dict(data_config, "shards")
    split_section = _expect_dict(shards_section, split)

    shard_pattern = _expect_str(split_section, "pattern")
    subset_manifest = split_section.get("subset_manifest")

    if subset_manifest is not None and not isinstance(subset_manifest, str):
        raise TypeError(
            f"Expected shards.{split}.subset_manifest to be a string or null."
        )

    download_patterns = [shard_pattern]
    if subset_manifest is not None:
        download_patterns.append(subset_manifest)

    local_root = download_repo_patterns(
        repo_id=repo_id,
        patterns=download_patterns,
        local_dir=local_cache_dir,
        repo_type=repo_type,
        revision=revision,
    )

    local_shards = find_local_matches(local_root, shard_pattern)
    local_shards = [path for path in local_shards if path.suffix == ".tar"]

    if not local_shards:
        raise FileNotFoundError(f"No .tar shards found for split {split!r}.")

    local_manifest_path: Path | None = None
    if subset_manifest is not None:
        manifest_matches = find_local_matches(local_root, subset_manifest)
        if len(manifest_matches) != 1:
            raise RuntimeError(
                f"Expected exactly one manifest for {subset_manifest!r}, found {len(manifest_matches)}."
            )
        local_manifest_path = manifest_matches[0]

    sample_keys = _expect_dict(data_config, "sample_keys")
    image_key = _expect_str(sample_keys, "image")
    metadata_key = _expect_str(sample_keys, "metadata")

    input_section = _expect_dict(train_config, "input")
    size_key = "train_size" if split == "train" else "eval_size"
    size_section = _expect_dict(input_section, size_key)

    height = _expect_int(size_section, "height")
    width = _expect_int(size_section, "width")

    preserve_aspect_ratio = bool(input_section.get("preserve_aspect_ratio", True))
    letterbox_value_raw = input_section.get("letterbox_value", 114)

    if not isinstance(letterbox_value_raw, int | float):
        raise TypeError("Expected input.letterbox_value to be numeric.")

    transform = ResizeDetectionSample(
        DetectionResizeConfig(
            height=height,
            width=width,
            preserve_aspect_ratio=preserve_aspect_ratio,
            letterbox_value=float(letterbox_value_raw) / 255.0,
        )
    )

    dataset = BDD100KWebDataset(
        local_shards,
        subset_manifest=local_manifest_path,
        image_key=image_key,
        metadata_key=metadata_key,
        shuffle_buffer=shuffle_buffer,
        transform=transform,
        shardshuffle=1000 if split == "train" else 0,
    )

    train_section = _expect_dict(train_config, "train")
    resolved_batch_size = (
        batch_size
        if batch_size is not None
        else _expect_int(train_section, "batch_size")
    )

    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )


def _expect_dict(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Expected {key!r} to be a mapping.")
    return value


def _expect_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise TypeError(f"Expected {key!r} to be a string.")
    return value


def _expect_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise TypeError(f"Expected {key!r} to be an integer.")
    return value

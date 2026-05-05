from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ane_drive_perc.config.access import (
    expect_mapping,
    expect_optional_str,
    expect_str,
)
from ane_drive_perc.config.load import load_yaml
from ane_drive_perc.data.shards import download_repo_patterns, find_local_matches

SplitName = Literal["train", "val"]


@dataclass(frozen=True)
class ResolvedDataSplit:
    split: SplitName
    local_shards: list[Path]
    local_manifest: Path | None
    image_key: str
    metadata_key: str
    local_root: Path
    data_config_path: Path | None
    data_config: dict[str, Any]


def resolve_data_split_from_config(
    data_config_path: str | Path,
    *,
    split: SplitName,
) -> ResolvedDataSplit:
    path = Path(data_config_path)
    return resolve_data_split(
        load_yaml(path),
        split=split,
        data_config_path=path,
    )


def resolve_data_split(
    data_config: dict[str, Any],
    *,
    split: SplitName,
    data_config_path: Path | None = None,
) -> ResolvedDataSplit:
    repo_id = expect_str(data_config, "hf_repo")
    repo_type = str(data_config.get("hf_repo_type", "dataset"))
    revision = str(data_config.get("revision", "main"))

    cache_section = expect_mapping(data_config, "cache")
    local_cache_dir = expect_str(cache_section, "local_dir")

    shards_section = expect_mapping(data_config, "shards")
    split_section = expect_mapping(shards_section, split)
    shard_pattern = expect_str(split_section, "pattern")
    subset_manifest = expect_optional_str(split_section, "subset_manifest")

    sample_keys = expect_mapping(data_config, "sample_keys")
    image_key = expect_str(sample_keys, "image")
    metadata_key = expect_str(sample_keys, "metadata")

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

    local_shards = [
        path
        for path in find_local_matches(local_root, shard_pattern)
        if path.suffix == ".tar"
    ]
    if not local_shards:
        raise FileNotFoundError(f"No .tar shards found for split {split!r}.")

    local_manifest: Path | None = None
    if subset_manifest is not None:
        manifest_matches = find_local_matches(local_root, subset_manifest)
        if len(manifest_matches) != 1:
            raise RuntimeError(
                f"Expected exactly one manifest for {subset_manifest!r}, "
                f"found {len(manifest_matches)}."
            )
        local_manifest = manifest_matches[0]

    return ResolvedDataSplit(
        split=split,
        local_shards=local_shards,
        local_manifest=local_manifest,
        image_key=image_key,
        metadata_key=metadata_key,
        local_root=local_root,
        data_config_path=data_config_path,
        data_config=data_config,
    )

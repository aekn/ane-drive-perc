import os
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def get_hf_token() -> str | bool | None:
    token = os.environ.get("HF_TOKEN")

    if token:
        return token

    return True


def list_hf_repo_files(
    repo_id: str,
    *,
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | bool | None = None,
) -> list[str]:
    api = HfApi(token=token if token is not None else get_hf_token())

    files = api.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
    )

    return list(files)


def list_matching_files(
    repo_id: str,
    pattern: str,
    *,
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | bool | None = None,
) -> list[str]:
    files = list_hf_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )

    matched = [path for path in files if fnmatch(path, pattern)]

    if not matched:
        raise FileNotFoundError(
            f"No files matched pattern {pattern!r} in {repo_type} repo {repo_id!r} at revision {revision!r}."
        )

    return sorted(matched)


def list_matching_shards(
    repo_id: str,
    pattern: str,
    *,
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | bool | None = None,
) -> list[str]:
    matched = list_matching_files(
        repo_id=repo_id,
        pattern=pattern,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )

    shards = [path for path in matched if path.endswith(".tar")]

    if not shards:
        raise FileNotFoundError(
            f"Pattern {pattern!r} matched files, but none were .tar shards."
        )

    return shards


def download_repo_patterns(
    repo_id: str,
    patterns: list[str],
    local_dir: str | Path,
    *,
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | bool | None = None,
) -> Path:
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        allow_patterns=patterns,
        local_dir=str(local_dir),
        token=token if token is not None else get_hf_token(),
    )

    return Path(local_root)


def find_local_matches(local_root: str | Path, pattern: str) -> list[Path]:
    root = Path(local_root)

    matched = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and fnmatch(str(path.relative_to(root)), pattern)
    )

    if not matched:
        raise FileNotFoundError(
            f"No local files matched pattern {pattern!r} under {root}."
        )

    return matched


def download_matching_shards(
    repo_id: str,
    pattern: str,
    local_dir: str | Path,
    *,
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | bool | None = None,
) -> list[Path]:
    local_root = download_repo_patterns(
        repo_id=repo_id,
        patterns=[pattern],
        local_dir=local_dir,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )

    matched = find_local_matches(local_root, pattern)
    shards = [path for path in matched if path.suffix == ".tar"]

    if not shards:
        raise FileNotFoundError(
            f"Downloaded files matching {pattern!r}, but found no .tar shards."
        )

    return shards

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

import hydra
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from adp.config import write_resolved_config
from adp.utils.io import write_json
from adp.utils.paths import ensure_run_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _read_checksum(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty checksum file: {path}")

    return text.split()[0]


def _verify_checksum(*, file_path: Path, checksum_path: Path) -> str:
    expected = _read_checksum(checksum_path)
    actual = _sha256(file_path)

    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {file_path}\n"
            f"expected: {expected}\n"
            f"actual:   {actual}"
        )

    return actual


def _hf_token(cfg: DictConfig) -> str | None:
    token_env = str(cfg.hf.token_env)
    token = os.environ.get(token_env)
    return token if token else None


def _download_hf_file(*, cfg: DictConfig, filename: str, download_dir: Path) -> Path:
    hf_hub_download(
        repo_id=str(cfg.hf.repo_id),
        filename=filename,
        repo_type=str(cfg.hf.repo_type),
        revision=str(cfg.hf.revision),
        token=_hf_token(cfg),
        local_dir=download_dir,
    )

    local_path = download_dir / filename

    if not local_path.exists():
        raise FileNotFoundError(
            f"Downloaded file was not found at expected local path: {local_path}"
        )

    return local_path


def _download_and_verify_pair(
    *,
    cfg: DictConfig,
    package_filename: str,
    checksum_filename: str,
    download_dir: Path,
) -> dict[str, Any]:
    package_path = _download_hf_file(
        cfg=cfg,
        filename=package_filename,
        download_dir=download_dir,
    )

    checksum_path = _download_hf_file(
        cfg=cfg,
        filename=checksum_filename,
        download_dir=download_dir,
    )

    if package_path.suffix != ".tar":
        raise ValueError(f"Expected package tar path, got: {package_path}")

    if checksum_path.suffix != ".sha256":
        raise ValueError(f"Expected checksum path, got: {checksum_path}")

    checksum = _verify_checksum(file_path=package_path, checksum_path=checksum_path)

    return {
        "package_filename": package_filename,
        "checksum_filename": checksum_filename,
        "package_path": str(package_path),
        "checksum_path": str(checksum_path),
        "sha256": checksum,
    }


def _safe_extract_tar(*, tar_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[adp] extracting: {tar_path}")
    print(f"[adp] extract dir: {output_dir}")

    with tarfile.open(tar_path, "r") as tar:
        output_root = output_dir.resolve()
        members = tar.getmembers()

        print(f"[adp] tar members: {len(members)}")

        for member in members:
            target = (output_dir / member.name).resolve()
            if not str(target).startswith(str(output_root)):
                raise ValueError(f"Unsafe tar member path: {member.name}")

        tar.extractall(output_dir, members=members)

    print(f"[adp] extracted: {tar_path}")


def _move_path(*, src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)

    if dst.exists():
        if not overwrite:
            raise FileExistsError(dst)

        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _move_tree_contents(*, src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)

    dst.mkdir(parents=True, exist_ok=True)

    for child in sorted(src.iterdir()):
        _move_path(src=child, dst=dst / child.name, overwrite=overwrite)


def _stage_extracted_package(
    *,
    extracted_package_root: Path,
    staged_root: Path,
    package_label: str,
    overwrite: bool,
) -> dict[str, Any]:
    if not extracted_package_root.exists():
        raise FileNotFoundError(f"Missing extracted package root: {extracted_package_root}")

    images_dir = extracted_package_root / "images"
    manifests_dir = extracted_package_root / "manifests"
    metadata_dir = extracted_package_root / "metadata"

    staged_images_dir = staged_root / "images"
    staged_manifests_dir = staged_root / "manifests"
    staged_package_metadata_dir = staged_root / "metadata" / "packages" / package_label

    print(f"[adp] staging package={package_label}")

    if images_dir.exists():
        print(f"[adp] moving images: {images_dir} -> {staged_images_dir}")
        _move_tree_contents(src=images_dir, dst=staged_images_dir, overwrite=overwrite)

    if manifests_dir.exists():
        print(f"[adp] moving manifests: {manifests_dir} -> {staged_manifests_dir}")
        _move_tree_contents(src=manifests_dir, dst=staged_manifests_dir, overwrite=overwrite)

    if metadata_dir.exists():
        print(f"[adp] moving metadata: {metadata_dir} -> {staged_package_metadata_dir}")
        _move_tree_contents(src=metadata_dir, dst=staged_package_metadata_dir, overwrite=overwrite)

    return {
        "package_label": package_label,
        "extracted_package_root": str(extracted_package_root),
        "staged_images_dir": str(staged_images_dir),
        "staged_manifests_dir": str(staged_manifests_dir),
        "staged_package_metadata_dir": str(staged_package_metadata_dir),
    }


def stage_dataset(cfg: DictConfig) -> dict[str, Any]:
    download_dir = Path(str(cfg.stage.download_dir))
    staged_root = Path(str(cfg.stage.output_dir))
    overwrite = bool(cfg.stage.overwrite)

    if staged_root.exists() and overwrite:
        shutil.rmtree(staged_root)

    staged_root.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    print("[adp] downloading train package")
    train_download = _download_and_verify_pair(
        cfg=cfg,
        package_filename=str(cfg.hf.files.train_package),
        checksum_filename=str(cfg.hf.files.train_checksum),
        download_dir=download_dir,
    )

    print("[adp] downloading val package")
    val_download = _download_and_verify_pair(
        cfg=cfg,
        package_filename=str(cfg.hf.files.val_package),
        checksum_filename=str(cfg.hf.files.val_checksum),
        download_dir=download_dir,
    )

    extract_dir = download_dir / "extracted"
    if extract_dir.exists() and overwrite:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    train_tar = Path(str(train_download["package_path"]))
    val_tar = Path(str(val_download["package_path"]))

    print("[adp] starting train package extraction")
    _safe_extract_tar(tar_path=train_tar, output_dir=extract_dir)

    print("[adp] starting val package extraction")
    _safe_extract_tar(tar_path=val_tar, output_dir=extract_dir)

    train_package_root = extract_dir / str(cfg.data.package_name)
    val_package_root = extract_dir / "bdd_det10_val"

    train_stage = _stage_extracted_package(
        extracted_package_root=train_package_root,
        staged_root=staged_root,
        package_label="train",
        overwrite=overwrite,
    )

    val_stage = _stage_extracted_package(
        extracted_package_root=val_package_root,
        staged_root=staged_root,
        package_label="val",
        overwrite=overwrite,
    )

    stage_report = {
        "repo_id": str(cfg.hf.repo_id),
        "repo_type": str(cfg.hf.repo_type),
        "revision": str(cfg.hf.revision),
        "download_dir": str(download_dir),
        "staged_root": str(staged_root),
        "downloads": {
            "train": train_download,
            "val": val_download,
        },
        "staged_packages": {
            "train": train_stage,
            "val": val_stage,
        },
        "manifests": {
            "train": str(staged_root / "manifests" / "train.jsonl"),
            "val": str(staged_root / "manifests" / "val.jsonl"),
            "overfit16": str(staged_root / "manifests" / "overfit16.jsonl"),
        },
    }

    write_json(staged_root / "metadata" / "stage.json", stage_report)
    return stage_report


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_paths = ensure_run_paths(cfg)
    write_resolved_config(cfg, run_paths.root / "config.resolved.yaml")

    report = stage_dataset(cfg)
    report_path = run_paths.reports / "stage.json"
    write_json(report_path, report)

    print(f"[adp] staged dataset: {report['staged_root']}")
    print(f"[adp] wrote stage report: {report_path}")
    print("[adp] use this staged root with:")
    print(f"  data.root={report['staged_root']}")


if __name__ == "__main__":
    main()

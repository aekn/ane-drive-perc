import hashlib
import shutil
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from adp.config import write_resolved_config
from adp.data.manifest import read_manifest, write_manifest
from adp.data.records import ImageRecord
from adp.utils.io import write_json, write_text
from adp.utils.paths import ensure_run_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _rewrite_record_path(record: ImageRecord, relative_path: Path) -> ImageRecord:
    return replace(record, file_name=relative_path.as_posix())


def _copy_metadata(
    *, source_metadata_dir: Path, package_metadata_dir: Path, overwrite: bool
) -> None:
    if not source_metadata_dir.exists():
        raise FileNotFoundError(f"Missing metadata directory: {source_metadata_dir}")

    package_metadata_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(source_metadata_dir.iterdir()):
        if src.is_file():
            _copy_file(src, package_metadata_dir / src.name, overwrite=overwrite)


def _package_manifest(
    *,
    manifest_path: Path,
    package_root: Path,
    split: str,
    output_manifest_path: Path,
    copy_images: bool,
    overwrite: bool,
) -> dict[str, Any]:
    packaged_records: list[ImageRecord] = []

    image_count = 0
    object_count = 0

    for record in read_manifest(manifest_path):
        source_image = Path(record.file_name)

        if not source_image.exists():
            raise FileNotFoundError(
                f"Missing image referenced by manifest: {source_image}"
            )

        relative_image_path = Path("images") / split / source_image.name
        packaged_image_path = package_root / relative_image_path

        if copy_images:
            _copy_file(source_image, packaged_image_path, overwrite=overwrite)

        packaged_record = _rewrite_record_path(record, relative_image_path)
        packaged_records.append(packaged_record)

        image_count += 1
        object_count += len(record.objects)

    written_count = write_manifest(output_manifest_path, packaged_records)

    return {
        "source_manifest": str(manifest_path),
        "packaged_manifest": str(output_manifest_path),
        "split": split,
        "image_count": image_count,
        "written_count": written_count,
        "object_count": object_count,
        "images_copied": copy_images,
    }


def _create_tar(
    *, package_root: Path, tar_path: Path, arcname: str, overwrite: bool
) -> None:
    if tar_path.exists():
        if not overwrite:
            raise FileExistsError(f"Tar already exists: {tar_path}")
        tar_path.unlink()

    tar_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "w") as tar:
        tar.add(package_root, arcname=arcname)


def _write_checksum(*, tar_path: Path, checksum_path: Path) -> str:
    checksum = _sha256(tar_path)
    write_text(checksum_path, f"{checksum}  {tar_path.name}\n")
    return checksum


def _manifest_for_package_split(
    cfg: DictConfig, split: str
) -> list[tuple[str, Path, str]]:
    if split == "train":
        return [
            ("train", Path(str(cfg.data.manifests.train)), "train"),
            ("overfit16", Path(str(cfg.data.manifests.overfit16)), "train"),
        ]

    if split == "val":
        return [
            ("val", Path(str(cfg.data.manifests.val)), "val"),
        ]

    raise ValueError(f"Unknown package.split={split!r}. Expected 'train' or 'val'.")


def package_dataset(cfg: DictConfig) -> dict[str, Any]:
    package_name = str(cfg.data.package_name)
    package_split = str(cfg.package.split)
    package_root = Path(str(cfg.package.output_dir))
    tar_path = Path(str(cfg.package.tar_path))
    checksum_path = Path(str(cfg.package.checksum_path))
    overwrite = bool(cfg.package.overwrite)

    if package_root.exists() and overwrite:
        shutil.rmtree(package_root)

    package_root.mkdir(parents=True, exist_ok=True)

    manifests_dir = package_root / "manifests"
    metadata_dir = package_root / "metadata"

    _copy_metadata(
        source_metadata_dir=Path(str(cfg.data.metadata_dir)),
        package_metadata_dir=metadata_dir,
        overwrite=overwrite,
    )

    split_reports: dict[str, dict[str, Any]] = {}

    for manifest_name, manifest_path, image_split in _manifest_for_package_split(
        cfg, package_split
    ):
        report = _package_manifest(
            manifest_path=manifest_path,
            package_root=package_root,
            split=image_split,
            output_manifest_path=manifests_dir / f"{manifest_name}.jsonl",
            copy_images=(manifest_name != "overfit16"),
            overwrite=overwrite,
        )
        split_reports[manifest_name] = report

    package_report = {
        "name": package_name,
        "package_split": package_split,
        "package_root": str(package_root),
        "tar_path": str(tar_path),
        "checksum_path": str(checksum_path),
        "manifests_use_relative_paths": True,
        "splits": split_reports,
    }

    write_json(metadata_dir / "package.json", package_report)

    _create_tar(
        package_root=package_root,
        tar_path=tar_path,
        arcname=package_name,
        overwrite=overwrite,
    )

    checksum = _write_checksum(tar_path=tar_path, checksum_path=checksum_path)

    package_report["sha256"] = checksum
    write_json(metadata_dir / "package.json", package_report)

    return package_report


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_paths = ensure_run_paths(cfg)
    write_resolved_config(cfg, run_paths.root / "config.resolved.yaml")

    report = package_dataset(cfg)
    report_path = run_paths.reports / "package.json"
    write_json(report_path, report)

    print(f"[adp] wrote package directory: {report['package_root']}")
    print(f"[adp] wrote tar: {report['tar_path']}")
    print(f"[adp] wrote checksum: {report['checksum_path']}")
    print(f"[adp] wrote package report: {report_path}")


if __name__ == "__main__":
    main()

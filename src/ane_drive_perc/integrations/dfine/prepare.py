from __future__ import annotations

import subprocess
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ane_drive_perc.config.access import (
    ConfigMapping,
    expect_bool,
    expect_dict,
    expect_int,
    expect_optional_str,
    expect_str,
)
from ane_drive_perc.config.load import load_yaml
from ane_drive_perc.data.coco_export import materialize_coco_from_local_shards
from ane_drive_perc.data.resolver import resolve_data_split_from_config
from ane_drive_perc.integrations.dfine.checkpoints import ensure_local_checkpoint


@dataclass(frozen=True)
class DfinePreparedRun:
    repo_dir: Path
    coco_dir: Path
    generated_dataset_config: Path
    generated_model_config: Path
    generated_train_script: Path
    train_command: list[str]


def prepare_dfine_teacher_run(
    *,
    config_path: str | Path,
    run: bool = False,
    install_requirements: bool | None = None,
    output_dir_override: str | None = None,
    summary_dir_override: str | None = None,
    resume_from_override: str | None = None,
    skip_materialize: bool = False,
) -> DfinePreparedRun:
    config = load_yaml(config_path)

    data_cfg = expect_dict(config, "data")
    dfine_cfg = expect_dict(config, "dfine")
    train_cfg = expect_dict(config, "train")
    runtime_cfg = expect_dict(config, "runtime")

    pretrained_cfg = config.get("pretrained")
    if pretrained_cfg is not None and not isinstance(pretrained_cfg, dict):
        raise TypeError("Expected 'pretrained' to be a mapping when provided.")

    data_config_path = expect_str(data_cfg, "config")
    coco_dir = Path(expect_str(data_cfg, "coco_dir")).expanduser()
    train_split = expect_str(data_cfg, "train_split")
    val_split = expect_str(data_cfg, "val_split")

    repo_url = expect_str(dfine_cfg, "repo_url")
    repo_dir = Path(expect_str(dfine_cfg, "repo_dir")).expanduser()
    ref = expect_str(dfine_cfg, "ref")
    base_config = expect_str(dfine_cfg, "base_config")
    generated_config_dir = Path(
        expect_str(dfine_cfg, "generated_config_dir")
    ).expanduser()

    output_dir = Path(
        output_dir_override or expect_str(train_cfg, "output_dir")
    ).expanduser()
    summary_dir = Path(
        summary_dir_override or expect_str(train_cfg, "summary_dir")
    ).expanduser()
    resume_from = resume_from_override or expect_optional_str(train_cfg, "resume_from")

    input_size = get_int(train_cfg, "input_size", default=640)
    train_batch_size = get_int(train_cfg, "train_batch_size", default=32)
    val_batch_size = get_int(train_cfg, "val_batch_size", default=train_batch_size * 2)
    num_workers = get_int(train_cfg, "num_workers", default=8)

    epochs = get_int(train_cfg, "epochs", default=80)
    close_aug_epoch = get_int(train_cfg, "close_aug_epoch", default=max(0, epochs - 8))

    lr = get_float(train_cfg, "lr", default=0.00025)
    backbone_lr = get_float(train_cfg, "backbone_lr", default=0.0000125)
    weight_decay = get_float(train_cfg, "weight_decay", default=0.000125)
    encoder_decoder_norm_weight_decay = get_float(
        train_cfg,
        "encoder_decoder_norm_weight_decay",
        default=0.0,
    )

    materialize_train = expect_bool(runtime_cfg, "materialize_train")
    materialize_val = expect_bool(runtime_cfg, "materialize_val")

    config_install_requirements = expect_bool(runtime_cfg, "install_requirements")
    should_install_requirements = (
        config_install_requirements
        if install_requirements is None
        else install_requirements
    )

    if run and skip_materialize:
        raise ValueError(
            "Cannot use --run with --skip-materialize because training needs materialized COCO data."
        )

    if skip_materialize:
        print("skipping COCO materialization")
    else:
        if materialize_train:
            materialize_split(
                data_config_path=data_config_path,
                split=train_split,
                output_dir=coco_dir,
            )

        if materialize_val:
            materialize_split(
                data_config_path=data_config_path,
                split=val_split,
                output_dir=coco_dir,
            )

    ensure_dfine_repo(
        repo_url=repo_url,
        repo_dir=repo_dir,
        ref=ref,
    )

    if should_install_requirements:
        install_dfine_requirements(repo_dir)

    generated_config_dir.mkdir(parents=True, exist_ok=True)

    generated_dataset_config = generated_config_dir / "bdd100k_detection.yml"
    write_dfine_dataset_config(
        path=generated_dataset_config,
        coco_dir=coco_dir,
        train_split=train_split,
        val_split=val_split,
        num_classes=10,
        num_workers=num_workers,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
    )

    generated_model_config = generated_config_dir / "dfine_hgnetv2_l_bdd100k.yml"
    write_dfine_model_config(
        base_config_path=repo_dir / base_config,
        output_path=generated_model_config,
        dataset_config_path=generated_dataset_config,
        output_dir=output_dir,
        input_size=input_size,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        epochs=epochs,
        close_aug_epoch=close_aug_epoch,
        lr=lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay,
        encoder_decoder_norm_weight_decay=encoder_decoder_norm_weight_decay,
    )

    tune_from = expect_optional_str(train_cfg, "tune_from")

    if resume_from is None and pretrained_cfg is not None:
        if bool(pretrained_cfg.get("enabled", False)):
            pretrained_url = expect_str(pretrained_cfg, "url")
            pretrained_local_path = expect_str(pretrained_cfg, "local_path")
            tune_from = str(
                ensure_local_checkpoint(
                    url=pretrained_url,
                    local_path=pretrained_local_path,
                ).resolve()
            )

    train_command = build_dfine_train_command(
        model_config=generated_model_config,
        devices=str(train_cfg.get("devices", "0")),
        nproc_per_node=expect_int(train_cfg, "nproc_per_node"),
        master_port=expect_int(train_cfg, "master_port"),
        seed=expect_int(train_cfg, "seed"),
        use_amp=expect_bool(train_cfg, "use_amp"),
        output_dir=output_dir,
        summary_dir=summary_dir,
        resume_from=resume_from,
        tune_from=tune_from,
    )

    generated_train_script = generated_config_dir / "train_dfine.sh"
    write_train_script(
        path=generated_train_script,
        command=train_command,
        repo_dir=repo_dir,
        dataset_config=generated_dataset_config,
        model_config=generated_model_config,
    )

    prepared = DfinePreparedRun(
        repo_dir=repo_dir,
        coco_dir=coco_dir,
        generated_dataset_config=generated_dataset_config,
        generated_model_config=generated_model_config,
        generated_train_script=generated_train_script,
        train_command=train_command,
    )

    print_prepared_run(prepared)

    if run:
        run_dfine_training(
            command=train_command,
            cwd=repo_dir,
            dataset_config=generated_dataset_config,
            model_config=generated_model_config,
        )

    return prepared


def materialize_split(
    *,
    data_config_path: str | Path,
    split: str,
    output_dir: str | Path,
) -> None:
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}.")

    resolved = resolve_data_split_from_config(
        data_config_path,
        split=split,
    )

    result = materialize_coco_from_local_shards(
        shards=resolved.local_shards,
        output_dir=output_dir,
        split=split,
        image_key=resolved.image_key,
        metadata_key=resolved.metadata_key,
        subset_manifest=resolved.local_manifest,
        category_id_base=0,
    )

    print(
        f"materialized {split}: "
        f"{result.num_images} images, {result.num_annotations} annotations"
    )


def ensure_dfine_repo(
    *,
    repo_url: str,
    repo_dir: Path,
    ref: str,
) -> None:
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", repo_url, str(repo_dir)],
            check=True,
        )

    subprocess.run(
        ["git", "fetch", "--all", "--tags"],
        cwd=repo_dir,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", ref],
        cwd=repo_dir,
        check=True,
    )


def install_dfine_requirements(repo_dir: Path) -> None:
    requirements = repo_dir / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError(f"D-FINE requirements.txt not found: {requirements}")

    subprocess.run(
        ["python", "-m", "pip", "install", "-r", str(requirements)],
        check=True,
    )


def write_dfine_dataset_config(
    *,
    path: str | Path,
    coco_dir: str | Path,
    train_split: str,
    val_split: str,
    num_classes: int,
    num_workers: int,
    train_batch_size: int,
    val_batch_size: int,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    coco_root = Path(coco_dir).resolve()

    dataset_config: dict[str, Any] = {
        "task": "detection",
        "evaluator": {
            "type": "CocoEvaluator",
            "iou_types": ["bbox"],
        },
        "num_classes": num_classes,
        "remap_mscoco_category": False,
        "train_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": str(coco_root / "images" / train_split),
                "ann_file": str(
                    coco_root / "annotations" / f"instances_{train_split}.json"
                ),
                "return_masks": False,
                "transforms": {
                    "type": "Compose",
                    "ops": None,
                },
            },
            "shuffle": True,
            "total_batch_size": train_batch_size,
            "num_workers": num_workers,
            "drop_last": True,
            "collate_fn": {
                "type": "BatchImageCollateFunction",
            },
        },
        "val_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": str(coco_root / "images" / val_split),
                "ann_file": str(
                    coco_root / "annotations" / f"instances_{val_split}.json"
                ),
                "return_masks": False,
                "transforms": {
                    "type": "Compose",
                    "ops": None,
                },
            },
            "shuffle": False,
            "total_batch_size": val_batch_size,
            "num_workers": num_workers,
            "drop_last": False,
            "collate_fn": {
                "type": "BatchImageCollateFunction",
            },
        },
    }

    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_config, f, sort_keys=False)


def write_dfine_model_config(
    *,
    base_config_path: str | Path,
    output_path: str | Path,
    dataset_config_path: str | Path,
    output_dir: str | Path,
    input_size: int,
    train_batch_size: int,
    val_batch_size: int,
    num_workers: int,
    epochs: int,
    close_aug_epoch: int,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
    encoder_decoder_norm_weight_decay: float,
) -> None:
    base_path = Path(base_config_path)
    if not base_path.exists():
        raise FileNotFoundError(f"D-FINE base config does not exist: {base_path}")

    with base_path.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    if not isinstance(base_config, dict):
        raise ValueError(f"Expected D-FINE base config to be a mapping: {base_path}")

    includes = base_config.get("__include__")
    if not isinstance(includes, list):
        raise ValueError(f"Expected '__include__' list in D-FINE config: {base_path}")

    rewritten_includes: list[str] = []
    replaced_dataset = False

    for include in includes:
        if not isinstance(include, str):
            raise TypeError(
                f"Expected include entry to be a string in {base_path}: {include!r}"
            )

        if include.endswith("custom_detection.yml"):
            rewritten_includes.append(str(Path(dataset_config_path).resolve()))
            replaced_dataset = True
        else:
            rewritten_includes.append(str((base_path.parent / include).resolve()))

    if not replaced_dataset:
        raise ValueError(
            f"Could not find custom_detection.yml include to replace in D-FINE config: {base_path}"
        )

    generated_config = dict(base_config)
    generated_config["__include__"] = rewritten_includes
    generated_config["output_dir"] = str(Path(output_dir).resolve())
    generated_config["epochs"] = epochs

    apply_dataloader_overrides(
        generated_config,
        input_size=input_size,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        close_aug_epoch=close_aug_epoch,
    )
    apply_optimizer_overrides(
        generated_config,
        lr=lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay,
        encoder_decoder_norm_weight_decay=encoder_decoder_norm_weight_decay,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(generated_config, f, sort_keys=False)


def apply_dataloader_overrides(
    config: dict[str, Any],
    *,
    input_size: int,
    train_batch_size: int,
    val_batch_size: int,
    num_workers: int,
    close_aug_epoch: int,
) -> None:
    train_dataloader = ensure_mapping(config, "train_dataloader")
    train_dataloader["total_batch_size"] = train_batch_size
    train_dataloader["num_workers"] = num_workers

    train_dataset = ensure_mapping(train_dataloader, "dataset")
    train_transforms = ensure_mapping(train_dataset, "transforms")
    train_transforms["ops"] = [
        {"type": "RandomPhotometricDistort", "p": 0.5},
        {"type": "RandomZoomOut", "fill": 0},
        {"type": "RandomIoUCrop", "p": 0.8},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "RandomHorizontalFlip"},
        {"type": "Resize", "size": [input_size, input_size]},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
    ]

    train_policy = ensure_mapping(train_transforms, "policy")
    train_policy["name"] = "stop_epoch"
    train_policy["epoch"] = close_aug_epoch
    train_policy["ops"] = [
        "RandomPhotometricDistort",
        "RandomZoomOut",
        "RandomIoUCrop",
    ]

    collate_fn = ensure_mapping(train_dataloader, "collate_fn")
    collate_fn["type"] = "BatchImageCollateFunction"
    collate_fn["base_size"] = input_size
    collate_fn["base_size_repeat"] = 4
    collate_fn["stop_epoch"] = close_aug_epoch

    val_dataloader = ensure_mapping(config, "val_dataloader")
    val_dataloader["total_batch_size"] = val_batch_size
    val_dataloader["num_workers"] = num_workers

    val_dataset = ensure_mapping(val_dataloader, "dataset")
    val_transforms = ensure_mapping(val_dataset, "transforms")
    val_transforms["ops"] = [
        {"type": "Resize", "size": [input_size, input_size]},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
    ]


def apply_optimizer_overrides(
    config: dict[str, Any],
    *,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
    encoder_decoder_norm_weight_decay: float,
) -> None:
    optimizer = ensure_mapping(config, "optimizer")
    optimizer["lr"] = lr
    optimizer["weight_decay"] = weight_decay

    params = optimizer.get("params")
    if not isinstance(params, list):
        return

    for group in params:
        if not isinstance(group, MutableMapping):
            continue

        selector = group.get("params")
        if not isinstance(selector, str):
            continue

        if "backbone" in selector:
            group["lr"] = backbone_lr

        if ("encoder" in selector or "decoder" in selector) and (
            "norm" in selector or "bn" in selector
        ):
            group["weight_decay"] = encoder_decoder_norm_weight_decay


def build_dfine_train_command(
    *,
    model_config: str | Path,
    devices: str,
    nproc_per_node: int,
    master_port: int,
    seed: int,
    use_amp: bool,
    output_dir: str | Path,
    summary_dir: str | Path,
    resume_from: str | None,
    tune_from: str | None,
) -> list[str]:
    command = [
        "env",
        f"CUDA_VISIBLE_DEVICES={devices}",
        "MPLBACKEND=Agg",
        "torchrun",
        f"--master_port={master_port}",
        f"--nproc_per_node={nproc_per_node}",
        "train.py",
        "-c",
        str(Path(model_config).resolve()),
        f"--seed={seed}",
        "--output-dir",
        str(Path(output_dir).resolve()),
        "--summary-dir",
        str(Path(summary_dir).resolve()),
    ]

    if use_amp:
        command.append("--use-amp")

    if resume_from is not None:
        command.extend(["-r", str(Path(resume_from).expanduser().resolve())])
    elif tune_from is not None:
        command.extend(["-t", str(Path(tune_from).expanduser().resolve())])

    return command


def write_train_script(
    *,
    path: str | Path,
    command: list[str],
    repo_dir: Path,
    dataset_config: Path,
    model_config: Path,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    command_text = " ".join(shell_quote(part) for part in command)

    script = f"""#!/usr/bin/env bash
set -euo pipefail

cd {shell_quote(str(repo_dir.resolve()))}

echo "Generated dataset config:"
echo {shell_quote(str(dataset_config.resolve()))}

echo "Generated model config:"
echo {shell_quote(str(model_config.resolve()))}

echo "Training command:"
echo {command_text}

{command_text}
"""

    output.write_text(script, encoding="utf-8")
    output.chmod(0o755)


def run_dfine_training(
    *,
    command: list[str],
    cwd: Path,
    dataset_config: Path,
    model_config: Path,
) -> None:
    print(f"running D-FINE training from {cwd}")
    print(f"dataset config: {dataset_config}")
    print(f"model config: {model_config}")

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def print_prepared_run(prepared: DfinePreparedRun) -> None:
    print("prepared D-FINE teacher run")
    print(f"  repo_dir: {prepared.repo_dir}")
    print(f"  coco_dir: {prepared.coco_dir}")
    print(f"  dataset_config: {prepared.generated_dataset_config}")
    print(f"  model_config: {prepared.generated_model_config}")
    print(f"  train_script: {prepared.generated_train_script}")
    print("  command:")
    print("    " + " ".join(shell_quote(part) for part in prepared.train_command))


def ensure_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if value is None:
        value = {}
        mapping[key] = value

    if not isinstance(value, dict):
        raise TypeError(f"Expected {key!r} to be a mapping.")

    return value


def get_int(mapping: ConfigMapping, key: str, *, default: int) -> int:
    value = mapping.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Expected {key!r} to be an integer.")
    return value


def get_float(mapping: ConfigMapping, key: str, *, default: float) -> float:
    value = mapping.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"Expected {key!r} to be numeric.")
    return float(value)


def shell_quote(value: str) -> str:
    if not value:
        return "''"

    safe_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:@%"
    )
    if all(char in safe_chars for char in value):
        return value

    return "'" + value.replace("'", "'\"'\"'") + "'"

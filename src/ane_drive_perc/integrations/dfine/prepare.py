import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ane_drive_perc.config.access import (
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

    materialize_train = expect_bool(runtime_cfg, "materialize_train")
    materialize_val = expect_bool(runtime_cfg, "materialize_val")

    config_install_requirements = expect_bool(runtime_cfg, "install_requirements")
    should_install_requirements = (
        config_install_requirements
        if install_requirements is None
        else install_requirements
    )

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
        num_workers=4,
    )

    generated_model_config = generated_config_dir / "dfine_hgnetv2_l_bdd100k.yml"
    write_dfine_model_config(
        base_config_path=repo_dir / base_config,
        output_path=generated_model_config,
        dataset_config_path=generated_dataset_config,
        output_dir=output_dir,
    )

    tune_from = expect_optional_str(train_cfg, "tune_from")

    if (
        resume_from is None
        and pretrained_cfg is not None
        and bool(pretrained_cfg.get("enabled", False))
    ):
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

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(generated_config, f, sort_keys=False)


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


def shell_quote(value: str) -> str:
    if not value:
        return "''"

    safe_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:@%"
    )
    if all(char in safe_chars for char in value):
        return value

    return "'" + value.replace("'", "'\"'\"'") + "'"

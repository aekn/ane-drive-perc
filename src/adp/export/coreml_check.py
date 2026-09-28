"""Export ANE-S to Core ML and validate the converted model."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image

from adp.model.registry import get as get_model_spec

if TYPE_CHECKING:
    import coremltools as ct


class _ExportWrapper(nn.Module):
    """Return the detector outputs as tensors instead of a mapping."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.detector(image)
        return output["pred_logits"], output["pred_boxes"]


def _build_model(
    *,
    img_size: tuple[int, int],
    num_classes: int,
    checkpoint: Path,
) -> nn.Module:
    spec = get_model_spec("ane_s")
    model = spec.build_model(
        num_classes=num_classes,
        device="cpu",
        img_size=list(img_size),
        backbone_pretrained=False,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["ema"] if payload.get("ema") is not None else payload["model"]
    model.load_state_dict(state, strict=True)
    model.eval()

    logger.info(
        "loaded {} (epoch {}, best mAP {:.4f})",
        checkpoint,
        payload.get("epoch", "?"),
        float(payload.get("best_score", float("nan"))),
    )
    return model


def _example_input(img_size: tuple[int, int]) -> torch.Tensor:
    height, width = img_size
    generator = torch.Generator().manual_seed(0)
    return torch.rand((1, 3, height, width), generator=generator)


def _deploy(model: nn.Module, example: torch.Tensor) -> nn.Module:
    wrapper = _ExportWrapper(model).eval()
    with torch.inference_mode():
        before = wrapper(example)

    model.deploy()
    model.eval()

    deployed = _ExportWrapper(model).eval()
    with torch.inference_mode():
        after = deployed(example)

    for name, expected, actual in zip(
        ("pred_logits", "pred_boxes"), before, after, strict=True
    ):
        max_abs = (expected - actual).abs().max().item()
        mean_abs = (expected - actual).abs().mean().item()
        logger.info(
            "deploy equivalence {}: max_abs={:.3e}, mean_abs={:.3e}",
            name,
            max_abs,
            mean_abs,
        )
        if not torch.allclose(expected, actual, atol=1e-4, rtol=1e-4):
            raise RuntimeError(f"deploy reparameterization changed {name}")

    params = sum(parameter.numel() for parameter in model.parameters())
    logger.info("deploy model: {:,} parameters ({:.2f}M)", params, params / 1e6)
    return model


def _trace(model: nn.Module, example: torch.Tensor) -> torch.jit.ScriptModule:
    wrapper = _ExportWrapper(model).eval()
    logger.info("tracing model at shape {}", tuple(example.shape))
    with torch.inference_mode():
        return torch.jit.trace(wrapper, example, strict=True, check_trace=True)


def _convert(
    traced: torch.jit.ScriptModule,
    *,
    img_size: tuple[int, int],
) -> "ct.models.MLModel":
    import coremltools as ct

    height, width = img_size
    logger.info("converting to FP16 MLProgram with ImageType input")
    return ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, height, width),
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.TensorType(name="pred_logits", dtype=np.float16),
            ct.TensorType(name="pred_boxes", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )


def _validation_image(img_size: tuple[int, int]) -> tuple[Image.Image, torch.Tensor]:
    height, width = img_size
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    tensor = (
        torch.from_numpy(array.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=torch.float32)
        / 255.0
    )
    return image, tensor


def _validate_coreml(
    traced: torch.jit.ScriptModule,
    mlmodel: "ct.models.MLModel",
    *,
    img_size: tuple[int, int],
) -> None:
    image, tensor = _validation_image(img_size)

    with torch.inference_mode():
        torch_logits, torch_boxes = traced(tensor)

    coreml_output = mlmodel.predict({"image": image})
    pairs = (
        ("pred_logits", torch_logits.numpy(), coreml_output["pred_logits"]),
        ("pred_boxes", torch_boxes.numpy(), coreml_output["pred_boxes"]),
    )

    for name, expected, actual in pairs:
        actual = np.asarray(actual)
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"Core ML {name} shape {actual.shape} != PyTorch {expected.shape}"
            )
        if not np.isfinite(actual).all():
            raise RuntimeError(f"Core ML {name} contains non-finite values")

        error = np.abs(expected - actual)
        logger.info(
            "Core ML parity {}: max_abs={:.4e}, mean_abs={:.4e}",
            name,
            float(error.max()),
            float(error.mean()),
        )


def _benchmark(
    mlpackage: Path,
    *,
    img_size: tuple[int, int],
    iterations: int,
) -> dict[str, float]:
    import coremltools as ct

    image, _ = _validation_image(img_size)
    units = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }
    results: dict[str, float] = {}

    for name, unit in units.items():
        try:
            model = ct.models.MLModel(str(mlpackage), compute_units=unit)
            for _ in range(5):
                model.predict({"image": image})

            start = time.perf_counter()
            for _ in range(iterations):
                model.predict({"image": image})
            elapsed = (time.perf_counter() - start) * 1000.0 / iterations
        except Exception as exc:
            logger.warning("{:12s}: {}", name, exc)
            results[name] = float("nan")
            continue

        results[name] = elapsed
        logger.info("{:12s}: {:7.2f} ms", name, elapsed)

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="ANE-S training checkpoint (best.pt)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/coreml/ane_s.mlpackage"),
    )
    parser.add_argument("--img-h", type=int, default=544)
    parser.add_argument("--img-w", type=int, default=960)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--n-iter", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    img_size = (args.img_h, args.img_w)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    import coremltools as ct

    logger.info("torch {}", torch.__version__)
    logger.info("coremltools {}", ct.__version__)

    model = _build_model(
        img_size=img_size,
        num_classes=args.num_classes,
        checkpoint=args.checkpoint,
    )
    example = _example_input(img_size)
    model = _deploy(model, example)
    traced = _trace(model, example)
    mlmodel = _convert(traced, img_size=img_size)
    _validate_coreml(traced, mlmodel, img_size=img_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.is_dir():
        shutil.rmtree(args.out)
    elif args.out.exists():
        args.out.unlink()
    mlmodel.save(str(args.out))
    logger.info("saved {}", args.out)

    logger.info("benchmarking {} iterations after warmup", args.n_iter)
    _benchmark(args.out, img_size=img_size, iterations=args.n_iter)

    print()
    print(f"Core ML model: {args.out}")
    print("Inspect compute-device assignment in Xcode before making ANE residency claims.")


if __name__ == "__main__":
    main()

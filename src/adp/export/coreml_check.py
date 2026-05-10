import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from adp.model.registry import get as get_model_spec


class _ExportWrapper(nn.Module):
    """Wraps the detector to return tensors instead of dicts."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.detector(image)
        return out["pred_logits"], out["pred_boxes"]


def _build_deploy_model(
    *,
    img_size: tuple[int, int],
    num_classes: int,
    backbone_pretrained: bool,
    checkpoint: Path | None = None,
) -> nn.Module:
    spec = get_model_spec("ane_s")
    model = spec.build_model(
        num_classes=num_classes,
        device="cpu",
        img_size=list(img_size),
        backbone_pretrained=backbone_pretrained,
    )

    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # prefer EMA weights when available
        state = payload.get("ema") or payload["model"]
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing:
            logger.warning(f"missing keys: {missing}")
        if unexpected:
            logger.warning(f"unexpected keys: {unexpected}")
        epoch = payload.get("epoch", "?")
        score = payload.get("best_score", "?")
        logger.info(
            f"loaded checkpoint: {checkpoint.name} (epoch {epoch}, best_score {score})"
        )

    model.eval()
    # reparam the model (fold BN into preceding conv, RepMixer to single depthwise conv)
    model.deploy()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"ANE-S deploy-mode model built: {n_params:,} params ({n_params / 1e6:.2f}M)"
    )
    return model


def _trace_model(model: nn.Module, img_size: tuple[int, int]) -> torch.jit.ScriptModule:
    h, w = img_size
    example = torch.zeros(1, 3, h, w, dtype=torch.float32)
    wrap = _ExportWrapper(model).eval()

    logger.info(f"tracing model at input shape (1, 3, {h}, {w})...")
    with torch.no_grad():
        traced = torch.jit.trace(wrap, example, strict=False, check_trace=False)
    return traced


def _convert_to_coreml(
    traced: torch.jit.ScriptModule, *, img_size: tuple[int, int]
) -> "ct.models.MLModel":
    import coremltools as ct

    h, w = img_size
    logger.info("converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=(1, 3, h, w), dtype=np.float32)],
        outputs=[
            ct.TensorType(name="pred_logits"),
            ct.TensorType(name="pred_boxes"),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    return mlmodel


def _benchmark(
    mlpackage_path: Path, *, img_size: tuple[int, int], n_iter: int
) -> dict[str, float]:
    import coremltools as ct

    h, w = img_size
    example = np.zeros((1, 3, h, w), dtype=np.float32)

    units = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }
    results: dict[str, float] = {}

    for unit_name, unit in units.items():
        try:
            m = ct.models.MLModel(str(mlpackage_path), compute_units=unit)
        except Exception as e:
            logger.warning(f"  {unit_name:12s}: failed to load -> {e}")
            results[unit_name] = float("nan")
            continue

        # warmup so the runtime allocates buffers and (lil)jits anything it needs
        for _ in range(5):
            m.predict({"image": example})

        t0 = time.perf_counter()
        for _ in range(n_iter):
            m.predict({"image": example})
        elapsed_ms = (time.perf_counter() - t0) / n_iter * 1000.0
        results[unit_name] = elapsed_ms
        logger.info(f"  {unit_name:12s}: {elapsed_ms:7.2f} ms/iter")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-h", type=int, default=544)
    parser.add_argument("--img-w", type=int, default=960)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--backbone-pretrained",
        action="store_true",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/ane_s.mlpackage"),
    )
    parser.add_argument("--n-iter", type=int, default=50)
    args = parser.parse_args()

    img_size = (args.img_h, args.img_w)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    model = _build_deploy_model(
        img_size=img_size,
        num_classes=args.num_classes,
        backbone_pretrained=args.backbone_pretrained,
        checkpoint=args.checkpoint,
    )

    traced = _trace_model(model, img_size)

    mlmodel = _convert_to_coreml(traced, img_size=img_size)

    if args.out.exists():
        import shutil

        shutil.rmtree(args.out)
    mlmodel.save(str(args.out))
    logger.info(f"saved {args.out}")

    logger.info(f"benchmarking ({args.n_iter} iterations after 5-iter warmup):")
    results = _benchmark(args.out, img_size=img_size, n_iter=args.n_iter)

    cpu = results.get("CPU_ONLY", float("nan"))
    ane = results.get("CPU_AND_NE", float("nan"))
    if cpu == cpu and ane == ane and ane > 0:  # not NaN
        speedup = cpu / ane
        print()
        print("=" * 70)
        print(f"  CPU_AND_NE is {speedup:.1f}x faster than CPU_ONLY")
        if speedup >= 2.0:
            print(
                "  [okay] STRONG ANE residency confirmed;architecture is well-optimized"
            )
        elif speedup >= 1.3:
            print(
                "  [warn] Modest ANE speedup; some ops likely falling back. Inspect in Xcode."
            )
        else:
            print("  [uhoh] Little/no speedup; many ops falling back. Check the graph.")
        print()
        print(f"  Open {args.out} in Xcode")
        print("=" * 70)


if __name__ == "__main__":
    main()

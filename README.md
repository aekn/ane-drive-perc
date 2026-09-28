# ANE-S

ANE-S is a small DETR-style object detector I built to explore deployment-aware
driving perception on Apple hardware. The model detects the 10 BDD100K object
classes at 544x960, and the best student is deployed as an FP16 Core ML MLProgram
in an iPhone camera app.

This is an exploratory project, not a production driving system or a competitive
detection benchmark. The reported student runs used a 30k-image subset of
BDD100K for 40 epochs, and the validation curve was still improving at the end
of training.

## Results

COCO-style bounding-box AP on the 10k-image BDD100K validation split. Values
below are percentage points.

| Student run | AP | AP50 |
| --- | ---: | ---: |
| Ground-truth supervision only | 7.30 | 16.93 |
| D-FINE-M distillation | 9.35 | 21.29 |

Distillation improved AP by 28.1% relative to the ground-truth-only run. The
exact Colab commands used for these runs are in
`notebooks/ane_s_bdd30k_544x960_bs24_e40.ipynb` and
`notebooks/ane_s_distill_dfine_m.ipynb`.

The student is still undertrained: its best result was the final evaluated
epoch (39), so these numbers should be read as results from a limited experiment
rather than a converged model.

## Model

ANE-S has about 6.9M parameters and uses:

- a FastViT-T8 backbone with ImageNet-pretrained weights and feature maps at strides 8, 16, and 32;
- a 192-channel hybrid encoder with an AIFI block on the deepest scale and FPN/PAN feature fusion;
- a 3-layer, 300-query decoder with 8 attention heads and fine-grained box distributions;
- ANE-oriented attention blocks that keep tensors in `(B, C, 1, S)` layout, use 1x1 `Conv2d` projections, channel-axis normalization, and `matmul` attention.

For distillation, I use a frozen D-FINE-M teacher trained on the same BDD100K
split. Student and teacher queries are independently matched to ground-truth
objects, then queries assigned to the same ground-truth object are paired.
Training combines the normal supervised loss with box, class-distribution,
fine-grained regression-distribution, and deepest-scale feature distillation
losses.

## Core ML and iPhone deployment

The export path reparameterizes supported FastViT/convolutional blocks, traces
the fixed 1x3x544x960 model, and converts it to an FP16 Core ML MLProgram with
an RGB `ImageType` input. The image input uses a `1/255` scale, and the iOS app
uses Vision `.scaleFill` preprocessing to match the direct-resize training
pipeline.

The iOS app uses a serial latest-frame inference path with late-frame dropping,
then tracks detections in image space and predicts their display-time position
with `CADisplayLink`. The tracker is application-side only; ANE-S itself is a
frame-by-frame detector.

### iPhone 14 Plus benchmark

Measured on a physical iPhone 14 Plus with Xcode Instruments, CPU + Neural
Engine compute units, screen recording off, and steady-state 20-25 second runs.

| Core ML specialization | Median Neural Engine interval | Mean start-to-start cadence | Detector updates/s |
| --- | ---: | ---: | ---: |
| Standard | 75.6 ms | 82.8 ms | 12.1 |
| `fastPrediction` | 68.7 ms | 75.2 ms | 13.3 |

`fastPrediction` reduced the median Neural Engine interval by about 9.1% in this
controlled comparison. The update rate is derived from steady-state Neural
Engine interval spacing, not camera or display FPS. These measurements also do
not imply that every model operation runs exclusively on the Neural Engine.

## Running the project

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used for the
training/export code.

```bash
uv sync --locked
```

The experiment notebooks contain the exact commands used for the reported
training runs. The Hydra configs under `configs/experiment/` are starting points
for reproducing or extending them.

To export a trained ANE-S checkpoint to Core ML:

```bash
uv run \
  --with 'torch==2.7.0' \
  --with 'torchvision==0.22.0' \
  --with 'coremltools==9.0' \
  python -m adp.export.coreml_check \
  --checkpoint /path/to/best.pt \
  --out artifacts/coreml/ane_s.mlpackage
```

The iOS app is in `ios/ADPDetector`. Open
`ios/ADPDetector/ADPDetector.xcodeproj` in Xcode and run it on a physical
iPhone. The repository includes the exported `ane_s.mlpackage` used by the app.

## Limitations

- The student was trained on 30k BDD100K training images for only 40 epochs and had not converged.
- Detection accuracy is well below the D-FINE-M teacher and should not be interpreted as state of the art.
- The live confidence threshold (`0.45`) was chosen for the demo and has not been formally calibrated on the validation set.
- The Core ML exporter checks reparameterization equivalence, output shapes, finiteness, and logs raw FP16 conversion error, but I have not yet completed a detection-level PyTorch/Core ML parity study.
- The iOS motion tracker improves display alignment between detector updates; it does not add temporal information to the detector itself.

## References

- [BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning](https://arxiv.org/abs/1805.04687)
- [FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization](https://arxiv.org/abs/2303.14189)
- [D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/2410.13842)

Parts of the D-FINE training/model code are vendored from D-FINE-seg under
Apache 2.0. See `THIRD_PARTY_LICENSES/`.

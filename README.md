# ANE Driving Perception

Research codebase for driving perception models optimized for the Apple Neural Engine (ANE).

> Status: Early/preliminary. Scope is currently 2D object detection on a 30K subset of BDD100K. Models were not trained to convergence due to compute constraints.

## Overview

The objective is a compact, fast object detector for driving scenes that runs entirely on the ANE, enabling real-time inference on Apple Silicon without GPU or CPU fallback.

Training utilizes a 30K subset of the BDD100K dataset at 544x960 resolution. Two training approaches are compared for the student model (ANE-S):
1. GT-only: Trained from scratch using standard D-FINE detection loss.
2. Distilled: Trained against a larger, frozen teacher model (DFINE-M) using knowledge distillation.

## ANE-Resident Design

A model is "ANE-resident" when the entire forward pass executes on the ANE without CPU/GPU fallbacks. This requires specific architectural constraints:
- Convolutions only: Linear layers are replaced with Conv2d 1x1.
- Channels-first 4D layout: Tensors are shaped (B, C, 1, S).
- Custom normalization: Normalization occurs over the channel axis.
- Supported ops only: Deformable attention, grid_sample, and einsum are avoided.
- Reparameterization: Multi-branch structures are collapsed into single convolutions at export.

## Models

### Teacher: DFINE-M
The medium variant of the D-FINE detector. Initialized from Objects365 pretrained weights and fine-tuned on the 30K BDD100K subset. Used solely as a frozen teacher for distillation.

### Student: ANE-S
A compact 6.97M parameter detector built with ANE-friendly primitives.
- Backbone: FastViT-T8 (loaded via timm).
- Encoder: Hybrid encoder with a single AIFI transformer block and CSP-Rep blocks.
- Decoder: 3-layer DETR-style decoder with 300 queries and an FGL (Fine-Grained Localization) box head.

## Distillation

Because DETR models predict an unordered set of queries, the distillation pipeline uses Hungarian matching to pair student and teacher queries that are predicting the same ground-truth object.

- Query-level Distillation (on matched pairs):
  - Box and class KD: KL divergence on class logits, plus L1 and GIoU on boxes.
  - FDR feature distillation: KL divergence on the FGL edge distribution bins.

- Spatial Distillation (no matching required):
  - AIFI attention transfer: Cosine similarity matching the 2D attention maps of the 1/32 scale AIFI encoder block.

## Setup

Requires Python 3.12+ and uv.

```bash
uv sync
```

## Usage

Commands use Hydra for configuration.

Train the DFINE-M teacher:
```bash
uv run python -m adp.train.train +experiment=dfine_m_bdd30k
```

Train ANE-S, GT-only baseline:
```bash
uv run python -m adp.train.train +experiment=ane_s_bdd_finetune
```

Distill ANE-S from a teacher checkpoint:
```bash
uv run python -m adp.distill.train_distill \
  +experiment=ane_s_bdd_distill \
  distill.teacher_checkpoint=path/to/dfine_m_best.pt
```

Core ML export:
```bash
uv run python -m adp.export.coreml_check --model path/to/model.mlpackage
```

Real-time demo:
```bash
uv run python -m adp.export.realtime_demo --model path/to/model.mlpackage
```

## Limitations

- No temporal context (frame-by-frame processing).
- Missing lane and drivable-area segmentation.
- Bounding boxes only (no instance segmentation).
- No metric depth estimation.
- Bottlenecked by compute (subset training, early stopping).
- Training pipeline can definitely be improved.
- Model architecture can also be imporved.

## Acknowledgments and References

This project builds upon the following research and open-source work:

- D-FINE: Peng, Y., et al. "D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement" (2024). Source of the teacher architecture, criterion, matcher, and FGL head. Reference implementation is used under the Apache 2.0 license.
- FastViT: Vasu, P. K. A., et al. "FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization" (ICCV 2023). Used as the ANE-S backbone.
- ml-ane-transformers: Apple's reference implementation for ANE-resident LayerNorm, attention, and FFN design.
- Distillation Framework: Influenced by DETRDistill (ICCV 2023) and KD-DETR (CVPR 2024).
- Dataset: BDD100K (Yu et al., 2018, UC Berkeley).

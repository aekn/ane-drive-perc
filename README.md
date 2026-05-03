# traffic-object-detection

Object detection on BDD100K with a RepViT backbone and FCOS-family head.

## Setup

```bash
uv sync
```

## Train

```bash
uv run python scripts/make_subset.py                # build train manifests
uv run python scripts/make_val_manifest.py          # build val manifest
uv run python scripts/train.py configs/cell_a.yaml
```

## Evaluate

```bash
uv run python scripts/eval.py configs/cell_a.yaml runs/cell_a/best.pt
```

## Test

```bash
uv run pytest
```

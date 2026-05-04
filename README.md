# ANE Drive Perception

Compact driving perception models for Apple Neural Engine deployment.

This repo contains code, scripts, notebooks, and experiments for training and evaluating lightweight object detection models on curated BDD100K detection subsets. The initial focus is detection baselines and data efficient training on 3K/5K/10K BDD100K shards, with later work targeting ANE-aware architectures, Core ML export, knowledge distillation, and additional driving perception tasks such as lane segmentation and drivable-area segmentation.

## Repository layout
```text
scripts/data/           Data prep, sharding, export, and other scripts
src/ane_drive_perc/     Project library code
notebooks/              Colab training notebooks
configs/                Experiment and model configs
```

## Data and runs

Dataset shards live in a private Hugging Face dataset repo. Training outputs, checkpoints, and logs are saved on Google Drive for now.

"""Image+bbox augmentation pipelines built on Albumentations."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

ImgSize = tuple[int, int]  # (height, width)


def _resize(img_size: ImgSize) -> list[A.BasicTransform]:
    height, width = img_size
    return [A.Resize(height=height, width=width)]


def build_train_transforms(img_size: ImgSize) -> A.Compose:
    """Build train transforms.

    Uses direct resize, not letterbox, so D-FINE postprocessing can rescale
    predictions directly back to each image's original size during COCO eval.
    """
    return A.Compose(
        [
            *_resize(img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3,
            ),
            A.GaussianBlur(blur_limit=3, p=0.05),
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_area=1.0,
            min_visibility=0.1,
        ),
    )


def build_val_transforms(img_size: ImgSize) -> A.Compose:
    """Build validation transforms."""
    return A.Compose(
        [
            *_resize(img_size),
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
        ),
    )

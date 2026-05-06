import contextlib
import io
from pathlib import Path
from typing import Any

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = (
    "mAP",
    "mAP_50",
    "mAP_75",
    "mAP_small",
    "mAP_medium",
    "mAP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


class CocoEvaluator:
    def __init__(self, *, annotations_path: Path, category_id_base: int = 1) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco_gt = COCO(str(annotations_path))
        self.category_id_base = category_id_base
        self._predictions: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._predictions = []

    def update(
        self,
        *,
        image_ids: list[int],
        outputs: list[dict[str, torch.Tensor]],
    ) -> None:
        """Append predictions for a batch.

        outputs: list of {"boxes": [K,4] xyxy abs, "scores": [K], "labels": [K]}
        """
        for image_id, out in zip(image_ids, outputs, strict=True):
            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()

            # xyxy -> xywh as required by COCO
            xywh = boxes.clone()
            xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
            xywh[:, 3] = xywh[:, 3] - xywh[:, 1]

            for i in range(scores.shape[0]):
                self._predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(labels[i].item()) + self.category_id_base,
                        "bbox": [float(v) for v in xywh[i].tolist()],
                        "score": float(scores[i].item()),
                    }
                )

    def summarize(self) -> dict[str, float]:
        if not self._predictions:
            return {name: 0.0 for name in METRIC_NAMES}

        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = self.coco_gt.loadRes(self._predictions)
            evaluator = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

        stats = evaluator.stats.tolist()
        return dict(zip(METRIC_NAMES, stats, strict=True))

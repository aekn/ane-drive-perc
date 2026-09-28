"""Run ANE-S on a camera or video using the exported Core ML model."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BDD_CLASSES = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)

_COLORS = (
    (0, 114, 189),
    (217, 83, 25),
    (237, 177, 32),
    (126, 47, 142),
    (119, 172, 48),
    (77, 190, 238),
    (162, 20, 47),
    (76, 76, 76),
    (153, 153, 0),
    (255, 0, 127),
)


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: int

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def _topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    if values.size <= k:
        return np.argsort(values)[::-1]
    indices = np.argpartition(values, -k)[-k:]
    return indices[np.argsort(values[indices])[::-1]]


def _iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    union = a.area + b.area - intersection
    return intersection / max(union, 1e-12)


def _class_aware_nms(
    detections: list[Detection],
    *,
    iou_threshold: float,
    max_detections: int,
) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if any(
            previous.label == detection.label
            and _iou(previous, detection) > iou_threshold
            for previous in kept
        ):
            continue
        kept.append(detection)
        if len(kept) == max_detections:
            break
    return kept


def postprocess(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    *,
    score_threshold: float,
    nms_iou_threshold: float,
    topk: int = 300,
    max_detections: int = 80,
) -> list[Detection]:
    """Convert DETR outputs to normalized xyxy detections for visualization."""
    logits = np.asarray(pred_logits)[0]
    boxes = np.asarray(pred_boxes)[0]
    if logits.ndim != 2 or boxes.shape != (logits.shape[0], 4):
        raise ValueError(
            f"unexpected output shapes: logits={logits.shape}, boxes={boxes.shape}"
        )

    scores = _sigmoid(logits)
    flat_scores = scores.reshape(-1)
    class_count = scores.shape[1]

    candidates: list[Detection] = []
    for index in _topk_indices(flat_scores, topk):
        score = float(flat_scores[index])
        if score < score_threshold:
            break

        query = int(index // class_count)
        label = int(index % class_count)
        cx, cy, width, height = [float(value) for value in boxes[query]]
        x1 = max(0.0, min(1.0, cx - width / 2.0))
        y1 = max(0.0, min(1.0, cy - height / 2.0))
        x2 = max(0.0, min(1.0, cx + width / 2.0))
        y2 = max(0.0, min(1.0, cy + height / 2.0))
        if x2 <= x1 or y2 <= y1:
            continue

        candidates.append(Detection(x1, y1, x2, y2, score, label))

    return _class_aware_nms(
        candidates,
        iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )


def _model_image(frame: np.ndarray, *, height: int, width: int) -> Image.Image:
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _draw(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    height, width = frame.shape[:2]
    output = frame.copy()

    for detection in detections:
        x1 = int(round(detection.x1 * width))
        y1 = int(round(detection.y1 * height))
        x2 = int(round(detection.x2 * width))
        y2 = int(round(detection.y2 * height))
        color = _COLORS[detection.label % len(_COLORS)]

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        name = (
            BDD_CLASSES[detection.label]
            if detection.label < len(BDD_CLASSES)
            else str(detection.label)
        )
        text = f"{name} {detection.score:.0%}"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        top = max(0, y1 - text_height - 6)
        cv2.rectangle(
            output,
            (x1, top),
            (x1 + text_width + 6, top + text_height + 6),
            color,
            -1,
        )
        cv2.putText(
            output,
            text,
            (x1 + 3, top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def run(
    *,
    model_path: Path,
    video_source: int | str,
    model_height: int,
    model_width: int,
    score_threshold: float,
    nms_iou_threshold: float,
) -> None:
    import coremltools as ct

    model = ct.models.MLModel(
        str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video source: {video_source}")

    timings: list[float] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            image = _model_image(frame, height=model_height, width=model_width)
            start = time.perf_counter()
            output = model.predict({"image": image})
            inference_ms = (time.perf_counter() - start) * 1000.0

            detections = postprocess(
                output["pred_logits"],
                output["pred_boxes"],
                score_threshold=score_threshold,
                nms_iou_threshold=nms_iou_threshold,
            )
            visualization = _draw(frame, detections)

            timings.append(inference_ms)
            if len(timings) > 30:
                timings.pop(0)
            average_ms = sum(timings) / len(timings)
            hud = f"ANE-S  {average_ms:.1f} ms  {len(detections)} objects"
            cv2.putText(
                visualization,
                hud,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("ANE-S", visualization)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", default="0")
    parser.add_argument("--img-h", type=int, default=544)
    parser.add_argument("--img-w", type=int, default=960)
    parser.add_argument("--score-thresh", type=float, default=0.45)
    parser.add_argument("--nms-iou-thresh", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        video_source: int | str = int(args.video)
    except ValueError:
        video_source = args.video

    run(
        model_path=args.model,
        video_source=video_source,
        model_height=args.img_h,
        model_width=args.img_w,
        score_threshold=args.score_thresh,
        nms_iou_threshold=args.nms_iou_thresh,
    )


if __name__ == "__main__":
    main()

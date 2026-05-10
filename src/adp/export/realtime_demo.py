"""Real-time object detection demo using a CoreML ANE-S model.

Usage:
    python -m adp.export.realtime_demo --model artifacts/ane_s.mlpackage
    python -m adp.export.realtime_demo --model artifacts/ane_s.mlpackage --video path/to/file.mp4
    python -m adp.export.realtime_demo --model artifacts/ane_s.mlpackage --score-thresh 0.5
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# BDD-10 class names 0 indexed
BDD_CLASSES = [
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
]

_COLORS = [
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
]


def _letterbox(
    frame: np.ndarray, target_h: int, target_w: int
) -> tuple[np.ndarray, float, int, int]:
    """Resize with padding to preserve aspect ratio.

    Returns (padded, scale, pad_left, pad_top).
    """
    src_h, src_w = frame.shape[:2]
    scale = min(target_h / src_h, target_w / src_w)
    new_h, new_w = int(src_h * scale), int(src_w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas, scale, pad_left, pad_top


def _preprocess(
    frame_bgr: np.ndarray, h: int, w: int
) -> tuple[np.ndarray, float, int, int]:
    padded, scale, pad_left, pad_top = _letterbox(frame_bgr, h, w)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = rgb.transpose(2, 0, 1)[None]  # (1, 3, H, W)
    return chw, scale, pad_left, pad_top


def _postprocess(
    pred_logits: np.ndarray,  # (1, Q, C)
    pred_boxes: np.ndarray,  # (1, Q, 4) cxcywh normalized
    img_h: int,
    img_w: int,
    scale: float,
    pad_left: int,
    pad_top: int,
    score_thresh: float,
    nms_iou_thresh: float,
) -> list[tuple[int, int, int, int, float, int]]:
    """Returns list of (x1, y1, x2, y2, score, label) in original frame coords."""
    scores_all = 1.0 / (1.0 + np.exp(-pred_logits[0]))  # sigmoid, (Q, C)
    flat = scores_all.ravel()
    q, c = scores_all.shape
    boxes_cx = pred_boxes[0, :, 0]
    boxes_cy = pred_boxes[0, :, 1]
    boxes_w = pred_boxes[0, :, 2]
    boxes_h = pred_boxes[0, :, 3]

    # filter by score threshold
    mask = flat >= score_thresh
    if not mask.any():
        return []

    idx = np.where(mask)[0]
    scores = flat[idx]
    labels = idx % c
    qidx = idx // c

    x1 = (boxes_cx[qidx] - boxes_w[qidx] / 2) * img_w
    y1 = (boxes_cy[qidx] - boxes_h[qidx] / 2) * img_h
    x2 = (boxes_cx[qidx] + boxes_w[qidx] / 2) * img_w
    y2 = (boxes_cy[qidx] + boxes_h[qidx] / 2) * img_h

    # NMS; class agnostic 4 speed
    order = np.argsort(-scores)
    x1, y1, x2, y2 = x1[order], y1[order], x2[order], y2[order]
    scores, labels = scores[order], labels[order]

    keep = []
    suppressed = np.zeros(len(x1), dtype=bool)
    for i in range(len(x1)):
        if suppressed[i]:
            continue
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[i + 1 :])
        iy1 = np.maximum(y1[i], y1[i + 1 :])
        ix2 = np.minimum(x2[i], x2[i + 1 :])
        iy2 = np.minimum(y2[i], y2[i + 1 :])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_j = (x2[i + 1 :] - x1[i + 1 :]) * (y2[i + 1 :] - y1[i + 1 :])
        iou = inter / np.maximum(area_i + area_j - inter, 1e-6)
        suppressed[i + 1 :][iou > nms_iou_thresh] = True

    results = []
    for i in keep:
        # unpad + unscale back to original frame coords
        fx1 = int((x1[i] - pad_left) / scale)
        fy1 = int((y1[i] - pad_top) / scale)
        fx2 = int((x2[i] - pad_left) / scale)
        fy2 = int((y2[i] - pad_top) / scale)
        results.append((fx1, fy1, fx2, fy2, float(scores[i]), int(labels[i])))
    return results


def _draw(frame: np.ndarray, dets: list, class_names: list[str]) -> np.ndarray:
    out = frame.copy()
    for x1, y1, x2, y2, score, label in dets:
        color = _COLORS[label % len(_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = class_names[label] if label < len(class_names) else str(label)
        text = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            text,
            (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return out


def run(
    model_path: Path,
    video_source: int | str,
    model_h: int,
    model_w: int,
    score_thresh: float,
    nms_iou_thresh: float,
    class_names: list[str],
) -> None:
    import coremltools as ct

    print(f"loading {model_path} ...")
    model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source: {video_source}")

    print("press q to quit")
    frame_times: list[float] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        chw, scale, pad_left, pad_top = _preprocess(frame, model_h, model_w)
        pred = model.predict({"image": chw})
        logits = pred["pred_logits"]  # (1, Q, C)
        boxes = pred["pred_boxes"]  # (1, Q, 4)
        infer_ms = (time.perf_counter() - t0) * 1000

        dets = _postprocess(
            logits,
            boxes,
            model_h,
            model_w,
            scale,
            pad_left,
            pad_top,
            score_thresh,
            nms_iou_thresh,
        )
        vis = _draw(frame, dets, class_names)

        frame_times.append(infer_ms)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_ms = sum(frame_times) / len(frame_times)
        cv2.putText(
            vis,
            f"{avg_ms:.1f} ms  ({1000 / avg_ms:.1f} fps)  n={len(dets)}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("ANE-S demo", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True, help="path to .mlpackage")
    parser.add_argument(
        "--video", default=0, help="video file path or webcam index (default: 0)"
    )
    parser.add_argument("--img-h", type=int, default=544)
    parser.add_argument("--img-w", type=int, default=960)
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--nms-iou-thresh", type=float, default=0.5)
    args = parser.parse_args()

    video_source: int | str = args.video
    try:
        video_source = int(args.video)
    except (TypeError, ValueError):
        pass

    run(
        model_path=args.model,
        video_source=video_source,
        model_h=args.img_h,
        model_w=args.img_w,
        score_thresh=args.score_thresh,
        nms_iou_thresh=args.nms_iou_thresh,
        class_names=BDD_CLASSES,
    )


if __name__ == "__main__":
    main()

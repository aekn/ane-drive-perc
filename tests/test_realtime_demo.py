import numpy as np

from adp.export.realtime_demo import postprocess


def _logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


def test_postprocess_uses_global_topk() -> None:
    logits = np.full((1, 2, 2), _logit(0.01), dtype=np.float32)
    logits[0, 0, 0] = _logit(0.90)
    logits[0, 1, 1] = _logit(0.80)
    boxes = np.array(
        [[[0.25, 0.25, 0.20, 0.20], [0.75, 0.75, 0.20, 0.20]]],
        dtype=np.float32,
    )

    detections = postprocess(
        logits,
        boxes,
        score_threshold=0.1,
        nms_iou_threshold=0.6,
        topk=1,
    )

    assert len(detections) == 1
    assert detections[0].label == 0
    assert detections[0].score > 0.89


def test_postprocess_nms_is_class_aware() -> None:
    logits = np.full((1, 2, 2), _logit(0.01), dtype=np.float32)
    logits[0, 0, 0] = _logit(0.90)
    logits[0, 1, 1] = _logit(0.85)
    boxes = np.array(
        [[[0.50, 0.50, 0.60, 0.60], [0.50, 0.50, 0.60, 0.60]]],
        dtype=np.float32,
    )

    detections = postprocess(
        logits,
        boxes,
        score_threshold=0.1,
        nms_iou_threshold=0.5,
    )

    assert {detection.label for detection in detections} == {0, 1}


def test_postprocess_suppresses_same_class_overlap() -> None:
    logits = np.full((1, 2, 1), _logit(0.01), dtype=np.float32)
    logits[0, 0, 0] = _logit(0.90)
    logits[0, 1, 0] = _logit(0.80)
    boxes = np.array(
        [[[0.50, 0.50, 0.60, 0.60], [0.50, 0.50, 0.58, 0.58]]],
        dtype=np.float32,
    )

    detections = postprocess(
        logits,
        boxes,
        score_threshold=0.1,
        nms_iou_threshold=0.5,
    )

    assert len(detections) == 1
    assert detections[0].score > 0.89

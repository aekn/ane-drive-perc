from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResizeTransform:
    orig_height: int
    orig_width: int
    new_height: int
    new_width: int
    scale_y: float
    scale_x: float
    pad_top: float = 0.0
    pad_left: float = 0.0


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    x, y, w, h = boxes.unbind(dim=-1)
    return torch.stack((x, y, x + w, y + h), dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    return torch.stack((x1, y1, x2 - x1, y2 - y1), dim=-1)


def resize_boxes_xyxy(boxes: torch.Tensor, transform: ResizeTransform) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    out = boxes.clone()
    out[:, [0, 2]] = out[:, [0, 2]] * transform.scale_x + transform.pad_left
    out[:, [1, 3]] = out[:, [1, 3]] * transform.scale_y + transform.pad_top
    return out


def inverse_resize_boxes_xyxy(
    boxes: torch.Tensor, transform: ResizeTransform
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    out = boxes.clone()
    out[:, [0, 2]] = (out[:, [0, 2]] - transform.pad_left) / transform.scale_x
    out[:, [1, 3]] = (out[:, [1, 3]] - transform.pad_top) / transform.scale_y
    return out


def clip_boxes_xyxy(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    out = boxes.clone()
    out[:, [0, 2]] = out[:, [0, 2]].clamp(min=0, max=width)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(min=0, max=height)
    return out


def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = box_area_xyxy(boxes1)
    area2 = box_area_xyxy(boxes2)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-12)

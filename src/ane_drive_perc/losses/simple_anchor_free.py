from typing import Any, TypedDict

import torch
import torch.nn.functional as F


class DetectionLossOutput(TypedDict):
    loss: torch.Tensor
    cls_loss: torch.Tensor
    box_loss: torch.Tensor
    num_pos: torch.Tensor


def simple_anchor_free_detection_loss(
    predictions: dict[str, torch.Tensor],
    targets: list[dict[str, Any]],
    *,
    image_height: int,
    image_width: int,
    num_classes: int,
    cls_weight: float = 1.0,
    box_weight: float = 5.0,
) -> DetectionLossOutput:
    class_logits = predictions["class_logits"]
    pred_boxes = predictions["box_cxcywh"]
    batch_size, _, grid_height, grid_width = class_logits.shape
    device = class_logits.device

    target_classes = torch.zeros(
        (batch_size, num_classes, grid_height, grid_width),
        dtype=class_logits.dtype,
        device=device,
    )
    target_boxes = torch.zeros(
        (batch_size, 4, grid_height, grid_width),
        dtype=pred_boxes.dtype,
        device=device,
    )
    positive_mask = torch.zeros(
        (batch_size, 1, grid_height, grid_width),
        dtype=torch.bool,
        device=device,
    )

    stride_y = image_height / grid_height
    stride_x = image_width / grid_width

    for batch_index, target in enumerate(targets):
        boxes = target["boxes"].to(device=device, dtype=torch.float32)
        labels = target["labels"].to(device=device, dtype=torch.long)
        if boxes.numel() == 0:
            continue
        assign_targets_to_grid(
            batch_index=batch_index,
            boxes=boxes,
            labels=labels,
            target_classes=target_classes,
            target_boxes=target_boxes,
            positive_mask=positive_mask,
            stride_x=stride_x,
            stride_y=stride_y,
            image_width=image_width,
            image_height=image_height,
            num_classes=num_classes,
        )

    num_pos = positive_mask.sum().to(dtype=class_logits.dtype).clamp(min=1.0)
    cls_loss = binary_focal_loss_with_logits(
        class_logits, target_classes, num_pos=num_pos
    )

    if positive_mask.any():
        box_mask = positive_mask.expand_as(pred_boxes)
        box_loss = (
            F.l1_loss(pred_boxes[box_mask], target_boxes[box_mask], reduction="sum")
            / num_pos
        )
    else:
        box_loss = pred_boxes.sum() * 0.0

    total_loss = cls_weight * cls_loss + box_weight * box_loss
    return {
        "loss": total_loss,
        "cls_loss": cls_loss.detach(),
        "box_loss": box_loss.detach(),
        "num_pos": num_pos.detach(),
    }


def assign_targets_to_grid(
    *,
    batch_index: int,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    target_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    positive_mask: torch.Tensor,
    stride_x: float,
    stride_y: float,
    image_width: int,
    image_height: int,
    num_classes: int,
) -> None:
    x1 = boxes[:, 0].clamp(min=0, max=image_width)
    y1 = boxes[:, 1].clamp(min=0, max=image_height)
    x2 = boxes[:, 2].clamp(min=0, max=image_width)
    y2 = boxes[:, 3].clamp(min=0, max=image_height)
    widths = (x2 - x1).clamp(min=1.0)
    heights = (y2 - y1).clamp(min=1.0)
    cx = x1 + 0.5 * widths
    cy = y1 + 0.5 * heights

    grid_width = target_classes.shape[-1]
    grid_height = target_classes.shape[-2]
    grid_x = torch.floor(cx / stride_x).long().clamp(min=0, max=grid_width - 1)
    grid_y = torch.floor(cy / stride_y).long().clamp(min=0, max=grid_height - 1)

    for obj_index in range(boxes.shape[0]):
        label = int(labels[obj_index].item())
        if label < 0 or label >= num_classes:
            raise ValueError(
                f"Label {label} is outside valid range [0, {num_classes})."
            )

        gx = int(grid_x[obj_index].item())
        gy = int(grid_y[obj_index].item())
        target_classes[batch_index, label, gy, gx] = 1.0
        target_boxes[batch_index, :, gy, gx] = torch.tensor(
            [
                cx[obj_index] / image_width,
                cy[obj_index] / image_height,
                widths[obj_index] / image_width,
                heights[obj_index] / image_height,
            ],
            dtype=target_boxes.dtype,
            device=target_boxes.device,
        )
        positive_mask[batch_index, 0, gy, gx] = True


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_pos: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    return loss.sum() / num_pos

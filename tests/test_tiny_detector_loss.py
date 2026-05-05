import torch

from ane_drive_perc.losses.simple_anchor_free import simple_anchor_free_detection_loss
from ane_drive_perc.models.tiny_detector import TinyAnchorFreeDetector


def test_tiny_detector_shapes() -> None:
    model = TinyAnchorFreeDetector(num_classes=10, width=16)
    output = model(torch.randn(2, 3, 64, 64))
    assert output["class_logits"].shape == (2, 10, 8, 8)
    assert output["box_cxcywh"].shape == (2, 4, 8, 8)


def test_simple_anchor_free_loss_backward() -> None:
    model = TinyAnchorFreeDetector(num_classes=10, width=16)
    images = torch.randn(2, 3, 64, 64)
    predictions = model(images)
    targets = [
        {"boxes": torch.tensor([[5.0, 10.0, 15.0, 20.0]]), "labels": torch.tensor([2])},
        {
            "boxes": torch.tensor([[10.0, 20.0, 40.0, 60.0]]),
            "labels": torch.tensor([9]),
        },
    ]
    loss_output = simple_anchor_free_detection_loss(
        predictions,
        targets,
        image_height=64,
        image_width=64,
        num_classes=10,
    )
    assert loss_output["loss"].item() > 0.0
    loss_output["loss"].backward()

import json
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from adp.config import write_resolved_config
from adp.model.dfine import build_model
from adp.model.dfine.postprocess import DFINEPostProcessor
from adp.train.augment import build_val_transforms
from adp.train.dataset import CocoDetectionDataset
from adp.utils.paths import ensure_run_paths


PRED_COLORS = {
    0: "red",  # pedestrian
    1: "orange",  # rider
    2: "deepskyblue",  # car
    3: "magenta",  # truck
    4: "yellow",  # bus
    5: "cyan",  # train
    6: "violet",  # motorcycle
    7: "dodgerblue",  # bicycle
    8: "white",  # traffic light
    9: "hotpink",  # traffic sign
}

GT_COLOR = "lime"


def _class_names(cfg: DictConfig) -> list[str]:
    return [str(name) for name in cfg.classes.names]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _image_lookup(coco_root: Path, annotations_file: str) -> dict[int, dict[str, Any]]:
    annotation_path = coco_root / "annotations" / annotations_file
    data = _read_json(annotation_path)
    return {int(image["id"]): image for image in data["images"]}


def _to_xyxy_from_cxcywh_norm(
    boxes: torch.Tensor,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - 0.5 * w) * width
    y1 = (cy - 0.5 * h) * height
    x2 = (cx + 0.5 * w) * width
    y2 = (cy + 0.5 * h) * height
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: torch.Tensor,
    *,
    color: str,
    width: int,
) -> None:
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    color: str,
    font: ImageFont.ImageFont,
) -> None:
    draw.text(xy, text, fill=color, font=font)


def _prediction_color(label_id: int) -> str:
    return PRED_COLORS.get(label_id, "red")


def _make_canvas_with_legend(
    image: Image.Image,
    *,
    class_names: list[str],
    gt_counts: Counter[int],
    pred_counts: Counter[int],
    top_predictions: list[tuple[int, float]],
    score_threshold: float,
) -> Image.Image:
    legend_width = 360
    padding = 14
    line_height = 20

    canvas = Image.new(
        "RGB",
        (image.width + legend_width, image.height),
        color=(24, 24, 24),
    )
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x = image.width + padding
    y = padding

    _draw_text(draw, (x, y), "Legend", color="white", font=font)
    y += line_height + 4

    _draw_text(draw, (x, y), "GT boxes: green", color=GT_COLOR, font=font)
    y += line_height

    _draw_text(
        draw,
        (x, y),
        f"Pred boxes: class color, score >= {score_threshold:.2f}",
        color="white",
        font=font,
    )
    y += line_height + 8

    _draw_text(draw, (x, y), "Prediction classes", color="white", font=font)
    y += line_height

    for class_id, class_name in enumerate(class_names):
        color = _prediction_color(class_id)
        count = pred_counts.get(class_id, 0)
        draw.rectangle([x, y + 4, x + 12, y + 16], outline=color, width=2)
        _draw_text(
            draw,
            (x + 20, y),
            f"{class_id}: {class_name} ({count})",
            color=color,
            font=font,
        )
        y += line_height

    y += 8
    _draw_text(draw, (x, y), "GT counts", color=GT_COLOR, font=font)
    y += line_height

    for class_id, count in sorted(gt_counts.items()):
        class_name = class_names[class_id]
        _draw_text(
            draw,
            (x, y),
            f"{class_name}: {count}",
            color=GT_COLOR,
            font=font,
        )
        y += line_height

    y += 8
    _draw_text(draw, (x, y), "Top predictions", color="white", font=font)
    y += line_height

    for class_id, score in top_predictions[:10]:
        color = _prediction_color(class_id)
        class_name = class_names[class_id]
        _draw_text(
            draw,
            (x, y),
            f"{class_name}: {score:.2f}",
            color=color,
            font=font,
        )
        y += line_height

    return canvas


def visualize_predictions(cfg: DictConfig) -> dict[str, object]:
    run_paths = ensure_run_paths(cfg)
    write_resolved_config(cfg, run_paths.root / "config.resolved.yaml")

    requested_device = str(cfg.train.device)
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    img_size = (int(cfg.train.img_size[0]), int(cfg.train.img_size[1]))
    class_names = _class_names(cfg)

    coco_root = Path(str(cfg.train.coco_root))
    annotations_file = str(cfg.train.val_annotations)
    checkpoint_path = Path(str(cfg.eval.checkpoint_path))
    output_dir = run_paths.debug / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not coco_root.exists():
        raise FileNotFoundError(f"COCO root not found: {coco_root}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    image_by_id = _image_lookup(coco_root, annotations_file)

    dataset = CocoDetectionDataset(
        coco_root=coco_root,
        annotations_file=annotations_file,
        transforms=build_val_transforms(img_size),
        category_id_base=int(cfg.coco.category_id_base),
    )

    model = build_model(
        model_name=str(cfg.train.model_name),
        num_classes=int(cfg.train.num_classes),
        enable_mask_head=False,
        device=device,
        img_size=list(img_size),
        pretrained_model_path=None,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    postprocessor = DFINEPostProcessor(
        num_classes=int(cfg.train.num_classes),
        num_top_queries=int(cfg.train.num_top_queries),
    ).to(device)
    postprocessor.eval()

    score_threshold = float(cfg.eval.score_threshold)
    max_images = min(int(cfg.eval.max_images), len(dataset))
    max_predictions_per_image = int(getattr(cfg.eval, "max_predictions_per_image", 80))
    draw_prediction_labels = bool(getattr(cfg.eval, "draw_prediction_labels", False))

    font = ImageFont.load_default()
    written: list[str] = []

    with torch.no_grad():
        for idx in range(max_images):
            image_tensor, target = dataset[idx]

            image_id = int(target["image_id"].item())
            orig_h, orig_w = [int(v) for v in target["orig_size"].tolist()]

            image_info = image_by_id[image_id]
            image_path = coco_root / str(image_info["file_name"])

            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            gt_boxes = _to_xyxy_from_cxcywh_norm(
                target["boxes"].cpu(),
                width=orig_w,
                height=orig_h,
            )
            gt_labels = [int(v) for v in target["labels"].cpu().tolist()]
            gt_counts = Counter(gt_labels)

            for box in gt_boxes:
                _draw_box(draw, box, color=GT_COLOR, width=2)

            batch = image_tensor.unsqueeze(0).to(device)
            orig_sizes = torch.tensor(
                [[orig_h, orig_w]],
                dtype=torch.long,
                device=device,
            )

            outputs = model(batch)
            result = postprocessor(outputs, orig_sizes)[0]

            scores = result["scores"].detach().cpu()
            labels = result["labels"].detach().cpu()
            boxes = result["boxes"].detach().cpu()

            keep = scores >= score_threshold
            kept_scores = scores[keep]
            kept_labels = labels[keep]
            kept_boxes = boxes[keep]

            if len(kept_scores) > max_predictions_per_image:
                top_scores, top_indices = torch.topk(
                    kept_scores,
                    k=max_predictions_per_image,
                )
                kept_scores = top_scores
                kept_labels = kept_labels[top_indices]
                kept_boxes = kept_boxes[top_indices]

            pred_label_ids = [int(v) for v in kept_labels.tolist()]
            pred_counts = Counter(pred_label_ids)
            top_predictions = [
                (int(label_id), float(score))
                for label_id, score in zip(kept_labels.tolist(), kept_scores.tolist())
            ]
            top_predictions.sort(key=lambda item: item[1], reverse=True)

            for box, label_id, score in zip(kept_boxes, kept_labels, kept_scores):
                class_id = int(label_id)
                color = _prediction_color(class_id)

                _draw_box(draw, box, color=color, width=3)

                if draw_prediction_labels:
                    x1, y1, _, _ = [float(v) for v in box]
                    class_name = class_names[class_id]
                    _draw_text(
                        draw,
                        (int(x1) + 2, max(0, int(y1) - 12)),
                        f"{class_name} {float(score):.2f}",
                        color=color,
                        font=font,
                    )

            canvas = _make_canvas_with_legend(
                image,
                class_names=class_names,
                gt_counts=gt_counts,
                pred_counts=pred_counts,
                top_predictions=top_predictions,
                score_threshold=score_threshold,
            )

            safe_name = str(image_info["file_name"]).replace("/", "_")
            output_path = output_dir / f"{idx:03d}_{safe_name}"
            canvas.save(output_path)
            written.append(str(output_path))

    report = {
        "checkpoint_path": str(checkpoint_path),
        "coco_root": str(coco_root),
        "annotations_file": annotations_file,
        "output_dir": str(output_dir),
        "score_threshold": score_threshold,
        "max_predictions_per_image": max_predictions_per_image,
        "draw_prediction_labels": draw_prediction_labels,
        "image_count": len(written),
        "images": written,
    }

    report_path = run_paths.debug / "prediction_visualization.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[adp] wrote {len(written)} prediction visualizations: {output_dir}")
    print(f"[adp] wrote report: {report_path}")
    print("[adp] ground truth = green")
    print("[adp] predictions = class colors shown in legend")

    return report


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    visualize_predictions(cfg)


if __name__ == "__main__":
    main()

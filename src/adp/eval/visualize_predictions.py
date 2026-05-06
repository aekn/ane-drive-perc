import json
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw

from adp.config import write_resolved_config
from adp.model.dfine import build_model
from adp.model.dfine.postprocess import DFINEPostProcessor
from adp.train.augment import build_val_transforms
from adp.train.dataset import CocoDetectionDataset
from adp.utils.paths import ensure_run_paths


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
    label: str,
    *,
    color: str,
    width: int,
) -> None:
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    draw.text((x1 + 2, max(0, y1 - 14)), label, fill=color)


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
            gt_labels = target["labels"].cpu().tolist()

            for box, label_id in zip(gt_boxes, gt_labels):
                class_name = class_names[int(label_id)]
                _draw_box(
                    draw,
                    box,
                    f"GT {class_name}",
                    color="lime",
                    width=2,
                )

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

            for box, label_id, score in zip(boxes[keep], labels[keep], scores[keep]):
                class_name = class_names[int(label_id)]
                _draw_box(
                    draw,
                    box,
                    f"P {class_name} {float(score):.2f}",
                    color="red",
                    width=3,
                )

            safe_name = str(image_info["file_name"]).replace("/", "_")
            output_path = output_dir / f"{idx:03d}_{safe_name}"
            image.save(output_path)
            written.append(str(output_path))

    report = {
        "checkpoint_path": str(checkpoint_path),
        "coco_root": str(coco_root),
        "annotations_file": annotations_file,
        "output_dir": str(output_dir),
        "score_threshold": score_threshold,
        "image_count": len(written),
        "images": written,
    }

    report_path = run_paths.debug / "prediction_visualization.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[adp] wrote {len(written)} prediction visualizations: {output_dir}")
    print(f"[adp] wrote report: {report_path}")
    print("[adp] green = ground truth; red = prediction")

    return report


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    visualize_predictions(cfg)


if __name__ == "__main__":
    main()

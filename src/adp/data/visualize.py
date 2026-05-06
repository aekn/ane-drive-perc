import hashlib
from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from adp.config import write_resolved_config
from adp.data.classes import build_class_map
from adp.data.manifest import read_manifest
from adp.data.records import ImageRecord, ObjectAnnotation
from adp.utils.io import write_text
from adp.utils.paths import ensure_run_paths


def _class_color(class_id: int) -> tuple[int, int, int]:
    digest = hashlib.md5(str(class_id).encode("utf-8")).digest()
    return digest[0], digest[1], digest[2]


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _record_image_path(record: ImageRecord, *, data_root: Path) -> Path:
    path = record.path
    if path.is_absolute():
        return path
    return data_root / path


def _draw_object(
    *,
    draw: ImageDraw.ImageDraw,
    obj: ObjectAnnotation,
) -> None:
    x1, y1, x2, y2 = obj.bbox_xyxy
    color = _class_color(obj.class_id)

    for offset in range(3):
        draw.rectangle(
            (x1 - offset, y1 - offset, x2 + offset, y2 + offset),
            outline=color,
        )


def _text_size(
    *,
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _legend_width(
    *,
    class_names: tuple[str, ...],
    font: ImageFont.ImageFont,
    min_width: int = 220,
) -> int:
    scratch = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(scratch)

    max_text_width = 0
    for idx, name in enumerate(class_names):
        text = f"{idx}: {name}"
        width, _ = _text_size(draw=draw, text=text, font=font)
        max_text_width = max(max_text_width, width)

    return max(min_width, max_text_width + 64)


def _draw_legend(
    *,
    canvas: Image.Image,
    image_width: int,
    class_names: tuple[str, ...],
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(canvas)

    panel_x0 = image_width
    panel_y0 = 0
    panel_x1 = canvas.width
    panel_y1 = canvas.height

    draw.rectangle((panel_x0, panel_y0, panel_x1, panel_y1), fill=(20, 20, 20))

    title = "Classes"
    title_x = panel_x0 + 16
    title_y = 16
    draw.text((title_x, title_y), title, fill=(255, 255, 255), font=font)

    y = title_y + 28
    swatch_size = 14
    row_gap = 10

    for class_id, class_name in enumerate(class_names):
        color = _class_color(class_id)
        text = f"{class_id}: {class_name}"

        swatch_x0 = panel_x0 + 16
        swatch_y0 = y + 2
        swatch_x1 = swatch_x0 + swatch_size
        swatch_y1 = swatch_y0 + swatch_size

        draw.rectangle((swatch_x0, swatch_y0, swatch_x1, swatch_y1), fill=color)
        draw.text((swatch_x1 + 10, y), text, fill=color, font=font)

        _, text_height = _text_size(draw=draw, text=text, font=font)
        y += max(swatch_size, text_height) + row_gap


def _render_record(
    *,
    record: ImageRecord,
    image_path: Path,
    output_path: Path,
    class_names: tuple[str, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        font = _font()

        legend_width = _legend_width(class_names=class_names, font=font)
        canvas = Image.new(
            "RGB",
            (image.width + legend_width, image.height),
            color=(20, 20, 20),
        )
        canvas.paste(image, (0, 0))

        draw = ImageDraw.Draw(canvas)

        for obj in record.objects:
            _draw_object(draw=draw, obj=obj)

        _draw_legend(
            canvas=canvas,
            image_width=image.width,
            class_names=class_names,
            font=font,
        )

        canvas.save(output_path, quality=95)


def _make_grid(
    image_paths: list[Path],
    output_path: Path,
    *,
    columns: int,
    thumbnail_size: int,
) -> None:
    if not image_paths:
        return

    thumbs: list[Image.Image] = []

    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumbnail_size, thumbnail_size))

            canvas = Image.new(
                "RGB",
                (thumbnail_size, thumbnail_size),
                color=(20, 20, 20),
            )
            x = (thumbnail_size - image.width) // 2
            y = (thumbnail_size - image.height) // 2
            canvas.paste(image, (x, y))
            thumbs.append(canvas)

    rows = (len(thumbs) + columns - 1) // columns
    grid = Image.new(
        "RGB",
        (columns * thumbnail_size, rows * thumbnail_size),
        color=(20, 20, 20),
    )

    for idx, thumb in enumerate(thumbs):
        x = (idx % columns) * thumbnail_size
        y = (idx // columns) * thumbnail_size
        grid.paste(thumb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, quality=95)


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    run_paths = ensure_run_paths(cfg)
    write_resolved_config(cfg, run_paths.root / "config.resolved.yaml")

    class_map = build_class_map(cfg)
    data_root = Path(str(cfg.data.root))

    split = str(cfg.samples.split)
    limit = int(cfg.samples.limit)
    columns = int(cfg.samples.grid_columns)
    thumbnail_size = int(cfg.samples.thumbnail_size)

    manifest_path = Path(str(cfg.data.manifests[split]))
    records = list(read_manifest(manifest_path))[:limit]

    samples_dir = run_paths.debug / "samples"
    rendered_paths: list[Path] = []

    lines = [
        "# Visualization report",
        "",
        f"split: {split}",
        f"data_root: {data_root}",
        f"manifest: {manifest_path}",
        f"limit: {limit}",
        "",
        "Boxes are color-coded by class. Labels are shown in the legend panel.",
        "",
    ]

    for idx, record in enumerate(records):
        image_path = _record_image_path(record, data_root=data_root)
        output_path = samples_dir / f"{idx:04d}_{record.image_id}.jpg"

        _render_record(
            record=record,
            image_path=image_path,
            output_path=output_path,
            class_names=class_map.names,
        )
        rendered_paths.append(output_path)

        lines.append(f"## {idx:04d} — {record.image_id}")
        lines.append("")
        lines.append(f"- source: `{record.file_name}`")
        lines.append(f"- resolved_source: `{image_path}`")
        lines.append(f"- rendered: `{output_path}`")
        lines.append(f"- size: `{record.width}x{record.height}`")
        lines.append(f"- objects: `{len(record.objects)}`")

        class_counts: dict[str, int] = {}
        for obj in record.objects:
            class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1

        if class_counts:
            lines.append("- class counts:")
            for class_name, count in sorted(class_counts.items()):
                lines.append(f"  - `{class_name}`: {count}")

        lines.append("")

    grid_path = run_paths.debug / "sample_grid.jpg"
    _make_grid(
        rendered_paths,
        grid_path,
        columns=columns,
        thumbnail_size=thumbnail_size,
    )

    report_path = run_paths.debug / "visualization_report.md"
    write_text(report_path, "\n".join(lines) + "\n")

    print(f"[adp] wrote rendered samples: {samples_dir}")
    print(f"[adp] wrote sample grid: {grid_path}")
    print(f"[adp] wrote visualization report: {report_path}")


if __name__ == "__main__":
    main()

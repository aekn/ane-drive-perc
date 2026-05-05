import io
import json
import tarfile
from pathlib import Path

from PIL import Image

from ane_drive_perc.data.coco_export import materialize_coco_from_local_shards


def _make_test_image_bytes() -> bytes:
    image = Image.new("RGB", (1280, 720), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _add_bytes_to_tar(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_materialize_coco_from_local_shards(tmp_path: Path) -> None:
    shard_path = tmp_path / "sample-000000.tar"
    output_dir = tmp_path / "coco"
    metadata = {
        "id": "99267ac5-d1d37e06",
        "split": "train",
        "width": 1280,
        "height": 720,
        "boxes": [
            {
                "category": "car",
                "category_id": 2,
                "source_category": "car",
                "xyxy": [100.0, 200.0, 300.0, 400.0],
                "occluded": False,
                "truncated": False,
            }
        ],
    }
    with tarfile.open(shard_path, "w") as tar:
        _add_bytes_to_tar(tar, "99267ac5-d1d37e06.jpg", _make_test_image_bytes())
        _add_bytes_to_tar(tar, "99267ac5-d1d37e06.json", json.dumps(metadata).encode())

    result = materialize_coco_from_local_shards(
        shards=[shard_path],
        output_dir=output_dir,
        split="train",
    )

    assert result.num_images == 1
    assert result.num_annotations == 1
    with result.annotation_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    assert coco["annotations"][0]["category_id"] == 3
    assert coco["annotations"][0]["bbox"] == [100.0, 200.0, 200.0, 200.0]

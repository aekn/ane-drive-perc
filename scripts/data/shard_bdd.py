import io
import json
from pathlib import Path
import argparse
import tarfile


from ane_drive_perc.utils import jsonl_read


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/bdd"))
    p.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    p.add_argument("--out-dir", type=Path, default=Path("data/hf/bdd100k-det"))
    p.add_argument(
        "--names",
        nargs="+",
        default=[
            "train_full",
            "val_full",
            "test_images",
            "train_003k_seed42",
            "train_005k_seed42",
            "train_010k_seed42",
        ],
    )
    p.add_argument("--samples-per-shard", type=int, default=1000)
    return p.parse_args()


def add_json(tar: tarfile.TarFile, name: str, row: dict):
    data = json.dumps(row, separators=(",", ":")).encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def shard_rows(
    *,
    root: Path,
    rows: list[dict],
    out_dir: Path,
    name: str,
    samples_per_shard: int,
) -> list[dict]:
    split_dir = out_dir / "shards" / name
    split_dir.mkdir(parents=True, exist_ok=True)

    shard_records: list[dict] = []

    for shard_id, start in enumerate(range(0, len(rows), samples_per_shard)):
        chunk = rows[start : start + samples_per_shard]
        shard_name = f"{name}-{shard_id:06d}.tar"
        shard_path = split_dir / shard_name

        with tarfile.open(shard_path, "w") as tar:
            for row in chunk:
                sample_id = row["id"]
                image_path = root / row["image"]

                tar.add(image_path, arcname=f"{sample_id}.jpg")
                add_json(tar, f"{sample_id}.json", row)

        shard_records.append(
            {
                "name": shard_name,
                "path": f"shards/{name}/{shard_name}",
                "samples": len(chunk),
            }
        )

        print(f"[shard] {name}: {shard_name} ({len(chunk)} samples)")

    return shard_records


def copy_manifest(manifest_dir: Path, out_dir: Path, name: str):
    dst_dir = out_dir / "manifests"
    dst_dir.mkdir(parents=True, exist_ok=True)

    src = manifest_dir / f"{name}.jsonl"
    dst = dst_dir / f"{name}.jsonl"

    dst.write_bytes(src.read_bytes())


def main():
    args = parse_args()

    index = {}

    for name in args.names:
        manifest_path = args.manifest_dir / f"{name}.jsonl"
        rows = jsonl_read(manifest_path)

        copy_manifest(args.manifest_dir, args.out_dir, name)

        shards = shard_rows(
            root=args.root,
            rows=rows,
            out_dir=args.out_dir,
            name=name,
            samples_per_shard=args.samples_per_shard,
        )

        index[name] = {
            "manifest": f"manifests/{name}.jsonl",
            "num_samples": len(rows),
            "num_shards": len(shards),
            "shards": shards,
        }

    index_path = args.out_dir / "shard_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"[done] wrote hf dataset files to {args.out_dir}")
    print(f"[done] wrote shard index to {index_path}")


if __name__ == "__main__":
    main()

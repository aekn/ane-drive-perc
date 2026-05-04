import argparse
import json

from pathlib import Path
import shutil
import tarfile
from huggingface_hub import HfApi, hf_hub_download

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="aekn/ane-drive-perc-bdd100k")
    p.add_argument("--split", required=True)
    p.add_argument("--out-dir", default="data/materialized/bdd100k-det")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--revision", default="main")
    p.add_argument("--token", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def list_split_shards(repo_id: str, split: str, revision: str, token: str | None):
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)

    prefix = f"{split}-"
    shards = [
        name
        for name in files
        if Path(name).name.startswith(prefix) and name.endswith(".tar")
    ]
    return sorted(shards)


def materialize_tar(tar_path: Path, split_dir: Path, overwrite: bool):
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    images = 0
    labels = 0

    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue

            name = Path(member.name).name
            suffix = Path(name).suffix.lower()

            if suffix in IMAGE_EXTENSIONS:
                dst = image_dir / name
                images += 1
            elif suffix == ".json":
                dst = label_dir / name
                labels += 1
            else:
                continue

            if dst.exists() and not overwrite:
                continue

            src = tar.extractfile(member)
            if src is None:
                continue

            tmp = dst.with_suffix(dst.suffix + ".tmp")
            with src, tmp.open("wb") as f:
                shutil.copyfileobj(src, f)

            tmp.replace(dst)
        return images, labels


def main():
    args = parse_args()

    split_dir = Path(args.out_dir) / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    shard_names = list_split_shards(
        repo_id=args.repo_id,
        split=args.split,
        revision=args.revision,
        token=args.token,
    )

    if not shard_names:
        raise SystemExit(
            f"No .tar shards found for split {args.split!r} in {args.repo_id!r}"
        )

    nimg = 0
    nlbl = 0

    for shard_name in shard_names:
        local_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=shard_name,
                revision=args.revision,
                cache_dir=args.cache_dir,
                token=args.token,
            )
        )

        images, labels = materialize_tar(
            tar_path=local_path,
            split_dir=split_dir,
            overwrite=args.overwrite,
        )

        nimg += images
        nlbl += labels

        print(f"[materialize] {shard_name}: images={images} labels={labels}")

    summary = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "split": args.split,
        "num_shards": len(shard_names),
        "images": nimg,
        "labels": nlbl,
        "layout": {
            "images": str(split_dir / "images"),
            "labels": str(split_dir / "labels"),
        },
        "shards": shard_names,
    }

    summary_path = split_dir / "_materialized.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"[done] split={args.split}")
    print(f"       images={nimg}")
    print(f"       labels={nlbl}")
    print(f"       out={split_dir}")


if __name__ == "__main__":
    main()

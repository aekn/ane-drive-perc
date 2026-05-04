import argparse
from pathlib import Path, PurePosixPath
import shutil
import zipfile


IMAGE_PREFIX = PurePosixPath("bdd100k/images/100k")
LABEL_PREFIX = PurePosixPath("bdd100k/labels/100k")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out-dir", type=Path, default=Path("data/bdd"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def target_path(member_name: str, out_dir: Path):
    src = PurePosixPath(member_name)

    if src.name == "":
        return None

    if src.is_relative_to(IMAGE_PREFIX):
        rel = src.relative_to(IMAGE_PREFIX)
        return out_dir / "100k" / "images" / Path(*rel.parts)

    if src.is_relative_to(LABEL_PREFIX):
        rel = src.relative_to(LABEL_PREFIX)
        return out_dir / "100k" / "labels" / Path(*rel.parts)

    return None


def extract_selected(zip_path: Path, out_dir: Path, overwrite: bool):
    count = 0

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            dst = target_path(info.filename, out_dir)

            if info.is_dir():
                continue

            if dst is None:
                continue

            if dst.exists() and not overwrite:
                count += 1
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(info) as src, dst.open("wb") as f:
                shutil.copyfileobj(src, f, length=1024 * 1024)

            count += 1
    return count


def main():
    args = parse_args()

    image_zip = args.raw_dir / "bdd100k_images.zip"
    label_zip = args.raw_dir / "bdd100k_labels.zip"

    n_img = extract_selected(image_zip, args.out_dir, args.overwrite)
    n_lbl = extract_selected(label_zip, args.out_dir, args.overwrite)

    print(f"[images] {n_img} files")
    print(f"[labels] {n_lbl} files")
    print(f"[done] wrote BDD100K to {args.out_dir}")


if __name__ == "__main__":
    main()

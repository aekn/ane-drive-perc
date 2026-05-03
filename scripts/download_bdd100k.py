import argparse
import subprocess
import sys
from pathlib import Path

DATASET = "marquis03/bdd100k"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()

    dst = Path(args.data_dir)
    dst.mkdir(parents=True, exist_ok=True)

    ret = subprocess.run(
        f"kaggle datasets download -d {DATASET} -p {dst} --unzip",
        shell=True,
    )
    if ret.returncode != 0:
        sys.exit("kaggle download failed")

    bdd = dst / "bdd100k"
    checks = {
        "images/train": ("dir", 70_000),
        "images/val": ("dir", 10_000),
        "images/test": ("dir", 20_000),
        "labels/bdd100k_labels_images_train.json": ("file", None),
        "labels/bdd100k_labels_images_val.json": ("file", None),
    }

    print()
    ok = True
    for rel, (kind, expected_n) in checks.items():
        p = bdd / rel
        if kind == "dir":
            n = len(list(p.glob("*.jpg"))) if p.is_dir() else -1
            flag = "ok" if p.is_dir() and n >= expected_n * 0.99 else "!!"
            print(f"  [{flag}]  {rel}  ({n:,} jpg)")
            if flag == "!!":
                ok = False
        else:
            flag = "ok" if p.is_file() else "!!"
            print(f"  [{flag}]  {rel}")
            if flag == "!!":
                ok = False

    if not ok:
        sys.exit("\nOne or more checks failed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import webdataset.compat as wds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("shard", type=Path)
    p.add_argument("--limit", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset = wds.WebDataset(str(args.shard), shardshuffle=False)

    for index, sample in enumerate(dataset):
        print(f"sample {index}")
        print("  key:", sample.get("__key__"))
        print("  fields:", sorted(k for k in sample.keys() if not k.startswith("__")))

        if index + 1 >= args.limit:
            break


if __name__ == "__main__":
    main()

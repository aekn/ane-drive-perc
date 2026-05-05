import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from webdataset.compat import WebDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("shard", type=Path)
    p.add_argument("--limit", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = WebDataset(str(args.shard), shardshuffle=False)
    samples = cast(Iterable[dict[str, Any]], pipeline)

    for index, sample in enumerate(samples):
        print(f"sample {index}")
        print("  key:", sample.get("__key__"))
        print("  fields:", sorted(k for k in sample.keys() if not k.startswith("__")))

        if index + 1 >= args.limit:
            break

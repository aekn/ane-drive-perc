"""Build a manifest containing every image in BDD100K val.

Usage:
    uv run python scripts/make_val_manifest.py
"""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LABEL_JSON = REPO / "data/raw/bdd100k/labels/bdd100k_labels_images_val.json"
OUT = REPO / "data/manifests/bdd_val_full.json"


def main() -> None:
    with open(LABEL_JSON) as f:
        raw = json.load(f)
    ids = sorted(r["name"] for r in raw)
    payload = {
        "name": "bdd_val_full",
        "version": "v1",
        "source": "BDD100K val split (full)",
        "source_path": str(LABEL_JSON.relative_to(REPO)),
        "seed": None,
        "size": len(ids),
        "stratification": None,
        "parent_manifest": None,
        "stratum_counts": None,
        "image_ids": ids,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {OUT.relative_to(REPO)}  size={len(ids)}")


if __name__ == "__main__":
    main()

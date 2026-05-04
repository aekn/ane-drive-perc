"""Build BDD-Edge-{3K,5K,10K} stratified subset manifests."""

from __future__ import annotations

import json
from pathlib import Path

from fkde.data.labels import filter_existing_images, load_records
from fkde.data.subset import nested_subset, stratified_subset

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS_JSON = REPO_ROOT / "data/raw/bdd100k/labels/bdd100k_labels_images_train.json"
IMAGES_DIR = REPO_ROOT / "data/raw/bdd100k/images/train"
MANIFEST_DIR = REPO_ROOT / "data/manifests"

SEED = 0
AXES = ("weather", "timeofday")
SIZES = (10_000, 5_000, 3_000)
VERSION = "v1"


def _manifest(
    name: str,
    size: int,
    image_ids: list[str],
    stratum_counts: dict[str, int],
    parent: str | None,
) -> dict:
    return {
        "name": name,
        "version": VERSION,
        "source": "BDD100K train split",
        "source_path": str(LABELS_JSON.relative_to(REPO_ROOT)),
        "seed": SEED,
        "size": size,
        "stratification": {
            "axes": list(AXES),
            "rule": "proportional with rare-stratum take-all + deficit redistribution",
        },
        "parent_manifest": parent,
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "image_ids": image_ids,
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def main() -> None:
    print(f"Loading labels from {LABELS_JSON.relative_to(REPO_ROOT)} ...")
    records = load_records(LABELS_JSON)
    print(f"  raw label rows: {len(records)}")
    records = filter_existing_images(records, IMAGES_DIR)
    print(f"  with image present: {len(records)}")

    # 10K set
    print(f"\nSampling BDD-Edge-10K (seed={SEED}, axes={AXES}) ...")
    ids_10k, counts_10k = stratified_subset(records, 10_000, axes=AXES, seed=SEED)
    print(f"  picked {len(ids_10k)} ids; strata used: {len(counts_10k)}")
    payload_10k = _manifest("bdd_edge_10k", 10_000, ids_10k, counts_10k, parent=None)
    out_10k = MANIFEST_DIR / "bdd_edge_10k_v1.json"
    _write(out_10k, payload_10k)
    print(f"  wrote {out_10k.relative_to(REPO_ROOT)}")

    # 5K subset
    print("\nSampling BDD-Edge-5K nested in 10K ...")
    ids_5k, counts_5k = nested_subset(ids_10k, records, 5_000, axes=AXES, seed=SEED)
    payload_5k = _manifest(
        "bdd_edge_5k", 5_000, ids_5k, counts_5k, parent="bdd_edge_10k_v1"
    )
    out_5k = MANIFEST_DIR / "bdd_edge_5k_v1.json"
    _write(out_5k, payload_5k)
    print(f"  wrote {out_5k.relative_to(REPO_ROOT)}")

    # 3K subset
    print("\nSampling BDD-Edge-3K nested in 5K ...")
    ids_3k, counts_3k = nested_subset(ids_5k, records, 3_000, axes=AXES, seed=SEED)
    payload_3k = _manifest(
        "bdd_edge_3k", 3_000, ids_3k, counts_3k, parent="bdd_edge_5k_v1"
    )
    out_3k = MANIFEST_DIR / "bdd_edge_3k_v1.json"
    _write(out_3k, payload_3k)
    print(f"  wrote {out_3k.relative_to(REPO_ROOT)}")

    print("\nStratum counts (10K):")
    for k, v in counts_10k.items():
        print(f"  {k:35s} {v:5d}")

    s3, s5, s10 = set(ids_3k), set(ids_5k), set(ids_10k)
    assert s3.issubset(s5), "3K not a subset of 5K"
    assert s5.issubset(s10), "5K not a subset of 10K"
    print("\n3K subset 5K subset 10K")


if __name__ == "__main__":
    main()

import argparse

from ane_drive_perc.data.shards import list_matching_shards


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--pattern", required=True)
    p.add_argument("--repo-type", default="dataset")
    p.add_argument("--revision", default="main")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    shards = list_matching_shards(
        repo_id=args.repo_id,
        pattern=args.pattern,
        repo_type=args.repo_type,
        revision=args.revision,
    )

    print(f"Found {len(shards)} matching shards.")

    for shard in shards[:20]:
        print(f"  {shard}")

    if len(shards) > 20:
        print(f"  ... {len(shards) - 20} more")

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

from .labels import ImageRecord


def _deterministic_seed(global_seed: int, key: tuple[str, ...]) -> int:
    payload = f"{global_seed}|" + "|".join(key)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


StratumKey = tuple[str, ...]


def _stratum_key(record: ImageRecord, axes: tuple[str, ...]) -> StratumKey:
    return tuple(getattr(record, axis) for axis in axes)


def _allocate(group_sizes: dict[StratumKey, int], n: int) -> dict[StratumKey, int]:
    total = sum(group_sizes.values())
    if total < n:
        raise ValueError(f"requested n={n} exceeds total available {total}")

    targets: dict[StratumKey, int] = {}
    raw: dict[StratumKey, float] = {}
    for k, size in group_sizes.items():
        raw[k] = n * size / total
        targets[k] = round(raw[k])

    pool = 0
    flexible: dict[StratumKey, int] = {}
    for k, size in group_sizes.items():
        if targets[k] > size:
            pool += targets[k] - size
            targets[k] = size
        else:
            flexible[k] = size

    if pool > 0 and flexible:
        flex_total = sum(flexible.values())
        flex_keys = sorted(flexible.keys(), key=lambda k: (-flexible[k], k))
        for k in flex_keys:
            if pool <= 0:
                break
            add = min(
                pool, flexible[k] - targets[k], round(pool * flexible[k] / flex_total)
            )
            add = max(add, 0)
            targets[k] += add
            pool -= add
        i = 0
        while pool > 0:
            k = flex_keys[i % len(flex_keys)]
            if targets[k] < group_sizes[k]:
                targets[k] += 1
                pool -= 1
            i += 1
            if i > 10 * len(flex_keys):
                raise RuntimeError("redistribution failed to converge")

    diff = sum(targets.values()) - n
    if diff != 0:
        ordered = sorted(targets.keys(), key=lambda k: (-group_sizes[k], k))
        i = 0
        while diff != 0:
            k = ordered[i % len(ordered)]
            if diff > 0 and targets[k] > 0:
                targets[k] -= 1
                diff -= 1
            elif diff < 0 and targets[k] < group_sizes[k]:
                targets[k] += 1
                diff += 1
            i += 1
            if i > 100 * len(ordered):
                raise RuntimeError("off-by-one fix failed to converge")

    return targets


def stratified_subset(
    records: list[ImageRecord],
    n: int,
    axes: tuple[str, ...] = ("weather", "timeofday"),
    seed: int = 0,
) -> tuple[list[str], dict[str, int]]:
    groups: dict[StratumKey, list[str]] = defaultdict(list)
    for r in records:
        groups[_stratum_key(r, axes)].append(r.name)

    for k in groups:
        groups[k].sort()

    sizes = {k: len(v) for k, v in groups.items()}
    targets = _allocate(sizes, n)

    picked: list[str] = []
    counts: dict[str, int] = {}
    for k in sorted(groups.keys()):
        pool = groups[k]
        rng = random.Random(_deterministic_seed(seed, k))
        shuffled = pool[:]
        rng.shuffle(shuffled)
        chosen = sorted(shuffled[: targets[k]])
        picked.extend(chosen)
        counts["/".join(k)] = len(chosen)

    if sum(counts.values()) != n:
        raise RuntimeError(f"target sum {sum(counts.values())} != requested n={n}")

    picked.sort()
    return picked, counts


def nested_subset(
    parent_ids: list[str],
    records: list[ImageRecord],
    n: int,
    axes: tuple[str, ...] = ("weather", "timeofday"),
    seed: int = 0,
) -> tuple[list[str], dict[str, int]]:
    parent = set(parent_ids)
    sub_records = [r for r in records if r.name in parent]
    return stratified_subset(sub_records, n, axes=axes, seed=seed)

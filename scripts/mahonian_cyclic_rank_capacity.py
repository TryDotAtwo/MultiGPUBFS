#!/usr/bin/env python3
"""Exact layer/owner counts and ideal 2-bank bytes for distinct Coxeter S_n.

Assumptions: identity root, all adjacent swaps, lexicographic factoradic ID,
cyclic logical owner = ID % world, full-domain visited bitmap, no remapping.
The full GPU memory requirement is larger; this estimates live payload only.
"""
import argparse
import json
import math


def layer_owner_counts(n, world):
    if not 2 <= n <= 20:
        raise ValueError("n must be 2..20 for a u64 factorial domain")
    if not 1 <= world <= 64:
        raise ValueError("world must be 1..64")
    # A Lehmer digit with k remaining symbols is in 0..k-1.  Its value
    # contributes that many inversions and digit*(k-1)! to the lexical ID.
    counts = {(0, 0): 1}
    for k in range(n, 0, -1):
        weight = math.factorial(k - 1) % world
        following = {}
        for (depth, owner), count in counts.items():
            for digit in range(k):
                key = (depth + digit, (owner + digit * weight) % world)
                following[key] = following.get(key, 0) + count
        counts = following
    return [[counts.get((depth, owner), 0) for owner in range(world)]
            for depth in range(n * (n - 1) // 2 + 1)]


def capacity_estimate(n, world):
    counts = layer_owner_counts(n, world)
    total = math.factorial(n)
    q, remainder = divmod(total, world)
    owners = []
    for owner in range(world):
        length = q + (owner < remainder)
        bitmap_bytes = ((length + 63) // 64) * 8
        peak_rows, peak_depth = max(
            (counts[depth - 1][owner] + counts[depth][owner], depth)
            for depth in range(1, len(counts))
        )
        peak_one_layer_rows, peak_one_layer_depth = max(
            (counts[depth][owner], depth) for depth in range(len(counts))
        )
        owners.append({
            "logical_owner": owner,
            "domain_ids": length,
            "bitmap_bytes": bitmap_bytes,
            "peak_live_rows": peak_rows,
            "peak_depth": peak_depth,
            "peak_one_layer_rows": peak_one_layer_rows,
            "peak_one_layer_depth": peak_one_layer_depth,
            "unpacked_min_bytes": bitmap_bytes + n * peak_rows,
            "packed_min_bytes": bitmap_bytes + 8 * peak_rows if n <= 16 else None,
        })
    return {
        "n": n,
        "world": world,
        "states": total,
        "assumptions": "identity root; all adjacent swaps; exact lexical Lehmer ID; cyclic ID%world; dense visited; two live layers",
        "packed_codec_supported_by_experiment": n <= 16,
        "excludes": "segment rounding, reserved unused rows, queues, transient copies, archive, CUDA/NCCL, allocator and other overhead",
        "by_logical_owner": owners,
        "max_packed_min_bytes": max(x["packed_min_bytes"] for x in owners) if n <= 16 else None,
        "max_unpacked_min_bytes": max(x["unpacked_min_bytes"] for x in owners),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", nargs="+", type=int)
    parser.add_argument("--world", type=int, default=8)
    args = parser.parse_args()
    for n in args.n:
        print(json.dumps(capacity_estimate(n, args.world), sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ideal storage bounds for Coxeter S_n with a dense visited bitmap.

This is arithmetic on independent Mahonian layer counts, not a GPU test or
an admission estimate. Rows are assumed to store each symbol in one byte or
a whole state in one u64. It does not include queues, allocator granularity,
archive snapshots, communication buffers, or per-rank imbalance.
"""
import argparse
import json
import math

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("n", type=int, nargs="+", help="Coxeter S_n values, 2..20")
args = parser.parse_args()
for n in args.n:
    if not 2 <= n <= 20:
        parser.error("n must be between 2 and 20")
    layers = [1]
    for k in range(1, n + 1):
        out = [0] * (len(layers) + k - 1)
        for i, count in enumerate(layers):
            for j in range(k):
                out[i + j] += count
        layers = out
    assert sum(layers) == math.factorial(n)
    live2 = max(a + b for a, b in zip(layers, layers[1:]))
    live3 = max(sum(layers[i:i+3]) for i in range(len(layers)))
    bitmap = (sum(layers) + 7) // 8
    result = {
        "graph": f"S_{n}, full adjacent-transposition Coxeter generating set",
        "states": sum(layers),
        "layers": len(layers),
        "largest_layer": max(layers),
        "max_two_adjacent_layers": live2,
        "max_three_adjacent_layers": live3,
        "dense_visited_bitmap_bytes": bitmap,
        "bitmap_plus_two_frontiers_bytes_unpacked": bitmap + live2 * n,
        "bitmap_plus_two_frontiers_bytes_u64": bitmap + live2 * 8 if n <= 16 else None,
        # These lower bounds apply only to the implementation's three-bank
        # inverse-closed segmented hash layouts with uncompressed u8 rows.
        "segmented_sorted_hash_3bank_state_and_index_bytes_lower_bound": live3 * (n + 12),
        "segmented_open_hash_3bank_state_and_index_bytes_lower_bound": live3 * (n + 16 + 8),
        "segmented_open_compact_hash_3bank_state_and_index_bytes_lower_bound": live3 * (n + 16 + 4),
        "scope": "ideal global storage only; no real-rank capacity or GPU acceptance",
    }
    print(json.dumps(result, sort_keys=True))

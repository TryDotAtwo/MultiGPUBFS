#!/usr/bin/env python3
"""Memory illustration for bitmap BFS of adjacent-swap S_n, not admission.

For distinct symbols and adjacent transpositions, BFS distance is inversion
count. A Lehmer code contributes digit d at factorial place k! and adds d
inversions. Dynamic programming computes its exact distribution by distance
and by the bitmap owner's `rank % world` without enumerating n! states.
"""
import argparse
import json
import math


def layers_by_owner(n, world):
    if type(n) is not int or not 1 <= n <= 20:
        raise ValueError('degree must be in 1..20 for the current u64 rank')
    if type(world) is not int or not 1 <= world <= 128:
        raise ValueError('GPU count must be in 1..128')
    layers = [[1] + [0] * (world - 1)]
    factorial = 1
    for place in range(n):
        target = [[0] * world for _ in range(len(layers) + place)]
        for depth, owners in enumerate(layers):
            for digit in range(place + 1):
                target_row = target[depth + digit]
                offset = digit * factorial
                for owner, count in enumerate(owners):
                    target_row[(owner + offset) % world] += count
        layers = target
        factorial *= place + 1
    return layers


def estimate(n, world, segment_rows):
    if type(segment_rows) is not int or not 1 <= segment_rows <= 1 << 32:
        raise ValueError('segment_rows must be in 1..2**32')
    layers = layers_by_owner(n, world)
    total = math.factorial(n)
    if sum(map(sum, layers)) != total:
        raise AssertionError('rank/distance distribution did not sum to n!')
    peak, depth, owner = max((count, d, o) for d, row in enumerate(layers)
                             for o, count in enumerate(row))
    live, live_depth, live_owner = max(
        (sum(layers[j][o] for j in range(max(0, d - 1), min(len(layers), d + 2))), d, o)
        for d in range(len(layers)) for o in range(world))
    # Cardinality differs by at most one across owners. The runtime uses
    # 64-bit words and allocates at least one word even for an empty owner.
    owners = [total // world + int(o < total % world) for o in range(world)]
    bitmaps = [max(8, ((count + 63) // 64) * 8) for count in owners]
    bitmap_bytes = max(bitmaps)
    segment_bytes = 3 * segment_rows * n + 24
    owner_peaks = [max(row[o] for row in layers) for o in range(world)]
    owner_live = [max(sum(layers[j][o] for j in range(max(0, d - 1), min(len(layers), d + 2)))
                      for d in range(len(layers))) for o in range(world)]
    segment_counts = [(v + segment_rows - 1) // segment_rows for v in owner_peaks]
    segment_count = max(segment_counts)
    logical_bytes = max(value * n + bitmaps[o] for o, value in enumerate(owner_live))
    segment_allocation_bytes = max(count * segment_bytes + bitmaps[o]
                                   for o, count in enumerate(segment_counts))
    # Hash owners differ from exact-rank modulo owners.  A perfectly even
    # split is an optimistic illustration, not a measured imbalance bound.
    peak_global_layer = max(map(sum, layers))
    balanced_peak = (peak_global_layer + world - 1) // world
    hash_segments = (balanced_peak + segment_rows - 1) // segment_rows
    hash_slots = 1 << ((6 * segment_rows - 1).bit_length())
    hash_segment_bytes = hash_slots * 8 + 3 * segment_rows * (n + 8) + 24
    return {
        'scope': 'adjacent-transposition S_n; distinct one-byte symbols; exact rank modulo GPU count',
        'degree': n, 'gpus': world, 'vertices': total, 'layers': len(layers),
        'peak_owner_layer_rows': peak, 'peak_depth': depth, 'peak_logical_owner': owner,
        'peak_owner_three_consecutive_layer_rows': live,
        'three_layer_center_depth': live_depth, 'three_layer_logical_owner': live_owner,
        'bitmap_max_per_gpu_bytes': bitmap_bytes,
        'logical_three_layers_plus_bitmap_max_per_gpu_bytes': logical_bytes,
        'hypothetical_rows_per_segment_bank': segment_rows,
        'ideal_segment_count_for_peak_owner': segment_count,
        'ideal_segment_banks_plus_bitmap_per_gpu_bytes': segment_allocation_bytes,
        'hash_assumption': 'perfectly balanced hash ownership; three-bank inverse-closed history; no shard fragmentation',
        'hash_balanced_ideal_peak_rows_per_gpu': balanced_peak,
        'hash_balanced_ideal_segments_per_gpu': hash_segments,
        'hash_balanced_ideal_segment_bytes_per_gpu': hash_segments * hash_segment_bytes,
        'excluded': ['shard fragmentation', 'queues', 'NCCL', 'CUDA allocations', 'reserve', 'allocator overhead'],
        'warning': 'Illustration only; actual BFS capacity and configuration must be checked on the target GPUs.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--degree', type=int, required=True)
    p.add_argument('--gpus', type=int, required=True)
    p.add_argument('--segment-rows', type=int, default=1 << 20,
                   help='hypothetical allocated rows/bank, not the 2**32 configurable maximum')
    args = p.parse_args()
    print(json.dumps(estimate(args.degree, args.gpus, args.segment_rows), indent=2))


if __name__ == '__main__':
    main()

# Experimental resident bitmap: independent receiver warps

This is a source checkpoint, not a release or a recommendation to use the
experimental engine for production. The default host-driven backend is unchanged.
The earlier public API and topology acceptance is described in
[resident-public-pack4](2026-11-22-resident-public-pack4.md).

The resident CTA now has four independent receiver warps, one per producer
lane. Each SPSC ring still has exactly one consumer. Receivers use a
nonblocking per-shard lease around exact bitmap acceptance; a busy shard is
retried while a receiver can service other source rings. Different shards
can use their preallocated exclusive extents concurrently. The reported
active-shard count records shards that accepted rows during a layer; it does
not prove that two were active at the same instant.

## Physical correctness

On 2 and 4 Tesla V100 SXM2 32 GB (SM70), CUDA 12.8.93, NCCL 2.30.7:

- CTest device-pack4-accept, device-pack4-library-bfs, and
  device-pack4-transport: passed.
- Persistent run_graph on 2 and 4 GPUs: complete **multisets** for every
  completed layer of S4, directed C6, and a repeated-symbol graph matched an
  independent CPU BFS. Both shards accepted rows. Tested an exactly sufficient
  rank cap, a cap one below it, and a duplicate at full capacity.
- Actual CayleyPy persistent API on 4 GPUs: layer sizes and the full last
  layer matched CPU BFS for S4, directed C6, repeated symbols and LRX3,
  including the resource stop and duplicate-at-full cases.
- On 2 GPUs a worker was killed during BFS. The group returned an error,
  invalidated the session and a new generation completed S4. Teardown took
  10.663 s.
- No physical 8-GPU resident run is claimed: this machine does not provide
  native peer atomics between all eight devices.

The exact-state layer gate is on small graphs. In the S9 timing below, all
37 layer sizes and the final reversed permutation were checked against an
independent Mahonian calculation, not the full set in every layer.

## Comparative observation

Coxeter adjacent-swap S9, 362,880 states and 2,903,040 generated edges,
on the same four V100s, two shards, atomic bitmap, packed frontier and no
autotuning. The selected limit was 65,536 rows per segment; actual admission
used 21 extents of 4,567 rows per rank for the resident path and two for the
host path. Times are BFS seconds, excluding setup and export.

| Engine | BFS seconds | Observations |
|---|---:|---|
| Resident, one receiver warp | 29.581, 29.011 | Same device transport |
| Resident, four receiver warps | 24.680, 24.752 | Same device transport |
| Existing host-driven bitmap | 0.440 | One observation; host/NCCL path |

The two resident orders were old→new and new→old; this is not a statistically
strong benchmark. The resident path remains about 56 times slower than the
host path on this graph, and both its performance on large layers and its
capacity on eight GPUs remain unverified. The runs do not have the same
allocated segment count, so these times do not isolate a pure transport cost.

The reverse-order resident run reported 2,903,040 sent packets for either
implementation, one per generated child. Its blocked-send attempts summed
to 26,749,615 with one receiver and 1,345,791 with four. Those are polling
attempts rather than bytes or hardware link transactions. Each resident rank
reported 2,302,272 B segment storage, 11,344 B bitmap, 393,312 B existing
queues, 2,432 B mailbox and 2,903,040 B layer log; allocator and CUDA/NCCL
costs are extra. NVML sampled at 0.5 s showed at most 2,162 MiB per tested GPU
for the resident pair and 1,282 MiB for the host run. These are sampled
device totals and may miss short peaks; they are not exact allocation peaks.
Resident peak_live_rows and per-layer times were not measured.

Executable SHA-256:
09065a90031c8cc15250435d0f24716335d5312bf26b87c41cea32b17901b084.
One-receiver CUDA library:
7b94710bc928afb5adb4cb7c45963f1df5950453a48c77559d6bb93c9cd198f2.
Four-receiver CUDA library:
7b43d95896241a09335f1ac902c51a3090e39934d96645e96a2146d6fc8590e5.
Each run verifies the loaded library checksum. Evidence, complete results,
sampled memory and both libraries are preserved separately on D:.

## Reproduction

Use the development executable and matching libraries, with a mutually
compatible 4-GPU group. The opt-in helper
tests/device_resident_s7_ab_gpu.py keeps S7 and the original alternating
order by default. Set RESIDENT_AB_N=9,
RESIDENT_AB_SEGMENT_ROWS=65536, and RESIDENT_AB_ORDER=old,new or
new,old for the resident S9 pair; RESIDENT_AB_ORDER=host measures the
existing host path. Set RESIDENT_OLD_LIB, RESIDENT_NEW_LIB,
RESIDENT_OLD_SHA, RESIDENT_NEW_SHA, and RESIDENT_LD_TAIL to the
corresponding directories, checksums and CUDA/NCCL paths.

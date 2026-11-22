# Resident hash: monotone parent-extent cursor

The attached candidate patch adds MGBFS_RESIDENT_HASH_PARENT_CURSOR=1 as an
experimental opt-in for the GPU-resident hash path. Each producer visits monotonically increasing parent
ordinals and keeps its position in the immutable read bank during one layer.
The cursor is reset at the next layer; it does not allocate a separate buffer
or change the owner, equality, archive, or default production path. A compiler
may nevertheless place thread-local data in local memory; device-wide peak
memory was not measured here.

The candidate request is agreed between ranks before GPU work. Capability query 5
prevents a new executable from silently requesting the cursor from an older
CUDA library. Unset or 0 uses the former lookup. The option is recorded
in each rank receipt and scoped correctly across persistent jobs.

## Validation

Source parent 5115dfc41871100459fdd1a0bdc085a084fa22f5; executable
SHA-256 e2ea1048dceca073a1ee4287082bd8259710f6c6f34930129448be8170f5c557 and loaded CUDA library SHA-256
537e72406f9763ae0a3eb0dee4d366663fafd39628d97342da37486586e00d87. Four physical V100-SXM2-32GB GPUs, CUDA 12.8.93,
NCCL 2.30.7, and a team with directed native peer atomics for every pair.

- Four GPUs: all state multisets of every completed layer matched an
  independent CPU BFS for S4, repeated symbols, and signed/wide symbols,
  including tiny segments, caps, duplicate-at-full, and persistent reuse.
  The native results confirmed two active shards.
- Two GPUs: a separate forced-collision run matched every full layer multiset.
  Its all-zero hash maps to one shard; it does not establish simultaneous
  full-hash collision and multi-shard replay in one integrated case.
- The public CayleyPy persistent path ran the cursor 0 → 1 → 0 in the same
  workers; a cap stop and next healthy job were also tested. Bitmap S4,
  directed C6, and repeated symbols matched full CPU layer multisets on two
  GPUs because the common producer loop changed.
- A GPU cursor fixture checked empty extents, jumps, overflow rejection,
  invalid backward traversal, and the virtual 2^32 address boundary. The
  latter allocates metadata only and is not a physical 2^32-row test.
- On two GPUs, an old CUDA library rejected the explicitly requested option,
  retained its default path, and an old executable also completed on the new
  library with the option unset. A rank-kill test rejected the poisoned worker
  generation and a deliberately new generation completed a healthy job.
- Rust capability unit test and the cursor CTest passed. The initial full
  layer fixture failed because it did not enable its own existing test-only
  profiling gate; that fixture was corrected and the reported cases rerun.

The measured benefit was small and has not been separated from run-to-run
variation on a large workload. The candidate code is therefore saved as a
[reproducible patch](patches/2026-11-22-resident-hash-parent-cursor.patch)
rather than integrated into this branch. Patch SHA-256: bdbe83f712edb892082708acc2d7fb7587d27331e98f19c9a29fe3460b8462ef.
For the tests, the first fault-test shell missed PYTHONPATH; the compatibility
fixture also initially used a batch larger than its selected capacity. Both
were corrected before the successful results above; their failure logs are
preserved.

## Measurements

S9 and S10 are adjacent-transposition graphs. All Mahonian layer counts and
the final reversed permutation were checked; full intermediate state
multisets were checked only for the small graphs above. Each row uses the same
new executable and CUDA library, four GPUs, hash/atomic, indexed extents,
parallel shard owners, two shards, eight lanes, eight records per packet,
one slot per ring, requested 8192 rows/segment, and no archival layer dump.

S9 actually used 4567 rows/bank and
81 extents per rank, with 362,880 states
and 37 layers:

| Cursor | BFS seconds | Observed wall seconds |
|---|---:|---:|
| off | 13.870499 | 20.226166 |
| on | 13.338529 | 20.016379 |
| on | 13.535656 | 19.711896 |
| off | 13.811016 | 20.948435 |

The two-pair mean changed from 13.841 s to 13.437 s
(+2.92% lower using this sign convention). The old executable and
library, measured separately on the same node, took 13.927
s BFS; it is not the same-binary control.

S10 actually used 8192 rows/bank and
444 extents per rank, with 3,628,800 states
and 46 layers:

| Cursor | BFS seconds | Observed wall seconds |
|---|---:|---:|
| off | 364.327166 | 371.877252 |
| on | 360.617698 | 368.495826 |

The single off/on pair changed by +1.02% using the same convention; it
is one observation per case and does not establish a stable general speedup.
A one-shot nvidia-smi utilization query occurred during the off run to
inspect apparent liveness. There was no continuous NVML sampler and no
device-wide peak measurement.

For S10 the per-rank receipt lists 429,205,920 B of
segment storage, including 109,117,440 B state
and 320,077,824 B index; queue allocation is
442,464 B per rank. The separately reported
2,147,483,648 B reserve is an admission allowance, not proof of a
matching allocation. These figures do not include all CUDA/NCCL allocations.
The off run reported about 239–255 million blocked send attempts per rank
with about 1.02 million sent packets each. Counts indicate contention or
waiting, but do not alone establish the dominant bottleneck.

The comparison can be repeated from the patched source and matching build with
the executable and library paths set for that build:
```sh
export MGBFS_RESIDENT_HASH_SHARD_OWNERS=1 MGBFS_RESIDENT_HASH_INDEXED_EXTENTS=1
export MGBFS_RESIDENT_LANES=8 MGBFS_RESIDENT_PACKET_RECORDS=8
export RESIDENT_AB_WORLD=4 RESIDENT_AB_N=10 RESIDENT_AB_SEGMENT_ROWS=8192
export RESIDENT_AB_ORDER=hash-device
for mode in 0 1; do
  MGBFS_RESIDENT_HASH_PARENT_CURSOR=$mode python3 tests/device_resident_hash_ab_gpu.py
done
```

[Machine-readable comparison](artifacts/2026-11-22-resident-hash-parent-cursor.json)
contains individual times and actual rank receipts. Full logs and test evidence
are retained separately. This is not an 8-GPU, B200, S14, production release,
large-memory limit, or Explore result. Directed resident hash, matrices,
resident warp/batch, max_capacity, and autotuning remain unsupported.

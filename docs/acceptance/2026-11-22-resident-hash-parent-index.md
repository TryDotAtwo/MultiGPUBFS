# Experimental resident hash parent-extent index

Status: opt-in, not a new default. This experiment applies to segmented24
with execution_engine=device. The ordinary host-driven backend remains
unchanged. The resident path is still much slower than the existing host path
in the earlier S9 comparison; those timings had different geometries and do
not isolate transport. See [the mailbox report](2026-11-22-resident-mailbox-slots.md).

## Change

For MGBFS_RESIDENT_HASH_PARENT_INDEX=1, a GPU thread constructs a checked
64-bit prefix over the current bank's extent counts once per layer. After the
block synchronization, producers use an upper-bound search to find the parent
extent and row. Empty extents are supported. The default 0 keeps the previous
scan. The additional (extent_count + 1) * 8 device bytes are preallocated per
rank, included in admission accounting, checked against the NCCL reserve, and
reported in segment_memory_by_rank.

The hash ABI advertises this optional pointer; an older library explicitly
rejects a request for the index, while the unindexed default remains usable.
Rank agreement includes the choice before launching device work. This change
does not change state equality, owner routing, the archive, segment cap, or
the publication of completed layers. It is not a physical test of 2^32 rows.

## Correctness checks

The tested executable SHA-256 was
3c3c2a519ef141cdd398bc39a657761d923d1b4e626327bccde3ac5ed5d3f141
and the loaded CUDA library SHA-256 was
4ab1b2fd55fc9afc023d15ee390e2f97df1dd97955ab318bf79ed312673daac3.
Hardware: 4 x V100-SXM2-32GB (SM70), CUDA 12.8.93, NCCL 2.30.7.

* The CPU prefix test exhaustively checks 4,096 six-extent count patterns,
  empty extents and 64-bit address boundaries. The 2^32 checks are arithmetic,
  not physical VRAM allocation.
* On 2 and 4 physical GPUs, exact state multisets on every completed layer of
  S4, repeated-symbol and wide hash fixtures agree with the CPU BFS oracle,
  including limited capacity and duplicate-at-full. Forced hash collisions
  were also checked on 2 GPUs.
* On the same 4-GPU library, bitmap S4, directed C6 and repeated-symbol
  fixtures pass the full-layer oracle. Public CayleyPy persistent jobs switch
  hash index 0 -> 1 -> 0 without replacing workers; a resource stop is
  followed by a healthy job. An older CUDA library rejects index 1.
* A deliberately killed hash rank produced a group error, and a new explicit
  generation completed a healthy job on 2 GPUs.

## S9 A/B on four V100

Adjacent transpositions from identity; all 37 Mahonian layer counts and the
exact last state checked, 362,880 vertices. This larger test did not compare
the complete state multiset of every intermediate layer. One native binary,
8 shards, 8 producer lanes, 4 mailbox slots, atomic strategy, requested
segment rows 8,192 and batch 128. Actual allocation was 325 extents per rank,
1,142 rows per bank in each extent; the index used 2,608 extra bytes per rank.
BFS timings exclude setup. Runs were made in order off/on/on/off in a
persistent group; the first observed wall time includes cold startup.

| Index | BFS seconds | Observed wall seconds |
| --- | --- | --- |
| off | 5.660250905 | 11.277226597 |
| on | 4.596084090 | 4.949242251 |
| on | 4.627542652 | 4.996506866 |
| off | 5.699198916 | 6.043751723 |

The mean BFS time falls approximately 18.8% for this small four-GPU case.
There are two observations per mode and no sampled NVML high-water; the
receipt records allocated index bytes, not a measured peak VRAM. The
[compact comparison receipt](artifacts/2026-11-22-resident-hash-parent-s9.json)
includes the per-rank geometry and layer counts.


## S10 A/B on four V100

The same binary and geometry were used: 8 shards, 8 lanes, 4 mailbox slots,
8 records per packet, batch 128 and 450 extents of 8,192 rows per bank on
each rank. The index added 3,608 allocated bytes per rank. Each rank reported
435,006,000 bytes of segment storage, 442,752 queue bytes and a 2,147,483,648
byte reserve; these receipts are not total peak VRAM.

The order was on/off in one persistent group and off/on in a new group. Each
group's first observed wall time includes a cold worker. All 46 Mahonian
layer counts and the exact final state were checked, for 3,628,800 vertices;
the full intermediate state multisets were not saved or compared.

| Index | BFS seconds | Observed wall seconds |
| --- | --- | --- |
| on | 52.286577816 | 58.437124718 |
| off | 72.139636868 | 73.072957141 |
| off | 74.594540765 | 80.610526972 |
| on | 53.136195286 | 54.098826774 |

The two-run mean fell by 28.2% for this S10 configuration. Two pairs do not
establish a universal default or performance on 8 GPUs, B200, or a larger
graph. NVML peak VRAM was not sampled. The
[compact comparison receipt](artifacts/2026-11-22-resident-hash-parent-s10.json)
contains per-rank allocations and counters.

## Host-driven reference on S10

The same node, native binaries, graph, four GPUs, requested eight shards,
8,192 segment rows, batch 128, atomic strategy and compact result were used
for two host-driven hash jobs in a separate persistent group. BFS times were
3.657727380 and 3.614450538 seconds; observed wall times were 15.815208856
(cold startup) and 3.729117746 seconds. All layer counts and the final state
were checked against the same oracle. On this case the indexed device engine
is still about 14.5 times slower by mean BFS time.

The host and device engines have different actual allocation behavior, so this
comparison is an end-to-end engine reference, not an isolated measurement of
transport. At the time the host receipt was emitted it reported eight segments,
7,733,440 bytes of segment storage and 442,752 queue bytes per rank; the device
engine preallocated 450 extents and reported 435,006,000 segment-storage bytes.
The receipt is not a measured host-engine peak. Both engines reported the same
2,147,483,648 byte reserve. NVML peak VRAM was not measured. The
[host receipt](artifacts/2026-11-22-resident-hash-parent-s10-host.json)
contains the recorded per-rank fields.

## Reproduction

On a compatible four-GPU LSA topology, after building the matching CUDA
library and native executable, run the explicit benchmark with those artifact
paths and their matching NCCL runtime in the loader path:

```sh
export PYTHONPATH="$PWD"
export MGBFS_EXECUTABLE=/path/to/mgbfs
export MGBFS_CUDA_LIB_FILE=/path/to/libmgbfs_cuda.so
export LD_LIBRARY_PATH=/path/to/cuda-lib:/path/to/nccl/lib:/usr/local/cuda/lib64
export MGBFS_TEST_TMP_ROOT=/path/to/evidence
export MGBFS_RESIDENT_LANES=8 MGBFS_RESIDENT_PACKET_RECORDS=8 MGBFS_RESIDENT_MAILBOX_SLOTS=4
export MGBFS_RESIDENT_HASH_SHARD_OWNERS=1 MGBFS_RESIDENT_HASH_INDEXED_EXTENTS=1
export RESIDENT_AB_WORLD=4 RESIDENT_AB_SHARDS=8 RESIDENT_AB_N=10
export RESIDENT_AB_PARENT_ORDER=1,0
python3 tests/device_resident_hash_parent_bench_gpu.py
```

The benchmark is an explicit command and is not part of the short CPU CTest
or the normal package path. Check hardware capability before use. Exact
full-layer oracle scripts are separate from this timing run.

A second experimental receiver layout with one receiver per source was also
tested and not merged: on this node S9 with two producer lanes it was slower
for hash and bitmap than the existing per-lane receiver. Its code and raw
results were retained separately; this is not evidence about other GPUs.

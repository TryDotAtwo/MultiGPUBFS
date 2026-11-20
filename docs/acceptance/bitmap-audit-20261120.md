# Segmented and permutation-bitmap BFS audit, 2026-11-20

These results apply to the tested one-host RTX configurations and the exact
Coxeter adjacent-transposition graph. They are not acceptance on B200 or B300,
or a general throughput or memory-fit guarantee for arbitrary generators.
The bitset uses the entire multiset-permutation domain, even when the actual
orbit of a generator set is a much smaller subgroup.

The implementation stores one exact visited bit per owner-local multiset rank,
and stores current/previous/next state banks separately. Exact ranks are
computed on the GPU. With the one-byte state encoding used below, state banks
are 13–15 bytes per row. The configured segment row limit is an upper bound,
with default `2**32`; actual allocations can be smaller, and overflow uses
independent segments. None of these measurements allocated a full `2**32`
row segment on a GPU.

## Scope and runtime

Source before these changes: `745026e501b426c8188d05ba86468f8aacc9fd11`.
Measurements used native SM86 code with CUDA runtime 12.8, NCCL 2.30.7, no
concurrent GPU workload, and all completed layer populations compared to the
independently computed Mahonian inversion polynomial. Two-GPU machine: 2×RTX
3060 12 GiB, driver 580.173.02. Eight-GPU machine: 8×RTX 3090 24 GiB,
driver 580.178.04. Performance values are not claims of statistical confidence.

The tested normal CUDA library SHA-256 was
`fb05417f10ec9124edee2c9ad9376e4b69209fd8e9467db898e12b751140cd97`.
The final Rust executable SHA-256 for the two-GPU acceptance was
`eaa1ad9a9f70b67dc72a55ff50e694488826369e6ed2f243bb2c67655a21bf00`.
The large exchange-comparison runs used the preceding CLI
`8d3e0dded827a836fbbeea5d45f36414ec3942ba407fe16da1000e2b5429d431` with the
same CUDA library, before the new transport-mode consensus check. The final
CLI adding the advertised capability is
`ed3ee654a7235f40d4c35d54cae84a8ce69872414aec3f1e5bae932b792c120d`.

## Two RTX 3060: S12

With eight shards, eight lanes, 8192 source rows per batch, 1,048,576 rows per
segment bank, and exact-length exchange, one paired diagnostic measurement was
65.49 s for segmented hash and 71.36 s for bitmap. Instrumentation assigns CPU
wall intervals per rank, with asynchronous GPU work and waits included: in the
bitmap run the owner/accept interval was 45.1–56.1 s, route-status collective
6.7–17.6 s, and count/exchange interval 8.2–8.3 s. These intervals do not
separately measure kernel execution, NCCL transfer, or device idle time.
Hardware performance counters were unavailable (`ERR_NVGPUCTRPERM`).

The bitmap source-batch experiment (same graph, exact exchange) measured:

| Shards | Lanes | Batch | BFS seconds | Runs |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 8 | 8192 | 68.63, 73.01 | 2 |
| 4 | 4 | 8192 | 158.08 | 1 |
| 16 | 8 | 8192 | 62.97 | 1 |
| 8 | 8 | 16384 | 95.15 | 1 |
| 8 | 8 | 32768 | 179.72 | 1 |

Source batches, GPU count, hardware topology and layer sizes affect the
result. Do not extrapolate these shard/batch choices to other GPUs.

## Eight RTX 3090: S13

The paired comparison completed all 6,227,020,800 states and 79 layers
with both exchange modes: padded payload took 1757.28 and 1760.75 s of BFS; exact-length
payload took 594.15 and 595.57 s in two runs (approximately 2.95–2.96x faster). Each case had two source queue retries per
rank and matched every layer to the independent oracle. A rank allocated
1,635,779,520 bytes in segment state banks and 97,297,200 bitmap bytes. These
allocation counters are not total VRAM. A one-second `nvidia-smi` sampler saw a
maximum of 2206 MiB on a GPU in each case; this sampling can miss brief peaks.
Rank 0 counted 2,802,159,360 real remote records and 77,872,913,664 logical
padding records in the padded case. The count is not a measurement of physical
link bytes or link utilization. The complete retained last two layers matched
as unordered per-rank state sets in all four runs; one rank wrote the
same states in a different order, changing the file SHA. This is not a
comparison of all intermediate state sets. [Rank receipts and retained-layer digests](bitmap-audit-8gpu-s13-20261120.json) contain no raw state files.

A final matched backend comparison on the same executable, library, graph,
eight GPUs, shards, lanes, batch size, segment size and exact-length transport
completed in **393.32 s** for segmented hash and **593.55 s** for bitmap.
The bitmap time was 1.51x that of hash in this one paired run; this is not a
statistical or cross-GPU performance guarantee. All layers matched the
independent oracle, and the complete final two layers were equal as global
sets despite different owner mappings. The one-second GPU sampler recorded
a maximum of 5632 MiB per GPU for hash and 2206 MiB for bitmap, including
CUDA and communication allocations; sampling can miss brief peaks.
[Backend receipts and retained-layer digests](bitmap-audit-8gpu-backends-s13-20261120.json)
contain no raw state files.

The optional exact-length exchange is enabled with
`MGBFS_SEGMENT_EXACT_EXCHANGE=1`. It exchanges queue counts and then only valid
payloads; all ranks must select the same mode. The runtime checks agreement
before the first layer. For segmented autotuning with at least two ranks and
no explicit setting, a matching rebuilt native runtime advertises exact exchange;
the tuner then probes padded and exact exchanges on one matched geometry,
verifies per-rank receipts, and compares their common completed
prefix with other admitted shard/lane/batch candidates. Explicit `0`/`1`
settings remain fixed; a too-small or short workload retains the conservative
default without claiming a measured winner. Older native components remain on
padded exchange; an explicit exact request with an older runtime fails clearly.
Compatibility was exercised with the actual published dev38 executable and
SM86 library on eight GPUs, including measured tuning and S10 in both modes.
Publishing these source changes alone does not replace the previously released
native components. These bounded pilots cannot prove the fastest policy on
substantially larger layers. The final automatic selection was checked on
eight GPUs with fixed shards and both memory backends. A common queue ceiling is retained
after a recoverable source overflow so later layers do not repeatedly retry a
known-too-large batch. No production BFS is automatically restarted.

## Analytical capacity illustration, not GPU acceptance

For 8 equal exact-rank owners, one-byte state entries, three simultaneously
stored state banks, 1,048,576 rows per bank in each segment, and ideal use of
segments, enumerating the Mahonian layer populations yields:

| Graph | States | Largest bitmap owner layer | Bitmap plus state banks / GPU | Segmented hash, balanced ideal / GPU |
| --- | ---: | ---: | ---: | ---: |
| S13 | 6,227,020,800 | 37,379,751 | ~1.46 GiB | ~4.46 GiB |
| S14 | 87,178,291,200 | 469,860,208 | ~19.68 GiB | ~56.49 GiB |
| S15 | 1,307,674,368,000 | 6,371,939,916 | ~286.08 GiB | ~783.97 GiB |

For distinct-symbol S16 the bitmap alone needs about 304.47 GiB per GPU
when split over eight GPUs. That is already more than one 180 GB B200,
before storing frontier states; the current full-domain bitmap representation
therefore cannot fit that domain on eight such GPUs. For S15 the bitmap alone
is about 19.03 GiB per GPU, but that does not imply the active state banks fit.

The bitmap owner counts are exact for rank modulo eight. The hash numbers
assume a perfectly even split, inverse-closed three-bank history and no shard
fragmentation; they include each segment's hash slots and position array but
are not measured capacity. Both estimates exclude shard fragmentation,
transport buffers, NCCL, driver allocations, allocator behavior, and reserves. For S15 the bitmap
plus simultaneously live rows alone is ~285.26 GiB per owner. Consequently this
specific in-memory bitmap representation cannot fit S15 on 8×180 GB B200 under
these assumptions. The estimator can be reproduced without a GPU:

```sh
python scripts/estimate_coxeter_bitmap.py --degree 14 --gpus 8 --segment-rows 1048576
```

Its owner/layer distribution was checked against direct permutation enumeration
for `n <= 7`, including uneven owner counts. Different generator sets have different layer populations;
this is not an S14/S15 guarantee for arbitrary Cayley graphs. S14 has not been
run here on B200, so its complete execution is unproven. The legacy `graph-info`
transport estimate does not fully model segmented and bitmap storage admission;
actual runtime allocations and explicit diagnostics remain authoritative.

## Validation and known limitations

On two GPUs, the candidate passed exact bitmap and segmented BFS checks,
full-bank duplicates, recoverable queue skew, asymmetric real allocation
failure, rank-failure cleanup, fixed-shard measured autotuning, and inconsistent
transport-mode rejection. On eight GPUs the earlier candidate passed small
permutation, repeated-symbol, directed and wide-state bitmap graphs, segmented
matrix graphs, rank caps, persistent-worker reuse, and the full S10 Mahonian
layers. The latest transport-mode agreement check, fixed-shard measured tuner,
preflight caching and the large-graph exchange/backend comparisons also passed
on eight GPUs. Seventy-three focused Python tests passed, with one test deselected
because CayleyPy was unavailable in the host Python environment. The graph-info
planner still uses a legacy storage estimate for these new backends.
An eight-GPU repeated call on the same graph reused preflight in 0 s; changing
the graph or backend caused a new prefix rehearsal of about 7.8 s, which is
not a millisecond graph transition.

The rank kernel still performs per-child multiset ranking and the owner path
still performs atomic operations and synchronization. An alternative unique-
symbol rank kernel was evaluated on 2×3060; it did not demonstrate a reliable
S11/S12 improvement and is not enabled. The exact-length protocol also inserts
count exchange and synchronization, and may be worse for other configurations.

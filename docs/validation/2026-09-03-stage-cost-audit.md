# Native BFS stage cost and scheduling audit

Read-only implementation audit of measured source
`5482f3bb9d20db5780bf6b5c915c4d93c8cd321c`. No new performance experiment or
runtime modification is part of this audit. Raw evidence:
`test_results/native-runtime-v4/native-runtime/summary.json` and worker logs.
Verification command with `--require-async-archive` passed before this audit.

## Scope and measurements

Single T4, DENSE, generation variant 0, pre-dedup ON, batch 524280 parents,
256 buckets, 16 shards, 16 buckets per owner job. No native NCCL exchange or
BMMA in this measured path. U4(m) has **six generators**, not 24: 24 is the
number of rows in the stacked generator matrix. States have 16 bytes.

At m16: 16,777,216 states expanded, 100,663,296 candidate children; median
search 1.020810099 s, MAD 0.005238639 s. Baseline median 1.615000901 s.
Native sampled whole-device peak 2753 MiB, baseline 6283 MiB. Native requested
allocations 2,761,413,331 bytes; pinned RAM 2,701,090,560 bytes. Median durable
archive completion from search start 5.663834870 s. All 10 T4 validation runs
passed, including asynchronous archive fixtures. Full m8 state digests agree;
m20/m24 layer counts agree. m20/m24 are single trials, not medians.

No per-stage timing for this assembled revision or successful overlap timeline
exists. Stage percentages, achieved DRAM bandwidth and tensor utilization cannot
be inferred from overall BFS time. The earlier generation-only experiment is
different source/workload instrumentation, not the current full-pipeline profile.

## Operations: integer TOPS, not floating-point TFLOPS

Count multiply and add as two operations. For one candidate:

- General dense 4x4 generation: 2*4*4*4 = 128 useful integer operations.
- GEMM limb hash: 2*16*16 = 512 useful integer operations, excluding residue
  reconstruction/reduction. This counts the chosen limb representation, not a
  minimum-operation hash algorithm.
- Archive rehash: another 512 operations per accepted state archived.

At m16 these are 12.884902 billion generation ops, 51.539608 billion candidate
hash ops and 8.589935 billion archive hash ops. Dividing their sum by complete
search time gives 0.071526 useful TOPS. This is **not tensor-pipe utilization**:
it excludes padded MMA work, sorts, scans, comparisons, modular arithmetic and
all non-GEMM work.

NVIDIA lists 130 INT8 TOPS for T4 and approximately 320 GB/s memory bandwidth:
[Turing whitepaper](https://images.nvidia.com/aem-dam/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf).
These are theoretical compatible-instruction ceilings, not measured application
capacity. The resulting compute/bandwidth crossover is about 406 integer ops/B.

## Stage accounting

Traffic below is logical global-memory payload, not measured DRAM transactions:
caches, sector amplification, repeated loads, padding and library internals matter.

| Stage | Work and logical traffic | Specific implementation issue |
|---|---|---|
| Parent packing | 16 B state read, 64 B padded columns written per parent, then consumed by GEMM | Explicit transpose/padding pass for n=4 |
| Generation GEMM | [24,16] x [16,4P], 512 logical padded ops/child vs 128 useful; writes 64 B s32/child | Default CTA tile 64x32x64 mismatches both M and K |
| Modular output | Reads 64 B s32, writes 16 B child | Separate full pass; default scalar output reads products in a strided layout |
| Candidate hash | 512 useful ops; reads 16 B state, writes 64 B limb sums, reads those 64 B, writes 16 B Hash128 | 160 B/child before coefficient reads; separate finish_hash kernel |
| Radix sort/pre-dedup | 16 B key + 8 B ref per record per radix pass, plus flags, scan/select, gather/compact | Sort over all 128 bits; exact pass count is library-policy dependent. Pre-dedup OFF still copies 24 B/record from internal result to output (48 B read+write) |
| Bucket directory | Two binary searches per bucket, then 4096 B directory read to host | A single 256-thread CTA at B=256; small data but another launch/readiness boundary |
| Owner compare | Three merge traversals: incoming vs previous layer, current layer and accepted next-layer prefix | Repeats old-bucket traversal for each incoming batch, including already rejected incoming rows |
| Owner compact/commit | One CTA per bucket compacts with repeated block scans; merge accepted+survivors to scratch, then copy back | Rewrites old accepted prefix for each commit, even when no new rows survive |
| State materialization | selected/ref gathers, 16 B state read and 16 B dense state write per survivor, plus validation | Destination coalesced; source is hash-sorted indexed gather, not globally contiguous. Launch size based on full candidate capacity, not owner-job survivor count |
| FinalizeDepth | Copy all next-layer hashes from fixed buckets to compact arena: 32 B/survivor read+write | One CTA/bucket loops through its whole range; host drain remains |
| Archive | Rehash states, copy 32 B/state over PCIe; worker hashes frame bytes and writes/fsyncs | Independent stream/plan; hash and D2H serialize within that stream. Enqueues the full frontier before host begins advance |

Generation and hash s32 intermediates alone cause 256 B/child of logical
write+read traffic: **25.77 GB at m16**, **293.53 GB at m24**. This excludes
sort, owner work, states, archive and packing. It is not a measured DRAM count
or a promise that fusion can remove all of it without costs.

Useful hash arithmetic intensity is approximately 512/160 = 3.2 ops/B before
weight traffic, far below the T4 compute/bandwidth crossover. Merely adding more
MMA work does not imply improved throughput for this pipeline.

The default generator's mainloop uses a 64x32x64 tile for an actual n=4 problem.
For aligned large batches, useful dense work covers (24/64)*(4/64) = 2.34% of
its tile arithmetic; hash uses (16/32)*(16/64) = 12.5%. These are static tile
work fractions, **not measured occupancy/utilization**. CUTLASS's pipelined K
loop still performs the tile's warp MMA iterations on predicated-zero padding.
Confirm executed instruction counts before making instruction-rate claims.

For this particular U4 workload generators are identity plus/minus one elementary
off-diagonal entry. A specialized successor is a row add/subtract, not a dense
matrix multiply. A generic GEMM backend can remain valid, but must be compared
with that workload-specific incumbent before calling generation optimal.

## Shards and repeated work

`jobs::split` does not cross a shard boundary. With all buckets populated, one
parent batch produces 16 jobs. All jobs run serially on the same owner stream,
sharing one scratch/control/selected allocation, with a host stream synchronize
and control/extent readback after each job. They are not 16 independent lanes.

Each job launches 20 kernels: bind 1, compare 7, reserve 3, commit 4,
materialize 5. A full batch can therefore require 320 owner-side launches, before
producer/CUB kernels. Owner compact has only 16 CTAs in a full job. In contrast,
materialization validation/copy each launch 4096 CTAs for the fixed candidate
capacity 3,145,680, even when the job has far fewer rows or zero survivors.

No bucket histogram/counter history is exported, so skew is unmeasured. A uniform
hash does not fix serial scheduling or eliminate repeated scans. Merely increasing
shard count can increase launch/readback overhead rather than improve balance.

At m24 peak depth 18, previous/current/next sizes are
17,397,812 / 17,784,471 / 17,362,824. At least 34 parent batches are needed.
If each batch touches every bucket, the prev+curr merge scans alone read about
16*34*(17,397,812+17,784,471) = **19.14 GB** of logical old-key payload for
that depth, ignoring probes and incoming keys. With approximately uniform
survivor arrivals, repeated accepted-prefix compare + merge/copy-back adds
about 32*17,362,824*(34-1) = **18.34 GB** of old-prefix traffic.
The second value is a model, not a counter; actual within-depth survival order
and bucket coverage change it. Work scales with batch count times window size,
not only with newly generated candidates.

Increasing bucket count alone does not remove that total scan volume when every
batch still touches all buckets. Candidate fixes require indexed/range-limited
membership or bounded immutable sorted runs / a hierarchical merge policy, with
immediate irreversible owner acceptance preserved. These are proposals, not a
change to dedup semantics or a selected implementation.

## Actual overlap and memory lifetimes

Producer stream serializes pack -> generation -> conversion -> hash -> sort ->
pre-dedup -> directory. Next batch generation cannot overlap the current batch's
sort on this stream. Ping-pong overlaps this whole producer with owner work.
Archive has another stream, but shares SM/cache/DRAM resources with both.

Parents are retired after the entire coalesced physical extent is consumed,
not after every parent batch. StateRing capacity is 2*F+C records. Thus the
intended fine-grained parent-release memory saving is not fully implemented.
Archive lease safety is correct only if D2H finishes before reuse; removing that
dependency would introduce corruption, not performance optimization.

## Priorities, without claiming measured bottleneck shares

1. Measure repeated owner key visits, per-job counts, kernel launch counts and
   CPU synchronization time; remove repeated full-prefix work before microtuning.
2. Separate producer generation/hash from route-sort scheduling using explicit
   bank lifetimes; batch independent owner jobs without shared-scratch races.
3. Size materialization grids to useful work; revisit the one-CTA/bucket scan.
4. Test fused GEMM epilogues and workload-specific row updates; compare end-to-end,
   not only dense-equivalent TOPS. Avoid assuming that fusion always wins.
5. Add per-bucket population and per-stage timing/bytes telemetry plus a successful
   CUDA timeline. Quantify archive overhead with a matched diagnostic experiment;
   retain mandatory archives in all production performance claims.

Conclusion: correctness gates pass, but the implemented execution graph does not
yet realize the intended fully overlapped architecture. Optimality, saturation,
and negligible archive interference are **not established** by the current data.

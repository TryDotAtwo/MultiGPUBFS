# Four-V100 resident hash timing diagnostics

This is an experimental, opt-in GPU-side instrumentation of the already
reported resident hash path. The measured source was `f9bf30b1e0bb0959f16ef8fb5019ef401e3d8f4e`
(tree `12a90dae2c92f1f591b8a40e1ec0cbdfbc583171`); its code is published in
[commit 4d0ce32](https://github.com/TryDotAtwo/MultiGPUBFS/commit/4d0ce32be237a499b0286a4f604da47fb0702692),
which additionally contains the existing acceptance documents. It is not a
new default or a B200 test. Full machine-readable layer counts and recorded
stage arrays are in [the receipts](2026-11-23-v100-hash-stage-evidence.json).
The earlier uninstrumented comparisons and their limitations remain in
[the four-V100 report](2026-11-23-v100-resident-bfs.md).

Each case is one run on the same four V100-SXM2-32GB devices, with four
virtual buckets per shard, rehash enabled, 32 physical shards, 262,144-record
send and receive tiles, packed exact states and a single LSA exchange slot.
S12 used 262,144 rows per segment bank and an 8M per-rank cap; S13 used
1,048,576 rows and a 100M cap. The executable SHA-256 was
`5316b0d1745fbb48560e8fd1097145f50dcc78d9c9aa9aa27a6f153a500ddf03`;
the loaded CUDA library SHA-256 was
`b51269b1d1e12b19d71cc04e288e7bad6bf44694ebaec3e6938f620835601345`.

The GPU coordinator records target-specific PTX `globaltimer` intervals for
each **source-layer expansion**. The table sums source depths 28–38 for S12
and 34–44 for S13, inclusive. Times are seconds per rank and must not be
summed across ranks. They measure one single-slot diagnostic path; receive
includes deduplication and commit, not just lookup. The phase columns do not
cover all maintenance and coordination time. Records transferred to peers
are useful-record counts, not hardware link-byte measurements.

| Graph | Rank | Layer window | Generate / partition | Publish / wait ready | Receive / dedup / commit | Wait ACK |
|---|---:|---:|---:|---:|---:|---:|
| S12 | 0 | 11.077 | 0.941 | 0.079 | 9.668 | 0.212 |
| S12 | 1 | 11.077 | 0.937 | 0.082 | 9.493 | 0.392 |
| S12 | 2 | 11.077 | 0.938 | 0.075 | 9.536 | 0.351 |
| S12 | 3 | 11.077 | 0.959 | 0.066 | 9.583 | 0.296 |
| S13 | 0 | 186.883 | 13.843 | 1.403 | 158.348 | 10.677 |
| S13 | 1 | 186.883 | 13.959 | 1.163 | 158.080 | 10.979 |
| S13 | 2 | 186.883 | 13.781 | 1.359 | 158.203 | 10.930 |
| S13 | 3 | 186.883 | 13.973 | 1.196 | 159.574 | 9.498 |

S12 completed in 19.359 s and S13 in 354.018 s. Every layer size agreed
with independent Mahonian coefficients and the terminal state was checked
exactly; full intermediate sets of these large graphs were not compared.
The maximum 0.5-second NVML sample was 5826 MiB per GPU for S12 and
30450 MiB per GPU for S13. Sampling can miss short peaks.

The four-GPU S4 diagnostic on/off check compared the full state multiset at
all seven layers, exercised truncated diagnostics, a resource-limited job and
a healthy following job using the same persistent workers. The four-GPU S6
oracle compared the full multiset on each of 16 layers with normal hashes and
forced hash_bits=0, including tight caps and persistent reuse. A Rust unit test
for native rehash capability handling was run with the CUDA feature.

The complete evidence archive is `multicta-metrics-evidence.tgz`, SHA-256
`ab0ecd6454e66449ff5878c2a8f9b77aa3d8952041197a007593bb9bcec6c334`;
the smaller receipts archive is `multicta-metrics-receipts.tgz`, SHA-256
`b7c86492495124ca50369044369969fb8703e26287c50ba2438bd624ea68a46a`.
Neither proves device-resident eight-GPU operation, dynamic segment growth,
overlapped A/B processing, or maximum B200 capacity.

# RAM pipeline validation on 2×RTX 3060

Status: validation in progress; the full automatic sweep is still running.

## Scope and provenance

- Isolated branch: `codex/bfs-tail-archive`.
- Previous implementation: `6dd0e21a6fa7c83d2fe38a44a71eae2aa4037d92`.
- Matched GPU panel: `e89428790ab1482fd02a2d294ec1b1de08fc60fc`.
- Automatic sweep: `81f8a2f208ec22a58f4082da0bf01cc757f74f25`.
- Vast instance: 54283794, machine 28995, 2×RTX 3060 12 GiB, SM86.
- Authorized budget: $10. Quoted combined GPU and 100 GB disk rate: $0.153111/hour; traffic charged separately. Billing-read access is unavailable; guards estimate cost from the quote and interface traffic.
- Source, build, and test logs stay on the rental. No graph state payloads are downloaded to the user's computer.

## Changes tested

1. AS3 transmits full symbol states without the redundant per-state Hash128 archive plane. AS2 remains the compact prefix/terminal format; legacy AR1 readers remain supported.
2. Four reusable pinned slots of approximately 8 MiB per rank, independent of graph degree and compute batch size; bounded writer queues.
3. Closed rank parts become archived files by rename. Their incrementally calculated SHA-256 is reused, avoiding concatenation and rereading the same payload.
4. End-upload mode records per-layer metadata in RAM and seals durable files/manifests at graph completion. Search-upload mode still produces intermediate snapshots for publication.
5. Reproducible resource-pruning decisions are checkpointed in batches; attempted graphs still receive immediate durable records.
6. Exact packed CPU BFS is available for small orbits; fresh automatic configurations route orbits up to 65,536 states to it. This is a backend routing threshold, not a graph exploration cutoff. Large orbits still use native GPU admission.
7. End/background publication groups up to 2,048 graph cases, also bounded by the existing byte limit; graph-upload mode seals each graph separately.

## Matched full-state archive panel

Same two GPUs, three alternating old/new repetitions, resident runtime, warmed each pair; retain five complete layers. This panel does not compare CPU routing or automatic capacity probing. Times below include search and local archive completion, but exclude HF upload and warm-up.

| (n,r) | Previous median wall, s | New median wall, s | Wall speedup | Previous native, s | New native, s |
|---|---:|---:|---:|---:|---:|
| (8,1) | 0.4704 | 0.3398 | 1.38× | 0.2650 | 0.2626 |
| (10,4) | 0.4991 | 0.3896 | 1.28× | 0.3157 | 0.3161 |
| (12,6) | 0.6920 | 0.4829 | 1.43× | 0.3996 | 0.4035 |
| (12,5) | 1.8625 | 0.9729 | 1.91× | 0.7538 | 0.8973 |

All old/new completed-layer tables agree. Native time includes archival backpressure; the 19% increase for (12,5) needs further investigation before claiming unchanged native throughput. The overall wall reduction is 22–48%, configuration-specific.

Two earlier panel harness attempts failed due to wrapper argument/path mistakes; both are excluded. The six final workers completed and their tables were checked.

## Exact CPU routing experiment

Three repetitions on the same rental, including compact local archive completion; no CUDA construction. Python int transitions are independently compared with a full-word tuple BFS, including both hash seeds and complete retained state sets. Additional transition tests cover 8-bit symbols across 64-bit boundaries.

| (n,r) | Median CPU search + archive wall, s |
|---|---:|
| (8,1) | 0.1594 |
| (17,16) | 0.0406 |
| (33,32) | 0.0612 |
| (128,127) | 0.2113 |
| (128,126) | 0.4448 |

The (8,1) CPU result is 2.13× faster than the new five-layer GPU panel wall time, but retention policies differ, so this is an execution-routing observation rather than a matched archive comparison. CPU reports explicitly identify their backend and do not fabricate rank reports or claim GPU seed verification.

## Correctness and HF gate

Sixteen cases exercised compact COMPLETE, full five-layer AS3, byte-tail AS3, wide packed layouts, and confirmed resource exhaustion. COMPLETE cases used two seeds and independent tuple-BFS counts and retained state sets. Resource cases skipped the second run and exported no more than 1,000 states globally.

HF readback verified 28 manifests, 20 Parquet payloads, 870,821 bytes, all payload SHA-256 values, and the sweep ledger at [revision 25578a21](https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/commit/25578a21c4815f84811f798322c041f9e33a3ea2).

Local validation: 113 tail tests and six CPU tests pass; 16 Rust archive tests and the Linux CUDA/library-owner compile check passed for the RAM/AS3 implementation.

## Remaining evidence

Follow-up implementation after the measured panel (GPU validation pending): a host-ring planner targets aggregate GPU VRAM, bounded by 75% of available host RAM with explicit reader/OS reserves. Each rank pins one shared contiguous allocation, partitioned into reusable 8 MiB slots; exact cache geometry is required so retaining a subset cannot keep an oversized physical allocation. CPU reader output/shift arenas are also preallocated, and frame SHA-256 no longer concatenates the payload. Unit and Linux-target compile checks pass; these changes are not running in the existing `81f8a2f` sweep and their GPU timing is not yet claimed.

- Full automatic sweep totals, search time versus pair transitions/archive/publication, and its HF readback are pending.
- The native-time increase in the largest matched archive case needs a pool/backpressure measurement.
- Plain NCCL fallback is used on these RTX cards. Startup LSA capability failed with CUDA_STATUS_7; Graph32/LSA performance is not validated here or on B300.
- Naturally large final COMPLETE layers are not covered by the gate; forced terminal replacement and local descriptor tests cover the protocol separately.
- The rental is still active with independent budget/deadline guards. Final teardown requires remote token removal and an API-confirmed ABSENT result.

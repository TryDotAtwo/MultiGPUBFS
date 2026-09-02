# REF-014: warp aggregation and block compaction

Date: 2026-08-28 (Europe/Moscow)  
Status: pass; both optimizations rejected as universal defaults

## Question

Do warp-local equal-key aggregation and block-level output compaction improve
the exact atomic bitmap baseline across acceptance and duplicate profiles?

## Variants

All variants use the same persistent fixed-capacity context, inputs, bitmap,
CUDA-event timing, Rust oracle, and Docker image.

- `baseline`: every candidate performs `atomicOr`; every winner reserves one
  output slot with `atomicAdd`.
- `warp-aggregate`: `__match_any_sync` elects one claimant for equal keys in a
  warp; the elected lane performs the bitmap claim.
- `block-compact`: CUB `BlockScan` computes output offsets and one lane reserves
  all winning output positions for the block.
- `warp-block`: combines both mechanisms.

The warp optimization only combines *equal keys in the same warp*. It does not
combine different keys that share a bitmap word and does not find equal keys
in other warps.

## Correctness and artifact gates

Every variant was checked against the same exact accepted count and two
order-independent 64-bit fingerprints. The Rust artifact validator confirmed:

```json
{"status":"pass","validator":"rust-ref014-artifact-v1","rows":64,"outcome_groups":16}
```

This establishes complete unique configuration-row coverage of four variants,
four patterns, and four sizes, and agreement of the recorded accepted counts
and fingerprints for every pattern/size pair. Finite fingerprints are not an
injective encoding of arbitrary output sets, so that agreement does not prove
exact output-set equality for every sweep row. The self-test separately
compares complete output sets and exercises explicit capacity overflow; its
exact-equality evidence is limited to those self-test fixtures.

The final all-variant self-test passed in Docker under:

```text
memcheck:  0 errors
racecheck: 0 errors, 0 warnings
initcheck: 0 errors
synccheck: 0 errors
```

## Selected results

Median isolated kernel time at 16,777,216 candidates:

| pattern | baseline ms | warp ms | block ms | warp+block ms | best conclusion |
|---|---:|---:|---:|---:|---|
| all-new | 2.119 | 2.104 | 2.204 | 2.195 | warp +0.7%; effectively neutral |
| half-seeded-fourfold | 0.606 | 0.622 | 0.610 | 0.722 | baseline wins |
| all-seen | 0.547 | 0.572 | 0.595 | 0.703 | baseline wins |
| single-key | 10.154 | 0.373 | 10.168 | 0.374 | warp is 27.2x faster |

The `single-key` warp kernel reached 45.0 Gcandidate/s versus 1.65 for the
baseline. In contrast, warp aggregation was 2.5% slower on
`half-seeded-fourfold` and 4.3% slower on `all-seen` at the largest size.

Raw data are in `REF-014-bitmap-variant-sweep.jsonl`. Small `2^16` rows are
retained but are too sensitive to launch and timing noise for architecture
selection. Iteration timings include reset, upload, and download and are not
used to compare these kernels.

## Findings

1. Global duplicate multiplicity is insufficient to predict warp aggregation.
   `half-seeded-fourfold` has four occurrences per key, but its permutation
   spaces equal keys across the batch rather than within a warp. The warp path
   therefore adds voting work without removing claims.
2. Candidate order is an algorithmic performance variable. Generator layout,
   frontier layout, and any prior partition/sort determine whether duplicates
   become warp-local. A useful selector needs a within-warp collision estimate,
   not only accepted fraction or a global duplicate ratio.
3. Block compaction did not improve the high-acceptance workload. On `all-new`
   it was 4.0% slower than baseline at `2^24`. This rejects a net speedup for
   this tested configuration, not the possibility that output-counter atomics
   had substantial cost. Added scan/synchronization work could outweigh saved
   atomic work; that is a plausible explanation, not a retained phase-level
   measurement. The timing alone does not identify the original bottleneck.
4. Combining mechanisms is not free. `warp-block` retained the single-key win
   but provided no benefit over warp-only there and was the worst variant on
   the distributed rejection profiles.
5. For the tested broad/uniform batches, the historical baseline choice was
   retained. Warp aggregation was a conditional specialization for co-located
   hot keys. This is a workload-specific observation, not a current universal
   dispatch default or authorization for another measurement.

Interpretation correction, 2026-08-31: the sweep values above are unchanged.
Only the strength of the equality and bottleneck claims was corrected; no
measurement or sanitizer run was repeated in this documentation pass.

## Failed and corrected operations

- The first image build after adding the Rust formatting gate failed because
  the source was not rustfmt-clean. Formatting was performed in the dedicated
  Docker Rust toolchain image, then the gated build passed.
- Two rebuild commands used nonexistent default names (`Dockerfile` and
  `docker/Dockerfile`) before the actual `docker/Dockerfile.gpu` was selected.
- The first sanitizer invocation forgot that the runtime image has a Rust CLI
  entrypoint, so `compute-sanitizer` was parsed as an unknown CLI command. The
  corrected runs used `--entrypoint compute-sanitizer` and all four tools passed.

These were orchestration failures; none executed a partial CUDA measurement.

## Reproduction

```powershell
docker build -f docker/Dockerfile.gpu --target runtime -t multigpubfs-gpu:dev .
docker run --rm --gpus all -v "${PWD}\experiments:/output" `
  -e MGBFS_OUTPUT_PATH=/output/REF-014-bitmap-variant-sweep.jsonl `
  multigpubfs-gpu:dev bitmap-variant-sweep
docker run --rm -v "${PWD}\experiments:/input:ro" `
  multigpubfs-gpu:dev validate-bitmap-variant-artifact
docker run --rm --gpus all --entrypoint compute-sanitizer `
  multigpubfs-gpu:dev --tool memcheck --error-exitcode 99 `
  multigpubfs-gpu bitmap-self-test
```

Repeat the final command with `racecheck`, `initcheck`, and `synccheck`.

## Next experiment

Compare a CUB radix-sort/unique pipeline against the bitmap variants while
preserving these workloads and correctness fingerprints. Include both original
candidate order and deliberately grouped duplicates: sorting may create the
locality that makes later filtering cheap, but its full movement and temporary
storage cost must be charged to the pipeline.

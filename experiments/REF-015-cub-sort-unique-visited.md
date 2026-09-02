# REF-015: CUB radix sort/unique plus exact visited

Date: 2026-08-28 (Europe/Moscow)  
Status: pass; rejected as a default for the tested dense 32-bit workloads

## Question

Can a complete device pipeline that radix-sorts a candidate batch, removes
adjacent duplicates, and then claims the remaining keys in the exact visited
bitmap outperform direct atomic bitmap variants?

## Exact pipeline

Rust owns workloads, context lifetime, oracle checks, timing aggregation,
memory reporting, JSONL artifacts, and CLI. CUDA C++ is restricted to the GPU
translation unit and narrow C ABI. Every operation runs in Docker.

The measured device interval is:

1. CUB `DeviceRadixSort::SortKeys` over the significant key bits;
2. CUB `DeviceSelect::Unique` over sorted keys;
3. exact atomic bitmap claims for unique keys and output compaction.

The unique count remains a device scalar. The claim kernel reads it directly,
so there is no host synchronization between stages. The context queries CUB's
two-phase APIs once and allocates the maximum shared scratch requirement at
creation. No allocation occurs in a measured repetition.

This implementation deliberately charges the complete sort and unique work.
Timing only the final, cheap claim kernel would answer a different question.

## Correctness

The Rust self-test checks:

- same-batch duplicates plus pre-seeded visited keys;
- complete accepted output set;
- persistence of visited state across a second run;
- an empty batch;
- exact accepted count under output overflow;
- rejection of an out-of-range key.

A separate Rust artifact validator checked all 16 sweep rows and compared every
accepted count and both 64-bit fingerprints to the baseline rows from REF-014:

```json
{"status":"pass","validator":"rust-ref015-artifact-v1","rows":16,"cross_backend_outcomes":16}
```

The expanded self-test passed Compute Sanitizer `memcheck`, `racecheck`,
`initcheck`, and `synccheck` with zero errors or warnings.

## Environment and memory

The environment is the same RTX 3070 Laptop GPU (`sm_86`), CUDA 12.8.1 Docker
stack used by REF-013/014. At `2^24` candidates the persistent CUDA allocations
are exactly:

```text
total device allocations  339,776,015 bytes (324.04 MiB)
CUB temporary storage       69,243,391 bytes ( 66.04 MiB)
```

The total includes a 2 MiB bitmap, input/sorted/unique/output key arrays, four
device scalars, and CUB scratch. It excludes CUDA event implementation overhead
and Rust host vectors. This is roughly 2.5x the explicit device allocation of
the direct bitmap context at this batch size, so sort/unique also reduces the
maximum feasible frontier capacity.

## Results

Primary `2^24` medians, with three warmups and ten measured repetitions:

| pattern | sort ms | unique ms | claim ms | pipeline ms | Gcandidate/s |
|---|---:|---:|---:|---:|---:|
| all-new | 1.475 | 0.384 | 2.108 | 3.966 | 4.230 |
| half-seeded-fourfold | 1.395 | 0.256 | 0.567 | 2.216 | 7.569 |
| all-seen | 1.389 | 0.226 | 0.176 | 1.793 | 9.357 |
| single-key | 1.378 | 0.215 | 0.069 | 1.662 | 10.095 |

Phase medians are computed independently and need not sum bit-exactly to the
median of total samples.

Comparison with the `2^24` REF-014 kernel medians:

| pattern | direct baseline ms | best bitmap ms | sort/unique ms | result |
|---|---:|---:|---:|---|
| all-new | 2.119 | 2.104 warp | 3.966 | sort pipeline is 1.88x slower |
| half-seeded-fourfold | 0.606 | 0.606 baseline | 2.216 | 3.66x slower |
| all-seen | 0.547 | 0.547 baseline | 1.793 | 3.28x slower |
| single-key | 10.154 | 0.373 warp | 1.662 | 6.11x faster than baseline, 4.46x slower than warp |

An independent repeat produced pipeline medians of 3.962, 2.214, 1.797, and
1.681 ms respectively. Large-run differences from the primary artifact were
0.10%, 0.12%, 0.20%, and 1.17%; the broad conclusions are stable. Small batch
rows remain launch-sensitive.

Raw data:

- `REF-015-sort-unique-sweep.jsonl`
- `REF-015-sort-unique-repeat.jsonl`

## Findings

1. Sort/unique is not the default for these dense ranked 32-bit states. It
   loses both throughput and capacity to direct bitmap claiming on broad and
   already-seen workloads.
2. Batch dedup can rescue an otherwise pathological algorithm: on `single-key`
   the pipeline is 6.11x faster than the naive per-candidate atomic baseline.
   It still loses 4.46x to the much cheaper warp-local specialization because
   all duplicates are already co-located.
3. The best primitive depends on where duplicate convergence occurs. Warp
   aggregation is cheap but local; sorting is global but expensive; direct
   bitmap claims defer convergence to visited atomics.
4. Sorting cost is largely insensitive to how many unique keys survive. At the
   largest size it consumed about 1.38-1.47 ms across all patterns, including
   the already sorted `all-new` input and the one-key input. The tested radix
   implementation does not provide an adaptive cheap path for these cases.
5. Dedup shrinks downstream work substantially in the fourfold and single-key
   profiles, but the saved claim work does not repay sorting when a bitmap or
   warp-local alternative is available.
6. Sort/unique may still be relevant when keys are not cheaply rankable, when
   the visited structure is a costly hash table, when records must be grouped
   for owner routing, or when sorting serves multiple downstream phases. Those
   are separate experiments; this result must not be generalized to them.

## Sources and implementation checks

- NVIDIA's [CUB API reference](https://nvidia.github.io/cccl/unstable/cub/api/index.html)
  identifies the device-wide radix-sort and selection primitives.
- The exact CUDA 12.8.1 headers inside the pinned Docker image were inspected
  before implementation. `DeviceSelect::Unique` uses a two-phase temporary
  storage API and an `int` item count, which is why the C ABI rejects capacities
  above `INT_MAX`.

## Next questions

- Does sort/unique become competitive for 64/128-bit non-rankable state keys
  against an exact fixed-capacity GPU hash table?
- Can sorting be amortized by reusing owner-grouped order for multi-GPU routing?
- What duplicate locality occurs in actual Cayley successor batches before and
  after generator-major versus parent-major layout?
- Would a histogram/sample selector distinguish direct, warp-local, and global
  convergence cheaply enough to dispatch per frontier?

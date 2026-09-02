# REF-013: exact dense-rank bitmap visited backend

Date: 2026-08-28 (Europe/Moscow)  
Status: pass; first GPU primitive baseline, not an end-to-end BFS result

## Question

Can a fixed-capacity, device-resident atomic bitmap implement exact concurrent
visited claims and compaction, and how does its throughput change with batch
size, accepted fraction, and duplicate concentration?

## Implementation

Rust owns workload construction, persistent host buffers, FFI lifetime,
correctness validation, timing aggregation, and JSONL output. CUDA C++ owns:

- a dense `uint32` bitmap, one bit per ranked state;
- persistent candidate/output buffers and scalar flags;
- seed and filter kernels;
- CUDA events for isolated kernel timing;
- explicit out-of-range and capacity-overflow reporting.

Every candidate performs an `atomicOr` claim. The first claimant performs a
global `atomicAdd` to reserve an output position. Output order is intentionally
unspecified; exact set semantics are required.

No device or host buffer allocation occurs in the measured repetition. The
implementation is a deliberately simple baseline: it has neither warp-level
duplicate suppression nor block-level output reservation.

Maximum swept allocation at 16,777,216 candidates was approximately:

- 2 MiB visited bitmap;
- 64 MiB candidate keys;
- 64 MiB output capacity;
- 64 MiB persistent Rust output buffer;
- small counters/events and input/seed host vectors.

## Correctness gates

The Rust self-test checked full output sets for:

- pre-seeded visited keys;
- same-batch duplicates;
- keys at both bitmap boundaries;
- persistence across a second invocation;
- zero candidates;
- exact explicit overflow with total accepted count retained;
- out-of-range candidate and seed rejection.

A separate 4,194,304-candidate fixture compared the complete sorted accepted
set of 524,288 keys with Rust's expected set. Sweep rows checked accepted count,
overflow, and two independent order-independent 64-bit fingerprints.

Compute Sanitizer on the reduced self-test reported:

```text
memcheck:  0 errors, 0 bytes leaked
racecheck: 0 errors, 0 warnings
initcheck: 0 errors
synccheck: 0 errors
```

`racecheck` does not validate arbitrary global-memory algorithms; exactness is
principally established by atomic claim semantics and the independent set
checks.

One final container verification invocation combined a read-only source mount
with ordinary `compileall`; tests and JSONL validation passed, but bytecode
creation failed with read-only filesystem errors. Re-running with
`PYTHONPYCACHEPREFIX=/tmp/pycache` preserved the read-only mount and compiled all
Python sources successfully.

## Environment

```text
GPU        NVIDIA GeForce RTX 3070 Laptop GPU, 8 GiB, sm_86
driver     572.70
CUDA       12.8.1 / nvcc 12.8.93
image      sha256:73655adbe0eb9bb47ee552a881e1b3960f134748537296d26b768a7e4081c68d
build      Release, native sm_86 cubin, -lineinfo
```

A post-run snapshot showed P0, 62 C, 28.80 W, 1290 MHz SM and 6000 MHz memory.
Clocks and power were observed, not locked; laptop thermal/power variability
remains a limitation.

## Sweep

Each row used 3 warmups and 10 measured repetitions. Candidate counts were
`2^16`, `2^20`, `2^22`, and `2^24`. Workload profiles:

- `all-new`: every candidate is unique and accepted;
- `half-seeded-fourfold`: four occurrences per key, half of unique keys already
  visited, so accepted occurrences are 12.5% of input;
- `all-seen`: every unique key is pre-seeded;
- `single-key`: every occurrence contends for one previously clear bit and only
  one is accepted.

Selected `2^24` median results:

| pattern | accepted | kernel ms | kernel Gcandidate/s | iteration ms | iteration Gcandidate/s |
|---|---:|---:|---:|---:|---:|
| all-new | 16,777,216 | 2.114 | 7.938 | 16.345 | 1.026 |
| half-seeded-fourfold | 2,097,152 | 0.602 | 27.864 | 9.541 | 1.758 |
| all-seen | 0 | 0.548 | 30.624 | 7.667 | 2.188 |
| single-key | 1 | 10.156 | 1.652 | 17.808 | 0.942 |

Raw rows are in `REF-013-bitmap-sweep.jsonl`.

## Observations

1. Accepted fraction alone is not a sufficient performance descriptor. Both
   `all-seen` and `single-key` accept essentially nothing, yet the large-batch
   kernels differ by 18.54x. Extreme concentration serializes atomic updates to
   one word/bit.
2. Accepting every state costs output reservation and a 64 MiB output write.
   `all-new` is 4.23x slower than `all-seen` at `2^24`, but their key working
   sets and reuse also differ, so this is not a controlled attribution to the
   output counter alone.
3. Kernel throughput rises sharply with batch size. `2^16` timings are heavily
   launch/measurement sensitive and some ten-sample maxima are several times
   their medians. Architecture choices should use sufficiently large levels.
4. The Rust iteration metric includes clearing/seeding visited, pageable host
   uploads, kernel work, scalar readback, and accepted-output download. It is
   intentionally much slower than the isolated kernel and is not an end-to-end
   GPU-resident BFS measurement.
5. The current global output `atomicAdd` and per-occurrence bitmap atomic are
   obvious optimization targets. Warp duplicate aggregation can attack hot-key
   contention; block scan plus one reservation per block can attack high-
   acceptance compaction.

## Reproduction

```powershell
docker build -f docker/Dockerfile.gpu -t multigpubfs-gpu:dev .
docker run --rm --gpus all multigpubfs-gpu:dev bitmap-self-test
docker run --rm --gpus all multigpubfs-gpu:dev bitmap-sweep
```

Sanitizer commands use the same image with `--entrypoint compute-sanitizer` and
the `bitmap-self-test` command.

## Boundaries and next comparison

This result establishes one exact specialization for rankable state spaces. It
does not show that bitmap wins against sort/unique or hashing, and it does not
include generator application. The next controlled experiment should preserve
these exact workload rows while adding:

1. warp-aggregated bitmap claims and block-level compaction;
2. CUB radix sort/unique plus prior-visited filtering;
3. a fixed-capacity 64-bit hash backend for non-rankable states.

# Macro publication contract: runtime validation

Date: 2026-10-01. Scope: existing single-rank macro runtime, not distributed macro completion.

## Changes

The macro reference path now uses the existing warmup configuration validator, rank-result writer and group-commit writer. It rejects invalid warmup configuration before outputs, refuses to overwrite an existing rank result, and publishes group-complete only after successful archive finalization. The existing debug archive-finalize fault hook is applied to this path. Warmup archives are released before measurement; warmup output explicitly carries an ephemeral scope and no durable-time claim.

No generation kernel, batch transport protocol or hot-path readback was added or replaced. Multi-rank macro remains explicitly unsupported rather than silently downgraded.

## Behavioral RED/GREEN

New Linux CUDA tests call the actual reference runtime, using U3(2), macro depth 2 and partial parent batches. Before their respective fixes they reproduced missing group publication, accepted finalize failure, retained warmup archive, and silently accepted invalid warmup. After integration all four tests passed.

The six existing macro_native tests also passed, including full-state oracle comparisons, nonidentity start and depth-10 coverage, archive verification, allocation contract and capacity/preflight failures.

Hardware: local RTX 3070 Laptop, sm86, CUDA 12.8, container multigpubfs-ref046-green:latest, Rust 1.75. Native CUDA library was built from current sources for sm86. This is not a 2xT4 or NCCL acceptance result.

Command: cargo test --locked --offline -p mgbfs-runtime --features cuda --test macro_publication --test macro_native -- --test-threads=1. Exit 0: 4/4 new publication tests and 6/6 existing macro tests.

Fresh ordinary Windows workspace suite: cargo test --locked, exit 0. Linux CUDA tests are cfg-gated out of that command; their execution is evidenced separately above. Existing dead-code warnings remain in the CPU build.

## Remaining gates

No four-sanitizer acceptance, paired performance claim, distributed weighted macro runtime or full project completion is claimed here. The finalize test injects an error after actual archive finish; it is not a physical disk-failure experiment. Durable timing currently ends after archive finalization, before publication metadata fsync, consistent with the existing reference path; it is not a timestamp covering every group-marker write.

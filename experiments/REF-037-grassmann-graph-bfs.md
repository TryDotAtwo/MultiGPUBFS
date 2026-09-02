# REF-037: exact BFS on small binary Grassmann graphs

## Question

Do complete small subspace graphs realize their q-binomial BFS formulas, and
can exact span identity expose the difference between bases and vertices?

## Method

- Restrict to `F_2^n`, `n<=6`.
- Enumerate independent vector selections and deduplicate their complete span
  membership masks.
- Connect two `k`-subspaces iff their intersection has dimension `k-1`.
- Run complete BFS and audit state count, degree, distance, layers,
  depth-conditioned neighbor classes, and shortest paths.
- Execute only in Docker with Rust.

## Retained failure

The first `rustfmt --check` reported four formatting differences and stopped
the command before compilation. The exact formatting changes were applied; the
same gate then passed.

## Result

All nine normalized fixtures passed. The largest was `J_2(6,3)` with 1,395
states, degree 98, diameter 3, and layers `[1,98,784,512]`. Every mismatch
counter was zero.

## Status

Pass after one formatting-only failed gate. This is an exact small-state
semantic probe, not performance evidence or a scalable enumerator.


# REF-036: exact BFS on Johnson graphs

## Question

Do complete small fixed-cardinality subset graphs realize the closed Johnson
distance, frontier, intersection, and shortest-path formulas?

## Method

- Represent each `k`-subset by an `n`-bit mask.
- Generate every selected/unselected membership exchange.
- Run complete BFS from `{0,...,k-1}`.
- Audit all states against intersection distance, closed layer counts,
  depth-conditioned neighbor classes, and shortest-path multiplicity.
- Run all normalized `J(n,k)` for `2<=n<=12` only in Docker.

## Result

All 36 fixtures passed with zero mismatches. The largest fixture was
`J(12,6)` with 924 states, degree 36, diameter 6, and layers
`[1,36,225,400,225,36,1]`.

## Status

Pass. This is exhaustive only for the declared small fixtures and is not
performance evidence.


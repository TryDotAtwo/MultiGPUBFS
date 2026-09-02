# REF-038: BFS frontiers in sparse Erdos-Renyi samples

## Question

How do finite BFS frontiers, root extinction, giant components, and outward
candidate collisions change across the sparse `G(n,c/n)` phase transition?

## Method

- `n=2000`, `p=c/(n-1)`.
- `c in {0.8,1.0,1.2,4.0}`.
- 20 deterministic seeds per value.
- Transparent all-pairs Bernoulli sampler, frozen undirected adjacency lists,
  exact component enumeration and BFS.
- Record fixed-root and largest-component observations separately.
- Docker-only Rust execution.

## Result

```text
c=0.8: largest mean 0.0207, range [0.0115,0.0350]
c=1.0: largest mean 0.0689, range [0.0310,0.1375]
c=1.2: largest mean 0.2996, range [0.1610,0.4335], predicted rho 0.3137
c=4.0: largest mean 0.9797, range [0.9745,0.9865], predicted rho 0.9802
```

The representative `c=4` largest-component frontiers were
`[1,8,31,98,351,765,591,105,9,1]`. Outward occurrence multiplicity per new
state peaked near 1.88, while early layers remained near one.

## Scope

These are 80 finite samples from one declared PRNG implementation. They
illustrate, but do not statistically establish, asymptotic random-graph
theorems. The `O(n^2)` sampler is intentionally transparent and not a benchmark.

## Status

Pass. No failed gate was observed.


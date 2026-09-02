# Multi-source balls superpose; distance frontiers do not

Multi-source BFS computes minimum distance to a source set. Its accumulated
balls distribute cleanly over union of source sets, but its exact-distance
frontiers generally do not. The visited subtraction is the point where a
seemingly linear wave propagation becomes history-sensitive.

## Distance and ball union

For source set `S`, write

```text
d_S(v) = min_(s in S) dist(s,v),
B_d(S) = {v : d_S(v)<=d},
F_d(S) = {v : d_S(v)=d}.
```

These definitions apply to directed or undirected unit graphs, with infinity
for unreachable vertices. For two source sets,

```text
d_(S union T)(v) = min(d_S(v),d_T(v)).
```

Therefore

```text
B_d(S union T) = B_d(S) union B_d(T).
```

This is an exact superposition law for bounded reachability: a vertex lies
within `d` of the combined sources exactly when it lies within `d` of at least
one constituent source set.

## Why the frontiers do not union layerwise

For `d>=1`, taking the delta between consecutive balls gives

```text
F_d(S union T)
  = (B_d(S) union B_d(T))
      minus (B_(d-1)(S) union B_(d-1)(T)).
```

Equivalently,

```text
F_d(S union T)
  = (F_d(S) minus B_(d-1)(T))
      union
    (F_d(T) minus B_(d-1)(S)).
```

A state exactly `d` away from `S` disappears from combined layer `d` when `T`
already reached it earlier. Thus in general

```text
F_d(S union T) != F_d(S) union F_d(T).
```

The right side classifies distance separately under two histories; the left
side classifies the minimum distance under one shared visited history.

## Path counterexample

Take the undirected path

```text
0 -- 1 -- 2
```

and source sets `S={0}`, `T={2}`. Separate BFS runs have

```text
F_0(S)={0}, F_1(S)={1}, F_2(S)={2}
F_0(T)={2}, F_1(T)={1}, F_2(T)={0}.
```

Their depth-two frontier union is `{0,2}`. But combined multi-source BFS starts
with `{0,2}`, reaches `{1}` at depth one, and then terminates:

```text
F_0(S union T)={0,2}
F_1(S union T)={1}
F_2(S union T)=empty.
```

No edge or reachable state was lost. The endpoint states moved from depth two
to depth zero under the changed source contract.

## Boolean propagation versus delta extraction

The neighbor operator and Boolean matrix product distribute over union:

```text
Post(X union Y)=Post(X) union Post(Y).
```

Accumulated least-fixed-point reachability therefore superposes. Exact BFS
deltas additionally subtract the combined old ball:

```text
F_(d+1)=Post(F_d) minus B_d.
```

That shared historical mask suppresses a state as soon as any source reaches
it. Calling Boolean propagation “linear” must not be extended to a claim that
first-discovery layers or their cardinalities add source by source.

## Source labels and ties

Scalar distance retains only the minimum value. If

```text
d_S(v)<d_T(v),
```

then `T` contributes no shortest combined-source path to `v`. If the two
distances tie, both source sets may contribute nearest-source labels and
shortest paths.

For source-distinguished path identity, the combined shortest-path count is

```text
sigma_(S union T)(v)
  = sum of sigma_source(v) over sources attaining the minimum distance.
```

It is not the sum over every source. A one-label Voronoi forest chooses among
tied nearest sources; a set-valued label or per-source count retains more
information. These are output-contract choices, not changes to scalar BFS
distance.

## Work and memory consequences

Running one BFS per source and unioning same-numbered frontiers is not equivalent
to one multi-source BFS. Separate searches may:

- expand the same state once per source;
- retain longer source-relative layers suppressed by the combined minimum;
- produce different queue/frontier peaks;
- preserve per-source distances and paths that combined scalar BFS discards.

Conversely, combined BFS can begin with a much wider `F_0` and create different
early duplicate convergence. Neither work profile can be inferred by simply
adding or subtracting the other frontier sizes.

For GPU or multi-GPU measurements, “number of sources” therefore belongs to the
semantic workload identity. A throughput comparison between separate and
combined searches must declare whether the requested output is minimum distance,
one nearest label, every tied label, or all per-source distances.

## Practical checks

For a claimed multi-source result, verify:

1. every distinct source is initialized at distance zero;
2. visited is shared when minimum-to-set distance is intended;
3. source labels have declared tie semantics;
4. per-source paths are not inferred from a scalar minimum field;
5. frontier comparisons use the combined-source metric, not unions of
   separately indexed layers;
6. work counters distinguish one combined traversal from repeated searches.

# REF-021: exhaustive counterexample to general-graph double sweep

## Question

Does two-sweep BFS always compute the exact diameter of a finite connected
undirected graph when the first sweep has a unique farthest vertex?

## Hypothesis

No. The theorem is special to trees and selected graph families; a general
graph should contain a start whose unique farthest vertex is not peripheral.

## Semantics and method

The bounded Rust probe enumerates every simple undirected labeled graph on
`4..=7` vertices in increasing vertex count and edge-bitmask order. It rejects
disconnected graphs, computes every all-pairs distance by ordinary exact BFS,
and derives every eccentricity and the true diameter.

For each start `r`, it asks whether:

1. `r` has exactly one farthest vertex `u`; and
2. `ecc(u) < diameter(G)`.

This is stronger than a tie-dependent failure: the first sweep is forced to
select `u`, and the second sweep returns only `ecc(u)`.

Source: `experiments/ref021_double_sweep.rs`.

## Environment

```text
runtime: Docker
image: multigpubfs-rust-toolchain:dev
rustc: 1.75.0 (82e1608df 2023-12-21)
architecture: x86_64
GPU: unused; container correctly reported no NVIDIA driver
date: 2026-08-28
```

## Command

```powershell
docker run --rm --mount "type=bind,source=C:\Users\Иван Литвак\Documents\ChatGPT\MultiGPUBFS,target=/work" -w /work multigpubfs-rust-toolchain:dev bash -lc "rustc --edition=2021 experiments/ref021_double_sweep.rs -o /tmp/ref021 && /tmp/ref021"
```

## Raw result

```text
REF021_DOUBLE_SWEEP_COUNTEREXAMPLE
vertices=7
edges=0-2,0-4,0-6,1-2,1-4,1-5,2-3
start=4
unique_first_farthest=3
first_distance=3
second_farthest=[4, 5, 6]
double_sweep_value=3
true_diameter=4
eccentricities=[3, 3, 2, 3, 3, 4, 4]
```

## Witness interpretation

The graph is

```text
      3
      |
6 -- 0 -- 2
     |    |
     4 -- 1 -- 5
```

The listed edge set remains the authoritative representation.

From start `4`, vertex `3` is the unique farthest vertex at distance `3`.
However, `ecc(3)=3`. The true diameter is `4`, witnessed for example by

```text
5 -- 1 -- 2 -- 0 -- 6.
```

Thus the forced two-sweep result is `3 < 4`.

## Correctness oracle

- Connectivity: every distance row contained no infinity sentinel.
- True diameter: maximum over all BFS distance rows.
- Unique first farthest: exhaustive equality check within the start row.
- Second-sweep value: eccentricity of that forced pivot.
- The reported diameter path can be replayed directly against the edge list.

## Observation

No qualifying graph was encountered before the reported seven-vertex witness
under this enumeration. This is evidence only about the probe's bounded labeled
search order, not a formal claim that seven is the minimum up to isomorphism.

## Interpretation

Two-sweep is an exact diameter algorithm on trees, but only a lower-bound
heuristic on general connected graphs. A unique first farthest vertex does not
repair the general theorem: a farthest vertex from an arbitrary start need not
be peripheral.

## Failure/negative result

The universal claim "unique first farthest makes two-sweep exact" is rejected.
There was no performance objective and no optimization attempt.

## Next question

Which algebraic symmetry conditions make one complete BFS sufficient for exact
diameter? For a finite connected Cayley graph, vertex transitivity supplies the
answer; generic Schreier/puzzle graphs need a separate proof.

# BFS with all transpositions: cycle-count distance and Stirling frontiers

Let every transposition of `S_n` be a unit-cost generator. One move either
merges two permutation cycles or splits one cycle into two. Therefore distance
from the identity is exactly the number of cycle mergers required, and complete
BFS layers are counted by unsigned Stirling numbers of the first kind.

This produces a fourth metric on the same permutation states studied with
adjacent, prefix-reversal, and star generators. It demonstrates both the power
and the limit of an exact scalar distance formula: vertices at one depth can
still have different inward/outward neighbor counts. This note adds no
optimizer, production implementation, benchmark, or GPU code. A small
exhaustive Rust probe checks `n=2..8`.

## 1. Complete-transposition contract

Let

```text
T = {(i j) : 1 <= i < j <= n}.
```

Define

```text
CT_n = Cay(S_n,T).
```

Every pair of positions may be swapped in one step. Hence

```text
|V(CT_n)| = n!,
degree = C(n,2),
|E(CT_n)| = n! C(n,2) / 2.
```

“Complete” describes the transposition generating graph on the `n` positions;
`CT_n` itself is not a complete graph on `n!` permutations.

## 2. One transposition changes cycle count by one

Include fixed points when counting disjoint cycles of permutation `pi`; write
the count as `c(pi)`. Multiplying by transposition `(i j)` has exactly two
possibilities:

- if `i,j` lie in different cycles, the cycles merge and `c` decreases by one;
- if they lie in the same cycle, it splits into two and `c` increases by one.

No edge preserves cycle count. Starting from the identity with `n` cycles,
every path to `pi` therefore needs at least `n-c(pi)` steps.

## 3. Exact cycle-count metric

A cycle of length `l` factors into `l-1` transpositions. Factoring every
nontrivial disjoint cycle independently uses

```text
sum_cycles (l-1) = n-c(pi)
```

transpositions. This meets the lower bound, so

```text
d_CT(e,pi) = n-c(pi).
```

For arbitrary endpoints, Cayley translation gives the same formula applied to
`pi^-1 tau` under the fixed composition convention.

## 4. Diameter and farthest layer

Cycle count is at least one, so

```text
diam(CT_n) = n-1.
```

Equality holds exactly for permutations consisting of one `n`-cycle. There are
`(n-1)!` such permutations, hence

```text
|F_(n-1)| = (n-1)!.
```

The farthest layer can therefore be a large fraction of the state space even
though its depth is only linear.

## 5. Exact Stirling frontier law

The unsigned Stirling number of the first kind

```text
[n over k]
```

counts permutations of `n` symbols with exactly `k` cycles. Therefore

```text
|F_d| = [n over n-d],    0 <= d <= n-1.
```

The complete sphere-generating polynomial is

```text
sum_d |F_d| q^d = product_(i=1)^(n-1) (1 + i q).
```

Setting `q=1` gives `n!`, proving that the layers exhaust the whole group.

## 6. Frontier recurrence across n

Unsigned Stirling numbers satisfy

```text
[n over k] = [n-1 over k-1] + (n-1)[n-1 over k].
```

Combinatorially, symbol `n` is either a singleton cycle or inserted after one of
the `n-1` existing symbols in cyclic notation. In distance coordinates this
recursively constructs the complete layer counts for `CT_n`.

This is a counting recurrence, not a level-synchronous implementation
requirement. An ordinary queue BFS and a coefficient recurrence compute
different output objects even when their final layer counts agree.

## 7. Bipartite orientation by cycle count

Every generator is an odd permutation and every edge changes `c(pi)` by one.
Thus

```text
depth parity = permutation sign,
```

and `CT_n` is bipartite. Relative to the identity root, every edge connects
adjacent BFS layers; there are no same-layer edges.

This makes cycle count a perfect scalar potential for edge orientation, but not
a complete vertex key.

## 8. Same depth, different local geometry

At permutation `pi`, a transposition moves one layer inward exactly when its two
symbols lie in the same cycle and split it. If cycle lengths are `l_1,...,l_k`,
then

```text
inward_degree(pi)  = sum_j C(l_j,2),
outward_degree(pi) = C(n,2) - inward_degree(pi).
```

At depth two:

- a 3-cycle has inward degree `C(3,2)=3`;
- two disjoint 2-cycles have inward degree `1+1=2`.

They share distance `n-c=2` but not intersection counts. Thus `CT_n` is not
distance-regular for `n>3`, despite its exact class-function distance and high
symmetry.

## 9. Early collision anatomy

Depth one contains every transposition:

```text
|F_1| = C(n,2).
```

Depth two contains two cycle types:

- one 3-cycle: `2 C(n,3)` states;
- two disjoint transpositions: `3 C(n,4)` states.

Hence

```text
|F_2| = 2 C(n,3) + 3 C(n,4).
```

Multiple two-step histories collide on each state: a 3-cycle has several
minimal transposition factorizations, while disjoint transpositions commute.
The large generator degree therefore creates duplicate pressure immediately.

## 10. Small exhaustive Rust probe

`experiments/all_transpositions_bfs_probe.rs` independently performs queue BFS,
computes `n-c(pi)`, checks inversion parity, and builds the expected layers by
the Stirling recurrence.

Observed in Docker with Rust 1.85.1:

```text
n  states  degree  diameter  layers
2       2       1         1  [1,1]
3       6       3         2  [1,3,2]
4      24       6         3  [1,6,11,6]
5     120      10         4  [1,10,35,50,24]
6     720      15         5  [1,15,85,225,274,120]
7    5040      21         6  [1,21,175,735,1624,1764,720]
8   40320      28         7  [1,28,322,1960,6769,13132,13068,5040]
```

For every checked state and layer:

```text
metric_mismatches = 0,
parity_mismatches = 0,
stirling_match = true.
```

This is exhaustive evidence through `n=8`; the general claims follow from the
cycle and Stirling proofs.

## 11. Four metrics on the same permutation universe

For `n=8`, exact probes now give:

```text
generator set          degree   diameter
adjacent swaps              7       28
prefix reversals            7        9
star transpositions         7       10
all transpositions         28        7
```

All four graphs contain exactly 40,320 permutations and can use the same Lehmer
rank. Degree, diameter, layer count, and successor transformation differ with
the generator set.

This table is descriptive for the declared metrics. It is not a ranking of
algorithms or hardware performance: higher degree can reduce depth while
increasing candidates, bytes, duplicate pressure, and per-level work.

## 12. Normal Cayley symmetry boundary

All transpositions form a conjugacy class in `S_n`, so the generator set is
invariant under conjugation. Consequently distance from identity is a class
function and entire conjugacy classes lie in one layer.

Star transpositions are not invariant under arbitrary conjugation because the
distinguished symbol `1` moves. Their distance requires the additional
`delta` information from note 137. Prefix reversals and adjacent swaps have
still different symmetry.

Even here, conjugacy class/cycle type is not authoritative state identity and
does not preserve all requested outputs such as a particular target or parent
word.

## 13. Shortest paths and factorization counts

The scalar distance `n-c(pi)` does not count shortest paths. Minimal
transposition factorizations can be numerous and depend on cycle lengths and
labels. BFS with one parent discards this multiplicity; a shortest-path DAG
retains predecessor edges; counting factorization words is another output.

An exact distance oracle can validate a BFS label without reproducing the BFS
parent or enumeration order.

## 14. GPU and multi-GPU boundary

All-transposition successor generation has quadratic logical degree
`C(n,2)` but each move swaps only two positions. Relative to star/pancake or
adjacent generators, this trades fewer BFS rounds for more candidates per
frontier state. The net execution effect is a measurement question.

Exact exhaustive BFS still has factorial visited capacity. High symmetry and a
closed distance oracle do not guarantee balanced owner hashing, coalesced rank
access, or low communication. Early commuting/factorization collisions can
make generated-transition throughput very different from new-state throughput.

Measurements should separate:

- logical degree and generated swap attempts;
- cycle-count oracle work and permutation-rank work;
- factorization histories, duplicate candidates, unique candidates, and new
  states;
- inward/outward frontier edges by cycle type;
- visited capacity and owner balance;
- synchronization rounds, routing bytes, and end-to-end time.

This is a conceptual workload comparison, not an optimization proposal.

## Sources

- E. Konstantinova,
  [*Vertex Reconstruction in Cayley Graphs*](https://doi.org/10.1016/j.disc.2008.07.039),
  *Discrete Mathematics* 309, 2009, for the all-transposition Cayley graph's
  order, degree, bipartiteness, diameter, and structural properties.
- NIST Digital Library of Mathematical Functions,
  [*Permutations: Cycle Notation*](https://dlmf.nist.gov/26.13), for the minimal
  number `n-c(pi)` of arbitrary transpositions in a permutation factorization.
- R. P. Stanley,
  [*Enumerative Combinatorics, Volume 1*](https://doi.org/10.1017/CBO9781139058520),
  Chapter 1, for unsigned Stirling numbers of the first kind, cycle counting,
  and their recurrence/generating polynomial.
- Notes 6, 10, 11, 27, 28, 32, 33, 35, 46, 51, 61, 64, 67, 68, 93, 136, and
  137 supply this repository's Cayley, adjacent/Mahonian, DAG, relation,
  identity, regularity, matrix, growth, memory, ownership, equality,
  multiplicity, parity, generator-change, word-metric, pancake, and star
  boundaries.

## Takeaway

With every transposition available, BFS depth is exactly `n-c(pi)`, diameter is
`n-1`, and layer sizes are unsigned Stirling numbers `[n over n-d]`. The same
scalar depth can hide different inward/outward degrees because cycle lengths
matter. Adding generators compresses distance but increases candidate degree
and early duplicate multiplicity; it does not make BFS work universally
smaller.

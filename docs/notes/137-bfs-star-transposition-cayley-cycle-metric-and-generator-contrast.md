# BFS on star-transposition Cayley graphs: cycle metric and generator contrast

The star graph and pancake graph use the same `n!` permutation states and the
same degree `n-1`, but different generators. Star moves swap the first position
with one other position. This change yields a bipartite Cayley graph, an exact
cycle-decomposition distance formula, a closed diameter, and frontier profiles
different from pancake BFS.

This is a controlled demonstration that the vertex set, degree, and
vertex-transitivity do not determine BFS geometry. This note adds no optimizer,
production implementation, benchmark, or GPU code. A small exhaustive Rust
probe checks `ST_2` through `ST_8`.

## 1. Star-transposition contract

Let `S_n` act on positions `1,...,n`. Define

```text
s_i = (1 i),    2 <= i <= n.
```

The star graph is

```text
ST_n = Cay(S_n, {s_2,...,s_n}).
```

Every generator swaps the first symbol with exactly one later symbol. Therefore

```text
|V(ST_n)| = n!,       degree = n-1,
```

and every generator is a fixed-point-free involution on permutation states.
The generator family creates every transposition and hence generates `S_n`, so
the graph is connected.

## 2. Same states and degree as pancake, different graph

Both `ST_n` and `P_n` are connected Cayley graphs on `S_n`, have `n!` vertices,
and have degree `n-1`. Their edge sets differ:

- star: swap positions `1` and `i`;
- pancake: reverse positions `1` through `i`.

The identity, exact permutation equality, and a Lehmer rank universe can be
shared. Distances, layers, path relations, diameter, parity behavior, and
per-candidate transformation work cannot be transferred without proof.

## 3. Exact distance from cycle structure

Write a permutation as disjoint cycles. Ignore fixed points, and define

```text
s = number of symbols in nontrivial cycles,
c = number of nontrivial cycles,
delta = 1 if symbol 1 lies in a nontrivial cycle, else 0.
```

Then star-transposition word length is

```text
d_ST(e,pi) = s + c - 2 delta.
```

Equivalently:

- a nontrivial cycle of length `l` containing `1` costs `l-1`;
- a nontrivial cycle of length `l` not containing `1` costs `l+1`.

A cycle containing `1` decomposes directly into star transpositions. For a
cycle avoiding `1`, one must bring `1` into the cycle, rotate through its
symbols, and remove `1`, causing the extra two operations relative to an
ordinary `l-1` transposition factorization.

## 4. Why the formula is a shortest-distance certificate

The constructions above provide an upper bound. For the lower bound, a star
transposition can only merge/split cycles through symbol `1`. Resolving a cycle
containing `1` needs at least `l-1` such changes. Resolving a disjoint cycle
requires entering it and leaving it in addition to touching its `l-1`
independent cycle structure.

Summing over disjoint cycles gives the stated lower bound and matches the
construction. Thus cycle decomposition is an exact distance oracle, not merely
a heuristic.

The oracle depends on the star generator set. Applying it to pancake prefix
reversals would answer the wrong metric.

## 5. Closed diameter

To maximize `s+c-2delta`, it is best to keep symbol `1` fixed and pack the other
symbols into as many short nontrivial cycles as possible:

- disjoint 2-cycles when `n-1` is even;
- one 3-cycle plus disjoint 2-cycles when `n-1` is odd.

This yields

```text
diam(ST_n) = floor(3(n-1)/2).
```

Unlike pancake diameter, this is a closed theorem for every `n`. The existence
of an exact oracle makes a one-root BFS useful for validation, but does not make
exhaustive traversal necessary to compute an individual distance.

## 6. Bipartite parity

Every star generator is one transposition and therefore changes permutation
sign. Hence sign supplies an exact two-coloring:

```text
depth parity = permutation parity.
```

The graph is bipartite and has no odd cycle. This contrasts with pancake
generators, whose signs vary with prefix length for `n>=4`.

Self-inverse generators alone do not imply bipartiteness; the common parity
homomorphism is the additional reason it holds here.

## 7. First layers from cycle types

The distance formula counts early spheres exactly:

```text
|F_1| = n-1.
```

Depth two consists of oriented 3-cycles containing `1`. Choose two other
symbols and either orientation:

```text
|F_2| = 2 C(n-1,2) = (n-1)(n-2).
```

Depth three has two disjoint cycle-type families:

- a 4-cycle containing `1`: `6 C(n-1,3)` states;
- one transposition avoiding `1`: `C(n-1,2)` states.

Therefore

```text
|F_3|
= 6 C(n-1,3) + C(n-1,2)
= (n-1)(n-2)(n-3) + C(n-1,2).
```

The same first two layer sizes as pancake do not imply the same third layer.

## 8. Exact small profiles

`experiments/star_transposition_bfs_probe.rs` exhaustively traverses `ST_2`
through `ST_8`, compares every BFS distance with the cycle formula, and checks
depth parity against inversion parity.

Observed in Docker with Rust 1.85.1:

```text
n  states  degree  diameter  layers
2       2       1         1  [1,1]
3       6       2         3  [1,2,2,1]
4      24       3         4  [1,3,6,9,5]
5     120       4         6  [1,4,12,30,44,26,3]
6     720       5         7  [1,5,20,70,170,250,169,35]
7    5040       6         9  [1,6,30,135,460,1110,1689,1254,340,15]
8   40320       7        10  [1,7,42,231,1015,3430,8379,13083,10408,3409,315]
```

Across all checked states:

```text
metric_mismatches = 0,
parity_mismatches = 0.
```

The probe is exhaustive through `n=8`; the general formula and diameter rely on
the cycle proof and cited results, not extrapolation.

## 9. Direct controlled comparison with pancake BFS

For each fixed `n`, both probes traverse exactly the same permutations with the
same degree. Yet:

```text
n    pancake diameter    star diameter
4           4                 4
5           5                 6
6           7                 7
7           8                 9
8           9                10
```

At `n=8`:

```text
pancake peak frontier = 15011 at depth 7,
star peak frontier    = 13083 at depth 7.
```

The difference cannot be attributed to state count, degree, root choice, or
visited representation. It is generated by the move relations and word metric.

These finite observations illustrate the distinction; they do not establish a
universal ordering of the two diameters or peaks for every `n`.

## 10. Frontier counts from cycle enumeration

Because star distance depends only on whether `1` is fixed/in a cycle and on
the lengths/counts of other nontrivial cycles, complete sphere counts can in
principle be obtained by enumerating cycle types with their permutation
multiplicities.

That is a special analytic shortcut. It does not mean a generic Cayley frontier
is determined by conjugacy class: the generator set is not invariant under all
conjugations of `S_n`, and arbitrary Cayley word length need not be a class
function.

## 11. Vertex transitivity versus cycle-type equivalence

Cayley translation makes all roots equivalent. It does not make all vertices in
one BFS layer equivalent under root-preserving automorphisms. Different cycle
types can share one star distance while having different shortest-factorization
counts and neighbor distributions across adjacent layers.

Thus one common layer profile coexists with heterogeneous vertices inside the
layer. Frontier size is an aggregate, not a complete local-geometry summary.

## 12. State identity and witness replay

As in pancake BFS, a full permutation or proved bijective rank is authoritative.
Cycle type alone is not identity: many permutations share a cycle type, and
even the exact distance formula intentionally maps them to the same scalar.

A distance certificate may use the cycle decomposition to construct a shortest
star-transposition word. A BFS parent chain is a different witness. Both need a
fixed composition convention and replay against the actual permutation action.

## 13. Network topology is not the computed state graph

Star graphs were proposed as logical interconnection networks because of their
regularity, symmetry, degree, diameter, and connectivity. If processors are
physically/logically connected as `ST_n`, shortest packet routing is BFS in that
network metric.

Using a star topology to execute BFS on some other state graph does not turn the
workload into star-transposition BFS. Computed edges, owner-routing edges, and
physical links remain separate graphs.

## 14. GPU and multi-GPU boundary

Compared with prefix reversal, a star successor changes two positions rather
than reversing a variable-length prefix. This suggests a different candidate
generation cost under the same state/rank representation, but it is not a
performance result.

Both families still require exact visited over `n!` states for exhaustive BFS.
Their different layer profiles change concurrency and synchronization epochs;
their different relations change duplicate timing. Logical vertex transitivity
does not guarantee owner balance or topology-aware traffic balance.

Measurements should separate:

- generator attempts and transformation bytes/operations;
- cycle-oracle work versus BFS work;
- histories, duplicates, unique candidates, and new states;
- rank/key and authoritative visited costs;
- frontier profile, diameter, and synchronization rounds;
- owner routing, physical hops, and end-to-end time.

This comparison is a semantic measurement design, not an optimization plan.

## Sources

- S. B. Akers and B. Krishnamurthy,
  [*A Group-Theoretic Model for Symmetric Interconnection Networks*](https://doi.org/10.1109/12.21148),
  *IEEE Transactions on Computers* 38, 1989, for the Cayley-network model and
  star/pancake graph families.
- E. Konstantinova,
  [*Vertex Reconstruction in Cayley Graphs*](https://doi.org/10.1016/j.disc.2008.07.039),
  *Discrete Mathematics* 309, 2009, for the star graph's order, regularity,
  bipartiteness, diameter, and short-cycle properties.
- J. Irving and A. Rattan,
  [*Minimal Factorizations of Permutations into Star Transpositions*](https://arxiv.org/abs/math/0610640),
  2006, for minimal star-transposition factorizations organized by permutation
  cycle structure.
- Notes 6, 10, 16, 27, 28, 32, 35, 46, 51, 61, 67, 68, 93, and 136 supply this
  repository's Cayley model, frontier growth, Schreier, relations, identity,
  regularity, growth, memory, ownership, equality, parity, generator-change,
  word-metric, and pancake comparison boundaries.

## Takeaway

Star and pancake BFS share the same `S_n` states, degree, dense rank universe,
and Cayley root symmetry, yet their distances and frontiers differ because the
generators differ. Star distance is exactly readable from cycle structure and
has diameter `floor(3(n-1)/2)`; every edge flips sign, so the graph is
bipartite. State-space size and degree alone do not determine BFS geometry or
execution shape.

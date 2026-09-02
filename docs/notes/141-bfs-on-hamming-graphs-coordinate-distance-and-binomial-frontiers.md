# BFS on Hamming graphs: coordinate distance and binomial frontiers

The Hamming graph `H(d,q)` is a controlled family where BFS geometry is exactly
separable by coordinates. It exposes how degree, same-layer edges, duplicate
multiplicity, shortest-path counts, and frontier peaks change when the binary
hypercube is generalized to a `q`-symbol alphabet.

No optimizer, production implementation, benchmark, or GPU code is added.

## 1. State and edges

Fix an alphabet of size `q>=2`. Vertices are words

```text
x=(x_1,...,x_d) in {0,...,q-1}^d.
```

Two words are adjacent exactly when they differ in one coordinate. Therefore

```text
|V| = q^d,
degree = d(q-1),
H(d,q) = K_q square ... square K_q  (d factors).
```

It is also the undirected Cayley graph of `Z_q^d` with every nonzero
single-coordinate increment as a generator. The full generator set matters:
using only `+e_j`, or only `+/-e_j`, defines a different graph when `q` changes.

## 2. BFS distance is coordinate disagreement

Root BFS at the all-zero word. Let `w(x)` be the number of nonzero coordinates.
Every move changes only one coordinate, so reaching `x` needs at least `w(x)`
moves. Setting each nonzero coordinate directly to its final symbol realizes
that bound. Hence

```text
d(0,x)=w(x).
```

By translation, pairwise graph distance is Hamming distance. The diameter is
`d`, and the farthest vertices are the `(q-1)^d` words with no zero coordinate.

This closed oracle validates distance but not enumeration order, parent choice,
or visited identity.

## 3. Exact frontier profile

To build a word at depth `i`, choose its `i` nonzero coordinates and then one of
`q-1` values in each. Thus

```text
|F_i| = C(d,i)(q-1)^i,
sum_i |F_i| = (1+q-1)^d = q^d.
```

The consecutive ratio is

```text
|F_(i+1)| / |F_i| = ((d-i)(q-1))/(i+1).
```

So the mode lies near `(q-1)d/q`, with the usual adjacent-mode exception when
the ratio equals one. Increasing `q` shifts mass toward the outer layers even
though the diameter remains `d`.

The normalized layer distribution is binomial with success probability
`(q-1)/q`. This is an exact counting identity, not a random-BFS assumption.

## 4. Intersection numbers

From a depth-`i` word, one coordinate change can:

- reset one of `i` nonzero coordinates to zero: `c_i=i` inward neighbors;
- change a nonzero symbol to another nonzero symbol: `a_i=i(q-2)` same-layer
  neighbors;
- set one of `d-i` zero coordinates to a nonzero symbol:
  `b_i=(d-i)(q-1)` outward neighbors.

Their sum is always `d(q-1)`. These counts depend only on depth, so `H(d,q)` is
distance-regular with intersection array determined by `b_i` and `c_i`.

The edge balance identity

```text
|F_i| b_i = |F_(i+1)| c_(i+1)
```

reduces exactly to the frontier recurrence above.

## 5. Binary versus nonbinary parity

For `q=2`, `a_i=0`: every edge changes Hamming weight parity, the graph is the
binary hypercube, and it is bipartite.

For `q>2`, each coordinate induces a clique `K_q`. Changing one nonzero symbol
to another creates same-layer edges, and any three symbols in one coordinate
give a triangle. Therefore `H(d,q)` is not bipartite.

This is a useful visited-semantics contrast. A bug that excludes only old
layers but fails to exclude the current frontier can remain hidden on `q=2`
and become visible immediately on `q>2`.

## 6. Exact duplicate multiplicity

Every vertex in `F_(i+1)` has exactly `i+1` inward predecessors. Consequently,
expanding the complete `F_i` generates

```text
|F_i| b_i = |F_(i+1)|(i+1)
```

outward candidate occurrences for only `|F_(i+1)|` new semantic states. The
outward candidate-to-new-state ratio is exactly `i+1`.

Additionally, each expanded depth-`i` state generates `i` candidates into the
previous layer and `i(q-2)` candidates into the current layer. These are
different rejection classes:

- earlier-visited hits;
- same-frontier hits;
- duplicate occurrences converging on next-frontier states.

The formulas predict counts, not where equal candidates land in a warp, owner
partition, or queue order.

## 7. Shortest paths

A depth-`i` target differs from the root in `i` coordinates. A shortest path
must set each such coordinate directly to its final value exactly once. The
coordinates may be handled in any order, giving

```text
sigma(0,x)=i!.
```

Thus all vertices in one layer have equal shortest-path multiplicity, while the
total number of shortest histories ending in the layer is

```text
|F_i| i! = d!/(d-i)! (q-1)^i.
```

A one-parent BFS keeps one of these paths. The frontier stores each endpoint
once. Generated word histories and unique states diverge increasingly with
depth.

## 8. Product and spectral views

The Cartesian-product distance theorem independently gives additive Hamming
distance. The frontier polynomial is

```text
(1+(q-1)z)^d,
```

whose coefficient of `z^i` is `|F_i|`.

Each `K_q` factor has adjacency eigenvalues `q-1` once and `-1` with
multiplicity `q-1`. Cartesian-product sums therefore give

```text
lambda_j = d(q-1)-qj,
multiplicity C(d,j)(q-1)^j,  0<=j<=d.
```

The spectral multiplicities equal the BFS layer sizes here because both come
from the Hamming association scheme. They are still different objects: one is
an eigenspace dimension, the other a root-distance count.

## 9. Codes, balls, and multi-source BFS

A radius-`r` Hamming ball has exact volume

```text
sum_(i=0)^r C(d,i)(q-1)^i.
```

Seeding a code `C subseteq V` in multi-source BFS computes distance to the
nearest codeword. Its maximum label is the code's covering radius; ties are
Voronoi ownership questions. Minimum distance between codewords is instead a
packing/separation property and is not returned by one merged source wave.

This connects note 139's fixed-center coverage to the classical Hamming scheme
without turning BFS into a code-construction algorithm.

## 10. Cayley and representation boundaries

Base-`q` ranking gives a dense bijection between words and `[0,q^d)`, making an
exact bitmap visited set possible. The state count remains exponential in `d`;
a compact rank does not remove capacity growth.

The Cayley description uses an abelian group, so coordinate moves commute. The
`i!` shortest paths are precisely the permutations of the `i` required
coordinate updates. Adding redundant generator encodings or intermediate
symbol changes can increase candidate histories without changing endpoints.

Quotienting words by coordinate permutation or symbol permutation changes
state identity and generally yields an orbit/Schreier problem, not the original
Hamming graph.

## 11. GPU and multi-GPU interpretation

`H(d,q)` is a clean calibration workload:

- fixed degree and exact per-layer inward/same/outward counts;
- dense base-`q` identity;
- predictable frontier and duplicate counts;
- tunable same-level traffic through `q-2`;
- short diameter `d` but exponentially many states.

Its regularity deliberately removes transformation-cost and degree imbalance.
Therefore good throughput on this family does not establish performance for a
puzzle with legality checks, wide states, irregular orbit sizes, expensive
ranking, or relation-dependent duplicates.

For multi-GPU runs, rank ownership may correlate with low base-`q` digits.
Balanced vertex counts do not imply balanced boundary traffic. Report logical
intersection counts, local versus remote candidates, owner skew, bytes,
synchronization, and end-to-end time separately.

This is a workload model, not an optimized implementation proposal.

## 12. Docker/Rust probe

`experiments/hamming_graph_bfs_probe.rs` exhaustively traverses all
`H(d,q)` for `d=1..5` and `q=2..4`. Across 15 fixtures and up to 1,024 states it
confirmed:

```text
state count q^d;
degree d(q-1) and diameter d;
all layers C(d,i)(q-1)^i;
zero Hamming-distance mismatches;
zero intersection-number mismatches;
zero per-state i! shortest-path mismatches.
```

For example:

```text
H(5,3): layers [1,10,40,80,80,32]
H(5,4): layers [1,15,90,270,405,243]
```

The probe is a small exact oracle, not a benchmark.

## Sources

- A. E. Brouwer, A. M. Cohen, and A. Neumaier,
  [*Distance-Regular Graphs*](https://doi.org/10.1007/978-3-642-74341-2),
  Springer, 1989, for Hamming graphs as a fundamental distance-regular family
  and their intersection structure.
- P. Delsarte, *An Algebraic Approach to the Association Schemes of Coding
  Theory*, Philips Research Reports Supplements 10, 1973, for the Hamming
  association scheme and Krawtchouk-polynomial framework.
- Notes 10, 11, 13, 32, 33, 46, 61, 64, 69, 70, 117, and 139 provide this
  repository's frontier, shortest-path, multi-source, distance-regular,
  spectral, capacity, equality, history, product, metric-net, and covering
  boundaries.

## Takeaway

In `H(d,q)`, BFS depth is Hamming weight, layers are
`C(d,i)(q-1)^i`, and every depth-`i` state has exactly `i`, `i(q-2)`, and
`(d-i)(q-1)` inward, same-layer, and outward neighbors. The family turns
frontier growth and duplicate pressure into closed formulas while showing
exactly where the binary hypercube's bipartite behavior stops generalizing.


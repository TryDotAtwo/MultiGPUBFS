# BFS on Grassmann graphs: subspace identity and q-binomial frontiers

The Grassmann graph `J_q(n,k)` is the finite-field `q`-analog of the Johnson
graph. Vertices are `k`-dimensional subspaces rather than `k`-subsets. This
preserves exact distance-regular formulas while making state equality and
canonical representation substantially more delicate.

No optimizer, production implementation, benchmark, or GPU code is added.

## 1. q-numbers and states

Let `q` be a prime power and let the ambient space be `F_q^n`. Write

```text
[m]_q = (q^m-1)/(q-1),
[n choose k]_q = product_(j=0)^(k-1) (q^(n-j)-1)/(q^(k-j)-1).
```

The Gaussian binomial coefficient counts `k`-dimensional subspaces. Therefore

```text
|V(J_q(n,k))| = [n choose k]_q.
```

Two vertices `U,W` are adjacent when

```text
dim(U intersection W)=k-1.
```

Equivalently, one one-dimensional direction is replaced while the subspace
dimension remains fixed.

## 2. Distance from intersection dimension

Fix root subspace `R` and let

```text
i = k-dim(R intersection U).
```

One adjacent step can increase the intersection dimension with `R` by at most
one, so at least `i` steps are required. A flag of intermediate subspaces can
replace the missing directions one at a time, attaining the bound. Hence

```text
d(R,U)=k-dim(R intersection U).
```

The diameter is `min(k,n-k)`. Orthogonal-complement duality under a chosen
nondegenerate form gives the parameter symmetry between `k` and `n-k`; it does
not identify those subspaces inside one visited set.

The constant-dimension coding metric is often

```text
d_S(U,W)=dim(U)+dim(W)-2dim(U intersection W)=2d_graph(U,W).
```

The factor of two must be stated when graph and coding distances are compared.

## 3. Degree

Choose a `(k-1)`-subspace of `U` in `[k]_q` ways. For each, the possible new
one-dimensional quotient directions outside `U` give `q[n-k]_q` distinct
extensions. Thus

```text
degree = q [k]_q [n-k]_q.
```

This counts distinct adjacent subspaces, not matrix operations, candidate
bases, or group elements that may realize the same neighbor.

## 4. Exact BFS layers

The number of vertices whose intersection deficit from `R` is `i` is

```text
|F_i| = q^(i^2) [k choose i]_q [n-k choose i]_q.
```

The Gaussian coefficients choose the internal and external quotient data; the
factor `q^(i^2)` counts the compatible linear maps between them.

The consecutive ratio is

```text
|F_(i+1)|/|F_i|
  = q^(2i+1) [k-i]_q [n-k-i]_q / [i+1]_q^2.
```

Summing the layers gives `[n choose k]_q`. This is the finite Grassmannian's
q-Vandermonde decomposition by intersection dimension.

## 5. Intersection numbers

At depth `i`, the distance-regular inward and outward counts are

```text
c_i = [i]_q^2,
b_i = q^(2i+1) [k-i]_q [n-k-i]_q,
a_i = q[k]_q[n-k]_q - b_i - c_i.
```

The balance identity

```text
|F_i| b_i = |F_(i+1)| c_(i+1)
```

is exactly the layer-ratio formula. The remaining `a_i` neighbors stay in the
same BFS layer, so current-frontier membership still matters.

Unlike a tree, one next-layer subspace has `[i+1]_q^2` shortest predecessors.
This convergence grows faster than the Johnson value `(i+1)^2` when `q>1`.

## 6. Shortest paths

Distance regularity gives the recurrence

```text
sigma_i = c_i sigma_(i-1),
sigma_0=1.
```

Therefore every depth-`i` target has

```text
sigma_i = (product_(j=1)^i [j]_q)^2 = ([i]_q!)^2
```

shortest paths. These paths represent chains of intermediate subspaces, not
choices of matrix bases. Basis multiplicity is an additional representation
multiplicity and must not be folded into graph-path counts.

## 7. The q-to-1 Johnson limit

Formally letting `q->1` gives

```text
[m]_q -> m,
[n choose k]_q -> C(n,k),
q^(i^2) -> 1.
```

The Grassmann formulas then become the Johnson formulas:

```text
|F_i| -> C(k,i)C(n-k,i),
c_i -> i^2,
b_i -> (k-i)(n-k-i),
sigma_i -> (i!)^2.
```

This is an algebraic q-analog relation, not an instruction to approximate a
finite-field traversal numerically with `q=1`.

## 8. Basis equality is not vertex equality

A `k`-subspace can be represented by many full-rank `k x n` matrices. Two
matrices represent the same vertex exactly when they have the same row space.
An ordered basis count contributes

```text
|GL(k,q)| = product_(j=0)^(k-1) (q^k-q^j)
```

representations per subspace before other encoding redundancies.

Using raw basis bytes as the visited key therefore overcounts states. Exact
options include:

- reduced row-echelon form under a fixed field and pivot convention;
- another proved canonical subspace encoding;
- a bijective rank of the finite Grassmannian;
- hash indexing followed by exact canonical equality.

A hash or rank formula without collision/bijection evidence is not enough.

## 9. Neighbor generation and duplicate sources

One conceptual generator chooses a hyperplane `H<U` and a new extension of `H`.
Naive enumeration can emit the same neighboring subspace through different
bases or scalar representatives. Canonicalization/deduplication belongs between
candidate representation and semantic neighbor identity.

There are at least four counts:

```text
matrix-operation attempts;
candidate bases;
distinct neighboring subspaces;
new BFS states.
```

Only the third is the simple graph degree `q[k]_q[n-k]_q`.

## 10. Schreier symmetry

`GL(n,q)` acts transitively on `k`-subspaces. The stabilizer of one subspace is
a large parabolic subgroup, so the natural action graph is Schreier-like rather
than a regular Cayley graph on all matrices.

Different group elements can fix the current subspace or send it to the same
neighbor. Vertex transitivity transfers scalar root profiles but does not make
group-element occurrences equal to graph edges.

Intersection dimension determines distance from one root but is far too coarse
for visited identity.

## 11. Spectrum

The adjacency eigenvalues are

```text
theta_j = q^(j+1)[k-j]_q[n-k-j]_q - [j]_q,
```

with multiplicities

```text
[n choose j]_q - [n choose j-1]_q,
```

for `0<=j<=min(k,n-k)`. These association-scheme parameters do not eliminate
the need for canonical state equality in traversal.

## 12. GPU and multi-GPU interpretation

Grassmann graphs are a useful stress model for implicit exact BFS because they
combine regular logical geometry with expensive representation semantics:

- state count roughly exponential in `k(n-k)`;
- canonical row reduction or nontrivial ranking;
- large and predictable shortest-predecessor convergence;
- multiple basis encodings per semantic state;
- transitive but non-free group action.

A GPU may generate matrix candidates quickly while canonicalization or exact
visited identity dominates. Multi-GPU ownership must be computed from a stable
canonical identity; assigning raw bases can route equal subspaces to different
owners and break exact deduplication.

Measurements should separate field operations, row reductions, candidate
bases, distinct neighbors, visited probes, accepted states, owner traffic,
synchronization, and end-to-end time.

This is a correctness and workload model, not an optimized implementation.

## 13. Docker/Rust probe

`experiments/grassmann_graph_bfs_probe.rs` studies only `q=2`, `n<=6`. It
enumerates independent vector sets, canonicalizes each subspace by its complete
membership bitmask, deduplicates equal spans, constructs adjacency from exact
intersection dimension, and runs complete BFS.

Nine normalized fixtures passed after one formatting-only failed gate. The
largest result was

```text
J_2(6,3): states=1395, degree=98, diameter=3,
layers=[1,98,784,512].
```

Every fixture had zero state-count, degree, distance, layer, intersection, and
shortest-path mismatches. The membership bitmask is exact only because the
ambient binary spaces are tiny; it is not proposed as a scalable encoding.

## Sources

- A. E. Brouwer, A. M. Cohen, and A. Neumaier,
  [*Distance-Regular Graphs*](https://doi.org/10.1007/978-3-642-74341-2),
  Springer, 1989, Section 9.3, for Grassmann graph parameters.
- P. Terwilliger, [*Grassmann Graphs, Degenerate DAHA, and Non-Symmetric Dual
  q-Hahn Polynomials*](https://doi.org/10.1016/j.laa.2019.11.027), *Linear
  Algebra and its Applications* 584, 2020, for the graph definition,
  intersection numbers, eigenvalues, and Q-polynomial context.
- Notes 16, 17, 20, 32, 45, 46, 55, 61, 62, 64, 70, 123, 142 provide this
  repository's Schreier, quotient, product-state, distance-regular, exact
  identity, capacity, oracle-validation, equality, full-state, history,
  group-action, covering, and Johnson boundaries.

## Takeaway

Grassmann BFS is governed by intersection dimension: layer `i` has
`q^(i^2)[k choose i]_q[n-k choose i]_q` states, inward degree `[i]_q^2`, and
outward degree `q^(2i+1)[k-i]_q[n-k-i]_q`. The formulas are a clean q-analog of
Johnson graphs, but exact traversal must canonicalize row spaces: different
bases are representation duplicates, not different graph vertices.


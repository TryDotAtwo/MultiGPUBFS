# BFS shortest-path DAG and betweenness centrality

Note 11 established that BFS can retain all shortest predecessors and count
shortest paths. Betweenness centrality asks a global question of that structure:
what fraction of shortest paths between other endpoint pairs passes through a
given vertex or edge? This note develops the semantics, reverse dependency
view, and a Cayley symmetry consequence. It adds no implementation, optimizer,
benchmark, or GPU code.

## 1. Raw shortest-path betweenness

For distinct vertices `s,t`, let `sigma_st` be the number of shortest paths from
`s` to `t` under a declared path identity, and let `sigma_st(v)` count those
having `v` as an internal vertex. In a connected undirected graph, raw vertex
betweenness under unordered endpoint pairs is

```text
C_B(v) = sum_{{s,t}: s!=v!=t} sigma_st(v)/sigma_st.
```

Endpoints are excluded here. Other conventions include endpoints, count
ordered pairs, or normalize by a graph-size factor. These values differ by more
than notation, so every result must state:

- ordered or unordered endpoint pairs;
- whether endpoints contribute;
- raw or normalized score;
- directed orientation and unreachable-pair policy;
- vertex, edge, or labeled-transition path identity.

## 2. Why one parent tree is insufficient

A deterministic BFS tree selects one shortest path per target. Counting how
often tree paths pass through `v` measures the chosen parent policy, not
shortest-path betweenness when alternatives exist.

In the diamond `s-a-t` and `s-b-t`, both length-two paths are shortest. Under
the fractional definition, `a` and `b` each receive one half from pair `{s,t}`.
A parent tree choosing only `a` would assign one to `a` and zero to `b`.

Exact betweenness therefore needs all relevant shortest-path multiplicity, not
merely exact distances or one replay witness.

## 3. Forward BFS builds sufficient source evidence

For one source `s`, exhaustive unweighted BFS can record:

```text
d_s(v)       shortest distance,
Pred_s(v)    every predecessor u with d_s(u)+1=d_s(v),
sigma_s(v)   number of shortest s-v paths.
```

The count recurrence is

```text
sigma_s(s)=1,
sigma_s(v)=sum_(u in Pred_s(v)) sigma_s(u).
```

The predecessor relation is acyclic by increasing depth. It compactly
represents possibly exponentially many shortest paths, subject to exact edge
identity and nonoverflowing arithmetic.

Distance labels alone cannot reconstruct path fractions when multiplicities
differ. Counts alone also do not identify which predecessor receives which
dependency; the reverse relation or equivalent successor access is needed.

## 4. Reverse-depth dependency accumulation

Brandes defines the dependency of source `s` on `v` and accumulates it from
deeper vertices toward the source:

```text
delta_s(v) = sum_(w: v in Pred_s(w))
               (sigma_s(v)/sigma_s(w)) * (1 + delta_s(w)).
```

The ratio is the fraction of shortest `s-w` paths whose final predecessor step
comes through `v`. The `1` accounts for target `w`; `delta_s(w)` represents all
farther targets already depending on `w`.

Processing vertices in nonincreasing BFS depth is essential. A forward pass
cannot finalize `delta_s(v)` before every deeper successor contribution is
known. After accumulating all sources, undirected ordered-source contributions
are divided by two to obtain the unordered raw convention above.

This is not a second BFS. It is dynamic programming on the completed
source-shortest-path DAG.

## 5. Calibration graphs

- **Complete graph:** every pair is adjacent, so internal-vertex betweenness is
  zero.
- **Path:** shortest paths are unique. Removing an internal vertex `v` leaves
  sides of sizes `L` and `R`, giving raw unordered `C_B(v)=L*R`.
- **Star:** every leaf pair has the center as its unique internal vertex, so the
  center has `binom(n-1,2)` and every leaf has zero.
- **Cycle:** alternate equal-length routes for antipodal pairs must be split
  fractionally; deterministic tie-breaking would destroy rotational symmetry.
- **Diamond:** its half-and-half contribution exposes why a tree is not enough.

High degree, high closeness, and high betweenness are not equivalent. A complete
graph has maximal degree and minimal distance but zero internal betweenness.

## 6. A global sum identity

Fix an unordered pair `{s,t}` at distance `d`. Every shortest `s-t` path has
exactly `d-1` internal vertices. Summing the fractional contribution over all
vertices gives

```text
sum_v sigma_st(v)/sigma_st = d(s,t)-1.
```

Now sum over all unordered pairs. Using note 120's Wiener index `W(G)`:

```text
sum_v C_B(v) = W(G) - binom(n,2).
```

This identity holds regardless of shortest-path multiplicity because every
path fraction participates in a partition whose total internal length is
`d-1`. It is both a conceptual bridge between closeness-like distance sums and
betweenness and a useful aggregate validation invariant.

For raw edge betweenness, every shortest path has `d` edges, so

```text
sum_e C_B(e) = W(G)
```

under the matching unordered-pair and fractional conventions. This edge
betweenness is not the edge-Wiener index, which measures distances between
edges and is a different invariant.

## 7. Cayley symmetry gives every vertex score from one BFS

In a finite connected undirected Cayley graph, translations act transitively by
graph automorphisms. They preserve shortest-path counts and fractions, so every
vertex has the same raw betweenness.

Let `T(e)` be the identity transmission from note 120. Since

```text
W(G)=n*T(e)/2
```

and the global vertex-betweenness sum is `W-binom(n,2)`, each vertex has

```text
C_B(v) = (T(e)-n+1)/2
```

for unordered endpoint pairs with endpoints excluded. Under ordered pairs the
value is `T(e)-n+1`. Under the common undirected normalization by
`binom(n-1,2)`, the value is

```text
(T(e)-n+1)/((n-1)(n-2)),  n>2.
```

Thus one complete identity-rooted BFS histogram determines every vertex's
scalar betweenness in this precise convention, without materializing every
source DAG. This does not reconstruct which endpoint pairs or generator paths
contribute; it derives only the symmetry-forced score.

Equal scores do not mean betweenness is zero or uninformative. They mean the
graph has no distinguished vertex under this automorphism-invariant statistic.

## 8. Edge symmetry is weaker

Vertex transitivity does not imply edge transitivity. A Cayley graph can have
several generator-edge orbits, and their edge betweenness values may differ.
If the graph is additionally edge-transitive, the aggregate identity gives

```text
C_B(e) = W(G)/|E|
```

for every edge under the raw unordered convention.

Without edge transitivity, one identity BFS plus a translation argument may
reduce work within each proved edge orbit, but total Wiener distance alone
cannot distribute the score among different orbits.

Generator labels, inverses, parallel semantic transitions, and collapsed
simple edges must be fixed before defining `sigma` or edge betweenness.

## 9. Schreier and quotient boundary

As in note 120, a Schreier state graph is not automatically vertex-transitive
under automorphisms preserving the fixed generator graph. Therefore equal
vertex betweenness and the one-root Cayley formula require an actual graph
automorphism proof.

Quotienting states can also merge paths and change fractions. Betweenness in a
quotient is not generally the aggregate betweenness of its fibers unless a
path-lifting and weighting theorem establishes that relation.

## 10. Directed, disconnected, and weighted variants

In a directed graph, ordered pairs are natural and `s->t` contributions need
not match `t->s`. Unreachable pairs are usually omitted or contribute zero, but
the convention must be explicit.

For disconnected undirected graphs, componentwise pair domains avoid division
by nonexistent `sigma_st`. Normalization by the whole graph versus the reachable
component produces different scores.

Weighted positive graphs replace BFS with a weighted shortest-path engine;
zero-weight edges complicate strict depth order and path counting. Ordinary
FIFO BFS/Brandes semantics apply directly only to unit-weight graphs.

## 11. Arithmetic and validation

Shortest-path counts may be exponential in DAG size. Fixed-width overflow can
silently corrupt both ratios and dependencies. Floating dependency accumulation
also depends on reduction order and is not exact rational arithmetic.

Useful validation layers include:

1. BFS distance and predecessor equations;
2. exact or overflow-detected `sigma` recurrence;
3. reverse-depth completion before finalization;
4. graph-automorphism checks before symmetry reduction;
5. `sum_v C_B(v)=W-binom(n,2)`;
6. `sum_e C_B(e)=W` under matching conventions;
7. calibration on paths, stars, complete graphs, cycles, and diamonds.

Passing an aggregate sum does not prove every vertex score, but failing it
proves at least one semantic, arithmetic, or accumulation error.

## 12. GPU and multi-GPU interpretation

For one source, forward BFS discovery is followed by shortest-path count
aggregation and a reverse-depth dependency wave. These are different dataflow
phases even if fused operationally:

- first discovery uses set-like visited semantics;
- equal-depth predecessor contributions use non-idempotent addition;
- reverse dependencies require all deeper contributions;
- floating accumulation is order-sensitive;
- retries must not duplicate `sigma` or dependency contributions.

In multi-GPU execution, every depth and source epoch needs globally complete
predecessor/count evidence before reverse finalization. Owner-local completion
is insufficient if remote successors still contribute.

Report separately traversal, predecessor/count storage, reverse accumulation,
cross-owner reductions, arithmetic mode, number of sources, sampling, symmetry
proof, and validation. A fast BFS frontier kernel alone is not a betweenness
measurement.

## Sources

- U. Brandes,
  [*A Faster Algorithm for Betweenness Centrality*](https://doi.org/10.1080/0022250X.2001.9990249),
  Journal of Mathematical Sociology 25(2), 2001. Gives the predecessor-count and
  reverse dependency accumulation framework.
- L. C. Freeman,
  [*Centrality in Social Networks: Conceptual Clarification*](https://doi.org/10.1016/0378-8733%2878%2990021-7),
  Social Networks 1(3), 1978/79. Establishes classical centrality conventions.
- Notes 11, 30, 53, 57, 69, 89, 90, and 120 provide this repository's
  shortest-path DAG, replay, sampling, finalization, product-count, dominator,
  disjointness, and distance-sum contracts.

## Takeaway

Betweenness is not obtained by choosing one BFS tree. It needs the complete
shortest-path DAG semantics: distances and counts forward, dependencies
backward. Summed over vertices, its fractional mass is exactly Wiener distance
minus one per endpoint pair. Cayley vertex transitivity then turns one complete
identity distance histogram into every vertex's scalar betweenness, but only
under a declared pair, endpoint, normalization, and graph-automorphism contract.

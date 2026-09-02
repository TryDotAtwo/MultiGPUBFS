# BFS, spanners, emulators, hopsets, and generator substitution

Sparse or shortcut graph structures can preserve distances approximately while
changing the number of stored edges or hops. They are useful abstractions, but
running BFS on them is not automatically exact BFS on the original graph.

Note 81 already introduced all-pairs tree stretch and multiplicative spanners.
This note focuses on spanner/emulator/hopset distinctions and Cayley generator
substitution. It adds no construction or optimization code.

## 1. Multiplicative and additive spanners

For a graph `G=(V,E)`, a spanning subgraph `H=(V,E_H)` is a multiplicative
`t`-spanner when

```text
d_G(u,v) <= d_H(u,v) <= t * d_G(u,v)
```

for every pair. The lower bound is automatic because `H` only removes edges.
An additive `beta`-spanner instead satisfies

```text
d_H(u,v) <= d_G(u,v) + beta,
```

and mixed `(alpha,beta)` guarantees use `alpha*d_G+beta`.

The guarantee is all-pairs unless explicitly restricted. Root-exactness of a
BFS tree is not such a guarantee: an omitted edge of a long cycle can have tree
distance nearly the whole cycle length.

## 2. What exact BFS on a spanner returns

Exact BFS on an unweighted spanner computes `d_H`, not `d_G`. It therefore gives
an upper bound on original distance and a valid original-graph path, because
every spanner edge is an original edge.

Approximate distance does not certify exact distance. If BFS on `H` returns
`U`, the spanner guarantee yields a range for `d_G`; exactness follows only when
independent lower and upper bounds collapse to the same integer.

A sparse spanner can also increase diameter and the number of BFS rounds. In
`K_n`, a center-star is a two-spanner with only `n-1` edges. From the center its
first frontier still has `n-1` vertices; from a leaf, the other leaves move to
depth two. Edge sparsity, peak frontier, and depth are separate quantities.

## 3. Emulators

An emulator is a graph on the same vertex set whose distances approximate those
of `G`, but it need not be a subgraph and may contain weighted virtual edges.
The declared guarantee should include a no-underestimation side such as

```text
d_G(u,v) <= d_H(u,v) <= alpha*d_G(u,v) + beta.
```

An emulator edge is not automatically an original graph transition. Returning
an original path requires an unpacking witness for every virtual edge. Without
that witness, an emulator distance is a numeric estimate rather than a
replay-valid puzzle move sequence.

Because emulators are commonly weighted, ordinary FIFO BFS is generally the
wrong distance engine. Replacing a weight-`w` virtual edge by one unit hop changes
the promised metric.

## 4. Hopsets

A `(hopbound,epsilon)` hopset adds weighted shortcut edges to `G` so that every
pair has a path using at most `hopbound` edges whose weight is within a factor
`1+epsilon` of the original shortest distance. Formally, the bounded-hop
distance in the augmented graph satisfies the declared approximation bounds.

Hop count and metric length are different. One shortcut edge may represent a
long original path and count as one hop while carrying a large weight. A small
hopbound can reduce the dependency depth of a relaxation algorithm without
reducing the original unweighted BFS distance or frontier count.

A hopset is therefore not just a denser BFS graph. Its semantics include edge
weights, a bounded-hop restriction, and usually approximation. Ignoring any one
of those changes the problem.

## 5. Spanners, emulators, and hopsets preserve different witnesses

- A spanner path is already a path in `G`, but may be longer than shortest.
- An emulator path may use virtual edges and needs unpacking into `G`.
- A hopset path uses original and shortcut edges and is evaluated by both weight
  and hop count.
- A BFS parent tree is exact only from its root and need not have bounded
  all-pairs stretch.

None of these structures automatically preserves the number of shortest paths,
the original predecessor DAG, canonical parents, BFS layer sizes, or duplicate
arrival patterns.

## 6. Cayley generator substitution theorem

Let `S` be a finite symmetric generating set and let `S'` be a symmetric subset
that still generates the same group. Then the Cayley graph for `S'` is a spanning
subgraph of the Cayley graph for `S`, so

```text
d_S(g,h) <= d_S'(g,h).
```

Suppose every generator `s` in `S` can be represented by an `S'`-word of length
at most `L`. Replace every letter of an `S`-word by its representing `S'`-word.
For every pair,

```text
d_S'(g,h) <= L * d_S(g,h).
```

Therefore `Cay(G,S')` is an `L`-spanner of `Cay(G,S)`. It is enough to certify
bounded replacement words for generators because any path is a concatenation of
generator edges.

If `S'` fails to generate the group, the subgraph is disconnected and has no
finite all-pairs stretch. A finite-radius BFS observation cannot replace the
global generator-word proof.

## 7. Generator removal changes more than edge count

Removing generators can:

- increase word distances and diameter;
- split one original BFS layer across several new layers;
- reduce degree and candidate occurrences;
- change shortest-word multiplicity and relation onset;
- increase the number of synchronization rounds even if each round is cheaper.

The maximum replacement length `L` is a worst-case multiplicative guarantee,
not a prediction that every pair stretches by `L`. Distribution of actual
stretch and frontier evolution require separate evidence.

For directed positive alphabets, every deleted generator needs a directed word
over the retained alphabet. An inverse-containing group identity is not a valid
directed replacement if inverse moves are unavailable.

## 8. Cayley versus Schreier lifting

An algebraic group-word identity gives a globally valid replacement under every
action. A path equality observed only at the identity state of a nonfree action
may instead rely on a stabilizer and need not represent equality of group
elements.

Thus a Schreier spanner claim needs either:

- genuine group-word substitutions, which act identically on all states; or
- complete action-specific evidence that every removed transition has a bounded
  retained-word replacement from every relevant state.

State-local shortcuts without such evidence are not universal generator
substitutions.

## 9. Exact-search certification boundary

Suppose an approximate structure finds a target path of original length `U`.
This is a valid upper bound if the path is replayable. It proves optimality only
when another argument gives `d_G>=U`.

Examples of possible lower-bound evidence include a completed exact BFS layer,
an admissible heuristic bound, or an independently validated distance
certificate. The approximation factor alone usually leaves multiple integer
distances possible.

Therefore approximate structures can guide or bound exact search, but their
names do not weaken the final exactness proof obligation.

## 10. GPU and multi-GPU boundary

Spanners may reduce stored edges while increasing rounds. Hopsets may reduce
relaxation depth while adding long-range weighted edges that increase routing or
replication. Emulators may compress adjacency but require virtual-edge unpacking.

Report separately:

- preprocessing/construction and validation cost;
- retained, virtual, or shortcut edge count and memory;
- approximation and hopbound parameters;
- weighted relaxation rounds versus original BFS levels;
- distributed communication created by long-range shortcuts;
- path-unpacking and replay validation;
- exact frontier/visited throughput on the original graph.

A reduced number of rounds is not automatically less total work, and an
approximate distance kernel is not an exact BFS implementation.

## Sources

- I. Althofer, G. Das, D. Dobkin, D. Joseph, and J. Soares,
  [*On Sparse Spanners of Weighted Graphs*](https://doi.org/10.1007/BF02189308),
  Discrete and Computational Geometry 9, 1993. Classical sparse multiplicative
  spanner guarantees.
- D. Peleg and A. A. Schaffer,
  [*Graph Spanners*](https://doi.org/10.1002/jgt.3190130114), Journal of Graph
  Theory 13, 1989. Foundational all-pairs spanner formulation used in note 81.
- E. Cohen,
  [*Polylog-Time and Near-Linear Work Approximation Scheme for Undirected
  Shortest Paths*](https://doi.org/10.1145/331605.331610), Journal of the ACM 47,
  2000. Hopsets and parallel approximate shortest paths.


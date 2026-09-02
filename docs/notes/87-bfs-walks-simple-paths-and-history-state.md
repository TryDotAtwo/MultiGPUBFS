# BFS walks, simple paths, and history state

Ordinary BFS finds shortest paths without storing the whole path history in its
visited key. That economy is not a generic property of path problems. It works
because cycles can be removed from a shortest unweighted walk.

When the query requires a simple path of an exact or long length, the set of
already used vertices affects future legality. Equal endpoint states can no
longer be merged safely.

## 1. Why shortest paths are automatically simple

Take any shortest directed or undirected unit-edge walk from `s` to `t`. If it
repeats a vertex, the segment between two occurrences is a nonempty closed
walk. Removing that segment leaves a shorter valid `s-t` walk, contradicting
shortestness.

Therefore every shortest walk is a simple path. This justifies ordinary BFS
state identity:

- reaching vertex `v` later cannot improve its distance;
- a shortest continuation depends only on `v` in a static memoryless graph;
- one global visited decision per vertex is exact for distance/reachability.

The argument proves existence of a simple shortest path. It does not identify
all longer simple paths or make all path-counting conventions equivalent.

## 2. Exact-length walks remain easy to propagate

For exact-length **walk** support, note 86 used

```text
W_0={s},
W_(i+1)=Post(W_i).
```

Repeated vertices and edges are legal. Boolean propagation for `k` steps
answers whether a walk of length exactly `k` reaches each endpoint. Arithmetic
adjacency powers can count walks under the declared edge-multiplicity model.

No per-walk used set is needed because history does not restrict the next arc.
The same vertex can be merged within one step's Boolean support.

## 3. Exact-length simple paths need history

A simple path may not repeat a vertex. After reaching `v`, the legal next
vertices depend on the used set `U`. The exact Markov state is therefore at
least

```text
(v,U),  with v in U.
```

A transition along `v->x` is legal only when `x not in U`, producing
`(x,U union {x})`.

Merging records solely by endpoint `v` forgets which successors remain legal.
This is the same product-state principle as note 20, with an exponentially
large history automaton whose state is a vertex subset.

## 4. Minimal endpoint-merging counterexample

Use directed vertices `s,a,b,v,t` and arcs

```text
s->a, a->v,
s->b, b->v,
v->b, b->t.
```

Two length-two histories reach `v`:

```text
H_a: s->a->v,  used={s,a,v}
H_b: s->b->v,  used={s,b,v}.
```

Continuation `v->b->t` is legal after `H_a` and yields the simple length-four
path

```text
s->a->v->b->t.
```

The same continuation is illegal after `H_b` because it repeats `b`. A visited
key containing only `v` can retain the wrong history and discard the right one.

Ordinary reachability is not endangered: the graph also has a shorter path
`s->b->t`, and any reachable target has some simple shortest path. What fails
is the **exact-length simple-path** output contract.

## 5. The expanded state space

An exact but direct formulation explores the product graph of states `(v,U)`.
For `n` base vertices there can be up to

```text
n 2^(n-1)
```

such endpoint/subset states before length bounds and reachability restrictions.
If only paths of length at most `k` matter, only subsets of size at most `k+1`
are relevant, but their number remains combinatorial.

BFS on this product graph is still BFS: every transition appends one edge, and
layer `i` represents simple histories of length `i`. The exponential object is
the declared state space, not a failure of FIFO layer ordering.

Specialized parameterized methods such as color-coding can search for a
`k`-path without enumerating every subset explicitly. Their existence does not
restore the correctness of a base-vertex visited bitmap.

## 6. Complexity boundary

When requested length is part of the input, general simple-path existence
contains Hamiltonian Path: on an `n`-vertex graph, a simple path of length
`n-1` visits every vertex exactly once. The decision problem is in NP because a
candidate vertex sequence is readily checked, and Hamiltonian variants are
NP-complete.

By contrast, ordinary unweighted shortest path and fixed-step walk support have
polynomial graph/dynamic-programming formulations. This is a problem-contract
change, not a small BFS implementation option.

For fixed or parameterized `k`, stronger algorithms and complexity bounds are
possible. The NP-completeness statement should not be misread as saying every
small-`k` instance requires exhaustive `2^n` search.

## 7. Nonbacktracking is weaker than simple

Forbidding only the immediate inverse of the previous edge remembers one move
and prevents length-two spurs. A nonbacktracking walk may still return to a
vertex after a longer cycle. Therefore

```text
simple path => nonbacktracking path
```

in a loopless undirected graph, but the converse fails whenever a longer cycle
can be traversed.

Likewise, a freely reduced Cayley word can revisit a group element through a
nontrivial relator. Word reduction, state simplicity, and shortestness are
distinct filters.

## 8. Cayley interpretation

Every geodesic word from the identity traces a simple state path: if two word
prefixes represented the same group element, deleting the intervening identity
subword would shorten the endpoint word.

Longer words representing the same element may revisit states, especially when
identity relations are inserted. Thus the eventual word-length spectra of note
86 concern walks/words, not self-avoiding state paths.

For a Schreier action, distinct group prefixes can already collapse to the same
state through the stabilizer. Simplicity must be defined in the declared state
graph, not inferred from distinct syntactic word prefixes.

## 9. GPU and distributed visited semantics

A vertex-keyed bitmap or hash table is ideal only when vertex identity is the
full semantic state. For exact simple-path histories:

- two records with the same endpoint and different used sets are not duplicates;
- owner hashing only the endpoint sends semantically distinct states together
  but does not authorize merging them;
- deduplicating generator words by endpoint destroys history-dependent legality;
- frontier size can reflect path-history multiplicity rather than unique base
  vertices.

Representing or compressing used sets is a separate algorithmic design problem.
No GPU representation is proposed here.

## 10. Evidence checklist

1. Walk, trail, nonbacktracking walk, or simple path.
2. Shortest, exact length, at most length, or longest query.
3. Base vertex versus `(vertex,history)` semantic identity.
4. Which history features change successor legality.
5. Whether repeated states, edges, or labels are forbidden.
6. Cayley word equality versus state-path simplicity.
7. Length encoding and any fixed/parameterized-`k` scope.
8. Visited/dedup key and proof that it preserves all required continuations.

## Sources

- R. M. Karp,
  [*Reducibility Among Combinatorial Problems*](https://doi.org/10.1007/978-1-4684-2001-2_9),
  Complexity of Computer Computations (1972), 85-103. Establishes foundational
  NP-completeness results including directed and undirected Hamiltonian cycle,
  from which standard Hamiltonian path reductions follow.
- N. Alon, R. Yuster, and U. Zwick,
  [*Color-Coding*](https://web.math.princeton.edu/~nalon/PDFS/colpr.pdf),
  Journal of the ACM 42 (1995), 844-856. Develops parameterized randomized and
  derandomized methods for simple paths and cycles of specified length.
- Notes 11, 20, 23, 33, 39, 53, 64, and 86 supply shortest-path, product-state,
  path-tree, walk-power, nonbacktracking, path-count, word-record, and
  exact-length distinctions used here.

## Takeaway

BFS can merge all arrivals at one vertex because a shortest walk never needs a
cycle. Exact-length walks also merge by endpoint because revisits are legal.
Exact-length simple paths cannot: future legality depends on the used-vertex
set, so `(v,U)` rather than `v` is the semantic state. This history expansion,
not queue ordering, creates the fundamental complexity gap.

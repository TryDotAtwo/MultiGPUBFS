# Batch incremental BFS: endpoint metric closure and its limits

## Question

For a finite batch of inserted unit arcs, can the updated source distances be
described exactly using the old graph without pretending that independent
one-edge sensitivities compose?

Note 187 shows that they do not compose: a shortest path can chain several new
arcs. The missing operation is closure over alternating old-graph subpaths and
inserted arcs. This note derives that closure and separates its distance
semantics from path counting and execution cost.

No dynamic algorithm implementation or performance result is claimed.

## 1. Path decomposition by inserted arcs

Let `G=(V,E)` be a finite directed unit-edge graph, source `s`, and let

```text
F = {e_i=(a_i,b_i)}
```

be a finite set of new semantic arcs. Every path in `G+F` has a unique ordered
sequence of occurrences from `F`. Between consecutive new arcs it follows only
old edges from `G`.

For a sequence

```text
e_(i_1), e_(i_2), ..., e_(i_k),
```

the best path with exactly that inserted-edge sequence has length

```text
d_G(s,a_(i_1)) + 1
+ d_G(b_(i_1),a_(i_2)) + 1
+ ...
+ d_G(b_(i_(k-1)),a_(i_k)) + 1
+ d_G(b_(i_k),v).
```

The `k=0` case is the old distance `d_G(s,v)`. Therefore the updated distance
is the minimum over all finite inserted-edge sequences.

With positive unit costs, a shortest path is simple and cannot repeat the same
inserted edge occurrence. Hence it is sufficient to consider `k<=|F|`, though
enumerating those sequences directly is not a performance prescription.

## 2. Bounded-use min-plus recurrence

Let `D_r(v)` be the shortest length among paths using at most `r` inserted
arcs. Initialize

```text
D_0(v) = d_G(s,v).
```

Then

```text
D_(r+1)(v) = min(
    D_r(v),
    min_((a,b) in F) [D_r(a)+1+d_G(b,v)]
).
```

Induction on `r` proves that `D_r` has the stated at-most-`r` meaning: a path
either already uses at most `r` inserted arcs or ends with one new arc after an
at-most-`r` prefix and an old-only suffix.

Because a shortest path needs at most `|F|` new edge occurrences,

```text
D_|F|(v) = d_(G+F)(s,v).
```

This is a min-plus fixed-point closure. Its round number counts inserted arcs
used, not original graph hops or BFS layers.

## 3. Endpoint metric graph

Let the terminal set contain

```text
T = {s} union {a : (a,b) in F} union {b : (a,b) in F}.
```

Construct a conceptual weighted graph `M_F` on `T` with:

- an old-metric arc `x -> y` of weight `d_G(x,y)` whenever finite;
- every inserted arc `a -> b` with weight one.

For scalar distance, shortest paths in `M_F` exactly represent alternation
between optimal old segments and new arcs. If `delta_F(s,t)` is its shortest
distance, then for any original target `v`,

```text
d_(G+F)(s,v)
  = min(d_G(s,v), min_(t in T) [delta_F(s,t)+d_G(t,v)]).
```

The old candidate is redundant if `s in T` and `d_G(s,v)` is included, but
writing it explicitly keeps the no-insertion case visible.

This is an exact semantic compression only when every required old terminal
distance is exact. Building or storing those rows may cost more than a direct
updated traversal; the theorem does not choose between them.

## 4. Why one-edge minima fail

The independent one-edge approximation restricts the sequence length to
`k<=1`. The isolated-vertex fixture from note 187 makes the gap exact:

```text
old graph: vertices s,x,t and no edges
insertions: (s,x), (x,t)
```

Every zero- or one-insertion candidate for `t` is infinite, while the two-edge
sequence has length two. The missing object is not another local comparison;
it is closure over interactions between new arcs.

## 5. Strict and equal fixed-point proposals

At every recurrence round, a proposal can be:

- strictly smaller than the current best label;
- equal to it, adding another shortest decomposition;
- larger and irrelevant for shortest distance.

Min-plus scalar closure is idempotent under repeated equal proposals. Richer
outputs are not:

- a predecessor DAG retains appropriate equal proposals;
- path counts add distinct semantic paths;
- a canonical word retains the minimum under its declared order;
- retries must not duplicate a non-idempotent contribution.

Thus convergence of the scalar vector `D_r` does not by itself close DAG,
count, or canonical-output obligations.

## 6. Endpoint compression is unsafe for naive path counting

The endpoint metric graph is exact for distances because `min` ignores
duplicate decompositions of the same old subpath. It is not automatically an
exact path-count graph.

Suppose an old shortest path from terminal `x` to terminal `z` passes through
another terminal `y`. The same concrete old path can appear in `M_F` as:

```text
x -> z
```

or as

```text
x -> y -> z.
```

Summing metric-graph paths double-counts it even though no inserted edge
distinguishes the two representations. Additional intermediate terminals create
more segmentations.

Exact counting must use the unique segmentation at inserted-edge occurrences,
or another proved canonical decomposition. Old segment multiplicities can then
be multiplied within one inserted-edge sequence and distinct semantic
sequences added, while collision of concrete paths is ruled out by their
unique new-edge occurrence sequence.

This is an algebraic boundary:

```text
min-plus distance tolerates redundant metric decompositions;
additive path counting does not.
```

## 7. Atomic batch versus sequential visibility

The final exact distance in the named graph `G+F` is independent of the order
in which a correct procedure relaxes proposals. Intermediate states are not.

If queries observe after each insertion, the service exposes a sequence of
graph versions. If the batch is atomic, only the old and fully closed new
versions are valid query epochs. Publishing `D_r` for `r<|F|` as the batch
answer can miss paths using more new arcs, even when no currently queued local
proposal looks smaller on one owner.

Canonical parents and words must also be finalized against the complete batch,
not the accidental arrival order of its inserted edges.

## 8. Undirected batches

An inserted undirected edge contributes two oriented unit arcs sharing one
semantic edge identity. Scalar endpoint closure can include both orientations.
A simple shortest path will not traverse the same undirected edge twice.

For richer outputs, treating the two orientations as independent physical
edges can double-count. The path contract must distinguish:

- one undirected edge used in one direction on a path;
- two directed arc records implementing that edge;
- genuinely parallel semantic edges with equal endpoints.

## 9. Cayley and Schreier generator families

On a finite action graph, adding a generator family can formally be represented
by a large batch `F` of translated arcs, so the closure theorem still holds.
But this expansion hides the algebraic structure and may contain one new arc
per state occurrence.

A shortest word may repeat the new generator label while using different
translated edge occurrences. The bound `k<=|F|` concerns concrete edges on a
simple path, not distinct generator labels. Therefore a label-level rule that
allows the new generator only once is generally false.

Any algebraic compression of the endpoint closure needs its own proof of path
lifting and exact identity. Group symmetry does not automatically preserve
owner locality, path counts, or canonical words.

## 10. GPU and multi-GPU meaning

The recurrence exposes several distinct work coordinates:

- number of inserted-edge-use closure rounds;
- number of strict and equal proposals;
- old-distance rows or on-demand old segment searches;
- proposal routing by tail/head ownership;
- duplicate decompositions harmless to `min` but harmful to addition;
- global closure of all lower candidate labels;
- separate closure of DAG/count/canonical metadata.

An implementation may realize the same fixed point by queues, buckets,
frontiers, sparse algebra, or recomputation. This note licenses none as
universally best. A GPU kernel timing for one proposal round is not batch
completion time, and endpoint closure preprocessing is part of the workload.

For exact distributed publication, a rank cannot declare completion merely
because its labels stopped changing: another rank may still hold a sequence
using additional inserted arcs that yields a lower or equal authoritative
proposal.

## 11. Counterclaims rejected

- **Take the minimum of all independent one-edge formulas.** This omits paths
  using two or more new arcs.
- **Closure round equals BFS depth.** It counts inserted arcs, while old metric
  segments can contain many hops.
- **The endpoint metric graph preserves path counts because it preserves
  distance.** Redundant subdivision at intermediate terminals overcounts old
  paths.
- **Scalar convergence finalizes every output.** Equal proposals can still
  change DAGs, counts, or canonical choices.
- **A new generator can occur at most once because it is one batch item.** The
  family contains many translated edge occurrences and a shortest word may
  repeat the label.

## Sources and dependencies

- Note 22 defines atomic/sequential dynamic graph versions.
- Notes 25 and 172 provide least-fixed-point and merge-algebra distinctions.
- Note 57 separates scalar, DAG, count, and canonical finalization.
- Note 187 gives the exact one-edge formula and chaining counterexample.
- The batch recurrence and endpoint metric equivalence are direct finite path
  decompositions under positive unit edge costs.

## Compact conclusion

A batch of inserted edges is exactly a min-plus closure over paths alternating
old metric segments with new unit arcs. Independent one-edge sensitivity is
only its first round. Endpoint metric compression preserves scalar distance but
can duplicate representations of old paths, so it cannot be reused for counts
without a canonical segmentation proof.

# BFS completeness and termination on finite and infinite graphs

The sentence "BFS is complete" is ambiguous.  At least three claims can be
intended:

1. **solution completeness:** if a target is reachable, BFS eventually finds
   one;
2. **enumeration completeness:** every reachable vertex is eventually emitted;
3. **decision termination:** BFS eventually answers either reachable or
   unreachable.

They coincide for a finite, effectively represented graph with sufficient
memory.  They separate on infinite or incompletely enumerable state spaces.

## The assumptions hidden in textbook BFS

For an executable graph search, mathematical adjacency is not enough.  Assume:

- the source and target predicates are computable;
- exact state equality is decidable;
- expanding any reached vertex enumerates all of its successors;
- each individual successor computation terminates;
- the schedule is fair among finite-depth work;
- no required state is silently lost to capacity limits.

On a finite explicit graph these are often taken for granted.  On an implicit
graph they are part of the algorithmic contract.  A nonterminating move
generator, undecidable equality, or infinite adjacency enumeration can invalidate
"BFS completeness" before queue order becomes relevant.

## Finite reachable component

If the component reachable from the source has finitely many vertices and every
adjacency enumeration terminates, exact graph-search BFS terminates:

- each vertex is accepted at most once;
- each accepted vertex is eventually expanded;
- after the last reachable vertex is expanded, no new state is produced;
- the next frontier is empty.

The entire ambient graph need not be finite.  A finite reachable component is
enough.  Conversely, knowing that a state encoding has finitely many bits proves
a finite *syntactic* universe only if every execution stays inside that fixed
encoding and equality is exact.

Termination here is mathematical.  A real implementation also needs enough
frontier, visited, metadata, and scratch capacity or an exact external-memory
mechanism.  Silent truncation is not termination with a smaller answer.

## Locally finite infinite graphs

A graph is locally finite when every vertex has finitely many relevant
successors (and, for an undirected notion, finite degree).  Starting from a
finite source set, every finite-radius ball is finite.  The proof is induction:

```text
B_0 is finite.
If B_d is finite and every vertex has finite successor set,
then the finite union of those successor sets is finite,
so B_(d+1) is finite.
```

No global maximum branching factor is required for this conclusion.  Degrees
may grow without bound from layer to layer; each individual finite ball remains
finite.

### Reachable target

If a target has finite distance `D`, BFS processes only the finite balls through
depth `D`.  Under terminating expansion and a fair level schedule, it therefore
finds the target after finite work.  This proves solution completeness on a
locally finite infinite graph.

The usual geometric estimate

```text
1 + b + b^2 + ... + b^D
```

requires a uniform branching bound `b`.  Local finiteness proves eventual
discovery without supplying such a uniform performance bound.

### Full enumeration

If the reachable component is infinite, full BFS never terminates.  It can still
be enumeration-complete: every particular reachable vertex lies at some finite
distance and is eventually discovered.  At every finite time, however, only a
finite prefix of the component has been enumerated.

### Unreachable target

An unreachable target need not produce a finite answer.  Consider the infinite
ray

```text
0 -> 1 -> 2 -> 3 -> ...
```

and a target `z` outside it.  Every frontier contains the next integer, so BFS
never finds `z` and never reaches an empty frontier.  The same algorithm is a
semi-decision procedure: it halts on reachable instances but may run forever on
unreachable ones.

An unreachable answer requires extra finite evidence, such as:

- exhausted finite reachable component;
- a proved depth/state bound;
- a separating invariant or abstraction;
- a finite quotient whose soundness is established;
- a domain-specific impossibility certificate.

## Infinite branching is a different failure mode

If a vertex has infinitely many successors, even a finite-depth ball may be
infinite.  A strict level-synchronous algorithm may never finish depth one and
therefore never begin depth two.

For example, let the root have children `c_0,c_1,...`, and let `c_0` have a
target child at depth two.  If BFS insists on completing the root's infinite
successor enumeration before advancing the layer, the depth-two target is
starved forever.

A streaming or dovetailing schedule can interleave work from countably many
enumerators, but then:

- the physical discovered set is not a completed finite metric ball;
- ordinary layer-boundary termination proofs no longer apply directly;
- fairness must be stated over generators and depths;
- memory and duplicate behavior may be qualitatively different.

Thus "the target is at finite depth" alone is insufficient.  Classical BFS
solution completeness needs finite branching, or a stronger effective/fair
enumeration argument replacing it.

## Graph search versus tree search

**Graph-search BFS** uses exact visited identity and expands each reachable
vertex once.  On a finite reachable component it terminates and can decide
unreachability.

**Tree-search BFS** treats every path occurrence as a new node.  With finite
branching it still reaches every finite path depth after finite work, so it can
find a reachable finite-depth target.  But cycles generate infinitely many path
occurrences.  Even a finite cyclic graph then produces an infinite search tree,
so exhaustive no-path termination is lost and memory/work can grow
exponentially.

Visited is therefore not required for the abstract shallowest-path ordering of
a finitely branching search tree.  It is required to turn repeated state
occurrences into finite graph traversal and to obtain finite-component
exhaustion.

## Fairness and ordering

For ordinary FIFO BFS with finite layers, fairness is automatic: every queued
item has finitely many items ahead of it.  Parallel and distributed versions
must reconstruct this property.

Potential fairness failures include:

- repeatedly servicing newly arrived work while an old partition starves;
- an infinite or nonterminating generator occupying a worker indefinitely;
- retry traffic preventing an owner from committing a finite-depth candidate;
- local progress without global progress because one rank never completes its
  epoch;
- bounded queues dropping or perpetually deferring one source.

A claim that "all finite-depth vertices are eventually processed" is a liveness
property, separate from the safety property that every assigned depth is
correct.

## Memory is part of operational completeness

Mathematical BFS can require storage proportional to a large frontier plus
visited history.  On a tree with branching `b`, the final frontier before a
depth-`D` target can dominate all earlier storage.  On convergent graphs,
visited may dominate frontier.  There is no fixed finite-memory implementation
that can retain arbitrary exact BFS balls of unbounded size.

Several responses change a different axis:

- iterative deepening trades repeated work for small path-stack memory;
- external-memory BFS changes the storage hierarchy;
- bidirectional BFS may reduce explored balls for a known target;
- canonicalization or quotienting changes identity and needs a proof;
- beam pruning bounds memory by giving up exact completeness/optimality.

An out-of-memory result is not evidence that the target is unreachable.  It is
an inconclusive execution unless the algorithm supplies a sound alternative
certificate.

## Finite and infinite Cayley graphs

For a finite move/generator collection `S`, every Cayley vertex has finitely many
outgoing generator occurrences.  Therefore every directed reachable ball is
finite, even when the group itself is infinite:

```text
|B_d| <= 1 + |S| + |S|^2 + ... + |S|^d
```

(relations usually make the inequality strict).

Consequences:

- in a finite group, exact BFS from the identity eventually exhausts the group;
- in an infinite group with a finite symmetric generating collection (or with
  both generators and inverses allowed), BFS eventually finds every group
  element, but full enumeration never ends;
- with directed positive generators only, BFS enumerates the reachable monoid;
  elements requiring unavailable inverse moves need not be reachable;
- failure to find an element so far is not a non-membership proof;
- finite generation gives finite branching, not necessarily practical growth.

### Equality is an algorithmic assumption

An abstract finite group presentation supplies finitely many generators and
relations, but it does not automatically supply an algorithm deciding whether
two generator words denote the same element.  The Novikov–Boone theorem shows
that finitely presented groups with undecidable word problem exist.

For such a presentation, the abstract Cayley graph is locally finite, yet a
generic exact visited operation on word representatives is unavailable.  This
separates two ideas that are easy to conflate:

- the mathematical graph has finite degree;
- the graph is effectively traversable with decidable vertex identity.

Puzzle groups avoid this particular obstacle when states have a concrete finite
representation—permutations/orientations with decidable equality—and moves act
computably on it.  The point is not that Cayley BFS is generally undecidable,
but that effective equality must come from the domain representation, not from
the word "Cayley."

## König's infinity lemma intuition

König's infinity lemma states, in one common form, that an infinite rooted tree
with finite branching contains an infinite ray.  Applied to the tree of finite
search paths, it explains why a locally finite reachable structure that never
exhausts can support arbitrarily deep paths rather than hiding infinitely many
vertices inside one completed finite layer.

The lemma is not needed for the elementary finite-ball induction above, but it
clarifies the geometry: under finite branching, infinity is encountered by
unbounded depth, not by an unfinishable finite layer.

## Termination in distributed BFS

For a finite graph, a distributed rank may have no local frontier while other
ranks still have:

- frontier states;
- generated candidates in send buffers;
- network messages in flight;
- owner-side visited decisions pending;
- accepted states not yet included in a global count.

Global termination requires a consistent statement that all such work is empty
or completed.  Bulk-synchronous BFS obtains this at a level boundary through a
collective/equivalent agreement.  An asynchronous system needs a distributed
termination-detection argument.  Local idleness is not graph exhaustion.

On an infinite reachable graph, a correct termination detector should never
announce full traversal completion.  It may still terminate a finite-depth or
target query when that query's proof conditions are met.

## Four outcomes that logs should distinguish

```text
FOUND        target reached with a shortest-distance/path certificate
EXHAUSTED    reachable component proved finite and fully traversed; target absent
BOUNDED      declared depth/state/resource boundary completed without target
INCOMPLETE   timeout, cancellation, OOM, overflow, dropped work, or unknown state
```

`BOUNDED` is not `EXHAUSTED` unless the bound is independently proved to cover
the entire reachable component.  `INCOMPLETE` must never be reported as
unreachable.

## Audit checklist

1. Is the reachable component finite, merely locally finite, or possibly
   infinitely branching?
2. Does every successor enumeration terminate and return all successors?
3. Is exact state equality decidable and implemented?
4. Are all finite-depth work items scheduled fairly?
5. Does the query ask for one target, full enumeration, or an unreachable
   decision?
6. What finite certificate justifies `EXHAUSTED`?
7. Are capacity and time limits semantic bounds or execution failures?
8. In distributed execution, what work-in-flight state participates in
   termination detection?
9. For a Cayley presentation, what concrete normal form/state representation
   makes equality effective?
10. Does a symmetry quotient preserve the query whose termination is claimed?

## Sources

- David Poole and Alan Mackworth,
  *Artificial Intelligence: Foundations of Computational Agents*, 3rd ed.,
  [search-space pruning section](https://www.cs.ubc.ca/~poole/aibook/3e/html/ArtInt3e.Ch3.S7.html),
  for the finite-branching condition behind breadth-first solution completeness.
- Dénes König, *Theory of Finite and Infinite Graphs*,
  [doi:10.1007/978-1-4684-8971-2](https://doi.org/10.1007/978-1-4684-8971-2),
  for the infinity lemma and locally finite graph setting.
- Shimon Even, *Graph Algorithms*, 2nd ed., section on the infinity lemma; an
  accessible statement is also present in
  [Wilson's Introduction to Graph Theory notes](https://sites.math.rutgers.edu/~zeilberg/akherim/wilsongraph.pdf).
- P. S. Novikov and William Boone's independent undecidability results are
  summarized with historical references in the
  [Princeton group-theory chapter](https://assets.press.princeton.edu/chapters/s7903.pdf).
- Local notes 03 and 06 supply the metric-ball and Cayley word-metric models to
  which these termination distinctions are applied.

## Current synthesis

BFS is solution-complete on an effectively presented locally finite graph: a
reachable target has finite depth, every finite ball finishes, and level order
eventually reaches it.  Full traversal terminates only when the reachable
component is finite.  An unreachable answer needs finite exhaustion or another
certificate.  Infinite branching, undecidable equality, unfair scheduling, and
resource truncation are separate reasons why a mathematical shortest-path
statement may fail to become an executable terminating algorithm.

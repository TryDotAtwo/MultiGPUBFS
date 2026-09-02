# BFS order, canonical shortest paths, shortlex, and LexBFS

BFS distances are canonical; a BFS traversal order and parent tree usually are
not. Several different requests are often compressed into "deterministic BFS":

1. deterministic distance/frontier sets;
2. deterministic order of vertices inside each frontier;
3. one deterministic shortest parent per vertex;
4. a lexicographically least shortest vertex path;
5. a shortlex-least move word in a labeled/Cayley graph.

Each stronger output needs an additional ordering contract. None follows from
the scalar BFS theorem alone.

## What ordinary BFS leaves unspecified

If two depth-`d` vertices both reach `v`, either is a valid parent. Reordering
same-level work preserves

```text
distance[v] = d+1
```

but may change:

- which parent wins;
- the order in which `F_(d+1)` is stored;
- which target candidate triggers early stop;
- the move string reconstructed later;
- locality, duplicate timing, and owner traffic.

Thus nondeterministic parents can coexist with fully deterministic mathematical
frontier sets.

## Shortest is not lexicographically least

Suppose edge labels come from a totally ordered alphabet. Ordinary BFS minimizes
word length. Among equal-length words, it may choose any first witness.

**Shortlex order** compares

1. length first;
2. lexicographic label order among equal lengths.

This matches "shortest path, then lexicographically least shortest path."
Pure lexicographic order without the length priority is a different objective:
an arbitrarily long word beginning with `a` may precede a one-edge word
beginning with `b`.

## When ordered FIFO BFS yields shortlex parents

Assume:

- one source;
- unit edges;
- a total order on outgoing edge labels/occurrences;
- the FIFO frontier is expanded in increasing canonical-word order;
- each parent's outgoing edges are generated in increasing label order;
- the first occurrence of a state wins exact deduplication.

Then first discovery assigns the shortlex-least word for every reached state.

Induction by depth explains why. At depth `d`, canonical parent words all have
equal length and are processed lexicographically. Every extension of an earlier
parent word precedes every extension of a later parent word; within one parent,
sorted labels order its extensions. Candidate words are therefore generated in
lexicographic order, and the first occurrence of each child is its least word
of length `d+1`.

The proof is about **path-word order**, not state-key order.

## Sorting a frontier by state ID is not enough

Let state key `a < b`, but let the canonical source-to-parent move words be

```text
word(a) = z
word(b) = a.
```

Suppose both parents reach child `v`, with labels

```text
a --a--> v    giving word z a
b --z--> v    giving word a z.
```

State-sorted expansion visits parent `a` first and may select `za`, while
shortlex requires `az`. Likewise, choosing the minimum parent state ID is not a
canonical-word rule.

To select a shortlex parent independently of execution order, compare something
equivalent to

```text
(canonical_word_rank(parent), edge_label, further_ties).
```

The parent rank must itself encode canonical path order, not merely vertex ID.

## Deterministic parent is still a valid separate contract

One may intentionally define

```text
parent[v] = minimum (parent_state_key, move_id)
```

over all depth-`d` predecessors. This is deterministic and shortest-depth
consistent. It simply need not be lexicographically least by full move word.

Such a reduction generally waits until all same-level proposals for `v` are
known, or uses an atomic/owner minimum under a total tuple order. "First GPU
winner" is not that reduction.

## Not every shortest-path tree is a FIFO BFS tree

Parent validity can be coupled across vertices. Let source `s` reach depth-one
vertices `u,v`, and let both `u` and `v` connect to both depth-two vertices
`x,y`.

The proposed tree

```text
parent[x]=u
parent[y]=v
```

contains valid shortest edges. But under ordinary first-discovery FIFO BFS,
whichever of `u` or `v` is expanded first discovers **both** `x` and `y`.
Adjacency order cannot make the second parent win one already discovered child.

Therefore "every parent edge is shortest-valid" does not imply that the whole
tree is realizable by some ordinary BFS adjacency ordering. A post-layer
canonical-parent reduction can deliberately produce trees outside the family of
first-winner FIFO trees while preserving all distances.

## LexBFS is BFS-consistent, but not sorted-adjacency BFS

Lexicographic Breadth-First Search (LexBFS/LBFS) does not mean FIFO BFS with
sorted adjacency. It refines BFS ties using the lexicographically ordered
history/label of **all** already selected neighbors. It was introduced for
structural graph orderings such as chordal-graph recognition.

Every LexBFS ordering is consistent with a possible BFS ordering and therefore
respects distance layers from its first vertex. A tempting counterexample is

```text
s -- a -- c
|
b
```

After selecting `s`, vertices `a` and `b` tie. If `a` is selected, `c` receives
a refinement from `a`; nevertheless `b` still has the earlier/more significant
source-neighbor label and is selected before `c`. The depth-one layer is not
overtaken.

The distinction is within the admissible BFS tie orders:

```text
sorted-neighbor BFS -> ties follow local adjacency/list choices
LexBFS              -> ties follow full selected-neighbor histories.
```

LexBFS can expose structural elimination properties that an arbitrary BFS order
does not, while retaining BFS distance consistency. It still does not mean
"shortlex smallest source-to-vertex move word"; those are different
lexicographic objects.

## Cayley shortlex normal forms

For an ordered generator alphabet `S`, exact BFS from identity can choose for
each group element the shortlex-least generator word representing it. This is a
normal-form choice relative to:

- the exact generator collection;
- generator order;
- left/right action and replay convention;
- whether duplicate labeled generators remain distinct;
- directed versus inverse-closed word alphabet.

Changing only generator order preserves distances and frontier sets but can
change nearly every selected word. Changing the generator set may change the
metric as well.

A BFS-derived shortlex representative is not automatically the same as a
domain-specific algebraic normal form or the output of a rewriting system. Two
normal forms can both be unique while optimizing different orders.

For a Schreier or symmetry-quotient search, the least word may reach the nearest
representative of an orbit rather than a fixed concrete target. Notes 16 and 17
still govern endpoint and lifting semantics.

## Labeled multigraph subtleties

If two distinct edge occurrences have the same visible label, label words may
not totally order paths. A deterministic output may need additional ties such
as generator occurrence ID, parent identity, or destination identity.

Removing duplicate generators preserves unlabeled vertex distances but changes
the set and multiplicity of labeled shortest paths. A claim of canonical move
output must say whether generator names, transformations, or edge occurrences
define equality.

## Multi-source ordering

For multi-source BFS, the ordering tuple may include

```text
(distance, source_label, path_word, parent/move ties).
```

Minimizing distance alone permits arbitrary nearest-source labels. Minimizing a
source ID and then a word defines a canonical Voronoi labeling, but equal-depth
label improvements may need propagation as described in note 13. A local first
winner does not implement the global tuple order.

## Parallel and multi-GPU determinism

Exact distances only require one successful discovery per state after the level
set is formed. Reproducible parents require a stronger global reduction over
all valid same-depth proposals.

Questions include:

- Is the total tuple order identical on every rank and GPU?
- Does local dedup retain the local minimum proposal rather than the first?
- Can owner-side reduction see every proposal before parent finalization?
- Are frontier path ranks stable across partition counts?
- Does reconstruction need the whole canonical word, a parent rank, or a
  second pass?

Sorting frontier vertices by state key can make output order reproducible while
still choosing non-shortlex paths. Deterministic bytes are not proof of the
intended deterministic semantics.

## Cost vocabulary without choosing an implementation

Richer deterministic contracts may require measuring:

```text
same-depth parent proposals per accepted state
bytes of tie metadata
local versus owner/global reductions
states whose first winner differs from canonical winner
frontier ordering/ranking work
extra storage for replay
postprocessing or second-pass reconstruction work.
```

These measurements describe the cost of an output contract. They do not imply
that canonical parents or shortlex words are always required.

## Counterexamples and rejected shortcuts

- **BFS returns a unique traversal order.** Only distance layers are canonical
  without tie rules.
- **Sorted neighbors imply the smallest parent ID.** Neighbor ordering controls
  candidates within a parent; parent expansion order controls cross-parent
  competition.
- **Minimum parent ID gives the lexicographically least move word.** Full parent
  path order can disagree with state ID.
- **Every valid shortest-path tree is producible by FIFO BFS.** Parent choices
  can impose incompatible first-expansion requirements.
- **LexBFS is sorted-adjacency BFS.** False: it uses full histories of selected
  neighbors, although its resulting order remains BFS-consistent.
- **A canonical quotient word is a concrete canonical solution.** It may target
  an orbit and still require a symmetry-frame lift.

## Audit checklist

1. Is the required object distance, vertex order, parent tree, or move word?
2. Does "lexicographic" mean pure lex or length-then-lex shortlex?
3. What alphabet and total tie order are declared?
4. Is frontier order based on state key or canonical path rank?
5. Are all same-depth parent proposals considered before finalization?
6. Are duplicate generator labels semantically distinct?
7. Is the desired tree required to be a first-winner FIFO BFS tree?
8. Is LexBFS's full-history tie rule actually the desired deterministic BFS
   order, or is local adjacency/shortlex path ordering intended?
9. Does a quotient word lift to the concrete fixed target?
10. Is determinism invariant across rank/GPU counts or only within one launch?

## Sources

- Donald J. Rose, R. Endre Tarjan, and George S. Lueker,
  *Algorithmic Aspects of Vertex Elimination on Graphs*, SIAM Journal on
  Computing 5(2), 1976,
  [DOI](https://doi.org/10.1137/0205021), for the original lexicographic graph
  search and its structural-ordering purpose.
- Derek F. Holt, Sarah Rees, and Claas E. Röver, *Groups, Languages and
  Automata*, Cambridge University Press, 2017,
  [book excerpt](https://assets.cambridge.org/97811071/52359/excerpt/9781107152359_excerpt.pdf),
  for the group-theoretic meaning of selecting the shortlex least representative
  of each element.
- Note 03 supplies distance/parent nondeterminism; notes 11, 13, 16, and 17
  supply shortest-path DAG, multi-source tie, action, and quotient contracts.

## Current synthesis

BFS canonically computes hop distance, not a canonical enumeration or solution
word. Ordered FIFO expansion can add a shortlex contract when it orders complete
path prefixes and labels, while state-key sorting or minimum parent ID define
different deterministic trees. LexBFS is a special BFS-consistent structural
tie ordering, not path-word shortlex. On parallel Cayley search, reproducibility
must name the exact ordered object before any reduction or sorting mechanism can
be judged correct.

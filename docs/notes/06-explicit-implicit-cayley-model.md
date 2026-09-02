# One BFS, three graph presentations

Explicit, implicit, and Cayley graphs do not require three definitions of BFS.
They differ in how the graph answers the questions that BFS asks.  A useful
minimal contract is

```text
(state identity, sources, outgoing transitions, optional labels/costs)
```

Ordinary exact BFS additionally assumes that every transition has unit cost.
Its layer recurrence is independent of whether transitions are stored in an
array, computed by a function, or induced by group generators.

## The graph-oracle view

Let `X` be a set of states, `~` the exact identity relation adopted for vertices,
and

```text
succ(x) = sequence or set of labeled successor states.
```

BFS operates on equivalence classes under `~`, not necessarily on raw byte
strings.  The next layer is

```text
F_(d+1) = {[y] | exists [x] in F_d and transition [x] -> [y]}
          minus B_d.
```

This formulation exposes four independent correctness duties:

1. **identity:** equal vertices compare equal, and unequal vertices do not merge;
2. **expansion:** `succ` enumerates every graph edge required by the contract;
3. **layering:** candidates from depth `d` receive tentative depth `d+1`;
4. **coverage:** the process continues until its declared stopping condition is
   proved.

Storage format is absent from the theorem.  It affects the feasibility and cost
of satisfying the duties, not the duties themselves.

## Explicit graphs

An explicit graph materializes adjacency.  In a CSR representation, a dense
vertex identifier indexes an offset interval, and that interval stores neighbor
identifiers.  The same integer can often serve as:

- the semantic vertex identity;
- the frontier payload;
- the visited-array or bitmap index;
- the adjacency lookup key;
- a partitioning key.

This convenient collapse is not a property of BFS.  It is a property of the
representation.  It explains why many explicit-graph techniques quietly assume
a known, enumerable universe `0..|V|`: dense bitmaps and pull traversal are
natural because every possible vertex already has an address.

The familiar `O(|V|+|E|)` statement charges each stored vertex and adjacency
entry a constant number of times.  It says little about state-generation cost,
because generation has already happened before traversal.

Explicit does not mean static or simple.  Parallel edges, self-loops, directed
edges, labels, and a CSR containing only part of a changing graph all change the
graph contract.  In particular, treating an evolving adjacency snapshot as
immutable during BFS is an extra assumption.

## Implicit graphs

An implicit graph provides a state and an expansion procedure rather than an
adjacency table.  Examples include puzzles, program states, game positions,
automata products, and combinatorial objects transformed by moves.

The graph may be finite even though its vertex set is never enumerated in
advance.  Consequently:

- `|V|` and `|E|` may be unknown until exploration finishes;
- generating a successor can dominate reading or storing it;
- a state can be wider than any compact visited key;
- pull traversal may be unavailable because "all unvisited vertices" cannot be
  enumerated cheaply;
- a missing move, incorrect legality predicate, or noncanonical state silently
  changes the graph being searched.

A useful level-work identity distinguishes occurrences from unique endpoints.
Let `C` count generated legal transition occurrences, let `U` be their distinct
endpoint set, and let `B_d` be the complete visited ball before accepting the
next layer (including the current frontier):

```text
C = (C - |U|) + |U intersect B_d| + |U minus B_d|
  = within-batch duplicate occurrences
  + unique endpoints already visited
  + accepted unique new states.
```

These terms are disjoint: duplicate occurrences of an old state belong only
to the first term, while one representative of that old endpoint belongs to
the second. Two moves reaching an already visited `s` give `2=1+1+0`, not
`2=1+2+0`. Invalid move attempts, when moves are partial, are charged separately
from `C`. The terms can have very different costs even when only one child is
stored.

### The expansion procedure is part of the specification

For an explicit graph, one can inspect the edge table independently of BFS.  In
an implicit graph, `succ` *is* the edge table.  A plausible state count does not
prove it correct.  Useful validation therefore includes:

- inverse or round-trip properties when moves are reversible;
- independent legality checks;
- exhaustive comparison on a small enumerable instance;
- known component sizes, diameters, or layer histograms;
- replaying every stored parent transition;
- deliberately colliding compact hashes while preserving full-state equality.

## Cayley graphs as structured implicit graphs

For a group `G` and an ordered generator collection `S`, choose a convention,
for example right multiplication:

```text
vertices:      g in G
labeled edge:  g --s--> g*s, for s in S.
```

Starting at the identity `e`, BFS depth is word length with respect to `S`: the
smallest number of generators whose product is `g`.  This is precisely an
ordinary hop metric on the Cayley graph.

Several details that look representational actually define the graph:

- changing `S` changes degree, distance, diameter, and duplicate relations;
- if `S` is not inverse-closed, the graph is directed under the chosen
  convention;
- BFS from `e` follows the positive monoid of actually allowed generators. In a
  finite group this equals the subgroup group-generated by `S`; in an infinite
  group it may be strictly smaller when inverse moves are absent, and neither
  need fill a larger ambient representation;
- left and right multiplication are both valid but must not be mixed during
  path replay;
- treating `S` as an ordered list preserves move labels and multiplicity, while
  treating it as a set removes duplicate labeled transitions.

The last point separates vertex BFS semantics from edge semantics.  Duplicate
generators do not change shortest vertex distances, but removing them can change
the number and labels of shortest move sequences.  An identity generator does
not change distances either, yet it adds a self-loop at every vertex.

### Relations explain duplicates

In an arbitrary implicit graph, two search paths may converge for incidental
domain reasons.  In a Cayley graph, convergence has algebraic form: two words
`u` and `v` are duplicates exactly when they represent the same group element.
Short relations produce predictable local duplication:

- `s*s^-1 = e` produces immediate backtracking;
- `s^2 = e` does the same for involutions;
- commuting generators give `s*t = t*s`;
- braid and other presentation relations make longer convergences.

Visited is therefore performing semantic word reduction indirectly: it does not
normally derive a normal form for the word, but recognizes that the resulting
element has already been reached.

## State, identity key, rank, and payload are different

An implementation may associate several representations with one vertex:

| Role | Question answered |
|---|---|
| Full state | What object do moves act on? |
| Equality/canonical key | Are these two objects the same adopted vertex? |
| Dense rank | Which exact integer slot represents this vertex? |
| Frontier payload | What must be retained to expand it next? |
| Parent record | How can a discovered path be reconstructed? |
| Wire record | What must another owner receive? |
| Owner key | Which partition has final visited authority? |

They may coincide, but assuming that they do creates subtle errors.

The generic omission rule is role-specific. A frontier payload may omit a
field needed for expansion only if the remaining payload plus declared shared
immutable context still determines the complete exact successor set. Equality
data need not travel in the frontier record if it is available without loss at
the visited/claim point. Parent labels or presentation fields may live in
separate records or be reconstructed later, but only when the requested replay
or output contract remains provable.

Thus there is no universally minimal “BFS record.” Minimality is relative to
four declared decoders:

```text
payload -> complete successors
candidate -> exact vertex identity
accepted record -> requested parent/path evidence
stored result -> requested user-visible state
```

A compact record that satisfies only the first line can still be insufficient
for exact deduplication or replay. Conversely, storing the full presentation
state in every frontier entry is not a BFS semantic requirement when the four
obligations are satisfied elsewhere.

### Hash is not identity

A many-to-one hash can select a table bucket or owner.  It cannot by itself be
an exact visited key.  A collision interpreted as equality is a false positive:
it can erase the only path to a reachable region.  Exactness needs either full
comparison after hashing or a proved bijective encoding over the reachable
domain.

### Rank is not automatically an expandable state

A bijective rank supports an exact dense visited bitmap.  It need not support
cheap successor generation.  One may have to unrank before applying every move,
or carry both state and rank.  The local symmetric-group oracle illustrates the
latter: permutations are frontier payloads, while Lehmer ranks index exact
visited membership.

### Canonicalization is not automatically equality

Canonicalizing rotations, reflections, colors, or other symmetries changes the
vertex identity unless the problem explicitly asks for equivalence classes.
To solve the original graph through a quotient one must prove that transitions
are well defined on classes and that any quotient path can be lifted to a valid
original path with the promised length.  A smaller visited table alone is not
that proof.

## The adjacent-transposition example

The local experiments use permutations of `0..n-1` and adjacent swaps
`s_i=(i,i+1)`.  This example is valuable because multiple views coincide while
remaining distinguishable:

- full state: the permutation array;
- generator application: swap adjacent positions;
- exact dense identity: Lehmer rank in `0..n!-1`;
- source: the identity permutation;
- word length: inversion count;
- diameter: `n(n-1)/2`, attained by the reversed permutation.

Each adjacent swap changes inversion count by exactly one.  Any word reaching a
permutation therefore needs at least its inversion count in swaps, while bubble
sorting supplies a word of exactly that length.  This proves the distance
formula; matching a measured histogram is then validation evidence rather than
the reason to believe it.

For `S_8`, the local oracle exhaustively verified 40,320 states, diameter 28,
and peak frontier 3,836.  REF-004 then changed only the generator collection:
an identity and a duplicate generator preserved vertex distances but added
predictable work, while adding 3-cycles changed the metric and introduced
same-level edges.  This is a compact demonstration that the state space alone
does not determine BFS behavior; the transition system does.

## Counterexamples that separate the models

### Compact key without exact comparison

Two puzzle states hash to the same 64-bit value.  If visited stores only that
value and treats it as identity, the second state is dropped.  Nothing in the
frontier schedule can repair the missing branch.

### Inverse generators do not create pull traversal

A puzzle supplies `move^-1`, so predecessors of a known state are easy.  But
there is no procedure to enumerate all unvisited legal states.  Scanning the
unvisited universe—the defining outer loop of ordinary pull BFS—is still
impossible.

### Removing a duplicate labeled generator

Two generator names happen to induce the same permutation on every state.
Removing one preserves the unlabeled vertex-distance graph.  It does not
preserve a request to enumerate move-labeled shortest solutions.  Whether the
preprocessing is valid depends on the output contract.

### Quotient key without path lifting

Two cube configurations related by a spatial rotation are merged.  The quotient
distance may be the minimum over all rotated representatives, not the distance
to the user's fixed-orientation target.  Returning the quotient path without
tracking orientation can produce an invalid replay.

### Rank over the wrong universe

A ranking function is bijective over syntactically valid arrays but the puzzle
has parity or conservation constraints.  The rank remains exact, yet a dense
bitmap pays for unreachable states.  Conversely, a "compressed" rank that
accidentally aliases two reachable states destroys correctness.

## What transfers across all three presentations

The following semantic ideas transfer unchanged:

- distance layers and metric-ball induction;
- exact state identity;
- separation of candidate occurrences, unique candidates, visited hits, and
  accepted next-frontier states;
- parent-depth and edge replay checks;
- level-complete termination arguments;
- deterministic tie-breaking as an optional extra contract.

The following mechanisms do **not** transfer automatically:

- dense bitmaps;
- cheap pull traversal;
- constant-time adjacency access;
- a known vertex universe or capacity;
- a single compact value serving simultaneously as state, identity, rank, and
  expansion payload;
- graph partitioning that preserves locality.

This is the practical boundary: the BFS theorem is portable, while the cheapest
way to realize its obligations is representation-specific.

## Questions to ask of a new state space

1. What mathematical objects are vertices, and what exactly makes two equal?
2. Is the requested graph directed, labeled, a multigraph, or a quotient?
3. Does expansion enumerate every valid successor exactly once, or merely at
   least once?
4. Is the reachable universe finite, known, enumerable, and densely rankable?
5. Can a compact key be proved injective over every reachable state?
6. What must a frontier entry retain to generate successors?
7. Are reverse transitions available, and are they truly inverse under the
   chosen left/right action convention?
8. Which invariants, layer counts, or small exhaustive models independently
   validate the oracle?
9. Does changing generators preserve only reachability, or also distances,
   labels, multiplicities, and required solutions?
10. If symmetry is factored out, how are paths lifted and replayed?

## Sources and evidence

- Scott Beamer, Krste Asanovic, and David Patterson,
  *Direction-Optimizing Breadth-First Search*, SC 2012,
  [paper](https://www.scottbeamer.net/pubs/beamer-sc2012.pdf), for the explicit
  top-down/bottom-up setting and its enumerable-vertex assumptions.
- D. H. Lehmer, *Teaching Combinatorial Tricks to a Computer* (1960), the source
  commonly associated with the permutation code; a modern explanation of its
  exact integer mapping is available in
  [Unakafov and Keller (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7514243/).
- Harald Helfgott and Akos Seress,
  *On the diameter of permutation groups*, Annals of Mathematics 179 (2014),
  [arXiv preprint](https://arxiv.org/abs/1205.1596), for generator-dependent
  word length and Cayley-graph diameter in symmetric groups.
- Local evidence: REF-002 (complete adjacent-transposition Cayley layers),
  REF-003 (per-level accounting), REF-004 (generator-set changes), REF-016
  (representation order/locality), and the exhaustive Lehmer-rank test in
  `rust/src/cayley.rs`.

## Current synthesis

An explicit graph stores `succ`; an implicit graph computes `succ`; a Cayley
graph derives `succ` from algebraic actions.  BFS itself only needs that this
oracle and vertex identity be exact.  Most portability mistakes arise when a
convenience of explicit dense-ID graphs is mistaken for a BFS invariant, or
when an implicit representation decision—hashing, canonicalization, generator
preprocessing—is mistaken for a semantics-preserving optimization without the
necessary proof.

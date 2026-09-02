# Distances, one BFS tree, and the shortest-path DAG

BFS distance labels are canonical for a fixed graph and source set.  A parent
tree usually is not.  Treating them as the same output leads to false validation
claims, accidental nondeterminism, and undercounted shortest solutions.

There are four increasingly rich objects:

```text
distance labels
< one selected shortest-path tree/forest
< the shortest-path predecessor DAG
< explicit enumeration of every shortest path.
```

Each answers a different question and has a different storage contract.

## Distance labels

For one source `s`, BFS computes

```text
d(v) = dist(s,v).
```

For a source set `S`, it computes `min_(s in S) dist(s,v)`.  These labels are
independent of within-layer order when the graph and vertex identity are fixed.

Distances alone answer reachability and shortest hop count.  They do not say
which source won a tie, which parent should be displayed, how to replay a move,
or how many shortest paths exist.

## One shortest-path tree or forest

For each non-source reachable vertex `v`, choose one parent `p(v)` satisfying

```text
(p(v),v) is an edge
d(p(v)) + 1 = d(v).
```

Every parent pointer decreases depth by one, so following pointers reaches a
depth-zero source and cannot cycle.  The selected edges therefore form a
shortest-path tree for one source or a forest for multiple sources.

The choice is usually nonunique.  "First discovery wins" converts iteration and
scheduling order into parent semantics:

- source order selects among equidistant sources;
- frontier order selects among parents;
- adjacency/generator order selects among parallel opportunities;
- parallel atomic arrival order selects among concurrent claimants.

All winners can certify the same distance while producing different paths.

### Minimal diamond example

```text
    a
  /   \
s     t
  \   /
    b
```

Both `s-a-t` and `s-b-t` are shortest paths of length two.  A BFS tree retains
only `a -> t` or `b -> t`.  The unchosen edge is not a duplicate error; it is
another valid shortest predecessor.

## The shortest-path DAG

Given exact distances, define

```text
E_sp = {(u,v) in E | d(u)<infinity and d(u)+1=d(v)}.
```

The reachable vertices together with `E_sp` form a directed acyclic graph:
depth strictly increases on every edge.  For each vertex,

```text
P(v) = {u | (u,v) in E and d(u)<infinity and d(u)+1=d(v)}
```

is its complete predecessor set.

Unreachable vertices have no source-shortest predecessors. The explicit
finite-distance guard excludes `infinity+1=infinity`: infinity is an absence
of a finite path, not a depth layer on which cycle edges can be retained.
Reachability is already implicit when constructing `E_sp` from actual frontier
layers; it must be explicit when filtering a full graph with infinity labels.

A BFS tree chooses one member of each nonempty `P(v)`.  It is a spanning
arborescence of the shortest-path DAG, not the full shortest-path structure.

For a labeled multigraph or Cayley move system, predecessor **edges** may need
to be retained rather than only predecessor vertices.  Two move labels can
connect the same states and represent distinct requested solutions even though
the vertex sequence is identical.

## Counting shortest paths without listing them

Let `sigma(v)` be the number of shortest paths from the source.  For a single
source,

```text
sigma(s) = 1
sigma(v) = 0 for unreachable v
sigma(v) = sum_(u in P(v)) sigma(u) for reachable v != s.
```

The recurrence is valid because every shortest path to `v` has exactly one
penultimate predecessor in `P(v)`, and appending `(u,v)` to a shortest path to
`u` produces a shortest path to `v`.

For multiple sources, initialize each distinct depth-zero source with one, or
with a declared source multiplicity if source labels are considered distinct.
Again, the output contract matters.

The recurrence must count predecessor edges when parallel/labeled edges define
different paths.  Deduplicating them by child vertex preserves distance but can
change `sigma`.

### Distance and multiplicity as one pair algebra

Represent a path family by `(d,c)`, where `d` is its minimum length and `c` is
the number of members attaining that minimum. Use `(infinity,0)` for no path.
Alternative families combine as

```text
(d1,c1) plus_alt (d2,c2) =
    (d1,c1)       if d1<d2,
    (d2,c2)       if d2<d1,
    (d1,c1+c2)    if d1=d2,
```

while concatenated families combine as

```text
(d1,c1) times_path (d2,c2) = (d1+d2, c1*c2).
```

These operations encode the exact candidate cases: a shorter proposal replaces
the old pair, an equal proposal contributes multiplicity, and a longer proposal
is ignored. In the diamond, the two alternatives give

```text
(2,1) plus_alt (2,1) = (2,2).
```

The addition assumes distinct semantic path alternatives. A delivery retry of
the same predecessor-edge contribution is not a new path and must not be added
again; parallel generator labels are distinct only when the output contract
says labeled paths are distinct. The pair algebra specifies semantic merging,
not physical message identity.

### Local-then-global pair reduction is exact

Partition all candidate path-family contributions for one target into groups
`C_1,...,C_r`. Reduce each group with `plus_alt`, then reduce the `r` local
results. This equals one reduction over their full union because `plus_alt` is
associative and commutative.

Operationally, a candidate longer than its local minimum can never be globally
minimal, since the global minimum is no larger than that local minimum. A local
count survives the second reduction exactly when its local distance equals the
global distance; all such counts are then added. For example,

```text
local A: (2,1) plus_alt (3,5) = (2,1)
local B: (2,2) plus_alt (4,9) = (2,2)
global:  (2,1) plus_alt (2,2) = (2,3).
```

The statement assumes one target, one graph/output epoch, complete candidate
coverage, and one occurrence of each semantic contribution. It does not make
duplicate physical delivery safe: repeating the same equal-distance summand
still inflates the count. This is an algebraic permission for hierarchical
grouping, not evidence that any particular GPU partition is complete or fast.

### Validity filtering does not generally commute with minimum reduction

The hierarchical theorem assumes every input is already a valid contribution
under the declared graph, source labels, and epoch. Suppose one local group
contains

```text
invalid candidate: (1,1)
valid candidate:   (2,1).
```

Reducing first retains `(1,1)` and discards `(2,1)`. If later validation removes
the winner, the result becomes absence. Filtering first removes the invalid
record and correctly retains `(2,1)`. Hence in general

```text
filter(reduce(C)) != reduce(filter(C)).
```

An invalid shorter witness may come from a stale graph epoch, a nonexistent
edge, a corrupted parent label, or another failed semantic premise. Local
minimum reduction is lossless only after validity is established, or when the
summary retains enough fallback information to survive later invalidation.
Associativity cannot recover a candidate that an earlier reduction discarded.

### One valid aggregate witness does not validate aggregate multiplicity

Suppose one real shortest contribution and one invalid equal-distance record
are merged:

```text
valid   (2,1) plus_alt invalid (2,1) = reported (2,2).
```

Replaying the real length-two path proves that distance two is attainable. It
does not prove that two distinct shortest paths exist. The valid summand masks
the invalid one in the distance coordinate while the count coordinate still
absorbs both.

Consequently, validating one minimum-length witness establishes an upper bound
and, with an independent lower-bound certificate, the scalar distance. Exact
multiplicity additionally requires complete validated predecessor/label
contributions, contribution provenance sufficient to check them, or an
independent recount. Range checks and one replayable path cannot certify the
aggregate count.

### Factorization through an intermediate BFS layer

Fix a target `t` with `d(t)=k` and an intermediate depth `0<=j<=k`. For
`a in F_j`, let `tau(a,t)` count shortest `a`-to-`t` suffixes of length `k-j`;
set it to zero when `dist(a,t) != k-j`. Then

```text
sigma_s(t) = sum_(a in F_j) sigma_s(a) * tau(a,t).
```

Every shortest `s`-to-`t` path visits exactly one vertex `a` at depth `j`.
Cutting there gives a unique shortest prefix and suffix. Conversely, combining
any counted prefix and suffix has total length `k`, so it is a shortest path.
This bijection proves the product-sum formula.

For scalar distance, retaining one `a` with `tau(a,t)>0` is sufficient. For an
exact path count, every nonzero layer contribution must survive, either as its
path structure or as an output-equivalent aggregate. In the diamond, retaining
only one of the two depth-one vertices preserves distance two but changes
`sigma(t)` from two to one.

For labeled multigraph semantics, `sigma` and `tau` count labeled edge paths,
not merely vertex sequences. The factorization remains valid, but collapsing
parallel labels before either factor changes the requested count.

### Frontier counts are sufficient statistics only for count continuation

At a completed layer `F_j`, the values `sigma_s(a)` summarize all shortest
prefix multiplicity needed to continue computing counts outward. If every
future shortest predecessor edge is processed exactly according to the path
identity convention, the usual recurrence propagates those prefix masses; the
interior predecessor DAG need not be traversed again merely to obtain later
scalar counts.

This summary is lossy. A record `sigma(a)=2` says that two shortest prefixes
exist but does not identify their parents, labels, or vertex sequences. It
therefore cannot by itself reconstruct either path, enumerate them, enforce a
canonical-path choice, or support backward sampling. Those outputs require the
old structure, a regenerable immutable graph plus sufficient distance/label
metadata, or another output-equivalent summary.

Thus `(a,depth,sigma(a))` can be a sufficient boundary state for exact count
continuation while being insufficient for richer shortest-path contracts.
There is no output-independent notion of a "complete" BFS checkpoint.

### Counts can be exponentially larger than the DAG

Create `k` layers with two vertices each, connect every vertex of one layer to
both vertices of the next, connect a source to both first-layer vertices, and
both last-layer vertices to a target.  The graph and shortest-path DAG have
`O(k)` vertices and edges, but the target has `2^k` shortest paths.

Consequences:

- a predecessor DAG can compactly represent exponentially many paths;
- explicit path enumeration necessarily takes output-proportional time;
- a fixed-width `sigma` counter can overflow even when the graph fits easily;
- "all paths" is not a small metadata extension to "one path."

Brandes's betweenness-centrality algorithm is a prominent use of this structure:
it stores predecessor sets and shortest-path counts during BFS, then performs a
reverse-depth dependency accumulation rather than enumerating every path.

## Cayley interpretation: elements versus geodesic words

For Cayley BFS, one vertex is a group element while a shortest path is a
geodesic generator word representing that element.  Relations can give one
element many geodesic words.

Adjacent-transposition `S_n` makes this concrete:

- depth is inversion count;
- a shortest predecessor is obtained by removing one inversion through an
  adjacent descent;
- the predecessor DAG contains all such depth-decreasing choices;
- a single BFS parent records one reduced decomposition;
- all paths correspond to all reduced decompositions under the chosen labeled
  generator convention.

The reverse permutation has maximal inversion count and many reduced words.
Storing its one parent chain says nothing about that multiplicity.

An identity generator never appears in a positive-length shortest path, because
it adds cost without changing the element.  A duplicate labeled generator can
multiply labeled geodesic paths without changing distances or predecessor
vertices.  This is why generator preprocessing is safe for distance-only BFS
but may be wrong for labeled solution enumeration.

## A parent chain is a witness, not a distance proof by itself

A valid parent chain of length `d(v)` proves

```text
dist(s,v) <= d(v).
```

It does not exclude a shorter path.  A deliberately wrong result can attach
`v` through a valid length-three chain while an ignored edge gives a length-two
path.

For a complete unweighted result, a useful proof pair is:

1. every non-source vertex has a real parent edge from depth one less;
2. every explored edge `(u,v)` satisfies `d(v) <= d(u)+1`, and every reachable
   child is present.

The parent chain gives an upper bound on true distance.  Applying the edge
inequality along any source-to-vertex path gives the matching lower bound on
recorded labels.  Together they establish equality.

This is why the local validator checks both parent validity and every outgoing
edge inequality.  REF-001 contains a negative fixture whose parent tree is
connected and internally consistent but nonminimal; the edge check catches it.

## Labeled replay adds another obligation

In an implicit graph, `(parent state, child state)` may not identify which move
was used.  One-path reconstruction commonly stores

```text
parent_state_or_key
parent_move_label
```

and validates

```text
apply(parent, move) == child.
```

Correct depths and a real unlabeled adjacency are not enough if the stored move
cannot reproduce the child.  The local labeled oracle and its negative fixture
test this exact failure.

For bidirectional search, reverse metadata has a directional convention:
following its stored pointer toward the target must yield a forward-replayable
move.  Joining two valid distance trees without checking move orientation can
produce an unreplayable solution.

## Determinism is an additional algorithm

Sequential first-discovery BFS appears deterministic only when all input
orders are deterministic.  Parallel execution makes arrival order unstable.

To promise a canonical parent, define a total rule such as

```text
min (source_id, parent_identity, move_label)
```

over every shortest predecessor opportunity.  Implementing the rule may
require finishing the relevant layer, retaining competing records, atomic
minimum/reduction, or owner-side convergence.  It can increase bytes and work
without changing any distance.

Important distinctions:

- deterministic within one GPU launch configuration is not necessarily stable
  across GPU counts;
- owner-local preference can reduce communication but changes the tie-break
  unless owner is explicitly part of the canonical rule;
- sorting frontier states makes one implementation reproducible but does not
  define a representation-independent parent unless the key/order is semantic;
- hash order must not be treated as a stable total order unless specified and
  collision-safe.

Determinism should therefore be priced and validated as an output feature, not
assumed as a side effect of exact BFS.

## Parallel discovery and lost shortest parents

For distance-only or one-tree BFS, an exact visited claim can discard all losing
same-layer candidates after one winner records the vertex.  For the complete
predecessor DAG, those losing candidates may contain required edges.

A correct all-predecessor flow must distinguish:

```text
candidate reaches unseen child at depth d+1
candidate reaches already discovered child at the same depth d+1
candidate reaches child at an older/smaller depth
```

The second case contributes another shortest predecessor; the third does not.
An ordinary visited boolean merges them.  Distance or epoch information is
needed to recover the distinction.

This changes the meaning of "duplicate removal": equal child records can be
collapsed for the next frontier, but their parent/move contributions cannot be
discarded if all shortest paths are requested.

## GPU storage contracts

The output choice changes memory and operations:

| Output | Typical per-vertex/edge information |
|---|---|
| Distance/reachability | visited bit or depth label |
| One path/tree | one parent key and possibly one move |
| Deterministic tree | competing parent reduction plus one winner |
| Path count | depth plus `sigma`, with overflow policy |
| Shortest-path DAG | variable-length predecessor edge records |
| Explicit all paths | DAG plus output-sized enumeration/materialization |

A fixed-capacity predecessor buffer needs the same explicit overflow semantics
as a frontier.  Silently dropping predecessors can leave distances correct
while corrupting counts, centrality, or all-path enumeration.

Path counts introduce arithmetic semantics too: wraparound is incorrect;
saturation answers a different question; modular counts must be declared; exact
big integers have variable cost and poor direct GPU fit.

## Multi-GPU parent ownership and reconstruction

If the child owner decides visited novelty, several parent strategies are
possible:

- send parent metadata eagerly with every candidate;
- send only child identity first, then request metadata for winners;
- prefer a producer colocated with the owner;
- store parent owner/key and perform distributed pointer chasing later;
- rerun or regenerate a constrained layer during reconstruction.

All can preserve distance, but they differ in bytes, rounds, determinism, and
reconstruction latency.  REF-011 rejected the universal claim that deferred
metadata always saves wire bytes: it helped selected deep runs and lost on every
tested shallow case.

For a complete predecessor DAG, a child can have shortest parents on many
ranks.  Owner authority for the child remains useful, but all contributing
records must converge or be recoverably referenced.  One accepted-state bitmap
returned to sources is insufficient to describe all predecessor edges.

## Early target stopping and path outputs

Stopping when a target is first discovered gives one shortest distance/path
under the appropriate BFS schedule.  It may occur before all other target
predecessors in the same generating layer have been processed.

Therefore:

- candidate-stop can return one shortest witness;
- complete-parent-batch stop may return some additional shortest parents;
- complete-layer stop can collect every predecessor edge into that target from
  the current layer;
- full shortest-path DAG to all vertices may require continuing beyond the
  target depth depending on the requested target set.

REF-008's stop granularities have identical distance semantics but should not be
mistaken for identical all-shortest-path semantics.

## Bidirectional all-shortest-path structure

One bidirectional answer joins one forward parent tree, one reverse next-pointer
tree, and one selected meeting.  To represent every shortest `s -> t` path, one
needs every compatible connector satisfying

```text
d_f(x) + d_b(x) = D
```

or crossing edge

```text
d_f(u) + 1 + d_b(v) = D,
```

plus complete forward/reverse predecessor structures in the relevant depth
ranges.  Distance-optimal `a+b>=D` termination can stop before every equality
case has been enumerated.  Note 08's distinction between distance-optimal and
enumeration-complete stopping is therefore operational, not merely formal.

## Validation matrix

| Claimed output | Minimum useful validation |
|---|---|
| Distances | sources at zero, reachability closure, edge inequalities, exact frontiers if returned |
| One tree | distance validation plus one real depth-decreasing parent per non-source |
| Labeled path | tree validation plus move replay to the exact target |
| Deterministic tree | compare against the declared global tie-break, not one incidental run |
| Path counts | complete predecessor-edge recurrence with overflow semantics |
| Shortest-path DAG | every and only edge satisfying `d(u)+1=d(v)` under the path-identity contract |
| All paths | DAG validation plus complete output enumeration/count agreement |

Passing a weaker row does not validate a stronger claimed output.

## Sources and local evidence

- Ulrik Brandes, *A Faster Algorithm for Betweenness Centrality*, Journal of
  Mathematical Sociology 25(2), 2001,
  [paper](https://snap.stanford.edu/class/cs224w-readings/brandes01centrality.pdf),
  [doi:10.1080/0022250X.2001.9990249](https://doi.org/10.1080/0022250X.2001.9990249),
  for predecessor sets, shortest-path counts, and reverse-depth accumulation.
- The [Graph500 BFS specification](https://graph500.org/?page_id=12) is an
  example of a one-tree contract: same-level parent races are permitted, while
  the returned parent array is validated for real edges, correct levels, and
  component coverage.
- REF-001 validates distance/tree separation; REF-002 validates labeled replay;
  REF-008 exposes early-stop granularity; REF-011 models parent wire records;
  notes 08 and 10 supply bidirectional and Cayley geometry context.

## Current synthesis

BFS distances describe the metric.  A parent tree is one compact witness chosen
from the shortest-path DAG.  Path counts and all-path enumeration are strictly
richer outputs whose information can grow from one pointer per state to
exponentially many paths.  Parallel "duplicate" candidates are therefore not
always disposable: some are redundant for frontier membership but essential as
shortest predecessors.  Exactness, determinism, replay, counting, and complete
enumeration must be named and validated separately.

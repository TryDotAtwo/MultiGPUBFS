# From BFS to 0-1 BFS and relaxation-based shortest paths

Ordinary BFS is not "Dijkstra without a heap" merely because both find shortest
paths.  BFS has a stronger structural fact: every outgoing edge from depth `d`
proposes exactly `d+1`.  That fact makes first discovery final and makes a FIFO
queue or complete frontier sufficient.

Once edge costs vary—even only between zero and one—the finalization proof
changes.  The central operation becomes relaxation:

```text
if dist[u] + weight(u,v) < dist[v]:
    dist[v] = dist[u] + weight(u,v)
    schedule v again as required
```

This note traces exactly where ordinary BFS ends and related shortest-path
algorithms begin.

## Why first discovery is final in ordinary BFS

Assume every edge has cost one and vertices are expanded in nondecreasing hop
distance.  While expanding a vertex at depth `d`, every newly encountered child
receives candidate distance `d+1`.

Could a later edge improve it?  Later expansions have depth at least `d`, so
their candidates cost at least `d+1`.  Therefore no strictly smaller candidate
can arrive later.  The first exact discovery is final.

This gives ordinary BFS its particularly simple state model:

```text
unvisited -> discovered/final distance
```

The vertex may still be physically queued and unexpanded, but its distance does
not need correction.

## Minimal counterexample for 0-1 weights

### Equal positive costs change units, not the search

The literal value one is a normalization. If every edge has the same constant
cost `c>0`, every path of `k` edges costs `c*k`. Multiplication by a positive
constant preserves the ordering and ties of all path lengths. Therefore

```text
weighted_dist(s,v) = c * hop_dist(s,v).
```

Ordinary BFS still finds all minimum-cost paths and their finite predecessor
DAG; return its hop label multiplied by `c` when cost is requested. For example,
two edges costing two each have hop distance two and cost four. No relaxation
algorithm is needed merely because the displayed cost is not one.

Positivity matters. If all edges cost zero, BFS still finds minimum-hop paths
and thereby some cost-optimal paths, but all reachable paths have cost zero.
Hop layers and the BFS predecessor DAG no longer enumerate every minimum-cost
path or walk. Thus zero is not just another harmless change of units. With
different positive costs, the common scaling argument likewise does not apply.

### Unequal zero-one costs break first-discovery finality

Consider directed edges in this enumeration order:

```text
s --1--> a
s --0--> b
b --0--> a
```

When `s` is expanded, a first-discovery algorithm can record `dist[a]=1` before
recording `dist[b]=0`.  But the path `s -> b -> a` has total cost zero.

If a visited boolean makes `a` irrevocable, the result is wrong.  Correct 0-1
BFS must allow the later relaxation

```text
dist[a]: 1 -> 0.
```

The problem is not adjacency ordering itself.  The problem is using ordinary
BFS finalization semantics after the unit-edge assumption has been removed.

## The 0-1 deque discipline

For edge weights in `{0,1}`, a deque is a specialized monotone priority queue:

- a successful zero-weight relaxation is pushed to the front;
- a successful unit-weight relaxation is pushed to the back.

At the moment a minimum-distance item of value `D` is processed, relevant deque
labels are organized around `D` and `D+1`.  Zero edges stay in the current cost
bucket; unit edges enter the next bucket.  Exhausting the zero-cost closure is
part of finishing distance `D`.

The deque is therefore not ordinary FIFO BFS with a cosmetic insertion rule.
It implements two moving distance buckets.

Depending on implementation details, a vertex whose label improves may appear
more than once, or an old queued entry may become stale.  Correctness needs a
policy for reactivation/stale entries.  A single "already visited" bit is not an
adequate substitute for tentative distance.

## Discovered, queued, and settled are different states

For relaxation-based shortest paths:

- **undiscovered:** tentative distance is infinity;
- **discovered/tentative:** at least one path is known, but a better path may
  still appear;
- **queued/active:** outgoing edges need processing for the current label;
- **settled/final:** no later relaxation can improve the label.

Ordinary BFS collapses tentative and final at first discovery because its queue
order and unit costs prove finality.  Dijkstra finalizes when a minimum
tentative-distance vertex is extracted.  Label-correcting methods may process a
vertex repeatedly without a permanent settled event until convergence.

Calling all of these states "visited" hides the key invariant.

## Cost balls replace hop balls

For nonnegative weights, define

```text
B(r) = {v | weighted_dist(s,v) <= r}.
```

With unit weights and integer `r=d`, these are the ordinary BFS balls.  With
zero-one weights, multiple graph hops can remain inside the same cost ball.
There may be edges within one distance layer:

```text
d(v) = d(u) when weight(u,v)=0.
```

Consequences:

- a weighted distance layer is not necessarily an independent frontier step;
- zero-cost closure may add vertices to the current bucket;
- hop depth and optimal cost are different metadata;
- a level count no longer equals path cost.

If a zero-cost cycle is reachable, distinct vertices—or repeated walks—can all
remain at one cost.  Finite graphs still have finite vertex closure, but the
number of shortest **walks** may be infinite because the zero cycle can be
traversed arbitrarily often.

## Dijkstra as the general nonnegative finalization rule

For arbitrary nonnegative edge weights, Dijkstra maintains tentative labels and
selects a vertex of globally minimum tentative distance.  When selected, that
label is final: any alternative path through an unsettled vertex has cost no
smaller because all remaining tentative distances and all edge weights are
nonnegative.

Ordinary BFS is the special case where all successful candidates from the
current minimum layer have the same next key, so FIFO layering implements the
priority order.  0-1 BFS is the special case where only the current and next
integer buckets need immediate distinction, so a deque implements the order.

The family resemblance is real, but the reusable theorem is "process tentative
labels in a way that makes finalization sound," not "use a queue and visited."

## Dial buckets

For bounded nonnegative integer weights, Dial's method uses cyclic/integer
distance buckets rather than a comparison heap.  0-1 BFS can be viewed as the
two-adjacent-bucket extreme of this idea.

Bucket width and maximum edge cost determine how far a relaxation can jump.  A
bucket data structure is an execution mechanism; correctness still relies on
processing buckets in nondecreasing distance and handling improved labels.

This is another naming boundary: bucketed SSSP may look level-synchronous, but
its levels are distance keys, not BFS hop frontiers.

## Delta-stepping and approximate buckets

Delta-stepping groups tentative distances into intervals of width `Delta` and
separates light and heavy edges.  It exposes parallel relaxation inside a bucket
at the price of possible repeated work and a more involved convergence proof.

It computes exact nonnegative shortest distances when its algorithmic
conditions are followed; "Delta" refers to scheduling granularity, not an
approximate answer.  However, it is not ordinary BFS, and choosing `Delta`
changes work/parallelism rather than the mathematical distances.

The relevant conceptual contrast is:

```text
ordinary BFS:  one final discovery per vertex under exact hop layers
delta-style:   tentative labels may improve while a distance interval closes
```

## Negative weights cross another boundary

Dijkstra, Dial, and 0-1 BFS rely on nonnegative weights.  A negative edge can
make a path through a later vertex improve an already settled label.  Negative
cycles reachable on a source-to-target route can make a finite shortest
distance undefined.

Bellman-Ford-style relaxation addresses negative edges by repeated global/local
edge relaxation and detects reachable negative cycles under its contract.  It
is not a BFS variant in the ordinary metric sense.

## Target stopping differs

In ordinary FIFO/level BFS, first target discovery proves its shortest hop
distance because discovery is final.

In 0-1 BFS or Dijkstra:

- first finite tentative label for the target is only an upper bound;
- a later zero/lower-cost route may improve it;
- target termination is safe when the algorithm's minimum-key/finalization rule
  proves that no outstanding label can beat it.

The three-edge counterexample above already shows why "return when target first
enters the deque" can be wrong.

Early stopping must name whether the target is discovered, extracted with a
current minimum key, or settled under the exact variant's proof.

## Parent and shortest-path structures with weights

For weighted distances, a shortest predecessor edge satisfies

```text
d(u) < infinity and d(u) + weight(u,v) = d(v).
```

With strictly positive weights, distances increase along these edges and the
shortest-path predecessor structure is a DAG.

With zero-weight edges, equal-distance predecessor edges can form cycles.  The
unit-weight DAG result from note 11 no longer transfers unchanged.  One must
distinguish:

- one simple shortest path;
- all simple shortest paths;
- all shortest walks, possibly infinite in number;
- a zero-cost strongly connected component quotient;
- path-count semantics under parallel/labeled edges.

Thus adding zero-cost moves changes not only queue mechanics but also the shape
of the shortest-path output object.

### Tight parent edges need not form a rooted tree

Consider exactly these directed edges:

```text
s --1--> a       s --1--> b
a --0--> b       b --0--> a
```

The exact labels are `d(s)=0`, `d(a)=d(b)=1`. Selecting `parent(a)=b` and
`parent(b)=a` passes the local check that every chosen edge is real and
`d(parent(v))+weight(parent(v),v)=d(v)`. Nevertheless those parent pointers
cycle forever and give no path from `s` to either state.

This is not a counterexample to 0-1 BFS distance correctness or to a carefully
specified parent-update algorithm. It rejects the weaker validation rule that
real tight parent edges plus correct distances automatically certify a rooted
witness tree. An arbitrary postprocessing choice among equal-cost parents can
break that contract even after distances are finalized.

In ordinary unit BFS, each parent reduces the finite distance by one, which
itself proves root termination. Zero-cost parent edges remove that strict
progress. A selected weighted parent structure therefore needs a separate
root-reaching/acyclicity certificate, or another proved strictly decreasing
rank. Keeping the complete tight-edge graph is also valid as a different
output, but it must not be called a parent tree or used with the unit-depth
path-count recurrence across its cycles.

## Multi-source and zero-cost super-sources

Multi-source ordinary BFS initializes every real source at hop distance zero.
Conceptually, this is equivalent to a virtual super-source with **zero-cost**
edges, not unit edges.

If one literally adds those zero edges, the resulting graph is a 0-1 weighted
problem.  Direct initialization avoids running a weighted algorithm because the
zero-cost closure is known in advance: it is exactly the declared source set.

If sources themselves are connected by additional zero-cost transitions, that
closure may be larger and must be handled explicitly.

## Weighted Cayley metrics

Assigning a nonnegative cost `c(s)` to each generator changes word length to
minimum total generator cost.  Several effects follow:

- a longer word can be cheaper than a shorter word;
- generator count is no longer distance;
- inverse generators may have asymmetric costs in a directed model;
- zero-cost generators put entire reachable orbits at distance zero;
- a zero-cost identity/relator creates infinitely many equal-cost words even
  though the vertex set is unchanged.

Ordinary Cayley BFS minimizes cost when every move has the same positive cost,
normalized to one, with output costs rescaled if needed. Unequal move costs
require a weighted shortest-path justification; ordinary hop minimality alone
does not establish cost minimality. All-zero costs preserve a BFS path as one
cost-optimal witness but do not preserve the full cost-optimal path family.

## Parallel/GPU implications without choosing an implementation

Ordinary level BFS offers a clean bulk-synchronous boundary: all accepted
children have one known next depth.  Relaxation-based algorithms introduce
additional physical obligations:

- atomic or owner-authoritative minimum updates rather than one visited claim;
- reactivation when a label improves;
- stale work detection;
- repeated processing within a cost bucket;
- global/local minimum nonempty bucket agreement;
- target stopping based on minimum outstanding cost;
- parent replacement or predecessor accumulation after improvements.

A high rate of relaxations/s can conceal low progress if the same labels improve
many times.  Useful counts include attempted relaxations, successful decreases,
reactivations, stale pops, settled vertices, bucket revisits, and final unique
distances.

For multiple GPUs, a state owner must arbitrate `min` over proposals arriving
from different ranks.  A local tentative improvement is not globally final
while a lower-cost message can still be in flight.  Level-empty termination from
ordinary BFS does not transfer; termination needs bucket/min-key and in-flight
work agreement.

These are conceptual consequences, not a request to build a weighted backend.

## Counterexamples to common equivalences

### 0-1 BFS is ordinary BFS with a deque

False as a semantic statement.  The deque supports relaxation and correction;
ordinary visited-on-discovery can return the wrong distance.

### A vertex in the queue has its final distance

True for ordinary BFS first discovery, not generally for 0-1 or label-correcting
SSSP.  Queue membership and finality are separate.

### Zero-cost edges can be contracted freely

Only after proving the contraction preserves direction, requested state/path
identity, labels, and reconstruction.  A directed zero-cost reachability
relation need not be symmetric; strongly connected zero-cost components are a
different object from one-way zero closure.

### Fewer buckets means less work

Coarser scheduling can expose more parallelism while causing more repeated
relaxations.  Delta/bucket choice is a workload-dependent execution trade, not
a monotone theorem.

### Correct distances imply correct path counts

Zero-cost cycles can make shortest-walk counts infinite, and losing equal-cost
predecessors can corrupt simple-path counts while distances remain exact.

## Audit checklist

1. Are all edge/move costs exactly one, zero-one, bounded integers, arbitrary
   nonnegative, or possibly negative?
2. Is path objective hop count or total cost?
3. When does a tentative label become final, and what proves it?
4. Can a vertex be requeued/reactivated after improvement?
5. How are stale entries detected?
6. What does "visited" mean: discovered, active, or settled?
7. What minimum outstanding key/bucket justifies target or global termination?
8. Can zero-cost edges create same-distance cycles?
9. Are parent/count outputs defined for paths, simple paths, or walks?
10. In distributed execution, where is `min` authoritative and how are in-flight
    lower proposals included in termination?

## Sources

- E. W. Dijkstra, *A Note on Two Problems in Connexion with Graphs*, Numerische
  Mathematik 1 (1959), 269-271,
  [full record and text](https://eudml.org/doc/131436),
  [doi:10.1007/BF01386390](https://doi.org/10.1007/BF01386390), for minimum-key
  label setting with nonnegative/positive branch lengths.
- Robert B. Dial, *Algorithm 360: Shortest-Path Forest with Topological
  Ordering*, CACM 12(11), 1969,
  [doi:10.1145/363269.363610](https://doi.org/10.1145/363269.363610), for integer
  bucket shortest paths.
- Ulrich Meyer and Peter Sanders, *Delta-Stepping: A Parallelizable Shortest
  Path Algorithm*, Journal of Algorithms 49(1), 2003,
  [doi:10.1016/S0196-6774(03)00076-2](https://doi.org/10.1016/S0196-6774(03)00076-2),
  for parallel bucket relaxation and its work/parallelism trade.
- Notes 03, 05, 09, and 11 provide the ordinary BFS finalization, variant,
  termination, and predecessor-DAG boundaries used here.

## Current synthesis

Ordinary BFS is unusually simple because unit edges align discovery order,
distance order, and finalization.  0-1 BFS preserves a narrow bucket structure
but not irrevocable first discovery; Dijkstra and related methods generalize the
minimum-key proof.  Once zero or varying costs appear, `visited`, frontier,
parent, target stopping, and distributed termination all need relaxation-based
definitions.  The data structure is secondary—the decisive question is what
makes a label final.

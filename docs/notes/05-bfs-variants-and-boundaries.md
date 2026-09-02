# BFS variants and the boundaries of the name

BFS is best identified by its semantic result, not by the presence of a queue
or by the word *breadth* in an algorithm's name.  For an unweighted graph and a
source set `S`, exact BFS computes

```text
d(v) = min { number of edges in a path from any s in S to v }.
```

Its layers are `F_k = {v | d(v) = k}`.  A genuine implementation variant may
change how a layer is enumerated, represented, scheduled, or distributed while
preserving these sets.  Other algorithms change the metric, the returned
object, or the guarantee.  They may be useful, but calling all of them BFS
without qualification hides the most important distinction.

## A taxonomy by the thing that changes

| Family | What changes | What remains true | Additional obligation | Frequent mistake |
|---|---|---|---|---|
| Multi-source BFS | Initial frontier is a set `S` | `d(v)=min_s d(s,v)` | All sources begin at depth zero | Giving a virtual super-source unit-cost edges and forgetting the extra level |
| Target or depth-limited BFS | Termination and returned subgraph | Reported settled distances are exact | Stop only when the target/depth claim is proved | Treating a truncated traversal as a complete BFS tree |
| Bidirectional BFS | Two searches approach one another | Their joined path can be shortest | Reverse edges for the backward side; a sound lower-bound stopping rule | Stopping at an arbitrary first contact |
| Push, pull, direction-optimizing BFS | How `F_(k+1)` is enumerated | The exact next layer is unchanged | Pull needs predecessor access, frontier membership, and an enumerable unvisited universe | Assuming pull applies naturally to every implicit graph |
| Level-synchronous parallel BFS | Execution schedule | Completed layers equal the sequential BFS layers | Exact visited authority and a global/equivalent level-completion proof | Letting delayed messages silently cross a distance boundary |
| 0-1 BFS | Edge weights are zero or one | Shortest weighted distances are computed | Deque scheduling plus relaxation | Using ordinary first-discovery visited semantics |
| Dial/Dijkstra families | General nonnegative edge costs | Shortest weighted distances are computed | Settle by distance priority, not hop layer | Calling every bucketed shortest-path method ordinary BFS |
| Lexicographic BFS | The requested result is a vertex ordering | A BFS-consistent graph-structural ordering is produced | Maintain lexicographic labels/partitions | Equating selected-neighbor-history ties with sorted-adjacency or path-word shortlex BFS |
| Beam search | Frontier is pruned by score/width | Only retained states are explored | State explicitly that completeness and optimality are lost | Describing it as memory-bounded exact BFS |

This table has two kinds of rows.  The first five can preserve the ordinary BFS
metric when their proof obligations hold.  The later rows either generalize the
metric or compute a different object.

## Multi-source BFS

Initialize `F_0` with every source and mark them all visited before expansion.
The recurrence is unchanged:

```text
B_0 = S
B_(k+1) = B_k union N(B_k)
F_k = B_k minus B_(k-1)
```

The result is distance to a *set*, and source ownership gives a graph Voronoi
partition.  A vertex equidistant from two sources has a deterministic distance
but not a unique owner unless tie-breaking is part of the contract.

A virtual super-source is a useful proof device only if its outgoing edges have
zero cost.  In an ordinary unit-edge graph those edges add one to every
distance.  Directly seeding all real sources at depth zero avoids this modelling
trap.

## Early termination changes the returned object

Several stopping contracts are distinct:

- **complete traversal** returns every reachable distance;
- **target search** may return as soon as the target's shortest distance is
  proved;
- **depth-limited BFS** returns the metric ball `B_L` (and perhaps the boundary
  `F_L`);
- **all shortest paths** requires retaining every predecessor edge `(u,v)` with
  `d(u)+1=d(v)`, rather than one arbitrary parent.

Discovering a target during expansion is enough for its distance under strict
FIFO or completed-layer semantics.  It is not automatically a certificate in
an asynchronous schedule where smaller-depth work may still be outstanding.
Likewise, stopping mid-layer can give a correct target distance but does not
mean the whole target layer has been enumerated.

## Bidirectional BFS

Bidirectional BFS does not merely run two ordinary searches and stop when two
visited sets happen to intersect.  It maintains two distance certificates:
`d_f(v)` from the source and `d_b(v)` to the target.  Every meeting vertex or
crossing edge provides a candidate path length.  Termination is safe when a
lower bound for all not-yet-seen connections cannot beat the best candidate.

Consequences:

- on a directed graph the backward search traverses the **reverse graph**;
- alternating sides and expanding the smaller frontier are scheduling
  policies, not correctness arguments by themselves;
- "first intersection" is safe only under sufficiently controlled layer
  semantics, not for an arbitrary interleaving of individual expansions;
- keeping one parent forest per side is enough for one path, while all shortest
  paths require more predecessor information.

The benefit is structural, not universal.  Roughly balanced branching can turn
work resembling `b^d` into two searches resembling `b^(d/2)`, but asymmetric
branching, large meeting layers, expensive reverse generation, or visited-set
overheads may erase that advantage.

## Push and pull are two enumerations of the same layer

In **push**, vertices in `F_k` enumerate their outgoing neighbours.  In
**pull**, an unvisited vertex asks whether any predecessor belongs to `F_k`.
If both are exact, they compute the same predicate:

```text
v in F_(k+1) iff v not in B_k and exists u in F_k with edge u -> v.
```

Direction-optimizing BFS switches between these enumerations to reduce edge
work.  It does not change search direction in the bidirectional-search sense.

Pull is natural when vertices have dense identifiers, the unvisited universe
is cheaply enumerable, frontier membership is fast, and incoming adjacency is
available.  Those assumptions often fail for implicit state spaces: enumerating
all not-yet-generated states may be harder than generating successors from the
frontier.  Thus a technique that is effective for explicit social/web graphs is
not automatically transferable to a Cayley graph represented only by moves.

## Parallel, distributed, and asynchronous schedules

Level-synchronous parallel BFS partitions the work of constructing one exact
next layer.  The partition may be by frontier vertices, edges, candidate keys,
or owner ranges.  These are execution choices; correctness still needs:

1. no state at depth `k+1` is accepted as though it were at another depth;
2. duplicate claims converge to one exact visited identity;
3. completion of a level is known globally, or an equivalent distributed
   termination invariant is proved;
4. target termination accounts for messages and work still in flight.

An asynchronous relaxation algorithm can compute the same distances, but its
proof resembles distributed shortest-path relaxation more than the simple BFS
layer induction.  Merely removing barriers does not preserve the proof for
free.

Multi-GPU BFS belongs here: GPUs alter ownership, communication, and the cost of
the schedule, not the mathematical distance.  A useful conceptual model is
that each state has one authoritative owner for exact visited membership, while
the global traversal must still prove that no shallower work remains.

## Weighted relatives

If every edge has one common positive cost `c`, ordinary BFS remains sufficient:
minimum cost is `c` times minimum hop count. This is a change of units, not a
change of scheduling or visited semantics. The weighted boundary below concerns
costs that differ between edges; zero costs additionally remove strict cost
increase along a path.

For weights in `{0,1}`, 0-1 BFS uses a deque: a zero-cost relaxation goes to the
front and a unit-cost relaxation goes to the back.  Unlike ordinary BFS, a
vertex first encountered through a costlier path may later receive a better
distance.  The central operation is therefore **relaxation**, not irrevocable
first discovery.

This is naturally understood as a special bucketed shortest-path algorithm.
Dial's algorithm extends the idea to bounded nonnegative integer weights, and
Dijkstra's algorithm uses a general distance priority.  They are relatives of
BFS, but their layers are weighted-distance buckets rather than hop-distance
frontiers.

## Algorithms whose names or shapes can mislead

**Lexicographic BFS (LexBFS)** produces an ordering by repeatedly selecting a
vertex with a lexicographically maximal history of already selected
neighbours.  It is important in structural graph algorithms such as chordal
graph recognition. It respects BFS distance layers from its first vertex;
the distinction is its richer within-layer tie order, not a different hop
metric. It is neither sorted-adjacency BFS nor path-word shortlex selection
(see note 19).

**Beam search** scores candidates and retains only a bounded subset.  Once a
valid frontier state is discarded, the BFS completeness proof is gone; the
discarded branch may contain the only solution or the shortest solution.  Beam
search is therefore a heuristic search shaped like a frontier traversal, not
memory-bounded exact BFS.

**Iterative deepening DFS** repeatedly performs depth-limited DFS.  Under finite
branching it can recover the shallowest-solution guarantee with low memory, but
it regenerates shallow work and does not maintain BFS frontiers.  It is best
treated as a different search schedule with a separately proved guarantee.

## A classification test

When encountering a purported BFS variant, ask:

1. What exact object is returned: hop distances, weighted distances, one path,
   all shortest paths, an ordering, or merely a candidate?
2. What is the identity relation for states?
3. Which operation makes a distance final, and why can it not later improve?
4. Does the method compute every member of the mathematical next frontier?
5. What work may still be outstanding when it terminates?
6. Are ordering and parent choices part of the promised output, or incidental?
7. Which graph operations does it assume: outgoing edges, incoming edges,
   enumerable vertices, inverse generators, or a dense rank?

If these questions have precise answers, the name matters less: one can tell
whether the ordinary BFS theorem still applies or a different proof is needed.

## Sources and local evidence

- E. F. Moore, *The Shortest Path Through a Maze* (1959), the classical
  unweighted shortest-path formulation.
- C. Y. Lee, *An Algorithm for Path Connections and Its Applications*, IRE
  Transactions on Electronic Computers 10(3), 1961,
  [doi:10.1109/TEC.1961.5219222](https://doi.org/10.1109/TEC.1961.5219222).
- Ira Pohl, *Bi-Directional Search*, Machine Intelligence 6, 1971,
  [paper copy](https://aitopics.org/download/aiclassics%3A630E1F02).
- Scott Beamer, Krste Asanovic, and David Patterson,
  *Direction-Optimizing Breadth-First Search*, SC 2012,
  [paper](https://www.scottbeamer.net/pubs/beamer-sc2012.pdf),
  [doi:10.3233/SPR-130370](https://doi.org/10.3233/SPR-130370).
- Robert B. Dial, *Algorithm 360: Shortest-Path Forest with Topological
  Ordering*, CACM 12(11), 1969,
  [doi:10.1145/363269.363610](https://doi.org/10.1145/363269.363610).
- D. J. Rose, R. E. Tarjan, and G. S. Lueker,
  *Algorithmic Aspects of Vertex Elimination on Graphs*, SIAM Journal on
  Computing 5(2), 1976,
  [doi:10.1137/0205021](https://doi.org/10.1137/0205021), for LexBFS and its
  graph-ordering context.
- Local experiments REF-007 through REF-010 exercise target stopping,
  bidirectional policies, directed reversal, and distributed owner routing.
  They are evidence for particular models, not universal performance claims.

## Current synthesis

The deepest commonality is not a queue.  It is a proof that accepted labels are
the least path lengths under a stated edge-cost model.  Multi-source,
bidirectional, push/pull, and parallel BFS can preserve the ordinary hop metric.
0-1 BFS changes the metric and finalization rule.  LexBFS changes the output.
Beam search removes the exact guarantee.  Keeping those axes separate prevents
implementation resemblance from being mistaken for algorithmic equivalence.

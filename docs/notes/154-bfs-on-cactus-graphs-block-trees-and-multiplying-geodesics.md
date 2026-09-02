# BFS on cactus graphs: block trees and multiplying geodesics

A cactus graph permits many cycles but forbids them from overlapping in a
complicated way: any two simple cycles share at most one vertex. Equivalently
for the finite connected simple setting used here, every edge belongs to at
most one simple cycle.

This is the next exact step after trees and unicyclic graphs. Local parity
signatures survive, while the block-cut structure tells us how distances and
shortest-path multiplicities compose globally.

No experiment is used. The statements are proved from the cactus block tree.

## 1. Blocks form a tree of choices

The nontrivial blocks of a cactus are:

- bridge edges;
- simple cycles.

Their incidence with articulation vertices forms a block-cut tree. Therefore a
source `s` and target `t` determine one sequence of blocks. A path from `s` to
`t` cannot choose a different block sequence without creating overlapping
cycles; it can choose only how to cross each cycle block on that sequence.

This separates two levels:

```text
global route = unique block-tree path,
local choice = one of two arcs inside each traversed cycle.
```

## 2. Distance is a sum of local block distances

For each bridge block on the route, the contribution is one. For a cycle of
length `l` entered at vertex `a` and exited at vertex `b`, let `delta` be their
cyclic separation in one direction. Its distance contribution is

```text
min(delta, l-delta).
```

Adding these bridge and cycle contributions gives `dist(s,t)`. The sum is exact
because every `s-t` path must traverse the same blocks in the same order, and a
shortest local crossing can be selected independently in each block.

## 3. Shortest-path counts multiply

A bridge has one local crossing. A cycle has:

- one shortest arc if `delta != l/2`;
- two shortest arcs if `l` is even and `a,b` are antipodal.

Let `a(s,t)` be the number of cycle blocks on the block-tree route whose entry
and exit are antipodal. The independent local choices give

```text
sigma(s,t) = 2^a(s,t).
```

This counts ordinary unlabeled vertex paths in a simple cactus. Parallel labels
or directed arcs require a different path identity.

Path multiplicity can therefore be exponential in the number of blocks while
every individual vertex has at most two immediate shortest predecessors. A
small local predecessor count does not bound the number of complete shortest
paths.

## 4. What BFS sees at each cycle

From the root side, every cycle block has one entry articulation (or contains
the root). BFS sends two waves around the cycle:

- odd cycle `2k+1`: the waves end at adjacent vertices in one layer, producing
  one same-layer edge;
- even cycle `2k`: the waves meet at one antipode, producing two shortest
  predecessors for that vertex.

These are exactly the note-153 signatures, repeated once per cycle block. Trees
hanging from the far side inherit the distance and path-count result of the
cycle crossing without adding immediate-parent convergence of their own.

## 5. Cycle rank counts independent duplicate events

For a connected cactus with `c` cycle blocks,

```text
m-n+1 = c.
```

One way to see this is to begin with its spanning block tree and add one closing
edge per cycle. Each cycle therefore contributes exactly one non-tree edge to
any spanning tree and exactly one parity-controlled BFS signature:

```text
odd cycle  -> same-layer non-tree edge,
even cycle -> adjacent-layer alternative predecessor edge.
```

The events are structurally independent in the cycle space because cactus
cycles are edge-disjoint. Their effects on path counts are not additive:
successive antipodal even crossings multiply.

## 6. A three-cycle mental fixture

Consider a chain of three cycle blocks joined at articulation vertices:

```text
C4 -- C5 -- C6
```

Choose endpoints so that the route crosses `C4` and `C6` antipodally and crosses
`C5` non-antipodally. Then:

- `C4` produces one double-parent meeting and multiplier two;
- `C5` produces one same-layer edge somewhere in its full root wave, but the
  chosen entry-exit crossing has one shortest arc;
- `C6` produces another double-parent meeting and another multiplier two;
- the target has `2*1*2=4` shortest paths.

Only two vertices on this route need two immediate predecessors. The target may
have one predecessor and still represent four complete shortest paths.

## 7. Frontier width does not compose as simply

Pairwise distance and path count follow one block-tree route. A BFS frontier is
global: at one depth it may occupy many branches and many cycle arcs
simultaneously. Its size is a sum over all active branches, with local waves
starting at different offsets.

Therefore no scalar product rule determines `|F_d|`. Exact frontier profiles
depend on:

- distances from the source to articulation vertices;
- attached-tree profiles;
- cycle lengths and parity;
- simultaneous activation and exhaustion of branches.

The graph is decomposable without its frontier being one-dimensional.

## 8. Visited and output contracts

Ordinary exact BFS with visited handles every cactus without knowing its block
decomposition. Specialized traversal may exploit the decomposition, but must
still preserve the requested output:

- distance only: either equal antipodal predecessor suffices;
- one path/tree: choose one predecessor consistently;
- shortest-path DAG: retain both at every even antipodal meeting;
- path counts: propagate the accumulated `sigma`, not merely local indegree;
- all paths: output can grow as `2^a`.

Discarding the losing parent is sound for one-tree output and unsound for the
last three richer contracts.

## 9. Multi-owner interpretation

The block-cut tree suggests partitions with few cross-owner articulation links,
but low cut does not imply balanced BFS work. A large attached tree or wide
cycle wave can remain on one owner for many levels.

Even-cycle meetings are small exact fixtures for owner authority: two owners
may route proposals to one antipode owner. Downstream path-count metadata must
combine the two contributions once and then propagate; double-counting retries
or dropping one legitimate predecessor changes the output without changing
distance.

Cycle count bounds structural convergence events in a cactus, but not frontier
volume, edge scans, subtree size, or communication caused by the chosen
partition.

## 10. Boundary of the model

The product formula relies on the block route being a tree and cycle choices
being independent. It fails directly for theta graphs, where two vertices are
joined by three internally disjoint paths and cycles overlap along path
combinations. General graphs can have interacting cycle-basis elements that do
not correspond to independent geodesic choices.

Likewise, a realistic Cayley graph contains translated and overlapping
relations. A cactus is a useful control model for independent relators, not a
generic Cayley geometry.

## Sources and internal dependencies

- Note 11 gives the shortest-path DAG and path-count recurrence.
- Notes 31 and 82 give odd same-layer and general fundamental-cycle parity.
- Notes 152-153 provide the tree recurrence and exact one-cycle signatures.
- Notes 60, 61, 66, and 82 explain why Cayley relations, state equality, trace
  classes, and cycle-basis elements must not be conflated.
- The block-tree distance and multiplicity formulas are proved directly above
  for finite connected simple cactus graphs.

## Takeaway

A cactus turns BFS cycles into composable local choices. Parity decides whether
each cycle produces a same-layer edge or a double-parent meeting; the block tree
makes distances additive and antipodal choices multiplicative. Local duplicate
events can stay sparse while global shortest-path multiplicity grows
exponentially.

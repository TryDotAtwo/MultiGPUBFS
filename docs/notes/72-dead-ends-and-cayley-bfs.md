# Dead ends, pockets, and radial progress in Cayley BFS

A Cayley graph is regular and vertex-transitive, but BFS distance is rooted.
Those facts allow a useful surprise: a vertex may have all its usual generator
neighbors and still have no neighbor farther from the BFS root. Such a vertex
is a **dead end** for the chosen word metric.

## Definition and convention

Let `G` have a finite symmetric generator set `S`, and write `|g|` for distance
from the identity in the undirected Cayley graph. An element `g` is a dead end
when

```text
|g s| <= |g|  for every s in S.
```

Equivalently, no geodesic from the identity ending at `g` can be extended by
one edge to a longer geodesic.

Let `n=|g|` and `B_n={x:|x|<=n}`. This note calls

```text
escape_depth(g) = dist(g, G minus B_n)
```

the number of edges required to reach any state strictly farther from the root.
A non-dead endpoint with an outward neighbor has escape depth one; a dead end
has escape depth at least two. Some papers shift this quantity by one or use
related notions called depth, retreat depth, or strong depth. Numerical claims
must therefore repeat the adopted definition rather than quote “depth” alone.

In a finite graph every diameter-layer vertex is trivially a dead end. The
interesting finite case is a dead end at depth smaller than the diameter; in an
infinite Cayley graph every dead end is interior to the unbounded graph.

## A small example on the integers

Take the additive group `Z` with symmetric generators

```text
S={-3,-2,2,3}.
```

The element `1` has word length two because `1=3-2`, and no generator equals
`1`. Its four neighbors are

```text
1+2=3   with length 1
1-2=-1  with length 2
1+3=4   with length 2
1-3=-2  with length 1.
```

None lies in `F_3`, so `1` is a dead end in `F_2`. Nevertheless the group is
infinite and later BFS layers exist. One escape is

```text
1 -> 4 -> 7,
```

where `|4|=2` and `|7|=3`; hence the adopted escape depth of `1` is two.

With the standard generator set `{-1,1}`, the same group has no dead ends:
from every integer, one of the two steps increases absolute value. Dead ends
are therefore properties of a group together with a generator set, not of the
abstract group alone.

## Why regularity does not prevent a dead end

Every vertex of a simple Cayley graph has the same local degree, but “outward”
is defined relative to the distinguished BFS root. At `g in F_n`, generator
occurrences can lead to:

- `F_(n-1)`;
- `F_n`;
- `F_(n+1)`.

Undirected unit edges cannot skip farther between layers. A dead end is exactly
a vertex for which the third category is empty. It is not a graph-theoretic
leaf and does not have smaller local degree.

Vertex transitivity moves both the vertex and the root. It does not imply that
every vertex has an outward edge with respect to one fixed root.

## Geodesic consequence

Every prefix of a shortest path from the root is itself shortest. Along such a
path, distance therefore increases by exactly one at every edge. A dead end can
be the endpoint of a root geodesic, but it cannot be an internal vertex of a
root geodesic to a farther state.

Escaping a dead end is still possible in an infinite connected Cayley graph,
but every escape path must first spend at least one step without increasing
root distance. This is not a contradiction: that escape path is not a geodesic
from the original root through `g`.

## What BFS sees

Ordinary exact BFS needs no special dead-end rule. When expanding a dead-end
record in `F_n`, it generates all declared moves, but accepts no child into
`F_(n+1)` from that parent.

This separates several quantities:

```text
generated occurrences from g = labeled degree
outward occurrences from g   = 0
new states accepted from g   = 0.
```

Thus low discovery yield can be genuine word-metric geometry rather than a
visited-table or duplicate-removal problem. A regular implicit generator loop
still performs regular generation work even when its semantic outward yield is
zero.

One dead end does not terminate global BFS. Other vertices in the same frontier
may have outward neighbors, and exact termination occurs only when the union of
all frontier successors adds no new state. Treating a parent-local lack of new
children as a global exhaustion signal is incorrect.

## Five meanings of “leaf” or “no children”

Several different events can produce a zero child counter. They must not share
one undifferentiated `dead_end` label.

### Graph-theoretic leaf

A leaf of an undirected simple graph has degree one. A nontrivial connected
regular Cayley graph does not contain isolated exceptional degree-one vertices:
translation preserves degree. This notion concerns local adjacency, not root
distance.

### BFS-tree leaf

A vertex is a leaf of one selected BFS parent tree when no reached vertex chose
it as parent. It may still have forward neighbors: those children may have
selected another shortest predecessor under the tie rule. Changing parent
order can change tree leaves without changing any distance or frontier set.

### Radial dead end

A radial dead end has `b_i(v)=0`: no graph neighbor is farther from the fixed
root. This is independent of which one-parent BFS tree was retained. It remains
meaningful when all shortest predecessors are stored or no parents are stored.

### Terminal-layer vertex

Every vertex in the last nonempty layer of a completed finite-component BFS is
a radial dead end, but an interior layer may contain dead ends mixed with
outward gateways. Global termination means every vertex of the current
frontier has zero outward degree after complete exact processing.

### Execution-policy zero

A parent may emit or retain no child because expansion stopped at a target,
capacity overflowed, a beam/top-k policy pruned its candidates, a generator was
omitted, or records were lost. None of these outcomes proves `b_i(v)=0` in the
declared graph. A geometric dead-end claim requires complete successor coverage
and exact state/distance classification, not merely a zero output count.

## Dead end versus peripheral vertex

For a fixed root `s`, every farthest-layer vertex is a dead end relative to
`s`, but an interior dead end is not farthest. “Peripheral” is different again:
a vertex `v` is peripheral when its own eccentricity equals the graph diameter.

In a finite connected Cayley graph, translation makes every vertex have the
same eccentricity, equal to the diameter. Thus every vertex is peripheral as a
possible BFS root, while only some elements are radial dead ends relative to
the currently fixed identity root. Vertex transitivity does not erase this
root-relative distinction.

## Depth and memory intuition

Deep dead ends form pockets inside a metric ball. Starting at such a state, a
path must remain in the current root ball for many steps before it can reach a
larger radius. This is a statement about the geometry of paths between states;
it is not the number of BFS layers needed when BFS is already expanding the
whole root sphere synchronously.

Consequently:

- dead-end depth does not delay the globally correct discovery depth of a
  farther state;
- it does obstruct greedy policies that insist every chosen step increase root
  distance;
- it warns against interpreting a state-local outward score as a completeness
  certificate;
- it can change the distribution of forward yield across a frontier even in a
  regular Cayley graph.

No claim about GPU speed follows without measuring how such states are ordered,
batched, and partitioned. The algebra predicts zero accepted children for the
dead-end parents, not their physical cost or locality.

## Literature boundary

The phenomenon can be arbitrarily severe for some group/generator pairs:

- Cleary and Taback prove dead ends of arbitrary depth for lamplighter groups
  and other wreath products with natural generating sets.
- Cleary and Riley give a corrected construction of a finitely presented group
  with unbounded dead-end depth.
- Lehnert emphasizes generator dependence, proves bounded-depth results for
  broad classes, and introduces a stronger depth notion.

These results establish existence and dependence; this note does not transfer
their unbounded-depth conclusions to Megaminx, Cube, or CayleyPy's particular
generator sets.

## Sources

- Sean Cleary and Jennifer Taback,
  [Dead end words in lamplighter groups and other wreath products](https://arxiv.org/abs/math/0309344),
  *Quarterly Journal of Mathematics* 56 (2005).
- Sean Cleary and Tim R. Riley,
  [A finitely presented group with unbounded dead-end depth, corrected version](https://pi.math.cornell.edu/~riley/papers/Dead-End_Depth/deadend.html).
- Jorg Lehnert,
  [Some remarks on depth of dead ends in groups](https://arxiv.org/abs/math/0703636).

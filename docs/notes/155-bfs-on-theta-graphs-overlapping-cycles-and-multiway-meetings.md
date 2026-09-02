# BFS on theta graphs: overlapping cycles and multiway meetings

A theta graph consists of two branch vertices `x,y` joined by three internally
vertex-disjoint paths of lengths `L_1,L_2,L_3`. In a simple graph, at most one
path has length one.

This is the smallest clean obstruction to the cactus product picture. There are
three simple cycles, one for each pair of paths, but cycle-space dimension two.
The paths interact through their common endpoints, so shortest choices are no
longer independent block choices.

No experiment is used. The formulas below are exact for BFS rooted at `x`.

## 1. Endpoint distance and multiplicity

The distance between the branch vertices is

```text
d = dist(x,y) = min(L_1,L_2,L_3).
```

The number of shortest `x-y` paths is exactly the number of path lengths equal
to `d`. It can therefore be one, two, or three.

In particular, if

```text
L_1=L_2=L_3=L >= 2,
```

then `y` has three distinct predecessors in `F_(L-1)` and BFS expansion sees

```text
three candidate occurrences -> one unique state y.
```

This already exceeds the at-most-two local meeting of a cactus cycle. The
shortest-path count is three, not a power of two.

## 2. Distances inside one path depend on all paths

Let `p_(i,j)` be the internal vertex at distance `j` from `x` along path `i`,
where `0<j<L_i`. It can be reached directly from `x`, or by first reaching `y`
along a globally shortest path and then walking backward on path `i`. Hence

```text
dist(x,p_(i,j)) = min(j, d + L_i-j).
```

The formula is exact: any route entering the interior of path `i` must enter
from `x` or `y`, because the path interiors are disjoint.

This is the failure of cactus locality. The layer of a vertex on path `i`
depends on `d`, which may be determined entirely by another path.

## 3. A long path becomes a secondary wave corridor

Suppose `L_i>d`. Before `y` is discovered, BFS advances directly from `x` along
the long path. After a shortest path reaches `y`, another wave advances backward
from `y` along the same path.

Their effective closed-walk length is `d+L_i`:

- if `d+L_i` is even, the waves meet at one vertex with two immediate shortest
  predecessors;
- if `d+L_i` is odd, they meet across one same-layer edge.

This resembles the parity signature of one cycle, but the `x-y` side of that
cycle is not a fixed independent block. It is whichever globally shortest path
or paths established `d`.

If `y` already has `q` shortest paths, the backward wave carries path count `q`.
A later vertex may have only one predecessor from the `y` side while inheriting
all `q` histories through it.

## 4. Cycle rank is two, simple cycles are three

The theta graph has

```text
m = L_1+L_2+L_3,
n = 2 + sum_i(L_i-1) = L_1+L_2+L_3-1,
m-n+1 = 2.
```

Yet its three simple cycles have lengths

```text
L_1+L_2,  L_1+L_3,  L_2+L_3.
```

Over `F_2`, the symmetric difference of any two of these cycles is the third.
Thus cycle rank counts independent binary cycle vectors, not simple cycles,
candidate multiplicity, shortest paths, or BFS duplicate events one-for-one.

## 5. Three fixtures with the same topology class

### Equal paths: `Theta(3,3,3)`

All three waves reach `y` at depth three. The only branch-vertex convergence is
three-way, and `sigma(x,y)=3`.

### One short, two odd-effective paths: `Theta(2,3,3)`

The length-two path discovers `y`. For each length-three path,
`d+L_i=5` is odd, so its direct and backward waves meet across a same-layer
edge. Two independent cycle-space directions appear as two same-layer events.

### One short, two even-effective paths: `Theta(2,4,4)`

Again `d=2`, but now `d+L_i=6`. Each long path contains one vertex where direct
and backward waves meet with two shortest predecessors. The endpoint `y` itself
has only one shortest path.

These fixtures have the same cycle rank two while moving duplicate work among a
three-way endpoint meeting, same-layer scans, and two separate double-parent
meetings.

## 6. Frontier and visited interpretation

The BFS frontier is a superposition of:

- direct waves launched from `x` on all paths;
- backward waves launched on long paths after `y` is reached;
- convergence or same-layer closure determined by `d+L_i` parity.

Visited is not merely eliminating repeated traversal around a named cycle. It
merges proposals whose causal histories may use different overlapping cycles.
Classifying a duplicate only by one selected cycle basis can hide the actual
meeting multiplicity.

For distance-only output, one winning proposal is enough. For a one-parent tree,
one predecessor is retained. For a shortest-path DAG or counts, every equal-
depth predecessor and its accumulated `sigma` contribution matters.

## 7. Multi-owner fixture

Place the three path interiors on three producers and let one owner be
authoritative for `y`. In `Theta(L,L,L)`, the owner receives three legitimate
same-level proposals.

This tiny graph checks that a distributed traversal:

- accepts `y` once into the next frontier;
- records distance `L` independently of arrival order;
- retains one, canonical, or all parents according to the output contract;
- sums exactly three shortest-path contributions when counts are requested;
- does not confuse message retry with a distinct graph predecessor.

The fixture tests semantics, not throughput. Its value is that all three
proposals are mathematically necessary for rich output but only one frontier
slot is valid.

## 8. Cayley interpretation

Three internally disjoint equal-length paths between two states correspond to
three geodesic word families with the same endpoint. Pairwise concatenation with
inverses yields three cycle words, but only two are independent in binary cycle
space.

Group relations are richer than binary edge sets: labels, orientation,
reduction, and translated copies matter. Still, the theta graph is a useful
warning that "one duplicate equals one relation" and "cycle rank equals number
of geodesic alternatives" are both false.

In a Cayley graph, translation can reproduce the same local theta relation at
every vertex, making a tiny semantic pattern a large global source of candidate
convergence.

## 9. Boundary and next step

The distance formula used three paths with disjoint interiors and no extra
edges. Chords or additional cross-links create more entry points and invalidate
the two-endpoint minimum formula.

More generally, overlapping shortest-path DAGs are the right object for path
multiplicity. A chosen cycle basis is an algebraic coordinate system, not a
decomposition of BFS work.

## Sources and internal dependencies

- Note 11 gives predecessor and shortest-path-count semantics.
- Note 74 separates candidate records from accepted frontier states.
- Notes 82 and 154 give cycle-space rank and the cactus independence boundary.
- Note 90 connects internally disjoint paths to connectivity rather than
  shortestness.
- Notes 60, 61, 66, and 82 provide the Cayley relation/equality distinctions.
- All theta formulas and fixtures above follow directly from the declared
  three-path graph.

## Takeaway

Overlapping cycles couple BFS waves through shared endpoints. A cycle space of
dimension two can contain three simple cycles, a three-way frontier meeting, or
two separate parity events. Cycle bases describe algebraic independence; they
do not count geodesics, duplicates, or accepted states.

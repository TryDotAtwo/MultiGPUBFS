# BFS degree--diameter capacity: the other Moore bound

Note 27 used collision-free BFS trees to lower-bound the order needed for a
regular graph of prescribed girth. The degree--diameter problem turns the same
tree picture around: with maximum degree and diameter fixed, how many vertices
can the entire graph contain? The resulting expression is also called a Moore
bound, but it is an upper bound on order. This note separates the two uses and
adds no graph construction, optimizer, benchmark, or GPU code.

## 1. The degree--diameter question

Let `G` be a finite connected simple undirected graph with maximum degree
`Delta` and diameter at most `D`. From any root `s`, every vertex belongs to
one of the BFS layers `F_0(s),...,F_D(s)`.

The root has at most `Delta` neighbors. Every later vertex already uses at
least one incident edge to connect toward an earlier layer, so each occurrence
in a collision-free search tree can offer at most `Delta-1` forward branches:

```text
|F_0| = 1
|F_i| <= Delta * (Delta-1)^(i-1),  1 <= i <= D.
```

Summing the layers gives the undirected Moore upper bound

```text
M(Delta,D) = 1 + Delta * sum_(j=0)^(D-1) (Delta-1)^j.
```

For `Delta>2`,

```text
M(Delta,D) = 1 + Delta * ((Delta-1)^D - 1)/(Delta-2),
```

and for `Delta=2`, `M(2,D)=2D+1`. Thus `|V|<=M(Delta,D)`.

This is a BFS counting proof. It needs neither regularity nor a chosen parent
policy: maximum degree and shortest-distance layers suffice.

## 2. Why `Delta-1` is a capacity, not measured branching

The bound imagines that every nonroot tree occurrence spends exactly one edge
toward its parent and every other edge discovers a new vertex. Real layers can
be smaller because of:

- degree below `Delta`;
- two or more previous-layer vertices reaching the same state;
- edges within one layer;
- additional edges back toward older layers;
- finite closure before depth `D`.

These effects all reduce new vertices, but they are not interchangeable event
counts. One convergence may correspond to several duplicate candidates, while
a missing incident edge creates no candidate at all. Therefore
`Delta(Delta-1)^(i-1)-|F_i|` is a layer-capacity deficit, not automatically the
number of duplicates produced by an implementation.

## 3. Diameter is stronger than one-root eccentricity

The counting argument for a chosen root only needs `ecc(s)<=D`. Graph diameter
at most `D` guarantees that condition for every root. Conversely, a shallow
BFS from one central root does not prove graph diameter `D`; another pair may
be farther apart.

If the graph is disconnected, ordinary finite diameter is not available for
the whole vertex set. The bound applies to the connected component reached by
the root, with its own eccentricity or diameter contract.

## 4. Equality is rigid

A graph attaining `|V|=M(Delta,D)` is called a Moore graph. Equality forces
every layer capacity to be tight from every root:

- every vertex has degree `Delta`;
- no distinct non-backtracking root paths of length at most `D` converge;
- no capacity is lost to premature same-layer or older-layer closure;
- every vertex is reached within `D`.

Consequently an undirected Moore graph has the corresponding geodetic,
large-girth structure; equality is rare, not the generic behavior of a
bounded-degree graph.

Calibrations include:

- `K_(Delta+1)` for diameter one;
- the odd cycle `C_(2D+1)` for `Delta=2`;
- the Petersen graph for `(Delta,D)=(3,2)`, with `10=1+3+6` vertices.

By contrast, the three-cube `Q_3` has degree three, diameter three, and eight
vertices, while the Moore capacity is `1+3+6+12=22`. Its many square relations
make the tree capacity extremely loose.

## 5. Defect can be localized by BFS layer

For a root with `ecc(s)<=D`, define capacities

```text
C_0 = 1,
C_i = Delta*(Delta-1)^(i-1),
delta_i(s) = C_i - |F_i(s)| >= 0.
```

Then the total Moore defect is exactly

```text
M(Delta,D) - |V| = sum_(i=0)^D delta_i(s).
```

This identity says where potential tree capacity was lost. It still does not
say why it was lost. Explaining a positive `delta_i` needs degree, boundary-edge,
relation, and duplicate evidence such as the distinctions in notes 10, 27, 60,
and 63.

In a vertex-transitive graph, including a genuine Cayley graph, the sphere-size
profile and hence the deficit profile are root-independent. In an irregular
graph, the same total defect can be distributed differently across roots.

## 6. The girth Moore bound is the other direction

Note 27's odd-girth lower bound for a `Delta`-regular graph of girth `2D+1` has
the same algebraic sum:

```text
|V| >= 1 + Delta * sum_(j=0)^(D-1) (Delta-1)^j.
```

There the absence of short cycles forces a radius-`D` BFS tree to fit inside
the graph. Here diameter at most `D` forces the whole graph to fit inside such
a tree. A Moore graph makes both statements tight and connects diameter `D`
with girth `2D+1`.

For even prescribed girth, the cage lower bound grows from an edge-centered
tree and has a different expression. Calling every one of these formulas "the
Moore bound" without naming degree--diameter versus degree--girth reverses the
inequality and loses the root geometry.

## 7. Directed graphs use a different tree

For a digraph with maximum out-degree `d` and directed diameter at most `D`, a
root can have at most `d^i` newly reachable vertices at directed distance `i`:

```text
|V| <= 1 + d + d^2 + ... + d^D.
```

There is no automatic `d-1` factor because following an outgoing arc does not
consume an inverse outgoing arc at the child. Applying the undirected formula
to a positive move alphabet is therefore wrong unless the graph is first given
a justified symmetric undirected interpretation.

Maximum out-degree alone also says nothing about reverse reachability. Directed
diameter requires the declared strongly connected reachability convention.

## 8. Cayley and Schreier interpretation

For a finite Cayley graph with `n=|G|`, true simple degree `Delta`, and diameter
`D`,

```text
|G| <= M(Delta,D).
```

This is an information constraint on how many group elements a bounded-length
generator alphabet can cover. The label count is not always the simple degree:
identity generators create loops, duplicate permutations create parallel
labels, and inverse conventions change whether arcs are merged.

Restricting the degree--diameter problem to Cayley, Abelian Cayley, circulant,
or Schreier graphs can lower the best attainable order because algebraic
symmetry is an additional constraint. The unrestricted Moore bound remains a
valid ceiling, not evidence that a Cayley construction near it exists.

Changing generators changes both degree and diameter. Saying that an added
move reduced diameter is incomplete for degree--diameter comparison unless the
new degree, state count, and graph convention are recorded as well.

## 9. What the bound says about BFS resources

The Moore sum gives a worst-case reachable-state capacity under maximum degree
and depth. It can support conservative bounds such as

```text
|B_D(s)| <= min(|V|, M(Delta,D)),
|F_i(s)| <= min(|V|, Delta*(Delta-1)^(i-1)).
```

It does not predict:

- the actual peak frontier;
- candidate or duplicate multiplicity;
- state encoding size;
- edge-generation cost;
- owner balance or communication volume;
- whether the graph is explicit or implicit.

For multi-GPU reasoning, global Moore capacity can bound how many distinct
states exist in the declared depth scope. Per-device memory and routing still
depend on partitioning and the realized layer profile. A loose combinatorial
ceiling is not a hardware sizing measurement.

## 10. Research discipline

For every degree--diameter claim, record:

1. simple, directed, labeled, or multigraph semantics;
2. maximum degree versus generator-label count;
3. diameter, radius, or one-root eccentricity;
4. exact state count and connectivity scope;
5. layer capacities and realized layer sizes;
6. total defect and, when useful, its per-layer distribution;
7. whether a cited Moore expression is an order upper bound or a girth-based
   order lower bound;
8. separate mathematical capacity from measured GPU memory and throughput.

## Sources

- M. Miller and J. Sirán,
  [*Moore Graphs and Beyond: A Survey of the Degree/Diameter Problem*](https://doi.org/10.37236/35),
  Electronic Journal of Combinatorics, Dynamic Survey DS14. Gives the standard
  undirected and directed Moore bounds and surveys Cayley restrictions.
- R. Dougherty and V. Faber,
  [*The Degree-Diameter Problem for Several Varieties of Cayley Graphs I: The Abelian Case*](https://doi.org/10.1137/S0895480100372899),
  SIAM Journal on Discrete Mathematics 17(3), 2004. Connects the extremal
  problem to Abelian Cayley constructions and lattice coverings.
- Notes 10, 27, 35, 39, 46, 60, 63, and 93 provide this repository's frontier,
  girth, growth-series, word-tree, resource, relation, Megaminx, and generator
  change distinctions.

## Takeaway

The degree--diameter Moore bound is the capacity of an ideal collision-free BFS
tree large enough to contain the whole graph. It is an upper bound, not a growth
forecast. Its gap from reality can be decomposed by layer, but that defect is
not a duplicate counter. The same-looking girth formula points in the opposite
direction, and directed move graphs require the directed tree sum.

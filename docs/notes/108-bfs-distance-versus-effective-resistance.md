# BFS distance versus effective resistance

BFS distance asks for the length of one shortest route. Effective resistance
treats every undirected edge as an electrical conductor and lets all routes
carry current simultaneously. The two quantities agree on trees and diverge
when cycles provide parallel alternatives.

This note studies that geometric distinction and its random-walk connection.
It adds no Laplacian or resistance solver.

## 1. Unit-resistance graph model

Let `G` be a finite connected undirected graph. Replace every edge by a
one-ohm resistor. Inject one unit of current at `u` and remove it at `v`. The
resulting voltage difference is the effective resistance

```text
R_eff(u,v).
```

Equivalently, `R_eff` is the minimum energy of a unit `u-v` flow. Current can
split among every available path, unlike a shortest-path algorithm that
selects one minimum-length witness.

The standard model is symmetric and undirected. Directed graphs need a
separately defined generalization; silently symmetrizing arcs changes directed
BFS reachability.

## 2. Equality on a tree

In a tree there is exactly one `u-v` path. Every unit of current must traverse
each edge of that path. Unit resistors in series add, so

```text
R_eff(u,v)=d(u,v).
```

Branches hanging off the path carry no net current to the sink and do not alter
the value. Thus tree geometry is the special case where "all routes" and "the
shortest route" describe the same unique route.

## 3. Resistance never exceeds unit-edge distance

Take any shortest `u-v` path of length `d(u,v)` and ignore every other edge.
That path alone has resistance `d(u,v)`. Restoring additional edges cannot
increase effective resistance by Rayleigh monotonicity, because they provide
more possible current flow.

Therefore

```text
R_eff(u,v) <= d(u,v)
```

in every connected unit-resistance graph.

This is a one-way bound. Small resistance does not imply small BFS distance;
many long parallel paths can collectively have low resistance.

## 4. Parallel paths create an arbitrary gap

Suppose `u` and `v` are joined by `k` internally disjoint paths, each of length
`L`, and there are no other relevant edges. Each path is a series resistance
`L`; the `k` equal branches are in parallel. Hence

```text
d(u,v)=L,
R_eff(u,v)=L/k.
```

The BFS distance remains `L` as `k` grows, while effective resistance tends to
zero. Redundant routing is invisible to the minimum path length but central to
electrical flow.

An even sharper fixed-distance example has one direct edge plus `k` internally
disjoint two-edge paths. Then

```text
d(u,v)=1,
R_eff(u,v)=1 / (1 + k/2) = 2/(k+2).
```

Every graph in the family has the same BFS distance one.

## 5. Complete graph calibration

In `K_n`, every distinct pair has BFS distance one. Symmetry and the parallel
network calculation give

```text
R_eff(u,v)=2/n.
```

As the complete graph grows denser, distance remains one while resistance
shrinks. Degree and route multiplicity affect resistance even after BFS has
already reached its minimum possible nonzero distance.

## 6. A BFS profile does not determine resistance

Layer sizes record radial vertex counts from a root. Effective resistance also
depends on how edges connect within and between those layers and on alternative
routes outside one selected shortest-path corridor.

Two graphs can share a root distance profile while differing in same-layer
edges, cross-layer multiplicities, cycles, and cuts, as note 101 already shows.
Those differences can change current flow and resistance.

Conversely, one scalar resistance does not recover frontier sizes, exact
distance, parents, or a shortest path. It is a global network summary with a
different information loss.

## 7. Connection to random-walk commute time

For the simple random walk on a finite connected unweighted graph with `m`
undirected edges, let `H(u,v)` be expected hitting time from `u` to `v`. The
commute time satisfies

```text
H(u,v)+H(v,u)=2m R_eff(u,v).
```

This explains why BFS and random-walk time can differ so strongly:

- BFS follows minimum hop distance deterministically;
- random walk repeatedly backtracks and samples alternatives;
- effective resistance measures route redundancy;
- the global edge count supplies the remaining scale in commute time.

Distance alone cannot determine commute time. Even `R_eff` alone cannot do so
without `m` under this formula.

## 8. Edge weights use a different algebra

For an electrical edge with resistance `r_e`, conductance is `c_e=1/r_e`.
Series resistances add; parallel conductances add. A weighted shortest path
instead minimizes

```text
sum_(e in P) w_e
```

over individual paths. Even if `w_e=r_e`, the network effective resistance is
not generally the minimum path resistance because parallel paths cooperate.

Zero, negative, asymmetric, or time-dependent search weights do not fit the
ordinary positive resistor model without a new definition.

## 9. Laplacian viewpoint

Let `L` be the weighted graph Laplacian and `L^+` its pseudoinverse. Effective
resistance can be written

```text
R_eff(u,v) = (e_u-e_v)^T L^+ (e_u-e_v).
```

This formula exposes its global nature: changing an edge far from one chosen
shortest path can still change the solution if it creates a useful alternate
current route.

BFS uses local frontier expansion plus exact `visited`. Resistance computation
uses a global linear-system/energy object. Shared sparse-matrix primitives do
not make the semantics or stopping certificates equal.

## 10. Cayley interpretation

In a finite undirected unit-generator Cayley graph, left translation preserves
both word distance and the electrical network, so

```text
d(g,h)=d(e,g^-1 h),
R_eff(g,h)=R_eff(e,g^-1 h).
```

Translation invariance alone does not prove that resistance is a function of
word length. Vertex transitivity makes every root equivalent, but it does not
necessarily make all elements in one sphere equivalent under root-fixing
automorphisms. Radial dependence only on word length needs an additional
symmetry theorem or a direct resistance calculation; lack of that symmetry is
not by itself a counterexample.

Generator changes alter degree, relations, shortest paths, total edge count,
and electrical conductance simultaneously. Abstract group identity alone fixes
neither metric.

For Schreier graphs, the same orbit/state and action conventions are required
before either distance or resistance is meaningful.

## 11. GPU and multi-GPU interpretation

BFS and resistance workloads can both use sparse matrix-vector operations, but
their end-to-end work differs:

- BFS produces exact discrete layers and state identities;
- resistance commonly requires solving or approximating Laplacian systems;
- iterative solver convergence is not BFS frontier exhaustion;
- floating-point residual tolerance is not exact state equality;
- multi-GPU BFS communicates discoveries and owners;
- distributed linear solves communicate vector values and global reductions.

A fast Laplacian SpMV is useful primitive evidence, not an exact BFS benchmark;
the converse is equally false. No solver or hardware policy is selected here.

## 12. Evidence checklist

1. Finite connected undirected graph or declared generalization.
2. Edge resistance/conductance and shortest-path weight conventions.
3. BFS distance, effective resistance, hitting time, or commute time output.
4. Simple graph, parallel edges, and total undirected edge count.
5. Exact versus approximate linear-system tolerance.
6. One pair, all pairs, one root, or sampled pairs.
7. Cayley/Schreier action and generator semantics.
8. Frontier work versus Laplacian-solver work.

## Sources

- P. G. Doyle and J. L. Snell, [*Random Walks and Electric
  Networks*](https://math.dartmouth.edu/~doyle/docs/walks/walks.pdf),
  Mathematical Association of America, 1984; freely redistributed edition,
  2006. Electrical reductions, flows, potentials, and random-walk connections.
- A. K. Chandra, P. Raghavan, W. L. Ruzzo, R. Smolensky, and P. Tiwari,
  [*The Electrical Resistance of a Graph Captures Its Commute and Cover
  Times*](https://doi.org/10.1145/73007.73062), STOC 1989, 574-586. Establishes
  `commute(u,v)=2m R_eff(u,v)` for unit-resistance undirected graphs.
- Notes 10, 11, 32, 33, 46, 48, 60, 78, 93, 95, 101, and 107 provide frontier,
  shortest-path, intersection, walk, expansion, separator, relation, landmark,
  Cayley-metric, random-walk, profile, and spanning-tree context.

## Takeaway

BFS distance measures the nearest route; effective resistance measures how the
whole undirected network carries flow. They coincide on a tree and satisfy
`R_eff<=d` for unit edges, but parallel paths can make the gap arbitrarily
large. The commute-time identity connects resistance to random walks, not to
deterministic first-arrival BFS. Shared graph primitives do not erase those
semantic differences.

# BFS on random geometric graphs: spatial waves and boundaries

A random geometric graph places graph adjacency inside a metric space: two
sampled points are adjacent exactly when their geometric distance is at most a
radius `r`.  BFS is still an exact hop-distance algorithm, but its layers now
look like a noisy physical wave constrained by density, holes, and the domain
boundary.

This note adds no optimizer, production implementation, benchmark, or GPU code.
The retained Rust program is a transparent semantic probe.

## 1. Square and torus models

Sample `n` points independently and uniformly from `[0,1)^2`.  The probe builds
two graphs on the same points and radius:

- **square:** ordinary Euclidean distance;
- **flat torus:** each coordinate difference is
  `min(|dx|,1-|dx|)`, removing the external boundary.

Each unordered pair is tested once and the resulting graph is frozen before
BFS.  The torus is not merely an implementation trick: wraparound edges change
the graph, metric, components, and shortest paths.

In the torus, expected degree is approximately

```text
pi n r^2.
```

The square loses part of a radius disk near edges and corners, so boundary
vertices have lower expected degree and the graph-wide mean is smaller.

## 2. Connectivity scale

In two dimensions the leading connectivity scale is

```text
r_c approximately sqrt(log(n)/(pi n)).
```

This is an asymptotic threshold scale, not a statement that a finite graph is
connected exactly at multiplier one.  Boundary convention, the finite critical
window, and random isolated vertices still matter.  Penrose's result that the
minimum radius for connectivity asymptotically coincides with disappearance of
isolated vertices gives the scale its structural meaning.

The probe used multipliers `0.8,1.0,1.3,2.0` of this scale at `n=2000`.

## 3. Deterministic metric lower bound

Every graph edge has geometric length at most `r`.  By the triangle inequality,
every reachable pair satisfies exactly

```text
d_G(u,v) >= ceil(d_E(u,v)/r).
```

The same proof applies with torus distance.  This is a one-sided bound: geometry
alone does not supply intermediate sampled vertices.  Empty regions and narrow
passages can force detours, while disconnected pairs have infinite graph
distance.

The probe asserted this inequality for every reached center-root pair.  It
measured the dimensionless stretch `d_G r / d_E` only for `d_E>=r`.

## 4. Density heals detours

Pair-weighted center-root observations were:

```text
multiplier     square stretch    torus stretch   maximum hop excess
0.8                 2.452             2.901          72 / 88
1.0                 1.563             1.557          19 / 19
1.3                 1.344             1.342           8 / 7
2.0                 1.240             1.239           2 / 2
```

The maximum excess is `d_G-ceil(d_E/r)` over retained reached pairs.  Increasing
radius adds both direct edges and alternate intermediate points, so paths align
more closely with the geometric lower bound in these samples.

The sparse `0.8` rows compare different reachable-pair populations and must not
be read as a clean square-versus-torus ranking.  Pair-weighting also gives large
components more influence than small ones; the aggregation contract is part of
the result.

Asymptotic upper bounds relating graph and Euclidean distance require density
conditions.  The deterministic lower bound alone never proves small stretch.

## 5. Boundary changes degree and reach

The measured graph-wide means were:

```text
multiplier   square degree   torus degree   square connected   torus connected
0.8              4.74           4.86             0/20              0/20
1.0              7.36           7.58             0/20              6/20
1.3             12.34          12.83            16/20             19/20
2.0             28.63          30.39            20/20             20/20
```

The torus degrees closely match `multiplier^2 log n`; square means are lower
from clipped neighborhoods.  Connectivity counts are finite observations, not
threshold estimates.

At multiplier `1.3`, mean eccentricities of the vertices nearest the geometric
center/corner were

```text
square: center 20.55, corner 38.55
torus:  center 19.95, corner 19.95.
```

When a graph is disconnected, the probe's maximum finite BFS depth is only
component eccentricity.  It is not graph diameter and does not include infinite
distances.  The square corner also has farther geometric opposite points and a
clipped local neighborhood; the torus makes every location statistically
equivalent.

## 6. A frontier is an annulus with defects

In a dense homogeneous region, depth `d` roughly selects points around geometric
radius `dr`, but the layer is not an exact circle:

- edge lengths range from nearly zero to `r`;
- a shortest path may fail to advance a full radius each hop;
- local holes bend or split the wave;
- collisions merge many geometric routes at one vertex;
- square boundaries truncate the wave;
- disconnected pockets stop it entirely.

Thus Euclidean radius, BFS depth, layer thickness, and number of frontier
vertices are related but distinct random variables.  Note 105's fixed-grid
stencil waves and this random unit-disk wave are not interchangeable models.

## 7. Spatial ownership: traffic versus time-local balance

The probe compared two owners:

```text
spatial owner: x<0.5 versus x>=0.5,
striped owner: vertex ID parity.
```

Remote-edge fractions were:

```text
multiplier    square spatial   torus spatial   striped
0.8               0.0120          0.0239       about 0.50
1.3               0.0191          0.0371       about 0.50
2.0               0.0295          0.0565       about 0.50
```

The torus spatial cut has two interfaces—near `x=0.5` and across the wrap at
`x=0/1`—which explains roughly twice the square fraction.  Spatial ownership
strongly reduced total cross-owner edges in these fixtures.

It did not guarantee per-level balance.  In the representative square graph at
multiplier `1.3`, a corner-root wave remained entirely on owner zero through
depth 14 before crossing.  A center-root wave touched both owners almost
immediately and became roughly balanced near its broad middle layers.

Therefore low edge cut, balanced total vertex count, balanced current frontier,
and simultaneous GPU utilization are different objectives.  Hash-like
ownership produces near-half remote traffic but tends to distribute every wave
earlier; spatial ownership produces locality but can serialize the passage of a
wave across domains.

## 8. Measurement and implementation boundaries

A useful spatial BFS observation records:

- domain metric and boundary convention;
- point process, seed, radius, and root-selection rule;
- component conditioning and unreachable vertices;
- frontier size, degree mass, owner mass, and geometric extent by depth;
- hop lower-bound excess and its pair-weighting convention;
- local/remote edge occurrences and owner interfaces;
- generation, adjacency construction, visited, routing, synchronization, and
  end-to-end time separately.

Efficient neighbor generation would normally use spatial indexing, but this
probe intentionally checks all `O(n^2)` pairs.  It studies graph meaning, not a
neighbor-search implementation.

## 9. Docker/Rust probe and retained failures

`experiments/random_geometric_bfs_probe.rs` uses the same deterministic point
sets across radii and across square/torus variants.  It checks the metric lower
bound for every retained center-root pair and runs only in Docker.

The first gate stopped on three `rustfmt --check` changes.  The first successful
execution then exposed `NaN` stretch when a sparse component had no reached pair
with `d_E>=r`.  Those aggregate values were rejected.  The probe now accumulates
ratio sums and explicit eligible-pair counts across samples; every value was
recomputed.  The final format, compile, assertions, and execution passed.

The CPU-only container did not request GPU access.  None of these measurements
is a throughput benchmark.

## Sources

- M. D. Penrose,
  [*On k-connectivity for a geometric random graph*](https://doi.org/10.1002/(SICI)1098-2418(199909)15:2%3C145::AID-RSA2%3E3.0.CO;2-G),
  *Random Structures & Algorithms* 15(2), 1999, for the asymptotic equivalence
  between the minimum radii for `k`-connectivity and minimum degree `k`.
- J. Díaz, D. Mitsche, G. Perarnau, and X. Pérez-Giménez,
  [*On the relation between graph distance and Euclidean distance in random geometric graphs*](https://doi.org/10.1017/apr.2016.31),
  *Advances in Applied Probability* 48(3), 2016; also
  [arXiv:1404.4757](https://arxiv.org/abs/1404.4757), for the deterministic
  lower bound, connectivity scale, and density-dependent distance bounds.
- Notes 07, 10, 29, 47, 48, 51, 71, 73, 74, 105, 119, 144, and 146 provide this
  repository's GPU, geometry, separator, owner, arbitrary-frontier, measurement,
  capacity, random-graph, and partition boundaries.

## Takeaway

Random geometric BFS is a discrete spatial wave, but hop distance is not
Euclidean distance divided by `r`.  That quotient is only a lower bound; density
and holes determine the excess.  Domain boundaries alter degrees, eccentricity,
and connectivity, while spatial partitioning trades low communication against
depth-dependent owner imbalance.

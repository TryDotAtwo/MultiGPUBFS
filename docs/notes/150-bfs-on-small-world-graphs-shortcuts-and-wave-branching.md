# BFS on small-world graphs: shortcuts and wave branching

A small-world graph combines a local substrate with a sparse set of long-range
edges.  From the viewpoint of BFS, each shortcut can seed a new local wave far
ahead of the original one.  A few such seeds can collapse eccentricity and
average distance long before they materially change mean degree.

This note adds no optimizer, production implementation, benchmark, or GPU code.
The Rust probe is a fixed-count semantic experiment.

## 1. Exact model contract

The original Watts-Strogatz model rewires local edges.  The related
Newman-Watts model adds shortcuts without deleting the lattice.  The probe uses
an explicit fixed-count additive variant:

1. vertices are `0,...,n-1` on a ring;
2. every vertex is joined to offsets `+-1,+-2`, giving the square of a cycle
   `C_n^2` and degree four;
3. exactly `s` uniformly proposed nonloop, nonduplicate long-range edges are
   added;
4. the resulting graph is frozen before BFS.

Fixing `s` removes shortcut-count variance.  This is not silently identified
with either canonical paper ensemble; transfer requires matching the model.

Because edges are only added, connectivity is preserved and every new distance
obeys the deterministic monotonicity

```text
d_(G+shortcuts)(u,v) <= d_G(u,v).
```

The shortcut graph is a different unit-edge metric, not a faster
implementation of BFS on the original ring.

## 2. Baseline ring wave

For `C_n^2`, let `delta` be cyclic ring distance.  Exact graph distance is

```text
d(u,v)=ceil(delta(u,v)/2).
```

At `n=4096`, root `1024` has:

```text
eccentricity 1024,
mean distance 512.25,
frontier size four at almost every nonterminal depth,
frontier peak four.
```

This is a one-dimensional wave with two directions and two lattice steps per
hop.  Its huge depth offers little parallelism per level despite fixed degree
and complete connectivity.

## 3. A shortcut creates new wave sources

When BFS reaches one endpoint of a shortcut, the opposite endpoint appears in
the next layer and begins expanding through its own local ring neighborhood.
With several shortcuts, the search evolves through three overlapping phases:

- initial local growth before a useful endpoint is reached;
- branching into multiple spatially separated local waves;
- collision and depletion when those waves meet.

The frontier is not a branching tree: each spawned wave has a local boundary,
and later waves merge through visited.  Shortcut endpoint spacing suggests a
crossover length, but exact onset depends on root and placement.

## 4. Distance collapse at nearly fixed degree

Twenty deterministic samples at `n=4096` gave:

```text
shortcuts  mean degree  eccentricity  mean distance  frontier peak
0             4.000       1024.00         512.25          4.00
4             4.002        552.90         279.24         16.25
16            4.008        255.50         138.63         42.30
64            4.031         85.75          46.47        130.40
256           4.125         33.70          19.77        414.05
1024          4.500         14.65           9.30        999.15
```

Only 64 added edges—`1.56%` as many as vertices—cut mean root distance by a
factor of about eleven while adding `0.031` to mean degree.  Edge count and mean
degree therefore do not predict BFS depth, peak width, or synchronization count.

The fraction of vertices whose distance improved over the ring was

```text
0, 0.6755, 0.8582, 0.9538, 0.9801, 0.9908.
```

One shortcut can improve many vertices because paths use it and then continue
locally.  Shortcut count is not the count of affected destinations.

## 5. Frontier shape changes qualitatively

The zero-shortcut frontier is almost flat at four.  With four shortcuts, the
mean peak rose to `16.25`: a handful of separated local intervals were expanding
at once.  At 64 shortcuts, a representative peak window was

```text
[104,109,124,132,128,121,127].
```

At 1024 shortcuts the representative complete profile was

```text
[1,5,9,15,30,69,163,317,589,890,985,680,279,57,7].
```

This resembles random-graph expansion and contraction even though the local
ring still supplies most edges.  The same graph family moves from high-span,
narrow-level BFS to low-span, wide-level BFS as shortcuts are added.

## 6. Ownership trade-off

The probe compared:

- contiguous owners: first and second halves of the ring;
- striped owners: vertex-ID parity.

The root `n/4` lies in the middle of the first contiguous half, deliberately
away from both partition interfaces.  Results were:

```text
shortcuts  first mixed-owner depth  contiguous remote  striped remote
0                    512.00              0.0007             0.5000
4                    136.90              0.0010             0.5000
16                    58.30              0.0017             0.5000
64                    12.25              0.0045             0.5000
256                    5.50              0.0159             0.5001
1024                   1.95              0.0562             0.5006
```

Approximately half of the shortcut edges crossed the contiguous owner cut, as
expected for uniformly placed endpoints.  Yet total remote fraction stayed
small because the many local ring edges remained owner-local.

This exposes two opposing effects:

- shortcuts increase cross-owner communication;
- shortcuts allow distant owners to receive useful frontier work much earlier.

The no-shortcut partition has almost zero communication but leaves the second
owner without frontier vertices for about 512 levels.  Low edge cut is not
equivalent to low time-to-parallelism.  Conversely, an edge-fraction statistic
understates the global influence of one routed shortcut occurrence, which can
seed an entire remote wave.

## 7. Relation to Cayley generators

Adding a unit shortcut resembles adding a redundant long-range generator to a
Cayley or implicit graph.  Reachability may stay unchanged while word distance,
frontiers, shortest paths, duplicate structure, and owner traffic all change.

Therefore a shortcut-augmented traversal cannot be reported as an optimization
of exact BFS in the old generator metric unless shortcuts carry old path costs
and the required weighted semantics/unpacking are explicit.  Note 116 treats
that separate hopset/emulator boundary.

## 8. Measurement boundaries

A useful small-world BFS record includes:

- exact base graph and shortcut construction rule;
- rewiring versus addition, fixed versus random shortcut count;
- root position relative to partitions;
- frontier profile, eccentricity, and distance distribution;
- destinations improved relative to the declared baseline metric;
- base versus shortcut edge scans and routing;
- first depth with work on each owner and per-level owner imbalance;
- graph generation, visited, routing, synchronization, and end-to-end time.

Clustering coefficient is part of the historical small-world characterization,
but this probe did not measure it.  Short path length alone is not a complete
definition of a small-world network.

## 9. Docker/Rust probe and retained failure

`experiments/small_world_bfs_probe.rs` constructs every ring edge explicitly,
rejects duplicate/loop shortcut proposals, asserts distance monotonicity against
the exact ring formula for every vertex, and runs only in Docker.

The first gate stopped on three `rustfmt --check` changes.  After the mechanical
correction, format, compile, monotonicity assertions, and execution passed.  The
CPU-only container did not request GPU access.

The code is a small semantic generator and measurement instrument, not an
optimized graph builder or BFS benchmark.

## Sources

- D. J. Watts and S. H. Strogatz,
  [*Collective dynamics of small-world networks*](https://doi.org/10.1038/30918),
  *Nature* 393, 1998, for the rewired-ring model and joint short-path/high-
  clustering phenomenon.
- M. E. J. Newman and D. J. Watts,
  [*Scaling and percolation in the small-world network model*](https://doi.org/10.1103/PhysRevE.60.7332),
  *Physical Review E* 60, 1999; also
  [arXiv:cond-mat/9904419](https://arxiv.org/abs/cond-mat/9904419), for the
  shortcut-controlled crossover length and neighborhood-growth scaling.
- M. E. J. Newman, C. Moore, and D. J. Watts,
  [*Mean-field solution of the small-world network model*](https://doi.org/10.1103/PhysRevLett.84.3201),
  *Physical Review Letters* 84, 2000, for additive-shortcut distance and path-
  length distribution analysis.
- Notes 07, 10, 26, 29, 46, 47, 51, 71, 73, 74, 92, 116, 119, and 149 provide
  this repository's depth, work/span, ownership, arbitrary-profile,
  measurement, shortcut-metric, hopset, capacity, and spatial-wave boundaries.

## Takeaway

Rare shortcuts do not merely shave a few hops from a ring.  They seed new local
waves, converting a narrow thousand-level traversal into a wide traversal of a
few dozen levels while barely changing mean degree.  For multi-GPU reasoning,
they simultaneously add communication and expose remote parallelism earlier.

# BFS on Erdos-Renyi random graphs: branching, collisions, and giant components

In a sparse random graph, early BFS often resembles a branching process. That
analogy is local, probabilistic, and temporary. Finite-population depletion,
cycles, repeated parents, root conditioning, and phase transitions determine
when the tree picture stops describing the actual frontier.

No optimizer, production implementation, benchmark, or GPU code is added.

## 1. The sampled graph is the workload

In `G(n,p)`, every unordered vertex pair is independently an edge with
probability `p`. The sparse regime writes

```text
p = c/n
```

up to an asymptotically immaterial `n-1` convention, giving expected degree
approximately `c`.

A BFS run is exact on the one graph that was sampled. Its distance labels and
frontiers are deterministic conditional on that graph, but random across graph
samples. Therefore `n`, `p` or `c`, model version, RNG algorithm, seed, root
selection, and graph-generation method belong to workload identity.

Resampling an edge whenever it is queried does not traverse one `G(n,p)` graph.
It creates a temporal/annealed process. Lazy generation is valid only when every
unordered pair receives one stable, symmetric decision reused everywhere.

## 2. Early branching-process approximation

Before BFS has exposed a material fraction of the vertices, most candidate
neighbors are new and local cycles are rare. A random vertex has asymptotically
Poisson(`c`) degree, and the unexplored continuation from an encountered vertex
is also asymptotically Poisson(`c`) in the Erdos-Renyi model. This motivates a
Galton-Watson approximation.

It does not assert

```text
|F_(d+1)| = c |F_d|
```

for one sample. Offspring counts fluctuate, frontier members share candidate
neighbors, and conditioning on survival changes observed waves. The coupling
is strongest for bounded/local exploration and weakens as exposed mass and
collision probability grow.

## 3. Extinction, survival, and the giant component

For a Poisson(`c`) branching process, extinction probability `u` is the
smallest solution of

```text
u = exp(-c(1-u)).
```

The survival probability is `rho=1-u`, equivalently

```text
rho = 1-exp(-c rho).
```

For `c<=1`, `rho=0`; no linear-size giant exists asymptotically. For `c>1`,
`rho>0`, and `G(n,c/n)` has a unique giant component occupying asymptotic
fraction `rho`, under the classical theorem.

This is an ensemble/asymptotic statement. At finite `n`, subcritical samples
still have nonempty largest components, and the critical window has components
far larger than the logarithmic subcritical scale but not linear in `n`.

## 4. Root selection changes the observation

Three experiments answer different questions:

- BFS from a fixed label such as vertex zero samples the component containing a
  uniformly ordinary vertex;
- BFS from a uniformly selected vertex has the same distribution when labels
  are exchangeable;
- BFS from a vertex chosen inside the largest component conditions on survival
  and is strongly size-biased.

In the supercritical limit, a random root lies in the giant with probability
approximately `rho`; conditional on that event its component occupies fraction
approximately `rho`. Thus the expected component fraction of an unconditioned
root is roughly `rho^2`, not `rho`.

Reporting only the largest-component wave suppresses extinct roots and cannot
be presented as an ordinary-root frontier distribution.

## 5. How the tree approximation breaks

At layer `F_d`, generated occurrences can hit:

- the previous ball;
- another vertex in the current frontier;
- one next-layer vertex through several parents;
- a genuinely new next-layer vertex.

As the discovered ball occupies fraction `x` of the graph, a candidate has
fewer unexposed destinations. Multiple frontier edges also compete for the same
remaining vertices. The effective new-state branching factor contracts even
when physical mean degree stays near `c`.

The ratio

```text
outward edge occurrences / unique vertices in F_(d+1)
```

equals one on a tree and exceeds one when next-layer states have multiple
shortest predecessors. It is only one collision statistic: same-layer and
earlier-ball edges must be counted separately.

## 6. Frontier peak and depletion

In a surviving supercritical exploration, frontiers can grow roughly
exponentially at first, become macroscopic, peak, and then contract as the
unseen population is depleted. The diameter and peak depth are random and scale
with the ensemble parameters.

A small diameter does not mean little work: a giant frontier can expose a
linear number of vertices and many more edge occurrences in a few rounds. A
long thin subcritical tree can have greater eccentricity but far less total
work.

Peak width, component size, depth, and generated edges are distinct random
variables.

## 7. Criticality is not one smooth benchmark point

For `c<1` bounded away from one, components are typically small and mostly
tree-like. At `c=1`, the critical scaling window produces large fluctuations;
the largest component has order `n^(2/3)` rather than a stable positive
fraction. For `c>1` bounded away from one, the giant has linear size.

Consequently, averaging frontier curves by raw depth near criticality can hide
the distribution: many roots die immediately while a few explore large, deep
components. Quantiles, extinction frequency, component conditioning, and raw
replicate traces are more informative than one mean curve.

## 8. Degree sequences and excess degree

The Erdos-Renyi coincidence “mean offspring approximately mean degree” does not
generalize to arbitrary random graphs. Following a uniformly random edge
size-biases the degree distribution. For degree variable `D`, the configuration
model's branching mean is

```text
E[D(D-1)] / E[D],
```

not `E[D]`. The Molloy-Reed giant criterion

```text
E[D(D-2)] > 0
```

is the corresponding supercritical condition under its regularity assumptions.
Two graphs with the same average degree can therefore have different BFS
survival, frontier variance, and giant-component behavior.

## 9. Exact BFS versus probabilistic prediction

Branching-process equations can predict distributions and asymptotic component
fractions. They do not certify one computed BFS result. Exact validation for a
sample still checks:

- every reported parent is an edge of the frozen sample;
- every distance follows the ordinary BFS local certificate;
- the visited set equals the root component after exhaustion;
- frontier counts sum to that component size;
- graph generation is symmetric and reproducible.

An observed deviation from `rho` at finite `n` is not automatically a BFS bug.
Conversely, agreement with `rho` does not validate edge generation or
distances.

## 10. Implicit and distributed generation

An explicit CSR sample freezes every edge before BFS. A stateless implicit
sample may decide adjacency from a deterministic hash of the canonical pair
`(min(u,v),max(u,v),seed)`. Exactness then requires that all devices and owners
use the same hash, threshold, integer convention, and graph version.

If rank `A` accepts `(u,v)` while rank `B` rejects `(v,u)`, the graph ceases to
be undirected. If retry order consumes a shared RNG stream differently, results
can depend on scheduling. Reproducible pairwise decisions avoid that semantic
race, though they say nothing about performance.

## 11. GPU and multi-GPU interpretation

Sparse random graphs stress different mechanisms from Hamming, Johnson, and
Grassmann families:

- irregular Poisson-like degrees rather than exact regular degree;
- stochastic frontier width and extinction;
- a transition from low-collision local trees to a depleted giant wave;
- possible push/pull work trade-offs at macroscopic frontiers;
- owner traffic determined by both partition and sampled edges.

Performance reports should preserve every replicate, state root-conditioning,
and separate graph-generation time from BFS time. Report vertices, edge
occurrences, previous/same/next-layer classes, accepted states, routing bytes,
load skew, synchronization, and end-to-end time.

One seed is a case study, not an ensemble benchmark. An ensemble mean without
variance or quantiles is also incomplete near criticality.

This note proposes no optimized random-graph generator or traversal.

## 12. Docker/Rust probe

`experiments/erdos_renyi_bfs_probe.rs` generated 20 independent deterministic
samples for each `c in {0.8,1.0,1.2,4.0}` at `n=2000`, using
`p=c/(n-1)`. It used a deliberately transparent `O(n^2)` pair sampler and exact
CPU BFS in Docker.

Measured largest-component fractions were:

```text
c=0.8: mean 0.0207, range [0.0115,0.0350]
c=1.0: mean 0.0689, range [0.0310,0.1375]
c=1.2: mean 0.2996, range [0.1610,0.4335], rho=0.3137
c=4.0: mean 0.9797, range [0.9745,0.9865], rho=0.9802
```

Fixed root zero reached depth five in `3/20`, `9/20`, `11/20`, and `19/20`
samples respectively. At `c=1.2`, its mean component fraction was `0.1035`,
close in scale to `rho^2=0.0984` but not an asymptotic test. At `c=4`, one of 20
roots missed the giant, visibly lowering the finite-sample root mean.

The representative largest-component trace for `c=4` was

```text
[1,8,31,98,351,765,591,105,9,1].
```

Outward occurrences per new state rose from one in early layers to about
`1.88` during contraction. The subcritical representative happened to be a
tree, with ratio one at every layer; this does not claim every subcritical
component is acyclic.

The probe is a semantic/measurement illustration, not a scalable generator or
performance benchmark.

## Sources

- P. Erdos and A. Renyi, [*On the Evolution of Random
  Graphs*](https://static.renyi.hu/~p_erdos/1960-10.pdf), 1960, for the phase
  transition and giant-component regimes.
- M. Molloy and B. Reed, [*A Critical Point for Random Graphs with a Given
  Degree Sequence*](https://doi.org/10.1002/rsa.3240060204), *Random Structures
  & Algorithms* 6, 1995, for the excess-degree giant criterion.
- B. Bollobas and O. Riordan, [*An Old Approach to the Giant Component
  Problem*](https://doi.org/10.1016/j.jctb.2015.03.002), *Journal of
  Combinatorial Theory, Series B* 113, 2015, for the branching-process
  interpretation under degree distributions.
- Notes 07, 10, 15, 46, 51, 55, 71, 73, 74, 95, 96, and 98 provide this
  repository's GPU, frontier, external-memory, growth, ownership, oracle,
  arbitrary-profile, queue, random-walk, flooding, and percolated-Cayley
  boundaries.

## Takeaway

Sparse random-graph BFS is tree-like only while exploration is local and
collisions are rare. The branching process predicts extinction and the giant
phase, but one frozen graph still requires exact BFS. Root conditioning,
critical fluctuations, finite-population depletion, and repeated next-layer
parents determine the observed frontier and hardware work.


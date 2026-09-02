# BFS on configuration models: size-biased frontiers and hubs

A degree distribution describes a uniformly chosen vertex.  BFS after its root
usually encounters vertices by following edges, which produces a different,
degree-biased distribution.  This distinction explains why average degree alone
is a poor predictor of frontier growth and physical work.

This note adds no optimizer, production implementation, benchmark, or GPU code.
Its Rust program is a deliberately small semantic probe.

## 1. Frozen pairing model

Fix degrees `d_1,...,d_n` with even sum.  Give vertex `v` exactly `d_v` stubs,
uniformly pair all stubs, and contract each vertex's stubs.  The direct
configuration model is a random multigraph: a pair may form a self-loop and two
pairs may form parallel edges.

The probe retains those occurrences.  A loop cannot discover another vertex
and parallel edges do not change hop distance in the support graph, but both are
real candidate occurrences scanned by adjacency-based BFS.  Collapsing them
after sampling changes the degree sequence; conditioning the pairing on
simplicity defines a different distribution.  The choice must be explicit.

One pairing is frozen before traversal.  Pairing generation is not BFS work.

## 2. Root law versus edge-endpoint law

Let a uniformly chosen vertex have degree `D` with probabilities `p_k`.  The
root has that ordinary law.  A stub followed across a uniform random pairing
reaches degree `k` with probability

```text
P(D*=k) = k p_k / E[D].
```

This is the size-biased degree `D*`.  Its mean is

```text
E[D*] = E[D^2]/E[D]
      = E[D] + Var(D)/E[D].
```

Thus an edge endpoint has at least the vertex-average degree, with strict
inequality whenever degree varies.  This is the structural core of the
friendship paradox; no social interpretation is required.

The arrival edge has already been used, so the approximate number of available
children is `D*-1`, not `D*`.

## 3. Excess degree and the first two BFS generations

Before pairing collisions and depletion matter, configuration-model BFS is
approximated by a two-stage branching process:

```text
root children:       D,
later offspring:     D* - 1,
mean later offspring nu = E[D(D-1)] / E[D].
```

The expected early layer sizes therefore begin approximately as

```text
E[|F_1|] = E[D],
E[|F_d|] = E[D] nu^(d-1),  d>=1.
```

This is an ensemble/local approximation, not an exact recurrence for a finite
sample.  Degrees reached within a realized layer are dependent; multiple stubs
can meet one vertex, and high-degree vertices are depleted earlier.

## 4. Giant threshold is not mean-degree threshold

The Molloy-Reed expression is

```text
E[D(D-2)] = E[D] (nu-1).
```

Under the theorem's regularity assumptions, a positive value gives the
supercritical giant regime and a negative value gives the subcritical regime.
Consequently, two distributions with the same `E[D]` can lie on different
sides of the threshold.

Using generating functions

```text
G0(x) = sum_k p_k x^k,
G1(x) = G0'(x)/G0'(1),
```

the locally tree-like prediction uses the smallest solution `u=G1(u)` and

```text
giant fraction S = 1-G0(u).
```

The same moments that increase `nu` need not increase `S`: high-degree hubs can
create rapid surviving growth while positive mass at degree zero or one leaves
more vertices outside the giant.

## 5. Same mean, different frontier laws

The probe compared three exact degree multisets at `n=2000`, all with mean four:

```text
distribution       nu    E[D*]  largest fraction  root in giant
all degree 4       3.00    4.00       1.0000          20/20
half 2, half 6     4.00    5.00       0.9999          20/20
half 1, half 7     5.25    6.25       0.9380          18/20
```

For the last distribution,

```text
G1(x)=0.125+0.875x^6,
```

which predicts a giant fraction near `0.9375`; the finite mean `0.9380` is
consistent in scale but does not validate the asymptotic formula.

The fixed root label received a freshly shuffled degree in every sample, so it
models an unconditioned uniform root.  Its mean component fraction was `0.8435`
in the half-1/half-7 case because only 18 of 20 retained roots entered the
giant.  Twenty Bernoulli outcomes are too few for a precise survival estimate.

## 6. A frontier consumes the degree distribution

In the representative half-1/half-7 sample, the root happened to be a leaf.
The mean degrees of successive frontiers were

```text
1.00, 7.00, 7.00, 6.50, 6.08, 5.70, 3.80, 1.10, 1.00.
```

The early wave is hub-rich because edges select stubs.  Later layers become
leaf-rich because many hubs have already been discovered and their leaf
neighbors sit near the boundary.  The frontier degree distribution is thus
depth-dependent even though the graph's degree multiset is fixed.

For that same sample, outward occurrences per newly discovered state rose from
one to two near the frontier peak.  This records pairing convergence; it must
not be inferred from `nu` alone.

## 7. What regular graphs had hidden

Note 145's random regular graphs made `D*=D=4` and `nu=3`.  That removed both
root-degree variance and edge-endpoint size bias.  The heterogeneous cases show
that three quantities must be kept separate:

- frontier vertices `|F_d|`;
- adjacency occurrences `sum_(v in F_d) d_v`;
- distinct newly discovered vertices `|F_(d+1)|`.

A narrow frontier containing hubs can scan more occurrences than a wider
frontier containing leaves.  Vertex count is therefore not a complete work
measure.

## 8. GPU and multi-GPU interpretation

Degree heterogeneity creates conceptual execution pressures without prescribing
an implementation:

- equal vertex ownership need not balance incident-edge occurrences;
- one hub can dominate a thread, warp, block, or owner's candidate volume;
- hub edges can fan out across many owners even when the hub itself has one
  authoritative owner;
- repeated arrivals at hubs or their neighbors change visited contention;
- early hub depletion makes the workload distribution change by depth;
- maximum degree and degree second moment can matter when mean degree looks
  harmless.

Per-depth observations should retain frontier degree histograms, maximum and
sum degree by owner, old/same/outward occurrences, unique new states, routing
destinations, synchronization, and end-to-end time.  A “vertices per second”
number without scanned occurrences and degree composition is ambiguous.

Heavy-tailed regimes require extra care: moments may grow with `n`, maximum
degree may violate bounded-degree theorem assumptions, and a finite cutoff can
control both mathematics and hardware work.  This probe uses only bounded
degrees one through seven and makes no power-law claim.

## 9. Docker/Rust probe and retained failure

`experiments/heterogeneous_degree_bfs_probe.rs` uses an explicit shuffled stub
list, checks every realized adjacency-list length against the requested degree,
and runs exact BFS plus component enumeration for 20 deterministic pairings per
distribution.

The first Docker gate stopped because `rustfmt --check` required a function
signature to wrap.  After that formatting-only correction, format, compile,
degree assertions, and execution passed.  The CPU-only container reported no
NVIDIA driver because GPU access was neither requested nor needed.

This is finite pseudorandom multigraph evidence, not a uniform-simple-graph
test, scalability study, or performance benchmark.

## Sources

- M. Molloy and B. Reed,
  [*A critical point for random graphs with a given degree sequence*](https://doi.org/10.1002/rsa.3240060204),
  *Random Structures & Algorithms* 6(2-3), 1995, for the giant criterion under
  stated degree-sequence conditions.
- M. E. J. Newman, S. H. Strogatz, and D. J. Watts,
  [*Random graphs with arbitrary degree distributions and their applications*](https://doi.org/10.1103/PhysRevE.64.026118),
  *Physical Review E* 64, 2001; also
  [arXiv:cond-mat/0007235](https://arxiv.org/abs/cond-mat/0007235), for the
  size-biased excess distribution, generating functions, layer growth, and
  giant-size equations.
- Notes 07, 10, 29, 44, 47, 51, 73, 74, 119, 144, 145, and 146 provide this
  repository's complexity, GPU, load, owner, queue, measurement, random-graph,
  and multitype-frontier boundaries.

## Takeaway

BFS samples the root by vertices but samples later candidates through stubs.
That changes the relevant degree law from `D` to `D*`, and the local branching
mean from average degree to `E[D(D-1)]/E[D]`.  Hubs can make a surviving wave
grow faster while leaves reduce total giant coverage.  For execution reasoning,
frontier size, scanned degree mass, and newly accepted states are distinct.

# BFS on random regular graphs: tree bounds, pairing, and radial variance

Random `r`-regular graphs remove degree variance without making BFS frontiers
deterministic. Their local limit is the infinite `r`-regular tree, but cycles,
shared children, same-layer edges, and finite-population depletion eventually
break the tree profile.

No optimizer, production implementation, benchmark, or GPU code is added.

## 1. Three meanings of regular

The word regular must be qualified:

- **degree-regular:** every vertex has degree `r`;
- **distance-regular:** inward/same/outward neighbor counts depend only on BFS
  depth;
- **regular group action:** every state is reached by exactly one group element
  from a fixed state.

A random regular graph has the first property. It generally has neither of the
other two. Equal candidate count per frontier vertex does not imply equal radial
work, path multiplicity, owner traffic, or symmetry.

## 2. Pairing/configuration model

Give each of `n` labeled vertices `r` stubs and choose a perfect matching of all
`nr` stubs. Contracting each vertex's stubs produces an `r`-regular
multigraph. A pair within one vertex is a loop; two pairs connecting the same
vertices are parallel edges.

Conditioning a uniform pairing on producing no loops or parallel edges gives a
uniform labeled simple `r`-regular graph, because every such graph corresponds
to the same `(r!)^n` stub assignments.

For fixed `r`, the probability of a simple pairing tends to

```text
exp(-(r^2-1)/4).
```

Rejection sampling is therefore conceptually simple but increasingly wasteful
as `r` grows. Generation attempts and rejected configurations are not BFS work.

The probe uses a declared deterministic xorshift stream and unbiased bounded
index reduction. Its retained outputs are reproducible pseudorandom samples,
not a proof that a finite PRNG realizes the mathematical uniform ensemble.

## 3. The exact regular-tree envelope

Root the infinite `r`-regular tree. Its BFS layers are

```text
T_0=1,
T_d=r(r-1)^(d-1),  d>=1.
```

For every finite simple `r`-regular graph, these are exact upper bounds:

```text
|F_d| <= r(r-1)^(d-1).
```

The root has at most `r` children; every later vertex uses at least one edge to
the previous ball and has at most `r-1` outward edges. Equality through depth
`d` means the explored ball has not yet lost any potential child to cycles or
convergence in those transitions.

This is a deterministic Moore/tree bound. Randomness enters the probability of
remaining close to it at a given finite radius.

## 4. Why the branching factor is r-1

A random regular root has exactly `r` incident edges. Once BFS reaches a new
nonroot vertex along one edge, only `r-1` other incident edges remain as
potential forward branches. Thus the local tree approximation has

```text
first generation r,
later offspring r-1.
```

This contrasts with sparse `G(n,c/n)`, where the Poisson degree distribution is
special: size bias followed by subtracting the arrival edge again gives a
Poisson(`c`) excess distribution.

Graphs with the same mean degree can therefore have different early frontier
laws. In particular, mean degree four does not make a 4-regular sample locally
equivalent to `G(n,4/n)`.

## 5. Local weak tree limit

For fixed `r`, a uniformly sampled random `r`-regular graph converges locally
to the infinite `r`-regular tree. Informally, a bounded-radius neighborhood of
a random root is acyclic with probability tending to one as `n` grows.

This does not say the whole finite graph is a tree, nor that radius may grow
arbitrarily with `n`. Once the exposed ball becomes large enough, stub pairings
increasingly connect within the discovered region or merge forward branches.

The local-limit theorem predicts early structure across an ensemble. It does
not certify one sample's BFS output.

## 6. Radial counts vary inside one layer

For a vertex `v` at depth `d`, split its `r` edges into

```text
c(v): neighbors at depth d-1,
a(v): neighbors at depth d,
b(v): neighbors at depth d+1.
```

Always `a(v)+b(v)+c(v)=r`, but these values need not be constant across `F_d`.
The probe's representative 4-regular sample had, at depth four,

```text
c(v) in [1,2], a(v) in [0,1], b(v) in [2,3].
```

So fixed degree did not imply distance regularity. A kernel assigning one
thread per vertex has equal adjacency-list length but can still see different
visited outcomes and atomic contention.

## 7. Frontier contraction and collisions

The tree bound grows forever, but a finite graph has only `n` vertices. Actual
frontiers first track the tree, then fall below it, peak, and contract.

For the representative samples:

```text
r=3 actual: [1,3,6,12,24,46,90,...]
r=3 tree:   [1,3,6,12,24,48,96,...]

r=4 actual: [1,4,12,36,106,282,...]
r=4 tree:   [1,4,12,36,108,324,...]
```

The first deficits mark cycle/convergence effects in the exposed ball. They do
not by themselves identify whether the missing tree child became an
earlier-layer hit, same-layer edge, or repeated next-layer parent.

Outward occurrences per unique next state equal one on a tree and rise with
shared children. Ratios in the final tiny layers can be numerically large
because the denominator is small; they must be paired with absolute counts.

## 8. Connectivity and low-degree exceptions

For fixed `r>=3`, a uniform random `r`-regular graph is connected with
probability tending to one. This differs sharply from sparse Erdos-Renyi at the
same constant mean degree, which retains isolated and small components.

The restriction matters:

- `r=0` gives isolated vertices;
- `r=1` gives disjoint edges;
- `r=2` gives a union of cycles, not the same connected regime.

The probe observed connectivity in every 3- and 4-regular sample, but 20/20 is
finite evidence, not a proof of the asymptotic theorem.

## 9. Diameter and eccentricity

Tree growth suggests logarithmic distance scale with base `r-1`, but collisions
and tail structure add corrections. Root eccentricity, graph diameter, average
distance, and depth of the largest frontier are different quantities.

A single-root BFS proves only that root's eccentricity. Even vertex-exchangeable
sampling does not make one realized graph vertex-transitive.

Random label symmetry is an ensemble property; it is not an automorphism of a
sample.

## 10. Sampling correctness boundary

Several generators called “random regular” define different distributions:

- exact uniform simple graph;
- pairing multigraph without conditioning;
- pairing rejection conditioned on simplicity;
- edge-switching Markov chain after a heuristic number of steps;
- constructive heuristic with possible bias.

The model and failure/retry policy must be reported. Removing loops or merging
parallel edges after sampling changes degrees and is not equivalent to
conditioning on simplicity.

In distributed generation, all owners must agree on the same accepted pairing
or edge set. Locally repairing conflicts independently can create degree errors
or inconsistent adjacency.

## 11. GPU and multi-GPU interpretation

Random regular graphs isolate some performance effects:

- exact equal adjacency-list length removes degree-count imbalance;
- early tree-like layers have low duplicate convergence;
- later macroscopic layers create shared children and same/old hits;
- random labels tend to scatter memory and owner destinations;
- connectivity makes a root traversal typically touch the whole sample for
  `r>=3`.

They do not isolate cache locality, partition quality, or radial work. Equal
degree cannot prove equal per-thread time because visited outcomes and remote
destinations differ.

Measurements should separate generation/rejection time, CSR construction,
BFS edge scans, previous/same/next-layer outcomes, accepted states, owner skew,
routing bytes, synchronization, and end-to-end time. A generator throughput
claim is not a BFS throughput claim.

This note proposes no optimized sampler or traversal.

## 12. Docker/Rust probe

`experiments/random_regular_bfs_probe.rs` used `n=2000`, degrees three and four,
and 20 deterministic seeds per degree. It shuffled stubs, rejected nonsimple
pairings, verified every final degree, and ran exact BFS from vertex zero.

Results:

```text
r=3: connected 20/20
     mean pairing attempts 6.45, range [1,37]
     mean root eccentricity 12.90, range [12,13]
     representative peak 497

r=4: connected 20/20
     mean pairing attempts 46.80, range [2,134]
     mean root eccentricity 9.00, range [9,9]
     representative peak 716
```

The attempt means are consistent in scale with inverse limiting simplicity
probabilities `exp(2)` and `exp(3.75)`, but 20 samples do not estimate those
limits precisely.

The first gate failed because Rust 1.85 does not stabilize `repeat_n`; it was
replaced with stable `repeat().take()`. A later instrumentation gate failed only
on `rustfmt` spacing for nested tuple fields. Both failures are retained; the
final format, compile, degree assertions, and execution passed in Docker.

The probe is a semantic/measurement illustration, not a uniformity test,
scalable sampler, or performance benchmark.

## Sources

- B. Bollobas, [*A Probabilistic Proof of an Asymptotic Formula for the Number
  of Labelled Regular Graphs*](https://doi.org/10.1016/S0195-6698(80)80030-8),
  *European Journal of Combinatorics* 1(4), 1980, for the pairing model and
  limiting simplicity probability.
- N. C. Wormald, [*Models of Random Regular
  Graphs*](https://doi.org/10.1017/CBO9780511721335.010), *Surveys in
  Combinatorics 1999*, for uniform regular-graph models, pairing, connectivity,
  and random-regular methodology.
- Notes 07, 10, 21, 32, 46, 47, 51, 71, 73, 74, 119, and 144 provide this
  repository's GPU, frontier, eccentricity, distance-regular, capacity,
  parallelism, ownership, arbitrary-profile, queue, Moore-bound, and
  Erdos-Renyi boundaries.

## Takeaway

Random regular BFS begins like the `r`-regular tree—`r` children at the root and
roughly `r-1` thereafter—but finite collisions force the wave below the tree
bound. Equal degree removes adjacency-count imbalance, not radial variance,
duplicate convergence, or owner traffic. Pairing generation and rejection are
also separate work from BFS itself.


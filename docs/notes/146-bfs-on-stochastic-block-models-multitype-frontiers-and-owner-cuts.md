# BFS on stochastic block models: multitype frontiers and owner cuts

A stochastic block model (SBM) adds latent vertex types to a random graph.
Ordinary BFS still sees only vertices and edges, but type-resolved frontiers
show structure that a single frontier-size curve hides.

This note adds no optimizer, production implementation, benchmark, or GPU code.
Its Rust probe is deliberately small and quadratic: it is a semantic instrument.

## 1. Frozen graph and BFS contract

Let type `a` occupy fraction `alpha_a` of `n` vertices.  Conditional on the
types, each unordered pair of types `(a,b)` is joined independently with
probability

```text
c_ab / n.
```

One complete edge sample is frozen before BFS starts.  Resampling an incident
edge whenever adjacency is queried would define a changing random process, not
BFS on one graph.  The type labels are analysis metadata: exact ordinary BFS
does not require them to compute distances.

The probe uses two equal blocks and

```text
C = [[within, across],
     [across, within]].
```

Every tested case has `(within+across)/2=4`, so the expected degree is the same.

## 2. Local multitype branching approximation

Before collisions and depletion matter, exploration from a type-`a` vertex is
approximated by a multitype Poisson branching process with mean offspring
matrix

```text
M_ab = alpha_b c_ab.
```

For two equal blocks,

```text
M = 1/2 [[within, across],
         [across, within]].
```

If typed frontier counts are represented by a row vector, their early
expectation obeys `f_(d+1) approximately f_d M`; a column-vector convention
uses `M^T`.  Declaring the orientation prevents a silent transpose error.

This is a local probabilistic approximation, not an exact recurrence for one
finite BFS.  Visited collisions, same-layer edges, shared children, and
finite-population depletion eventually invalidate it.

## 3. Two eigenmodes with different meanings

The symmetric matrix has eigenvalues

```text
lambda_sum      = (within+across)/2,
lambda_contrast = (within-across)/2.
```

The positive all-types mode controls total early growth.  Under the standard
irreducible sparse inhomogeneous-graph assumptions, survival and the giant
threshold are governed by the Perron eigenvalue: a giant appears above one.

The contrast mode controls how type imbalance evolves:

- positive and near the Perron value: the root's type persists for many layers;
- zero: the expected type contrast disappears in one branching step;
- negative: the dominant type tends to alternate between consecutive layers;
- equal to the Perron value because `across=0`: types never mix at all.

Thus one scalar frontier profile can conceal radically different typed waves.
The second eigenvalue is not another BFS distance and is not, by itself, a
general community-detection guarantee.

## 4. Irreducibility is not cosmetic

For `across>0`, the two-type branching kernel is irreducible.  In the symmetric
case the survival probabilities have the symmetric fixed point

```text
s = 1-exp(-4s).
```

For `across=0`, the graph is instead the disjoint union of two independent
`G(n/2,8/n)` blocks.  The spectral radius is still four, but there is no unique
global giant spanning both types.  A root in block zero can never reach block
one; the largest component occupies about half of the global vertex fraction
that the irreducible cases occupy.

More generally, multitype Poisson survival satisfies

```text
s_a = 1 - exp(-sum_b M_ab s_b).
```

Reducibility requires solving and interpreting this by communicating class,
not quoting only `rho(M)>1` as if it implied one global component.

## 5. Retained finite probe

`experiments/stochastic_block_bfs_probe.rs` sampled 20 deterministic graphs for
each case at `n=2000`, ran exact BFS from vertex zero, found the largest
component, and measured two static two-owner partitions.

```text
case             eigs       largest   root-0    block remote  striped remote
segregated       (4,  4)     0.4912    0.4902       0.0000         0.5012
assortative      (4,3.5)     0.9796    0.9796       0.0626         0.4996
neutral          (4,  0)     0.9792    0.9792       0.4987         0.4982
disassortative   (4,-3.5)    0.9798    0.9798       0.9376         0.5000
```

Representative typed layers from a type-zero root were:

```text
segregated:     (1,0),(6,0),(16,0),(75,0),(234,0),(417,0),...
assortative:    (1,0),(6,1),(16,8),(74,32),(233,98),(408,296),...
neutral:        (1,0),(3,4),(12,9),(40,46),(145,145),...
disassortative: (1,0),(2,6),(23,4),(12,72),(266,67),(189,547),...
```

The finite observations match the qualitative branching modes: confinement,
slow mixing, immediate mean mixing, and alternating contrast.  They do not
validate an asymptotic theorem or estimate performance.

The first Docker gate stopped on one `rustfmt --check` line wrap.  After the
formatter-only correction, format, compilation, assertions implicit in the
complete traversal, and execution passed.  The container reported no NVIDIA
driver because this CPU-only probe did not request GPU access.

## 6. Owner cuts expose a trade-off

If owner equals block type, the expected remote-edge fraction in this equal
two-block model is

```text
across / (within+across).
```

The probe observed the corresponding range from zero to about `0.938`.  A
striped parity owner was near one half in every case.

Consequences:

- block ownership sharply reduces routing for assortative graphs;
- it is neutral when within/across probabilities are equal;
- it maximizes routing for strongly disassortative graphs;
- in the segregated graph it yields zero routing, but one BFS occupies only one
  owner and leaves the other owner with no reachable work.

Therefore “minimize edge cut” is not a complete multi-GPU objective.  Routing,
reachable-owner count, per-level load balance, memory capacity, and replication
must be considered separately.  A zero-cut partition can have poor utilization
for a single-source traversal.

If block labels are inferred rather than given, inference errors change the
partition and communication pattern, but not the distances in the original
frozen graph.  Community inference and BFS correctness remain separate tasks.

## 7. What a distributed observation should record

A future measurement may resolve, per depth and source type:

- frontier counts by vertex type and by owner;
- the type-to-type and owner-to-owner candidate matrices;
- previous-ball hits, same-layer edges, repeated next parents, and new states;
- local versus routed occurrences and bytes;
- maximum/mean owner load and idle owners;
- generation, visited, routing, synchronization, and end-to-end time.

These fields describe where BFS work goes.  They do not prescribe an optimized
implementation, and this note creates no such implementation backlog.

## Sources

- P. W. Holland, K. B. Laskey, and S. Leinhardt,
  [*Stochastic blockmodels: First steps*](https://doi.org/10.1016/0378-8733(83)90021-7),
  *Social Networks* 5(2), 1983, for the blockmodel formulation.
- B. Bollobas, S. Janson, and O. Riordan,
  [*The phase transition in inhomogeneous random graphs*](https://doi.org/10.1002/rsa.20168),
  *Random Structures & Algorithms* 31(1), 2007; also
  [arXiv:math/0504589](https://arxiv.org/abs/math/0504589), for the multitype
  branching-process connection, operator threshold, and irreducibility scope.
- Notes 07, 10, 51, 56, 71, 73, 74, 98, 144, and 145 provide this repository's
  GPU, frontier, ownership, distributed-completion, arbitrary-profile,
  measurement, percolation, Erdos-Renyi, and random-regular boundaries.

## Takeaway

The SBM makes visible two different BFS phenomena: total frontier growth and
movement of mass between vertex types.  Equal mean degree and equal Perron
growth can coexist with confinement, slow mixing, neutral mixing, or alternating
typed layers.  For multi-GPU reasoning, the same structure can reduce traffic,
destroy utilization, or maximize cross-owner routing depending on how ownership
aligns with the graph.

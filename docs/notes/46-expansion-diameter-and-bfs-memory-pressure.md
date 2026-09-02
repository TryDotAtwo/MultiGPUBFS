# Expansion, diameter, and BFS memory pressure

BFS frontiers are vertex boundaries of metric balls. Expansion theory therefore
connects graph geometry to frontier growth, but only after the kind of boundary
and the quantified vertex sets are stated precisely.

This note develops that connection and its memory consequences. It does not
propose a GPU implementation or a runtime switching policy.

## Balls, spheres, and the external vertex boundary

For an undirected graph and source `s`, let

```text
B_d = {v | dist(s,v) <= d}
S_d = {v | dist(s,v) = d}
partial_V(X) = N(X) minus X.
```

Then exact complete BFS satisfies

```text
S_(d+1) = partial_V(B_d)
B_(d+1) = B_d disjoint_union S_(d+1).
```

This equality is stronger than saying that the frontier is "related to" a
boundary. Every distinct outside neighbor of the completed ball has distance
exactly `d+1`, and every depth-`d+1` vertex has a predecessor in `B_d`.

The statement needs modification for directed graphs: the next frontier is the
external **out-boundary** of the ball under the declared arc orientation.

## Vertex expansion gives a frontier lower bound

For a finite graph on `n` vertices, define the small-set vertex-expansion
constant

```text
h_V = min_{0 < |X| <= n/2} |partial_V(X)| / |X|.
```

Whenever `|B_d| <= n/2`, the BFS identity immediately gives

```text
|S_(d+1)| >= h_V |B_d|
|B_(d+1)| >= (1+h_V)|B_d|.
```

Induction yields exponential **ball** growth for as long as every preceding
ball remains in the quantified size range:

```text
|B_d| >= (1+h_V)^d.
```

Thus a uniform positive expansion constant forces every source ball to exceed
half the graph within `O(log n / log(1+h_V))` levels. Two balls containing more
than half the vertices intersect, so applying the argument from both endpoints
also supplies an `O(log n)` diameter bound for a constant-expansion family.

The implication direction matters:

```text
proved global expansion -> bounded ball-growth/diameter conclusions
one observed fast-growing BFS -> not a proof of global expansion.
```

Expansion is a minimum over many vertex subsets, not a statistic of one trace.

## Constant expansion forces a large frontier

There is a useful memory consequence that is easy to miss. Suppose the graph
has maximum degree `Delta`, `n>1`, and `h_V>0`. Choose the last BFS ball `B_d`
whose size is at most `n/2`; the next ball exceeds `n/2`.

Because one vertex has at most `Delta` outside neighbors,

```text
|B_(d+1)| <= (1+Delta)|B_d|.
```

Since `|B_(d+1)| > n/2`, this implies

```text
|B_d| > n / (2(1+Delta)).
```

Expansion at `B_d` then gives

```text
|S_(d+1)| >= h_V |B_d|
            > h_V n / (2(1+Delta)).
```

Therefore a bounded-degree family with expansion bounded below by a positive
constant has a BFS layer containing `Omega(n)` vertices from every source.

This makes the trade-off concrete:

- strong expansion helps keep BFS depth logarithmic;
- the same expansion forces a wide middle boundary;
- a list frontier may need linear state/ID storage at some level;
- cumulative exact visited is already linear as the ball crosses half the graph.

Low diameter is not evidence for a small BFS memory peak.

## Edge boundary is not next-frontier cardinality

Define the undirected edge boundary

```text
partial_E(X) = {{u,v} in E | u in X, v notin X}.
```

`|partial_E(B_d)|` counts crossing edge occurrences. `|partial_V(B_d)|` counts
distinct outside endpoints and equals the next frontier size. Many crossing
edges can converge on one next-layer vertex.

With maximum degree `Delta`,

```text
|partial_V(X)| <= |partial_E(X)|
|partial_V(X)| >= |partial_E(X)| / Delta.
```

The factor-`Delta` gap is algorithmically meaningful:

- push expansion work often follows crossing and internal edge occurrences;
- exact next-frontier storage follows distinct external vertices;
- duplicate detection pays for the convergence between the two.

For a labeled Cayley multigraph, repeated or identity generators complicate
edge-occurrence counts further while leaving the simple external vertex
boundary unchanged.

## Conductance and spectral gap are indirect evidence

Conductance normalizes an edge boundary by volume rather than vertex count.
Cheeger-type inequalities relate conductance to eigenvalues for a specified
random-walk or normalized-Laplacian model. These are powerful aggregate bounds,
but they do not directly reveal an exact rooted sphere sequence.

Moving from a spectral statement to a BFS memory claim therefore requires the
whole chain:

```text
spectral assumptions
-> conductance/edge-expansion bound
-> degree-aware vertex-boundary bound
-> application to the actual ball B_d
-> representation bytes for S_(d+1).
```

Each arrow has hypotheses and may lose constants. A spectral gap does not say
which exact states occur in a frontier, how many generator occurrences collide,
or how owner routing distributes them.

This is consistent with note 33: random-walk mixing concerns normalized walk
mass, whereas BFS concerns first-positive support after subtracting visited.

## Cayley graphs: uniform local view, nonuniform conclusions across generators

A Cayley graph is vertex-transitive, so translating a set preserves its boundary
ratio. This makes expansion source-independent at the graph level. It does not
make every metric ball extremal for the global expansion constant, nor does
vertex transitivity imply expansion.

Examples separate the possibilities:

- finite cycles are vertex-transitive but their expansion tends to zero as the
  cycle grows; BFS frontiers remain of constant width;
- hypercubes are vertex-transitive and have binomial BFS layers, with a large
  middle frontier, while their degree grows with dimension;
- adjacent-transposition `S_n` has the exact Mahonian profile from note 10;
  changing the generating set changes both metric balls and boundary ratios;
- an expander Cayley family can have logarithmic diameter and linear-width BFS
  layers under bounded degree.

Consequently "Cayley" supplies symmetry, not one universal growth regime.

## Why the tree heuristic is incomplete

In a regular tree, the frontier grows exponentially and contains no convergence
between distinct branches. In a finite expander, early balls may look tree-like,
but cycles and convergence become unavoidable before saturation.

Both can have rapid growth, yet the hardware funnel differs:

```text
generated edge occurrences
-> distinct boundary endpoints
-> previously unseen states
-> stored next frontier.
```

Expansion lower-bounds the final distinct external endpoints for eligible
balls. It does not determine the number of candidate records, probes, atomics,
or equal-parent collisions used to obtain them.

## Bidirectional BFS does not escape the geometry automatically

The familiar tree estimate suggests two frontiers near radius `D/2` instead of
one near radius `D`. In a finite expander, half-radius balls may already contain
a substantial fraction of the entire graph. Meeting earlier in depth can still
require wide frontiers and large visited sets on both sides.

Bidirectional benefit therefore depends on actual forward and reverse ball
volumes and on the stopping proof, not on logarithmic diameter alone. The two
visited sets may also overlap heavily before the termination lower bound closes.

## GPU capacity implications without designing an optimizer

For a frontier record of `f` bytes and a persistent visited record of `v` bytes,
the semantic payload lower bounds at depth `d` include

```text
frontier bytes >= f |S_d|
visited bytes  >= v |B_d|.
```

If states are generated into a materialized candidate bag, a bounded-degree
graph can additionally expose up to

```text
Delta |S_d|
```

transition records before exact duplicate convergence. Hash-table slack,
bitmaps, parents, sorting, routing, and temporary storage are separate.

For constant-degree expanders, the linear-frontier result means that an exact
complete traversal cannot rely on every frontier staying sublinear. Streaming,
chunking, or externalization may change the physical peak, but each needs a
proof that the logical layer completes without silent loss. This is a semantic
constraint, not a recommendation to implement any particular mechanism.

## Multi-GPU expansion and communication are different cuts

Graph expansion concerns edges leaving a **vertex subset in the graph**.
Distributed owner routing concerns transitions whose endpoints have different
owners under a chosen partition. These cuts coincide only for a deliberately
related partition.

- A locality-preserving graph partition on an expander necessarily cuts many
  edges for balanced large parts; expansion can lower-bound communication.
- Hash ownership may balance states while making most transitions remote, but
  that follows from the hash/edge correlation, not from `h_V` alone.
- Replicated visited filters may reduce messages but do not remove the need for
  one authoritative exact decision.
- Duplicate candidates can cross several senders and converge at one owner, so
  bytes sent are not determined by distinct vertex boundary size.

Per-level evidence should therefore keep separate:

```text
graph edge boundary
distinct next-state boundary
owner-crossing candidate occurrences
distinct remote state identities
post-owner accepted states
per-owner frontier and visited peaks.
```

## Counterexamples and rejected shortcuts

### Small diameter implies narrow BFS

A complete graph has diameter one and a frontier of `n-1` from every source.

### Vertex transitivity implies expansion

Cycles are vertex-transitive and have constant-width BFS frontiers, while their
normalized boundary ratio vanishes with graph size.

### Many cut edges imply equally many next states

All cut edges can converge on far fewer outside endpoints; degree bounds are
needed to translate edge boundary into vertex boundary.

### A large spectral gap gives exact frontier sizes

It supplies aggregate expansion/mixing bounds under a declared operator. It
does not identify the rooted balls or their exact boundary sequence.

### Expansion predicts multi-GPU traffic directly

Traffic also depends on ownership, representation, local pre-deduplication,
duplicate convergence, and whether the graph cut aligns with owner partitions.

## Sources

- Shlomo Hoory, Nathan Linial, and Avi Wigderson,
  [Expander graphs and their applications](https://www.cs.huji.ac.il/~nati/PAPERS/expander_survey.pdf),
  surveys vertex/edge expansion, spectral connections, and diameter consequences.
- Alexander Lubotzky,
  [Expander Graphs in Pure and Applied Mathematics](https://arxiv.org/abs/1105.2389),
  provides expansion, spectral, and Cayley-graph background.
- Daniel Spielman,
  [Spectral Graph Theory](https://www.cs.yale.edu/homes/spielman/PAPERS/SGTChapter.pdf),
  develops conductance and spectral inequalities.
- Notes 10, 32, 33, and 35 provide the project's exact frontier identities,
  intersection-profile distinction, walk/spectral boundary, and Cayley growth
  series context.

## Current conclusions

1. The exact next BFS frontier is the external vertex boundary of the completed
   ball, not its edge boundary or generator-occurrence count.
2. Positive vertex expansion forces geometric ball growth until half coverage
   and gives logarithmic diameter bounds for constant-expansion families.
3. With bounded degree, constant expansion also forces at least one
   `Omega(n)`-wide BFS frontier; shallow does not mean memory-light.
4. Spectral gaps constrain BFS only through a hypothesis-bearing chain from
   eigenvalues to conductance, edge boundary, and vertex boundary.
5. Cayley symmetry does not imply expansion, a universal frontier shape, or
   uniform owner routing.
6. Multi-GPU traffic is an ownership cut over candidate occurrences, not the
   same object as the graph's distinct external vertex boundary.

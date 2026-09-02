# BFS, local message passing, and neural receptive fields

BFS and graph message passing both propagate information along edges in rounds.
That visual similarity is real but incomplete. Exact BFS maintains set identity
and first-arrival semantics; a neural message-passing layer usually compresses
a multiset of neighboring vectors into another fixed-width vector.

This note separates locality, exact simulation, expressive power, and learned
heuristics. It proposes no model or implementation.

## 1. The locality theorem

A standard local message-passing layer has the schematic form

```text
h_v^(i+1) = UPDATE_i(
    h_v^i,
    AGG_i({ MESSAGE_i(h_v^i,h_u^i,e_uv) : u in N(v) })
).
```

Assume shared local functions, no global read/write state, and initial features
attached only to vertices and edges. By induction, `h_v^r` depends only on
information inside the radius-`r` rooted attributed neighborhood of `v`:

- at round zero it depends only on `v`;
- one update reads only neighbors' round-`i` states;
- those states depend on their radius-`i` neighborhoods;
- their union lies within radius `i+1` of `v`.

Therefore information strictly farther than `r` edges cannot influence `v`
after `r` local rounds.

This is an upper bound on possible dependence, not a claim that the embedding
faithfully stores the entire `r`-ball.

## 2. Receptive radius is not BFS distance

An `r`-layer model may aggregate information from within `r` hops while losing
which vertex supplied it, how far away it was, how many shortest paths carried
it, or whether two messages came from the same semantic state.

Thus none of these implications is automatic:

```text
v is in the receptive field  -> its identity is recoverable;
source information arrived   -> exact source distance is known;
embedding stopped changing   -> BFS component is exhausted;
two embeddings are equal     -> two graph states are equal.
```

The BFS ball is a set of exact vertices. A learned embedding is a function of
an attributed neighborhood and is usually many-to-one.

## 3. Exact BFS can be written as message passing

The distinction is semantic, not a claim that message passing can never compute
BFS. Mark a source `s` and initialize

```text
d_0(s)=0,
d_0(v)=infinity for v!=s.
```

Use the exact synchronous update

```text
d_(i+1)(v) = min(d_i(v), 1 + min_(u in N(v)) d_i(u)).
```

After `i` rounds, `d_i(v)` equals the true distance when that distance is at
most `i`, and remains infinity otherwise. The proof is the usual bounded-walk
induction. Equivalently, Boolean reached sets obey

```text
B_(i+1) = B_i union N(B_i),
F_(i+1) = B_(i+1) minus B_i.
```

This is exact BFS/fixed-point propagation expressed in local rounds. Its
guarantee comes from exact `min`/set operations, source initialization,
synchronous completion, and enough rounds -- not from being a neural network.

## 4. The source marker is essential

Distance is relative to a source. If every vertex begins with the same feature
and the graph is regular and symmetric, a shared permutation-equivariant local
update may leave every vertex embedding identical forever.

In an unlabeled `k`-regular graph, identical states give every vertex the same
multiset of `k` identical neighbor states, so induction preserves equality. In
a consistently generator-labeled Cayley graph, every vertex likewise sees the
same labeled local pattern under translation.

No source-relative BFS layer can emerge from completely homogeneous inputs.
One must supply a source marker or some other symmetry-breaking information.
Even then, marking the source only makes distance information available; it
does not force a learned aggregation to preserve it exactly.

## 5. Relation to 1-WL

Under the standard neighborhood-aggregation assumptions, message-passing GNNs
are bounded in graph-distinguishing power by one-dimensional
Weisfeiler-Leman/color refinement. Architectures with injective aggregation and
updates can match that bound; noninjective mean or max aggregation may be
strictly weaker.

The qualification matters:

```text
ordinary MPNN <= 1-WL
```

is not a statement about architectures with global attention, higher-order
tuple states, unique identifiers, positional encodings, or externally computed
distance features.

The prism/`K_(3,3)` witness from note 101 transfers directly. With uniform
features, each graph has six degree-three vertices receiving identical
multisets at every local layer. A permutation-invariant sum readout sees six
copies of the same vector in both graphs, even though one graph has triangles
and the other is bipartite.

## 6. Walk aggregation is not first arrival

Repeated linear or sum-style message passing naturally combines contributions
from many walks. A vertex can contribute repeatedly through cycles and through
multiple paths. BFS instead records the first radius at which a state belongs
to the metric ball and removes the accumulated visited set from later
frontiers.

This reprises note 33:

```text
message/walk mass at round r != exact distance-r sphere.
```

An exact BFS simulation needs idempotent reached-state semantics or exact
minimum-distance relaxation. Ordinary sum aggregation has neither property by
itself.

## 7. Over-squashing and frontier growth

When the number of relevant vertices in an `r`-hop neighborhood grows rapidly,
a fixed-width vector must carry information influenced by many sources. Narrow
cuts can funnel many dependencies through a small number of intermediate
states. This is the over-squashing phenomenon studied for GNNs.

BFS feels the same geometry differently:

- wide spheres require explicit frontier and visited capacity;
- bottlenecks can make the frontier narrow and later burst;
- exact BFS retains separate state identities rather than compressing the
  whole ball into one vector;
- message passing may keep fixed per-node width but lose task-relevant distant
  distinctions.

Exponential neighborhood growth is a tree-like or expanding-family condition,
not a universal law. Cycles, relations, polynomial-growth Cayley graphs, and
finite saturation can change the rate while leaving long-range compression
issues possible.

## 8. Learned guidance is not exact exploration

A learned embedding or predictor can rank states, estimate distance, or guide
beam search. Its output does not establish:

- that every legal successor was generated;
- that equal embeddings are equal states;
- that a predicted distance is admissible or exact;
- that discarded states cannot contain a shortest path;
- that an empty retained beam proves unreachable.

Those claims require the exact contracts developed in notes 24, 28, 41, 49,
and 50. A neural score can coexist with exact BFS only when it is advisory or
when a separately proved bound/pruning condition preserves completeness.

## 9. Cayley and Schreier qualifications

Translation symmetry makes homogeneous Cayley neighborhoods especially
difficult to distinguish without a marked goal/source or state attributes.
Generator labels can preserve move type but still do not identify the group
element: every translated vertex has the same labeled local template.

For a Schreier action, stabilizers can make distinct group words reach the same
state. An MPNN operating on word-generation records may therefore aggregate a
different object from one operating on deduplicated orbit states. The semantic
vertex contract must precede any expressiveness claim.

## 10. GPU and multi-GPU interpretation

Both BFS and message passing can expose bulk edge-parallel work, but their
storage and correctness obligations differ:

- BFS frontier size varies and `visited` enforces exact set identity;
- an MPNN commonly stores one fixed-width vector per materialized vertex;
- implicit Cayley BFS generates vertices on demand, whereas many GNN workloads
  assume a materialized graph;
- multi-GPU feature exchange communicates boundary embeddings, while exact
  BFS routing communicates candidate identities and discovery ownership;
- high message throughput does not prove exact unique-state throughput;
- approximate or low-precision embeddings cannot replace exact equality.

No performance transfer should be inferred without declaring graph
materialization, feature width, frontier shape, aggregation, synchronization,
and the exact output contract.

## 11. Evidence checklist

1. Local-only architecture or presence of global/positional channels.
2. Source/goal marker and all initial attributes.
3. Number of completed message-passing rounds.
4. Exact discrete operation versus learned finite-dimensional aggregation.
5. Walk multiplicity, state identity, and duplicate semantics.
6. Claimed radius of influence versus claimed recovered information.
7. 1-WL assumptions and any higher-order or individualized features.
8. Advisory prediction versus exact search certificate.

## Sources

- K. Xu, W. Hu, J. Leskovec, and S. Jegelka, [*How Powerful Are Graph Neural
  Networks?*](https://arxiv.org/abs/1810.00826), ICLR 2019. Neighborhood
  aggregation, 1-WL expressiveness bounds, and injective aggregation.
- C. Morris et al., [*Weisfeiler and Leman Go Neural: Higher-Order Graph Neural
  Networks*](https://doi.org/10.1609/aaai.v33i01.33014602), AAAI 2019.
  Formal connection between standard GNNs and 1-WL and higher-order scope.
- U. Alon and E. Yahav, [*On the Bottleneck of Graph Neural Networks and Its
  Practical Implications*](https://openreview.net/pdf?id=i80OPhOCVH2), ICLR
  2021. Fixed-width long-range information bottlenecks and over-squashing.
- Notes 10, 24, 25, 28, 33, 41, 46, 49, 50, 61, 93, and 101 provide frontier
  growth, beam-search, fixed-point, identity, walk, certificate, expansion,
  heuristic, stabilizer, Cayley-growth, and color-refinement context.

## Takeaway

`r` rounds of local message passing can be influenced only by an `r`-hop
neighborhood, but receptive radius is not an exact BFS result. Exact `min` or
set propagation with a marked source can simulate BFS; learned vector
aggregation generally compresses identities and walk histories. In symmetric
Cayley settings, source markers and exact state semantics are indispensable,
and learned embeddings remain guidance rather than reachability certificates.

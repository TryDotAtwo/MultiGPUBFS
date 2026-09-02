# Distance regularity: what frontier sizes do not tell BFS

The sequence of BFS frontier sizes describes metric volume growth.  It does not
usually determine how individual vertices in a layer connect backward, within
the layer, or forward.  Distance-regular graphs are the exceptional setting in
which those local transition counts depend only on distance.

This note develops that distinction as a way to read BFS measurements.  It does
not propose an optimized traversal.

## Per-vertex intersection profile

Let `G` be a connected finite simple undirected graph, fix root `s`, and write

```text
F_i(s) = {v | dist(s,v)=i},
w_i(s) = |F_i(s)|.
```

For `v in F_i(s)`, every neighbor lies in one of three layers because adjacent
distances differ by at most one.  Define

```text
c_i(s,v) = |N(v) intersect F_(i-1)(s)|   backward neighbors
a_i(s,v) = |N(v) intersect F_i(s)|       same-layer neighbors
b_i(s,v) = |N(v) intersect F_(i+1)(s)|   forward neighbors.
```

If `G` is `k`-regular, then pointwise

```text
c_i(s,v) + a_i(s,v) + b_i(s,v) = k.
```

These are semantic edge classes, not implementation counters.  Generator
occurrences, parallel labels, rejected inverse moves, and a simplified
undirected edge set can give different physical counts.

## Distance-regularity

A connected graph of diameter `D` is distance-regular when, for every pair
`s,v` at distance `i`, the backward and forward counts depend only on `i`, not
on the particular pair:

```text
c_i(s,v) = c_i,
b_i(s,v) = b_i.
```

Regularity then gives `a_i = k-b_i-c_i`.  The intersection array is

```text
{b_0,b_1,...,b_(D-1); c_1,c_2,...,c_D}.
```

Distance-regularity is stronger than ordinary degree regularity, but is
**incomparable with vertex transitivity**: neither implies the other.
The Cayley example below is vertex-transitive but not distance-regular.
Conversely, the twisted Grassmann graphs constructed by van Dam and Koolen
are distance-regular but not vertex-transitive, as described in
[van Dam's author abstract](https://www.maths.dur.ac.uk/lms/099/abstracts.html)
for the 2013 Durham symposium. Distance-transitivity implies both properties;
neither property by itself implies distance-transitivity.

## Edge balance between consecutive layers

Count edges between `F_i` and `F_(i+1)` from both sides.  In a distance-regular
graph,

```text
w_i b_i = w_(i+1) c_(i+1),
```

and therefore

```text
w_(i+1) = w_i b_i / c_(i+1).
```

The identity explains sphere growth as a competition between outward choices
per current vertex and convergence multiplicity per next-layer vertex.

The balance equation itself does not require distance-regularity.  For an
arbitrary rooted graph it becomes

```text
sum_(v in F_i) b_i(s,v)
  = |E(F_i,F_(i+1))|
  = sum_(u in F_(i+1)) c_(i+1)(s,u).
```

Only the replacement of these sums by uniform constants is special.

## Frontier sizes do not recover the intersection profile

Knowing `w_i` and `w_(i+1)` gives at most a ratio of aggregate edge counts.  It
does not reveal:

- the number of same-layer edges;
- how backward multiplicity is distributed among next-layer vertices;
- which vertices are convergence hotspots;
- how many labeled generator occurrences realize each simple edge;
- correlations between a state representation and its local profile.

A minimal visual witness starts from cycle `C_6`, rooted at vertex zero.  Its
frontier sizes are

```text
1, 2, 2, 1.
```

Adding a chord between the two depth-two vertices leaves every distance and
frontier size unchanged.  It nevertheless adds a same-layer edge, an odd
cycle, two adjacency inspections, and different duplicate behavior.  Metric
volume alone cannot distinguish the two rooted graphs.

## A Cayley graph need not be distance-regular

Consider the undirected Cayley graph

```text
Cay(Z_8, {+1,-1,4}).
```

From identity `0`, its BFS layers are

```text
F_0 = {0}
F_1 = {1,7,4}
F_2 = {2,3,5,6}.
```

Every vertex has degree three and translation makes the graph
vertex-transitive.  But vertices in `F_2` do not have a uniform backward
count:

```text
N(2) = {1,3,6}: c_2(0,2)=1, a_2(0,2)=2
N(3) = {2,4,7}: c_2(0,3)=2, a_2(0,3)=1.
```

The other two depth-two vertices repeat these two profiles.  Thus the graph is
not distance-regular.

The layer boundary contains six edges.  On the `F_1` side every vertex has two
forward neighbors, while the four `F_2` vertices have backward counts
`1,2,2,1`.  Their average is `3/2`, not an integer intersection number.  The
aggregate balance is correct:

```text
3*2 = 1+2+2+1 = 6,
```

but no uniform `c_2` exists.

This example blocks a tempting inference:

```text
Cayley -> vertex-transitive -> same sphere sizes from every root
       -/-> uniform local behavior within each sphere.
```

## Useful examples where the profile is uniform

For the `n`-dimensional hypercube, a state at Hamming distance `i` has

```text
c_i = i,
a_i = 0,
b_i = n-i.
```

Consequently

```text
w_(i+1) = w_i (n-i)/(i+1),
```

which yields `w_i = binomial(n,i)`.  Here the frontier sequence and the
intersection array reflect coordinate symmetry exactly.

For a cycle, vertices before the antipodal boundary have one backward and one
forward neighbor.  At the terminal layer the profile changes according to odd
or even cycle parity.  These examples are structured graph families, not a
generic model for puzzle Cayley graphs.

## Relation to duplicate classes

For an inverse-closed simple graph, expanding `F_i` produces occurrences in
three semantic directions:

- `c_i`-type occurrences hit `F_(i-1)` and are already visited;
- `a_i`-type occurrences hit the same layer and expose parity/cycle structure;
- `b_i`-type occurrences target `F_(i+1)`, but several parents may converge on
  the same child.

Even in a distance-regular graph, `b_i` alone is not the number of unique new
states per parent.  Uniqueness is a property after merging across parents, and
`c_(i+1)` measures the uniform multiplicity of boundary edges into each child.
The balance equation performs that global accounting.

Outside distance-regularity, averages can hide variance.  Two layers with the
same total edge count can produce different contention, owner skew, hash-probe
locality, and warp divergence because their `c/a/b` distributions differ.

## Dead ends are the zero mass of the forward profile

Note 72 calls `v in F_i` a dead end when it has no neighbor in `F_(i+1)`.
In this notation that is exactly

```text
b_i(s,v)=0.
```

Let `m=|F_i|`, let `z` be the number of such zero-forward vertices, and write

```text
E_i = sum_(v in F_i) b_i(s,v),
mean_b = E_i/m.
```

Suppose `B>0` and every `b_i(s,v)` is at most `B`.  Every non-dead vertex contributes at
least one and at most `B` boundary edges, so

```text
m-z <= E_i <= B(m-z).
```

Equivalently, for dead-end fraction `p_0=z/m`,

```text
max(0,1-mean_b) <= p_0 <= 1-mean_b/B.
```

The integer upper bound is `z <= m-ceil(E_i/B)`.  In a simple undirected
`k`-regular graph and a non-root layer, one may take `B<=k-1` because every
vertex has at least one backward neighbor.  Labeled multigraph occurrences
need their own declared maximum instead of silently using simple degree.

The bounds expose what an aggregate can and cannot certify:

- `E_i=0` means every current parent is a dead end and `F_(i+1)` is empty;
- `E_i>0` means at least one parent has an outward neighbor;
- an intermediate mean usually leaves a wide range of possible dead-end
  fractions.

For a sharp small witness, take four current-layer vertices and four next-layer
vertices.  Both of the following simple layered boundary patterns have four
edges, four unique next states, and backward profile `[1,1,1,1]` on the next
layer:

```text
current forward degrees [1,1,1,1] -> zero dead ends
current forward degrees [4,0,0,0] -> three dead ends.
```

They have the same `|F_i|`, `|F_(i+1)|`, and total forward occurrence count.
Only the per-parent distribution reveals whether outward progress is uniform
or concentrated through one gateway.  Thus even adding aggregate boundary work
to the complete frontier profile does not recover radial heterogeneity.

## What should be measured

Alongside `w_i`, a conceptual BFS profile can record:

- histograms of per-state backward, same-layer, and forward occurrences;
- the explicit zero-forward mass and adopted dead-end/depth convention;
- mean, variance, extrema, and quantiles of those counts;
- unique next states and their number of frontier parents;
- labeled occurrences versus simple endpoint edges;
- owner-conditioned versions of the same distributions;
- correlations with state encoding, generator label, and partition owner.

These measurements explain work; they do not alter BFS semantics.  A histogram
matching an intersection array at sampled depths is evidence of regularity, not
a proof over the whole graph.

## GPU and multi-GPU interpretation

Uniform intersection numbers make aggregate cost more predictable: every state
at depth `i` has the same semantic split of transitions.  They do not guarantee
uniform execution time, because state generation, encoding, memory placement,
and owner routing may still differ physically.

Conversely, a broad backward-multiplicity distribution predicts uneven
candidate convergence.  On one GPU it may concentrate atomics or hash probes;
across GPUs it may concentrate messages at particular owners.  Frontier size
alone cannot predict either effect.

No scheduling or kernel policy follows automatically.  The appropriate first
step is to distinguish a graph-theoretic imbalance from a representation or
partition imbalance using measured distributions.

## Rejected shortcuts

- **A regular graph has one intersection array.** Degree regularity fixes only
  `a_i+b_i+c_i` pointwise.
- **Every Cayley graph is distance-regular.** `Cay(Z_8,{+1,-1,4})` is an
  explicit counterexample.
- **Equal frontier sizes imply equal BFS work.** Same-layer edges and boundary
  multiplicities may differ.
- **The ratio `w_(i+1)/w_i` determines `b_i` and `c_(i+1)`.** It constrains only
  their ratio under uniformity, or aggregate sums without it.
- **Uniform semantic counts imply uniform GPU cost.** Physical state and memory
  operations remain separate.
- **A sampled uniform profile proves distance-regularity.** The definition
  quantifies over every root-state pair at every distance.

## Sources

- van Dam, Koolen, and Tanaka,
  [Distance-Regular Graphs](https://www.combinatorics.org/ojs/index.php/eljc/article/view/DS22),
  is a modern survey of the subject.
- Edwin van Dam,
  [Eigenvalues and distance-regularity of graphs, Durham 2013 author abstract](https://www.maths.dur.ac.uk/lms/099/abstracts.html),
  describes his joint twisted-Grassmann construction with Koolen and explicitly
  identifies distance-regular graphs that are not vertex-transitive. The
  existence claim, not the abstract's historical "only known family" status,
  is used here.
- Elena Konstantinova,
  [Some Problems on Cayley Graphs](https://zalozba.upr.si/ISBN/978-961-6832-50-2.pdf),
  gives the sphere/intersection-number definitions, intersection arrays, and
  examples separating Cayley, vertex-transitive, distance-regular, and
  distance-transitive properties.
- Note 10 supplies the existing frontier-growth and duplicate-work metrics;
  notes 16 and 21 supply the Cayley action and symmetry boundaries.

## Current conclusion

Frontier sizes are a radial volume profile.  Intersection numbers, when they
exist, describe the local flow of edges through that profile.  Distance-
regularity makes this flow uniform; ordinary Cayley vertex transitivity does
not.  For general puzzle graphs, distributions of backward, lateral, and
forward multiplicity are more informative than treating each layer as a single
homogeneous number.

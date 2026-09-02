# Frontier growth: BFS as the moving boundary of a metric ball

BFS does not simply "make a queue larger and then smaller."  At depth `d` it
knows the metric ball `B_d`, while the frontier `F_d` is its newest sphere.  The
next frontier is the outer vertex boundary:

```text
F_(d+1) = N_out(B_d) minus B_d
        = N_out(F_d) minus B_d.
```

The word *ball* is essential: `B_d` must contain every vertex whose true
source distance is at most `d`, not merely some correctly labeled vertices.
Consider the undirected edges

```text
s--a,  s--b,  a--x
```

and the incomplete set `B'_1={s,a}`. Every stored depth is correct, but `b` at
true depth one is missing. Its external boundary is

```text
N(B'_1) minus B'_1 = {b,x},
```

which mixes true layers one and two. Expanding only the partial newest set
`{a}` instead yields `{x}` and misses `b` entirely. Therefore neither boundary
formula defines an exact next sphere until all distances through `d` are
closed. Layer closure is semantic completeness, not just a synchronization
convenience.

The mixed set `{b,x}` is not intrinsically an invalid frontier. If the
incomplete visited set is re-declared as a new multi-source set
`A={s,a}`, then every member of `A` has new distance zero and

```text
F_1(A) = N(A) minus A = {b,x}
```

is exact for that new metric. What fails is treating this restart as a
continuation of the original single-source distances, where `a` has depth one,
`b` has depth one, and `x` has depth two. The same set operation can therefore
be correct while answering a different BFS problem; source initialization is
part of the semantic contract.

This geometric view separates three phenomena often compressed into the word
*branching*:

1. how many transition occurrences are attempted;
2. how many distinct vertex identities those occurrences represent;
3. how many distinct identities lie outside the accumulated ball.

Only the third number is frontier growth.

## A per-level vocabulary

Let

```text
w_d = |F_d|                              frontier width
V_d = |B_d| = sum_(i=0)^d w_i            cumulative visited volume
e_d = sum_(u in F_d) out_degree(u)        transition occurrences
c_d = |N_out(F_d)|                        unique candidate identities
h_d = |N_out(F_d) intersect B_d|          unique visited hits
```

Then

```text
c_d = h_d + w_(d+1)
e_d = duplicate_occurrences_d + h_d + w_(d+1).
```

Two useful but different ratios are

```text
growth ratio       rho_d   = w_(d+1) / w_d
discovery yield    alpha_d = w_(d+1) / e_d.
```

`rho_d` describes the geometry of spheres.  `alpha_d` describes how much
transition work discovers new states.  A level can have a growing frontier and
poor discovery yield at the same time if many paths converge.

For a `k`-regular Cayley graph, `e_d=k*w_d`, but neither `c_d` nor `w_(d+1)` is
fixed by regularity.  Relations determine how generator words collapse to
equal elements and how much of the neighbor set is already inside the ball.

## Edge distance constraints

For any directed unit edge `u -> v`, source distance satisfies

```text
d(v) <= d(u) + 1.
```

Thus an edge out of `F_d` can reach the next layer or any already visited
layer, but never skip forward beyond `d+1`.

For an undirected edge, the triangle inequality works in both directions:

```text
|d(u)-d(v)| <= 1.
```

Every edge touching `F_d` therefore connects only `F_(d-1)`, `F_d`, or
`F_(d+1)`.  In a bipartite graph, same-level edges are impossible because they
would join vertices of equal parity.  This makes some accounting especially
clean, but it is not a general BFS property.

## Four archetypal growth shapes

### Tree-like exponential growth

In a rooted directed `b`-ary tree,

```text
w_d = b^d,
V_d = (b^(d+1)-1)/(b-1),
alpha_d = 1.
```

Every generated child is new and has one parent.  This is the mental model
behind the familiar `O(b^d)` BFS bound and the optimistic bidirectional estimate
near `2b^(d/2)`.

An undirected `k`-regular tree differs slightly: after the root, one edge per
vertex goes back toward the source, so spheres grow by approximately `k-1`, not
`k`.  Even a relation-free undirected graph has visited back-edges.

### Polynomial growth and massive word convergence

In the standard Cayley graph of `Z^m`, ball volume grows polynomially like
`d^m` and sphere width like `d^(m-1)`.  Yet the number of length-`d` generator
words is exponential.  Commutation relations cause many words to reach the same
lattice point.

This is a sharp counterexample to treating branching factor as frontier growth:
the attempted word tree and the quotient state graph have different geometry.

### Bottlenecks and bursts

A graph can have a narrow bridge leading into a large region.  Frontier width
may shrink to one and then expand sharply.  Degree statistics averaged over the
whole graph do not predict the local boundary of the particular source ball.

This also defeats policies based only on the immediately preceding growth
ratio: `rho_(d-1)` need not predict `rho_d` on an irregular graph.

### Finite saturation

In a finite connected graph, `V_d` eventually reaches `|V|` and the next
frontier becomes empty.  Frontier width often rises and falls, but neither
unimodality nor symmetry is guaranteed for an arbitrary graph.

Near saturation, many generated transitions point into the known ball.  Low
discovery yield is then a consequence of finite geometry, not necessarily an
inefficient visited implementation.

## Boundary size is the real expansion signal

The set `F_(d+1)` is the external vertex boundary of `B_d`.  Its size depends on
how tightly the current ball connects to itself and how many distinct outside
vertices its outgoing edges reach.

Two balls can have the same volume and edge cut but different next-frontier
widths:

- many boundary edges may converge on few outside vertices;
- few edges may each reach a different outside vertex.

Thus edge boundary, vertex boundary, and generated occurrence count are
separate quantities.  GPU work may follow the edge boundary while future
frontier memory follows the vertex boundary.

An isoperimetric or expansion statement relates boundary size to ball volume.
It can explain growth for a graph family, but measuring one BFS trace does not
establish a global expansion constant.

## Generator words, vertices, and geodesics

For a Cayley graph, there are three counts at depth `d`:

- all length-`d` words;
- distinct group elements reachable by such words;
- elements whose **shortest** word length is exactly `d`.

Only the last count is `w_d`.  A word may reduce to a shorter element, and many
geodesic words may represent the same frontier element.

Relations affect these counts at different lengths:

- inverse cancellation creates immediate returns;
- involutions make `s^2=e`;
- commutation makes two length-two words coincide;
- braid relations create longer convergences;
- identity or duplicate labeled generators add occurrences without new
  vertices;
- genuinely new generators can shorten distances and reshape every later
  sphere.

The shortest relation length gives clues about when a word tree first deviates
from a state graph, but it does not alone determine all sphere sizes: relations
overlap and combine.

## Exact geometry of adjacent-transposition `S_n`

For permutations generated by adjacent swaps, distance from the identity is
inversion count.  Therefore `w_d` is the number of permutations with exactly
`d` inversions: a Mahonian number.  The sphere generating polynomial is

```text
sum_d w_d q^d
= product_(i=1)^n (1 + q + ... + q^(i-1)).
```

The product has degree

```text
D = n(n-1)/2,
```

and is palindromic: replacing every permutation by the complementary/reversed
one maps inversion count `d` to `D-d`.  Hence

```text
w_d = w_(D-d).
```

For `S_8`, `D=28`, and the retained exact frontiers are

```text
1, 7, 27, 76, 174, 343, 602, 961, 1415, 1940,
2493, 3017, 3450, 3736, 3836, 3736, 3450, 3017,
2493, 1940, 1415, 961, 602, 343, 174, 76, 27, 7, 1.
```

The symmetry and middle peak are therefore combinatorial structure, not a
generic "BFS bell curve."

## A stronger `S_8` conservation identity

Every adjacent transposition is an involution and changes inversion parity and
count by exactly one.  Thus the graph is bipartite and

```text
N(F_d) = F_(d-1) union F_(d+1)
```

as a set of distinct vertices (with the missing boundary layer interpreted as
empty).  With seven generators,

```text
7*w_d
= batch_duplicate_occurrences_d
+ w_(d-1)
+ w_(d+1).
```

At depth 14:

```text
w_14                         = 3,836
generated = 7*w_14           = 26,852
unique previous-layer hits   = 3,736
accepted next layer w_15     = 3,736
batch duplicate occurrences = 19,380
```

The duplicate count is not unexplained noise; it is the exact remainder after
the two neighboring spheres are removed from the seven labeled edge
occurrences.  A read-only Docker check recomputed this equality from the retained
REF-003 CSV for all 29 levels (`S8_GEOMETRY_IDENTITY_PASS levels=29`).

At depth 28, the single reversed permutation generates seven transitions, all
to `F_27`, so discovery yield is zero and traversal exhausts.  The kernel could
execute those seven moves perfectly; geometry still supplies no new state.

## What REF-004 changes geometrically

The four `S_8` generator collections make a useful controlled comparison:

- adding identity preserves all spheres but adds a self-loop occurrence at
  every vertex;
- duplicating `s_0` preserves all spheres but increases occurrence
  multiplicity;
- adding a 3-cycle and its inverse changes the word metric, reduces diameter
  from 28 to 22, increases peak frontier from 3,836 to 4,420, and creates
  same-level edges.

This rejects three tempting equivalences:

```text
smaller diameter  != smaller peak frontier
more generators   != proportionally faster growth
more transitions  != more newly discovered states.
```

The generator set determines the geometry and the physical work together, but
not in the same direction.

## Frontier memory and visited memory peak at different times

Frontier storage follows `w_d`; exact visited storage follows `V_d`.  For the
retained `S_8` trace:

- peak frontier is 3,836 at depth 14;
- visited through depth 14 is 22,078;
- full visited at exhaustion is 40,320.

A double-buffered frontier can fit while cumulative visited does not, or the
reverse can occur when a candidate bag is materialized.  Capacity planning must
separate at least:

```text
current frontier
raw candidates (if materialized)
next frontier
cumulative visited
parent/path metadata
temporary dedup/routing storage.
```

No single "number of states" captures all peaks.

## Bidirectional search is ball geometry too

Bidirectional BFS explores a forward ball and a reverse ball until their
distance bounds meet.  Its work depends on the volumes and boundaries of both
sides, not merely on target distance.

Tree-like growth makes two half-radius balls attractive.  In a finite
saturating graph, two large-radius balls may each contain much of the component.
The retained `S_8` sweep illustrates this: reduction versus level-complete
unidirectional work was large around middle depths but fell to 9.51% at diameter
28.

The result is not paradoxical.  The farthest target forces both symmetric balls
near the high-volume middle before the lower bound closes.

## GPU and multi-GPU implications without an optimizer

The geometric quantities map to different physical pressures:

| Geometry/work | Likely physical pressure |
|---|---|
| Large `w_d` | frontier storage and available parallelism |
| Large `e_d` | expansion instructions/adjacency traffic |
| Large `e_d-c_d` | duplicate convergence work |
| Large `h_d` | visited probes/claims with low discovery yield |
| Large `V_d` | cumulative visited and parent capacity |
| Concentrated equal candidates | local atomic/warp contention opportunity |
| Cross-owner convergence | communication plus owner-side dedup |
| Rapidly changing `w_d` | load/capacity variation across levels |

These are explanatory variables, not a policy selector.  REF-014 through
REF-017 show that even the same global counts can behave differently when
candidate order changes their spatial locality.

For multiple GPUs, ownership partitions the boundary.  Useful per-level
geometry includes not only total `w_d` and `e_d`, but each owner's share,
remote boundary edges/candidates, and where duplicates converge.  A partition
can reduce remote boundary traffic while worsening load skew, as REF-006/010
demonstrate in simulation.

## Counterexamples to common growth intuitions

### Degree predicts frontier width

A clique has high degree but BFS from any vertex finishes after one wide layer;
subsequent discovery yield is zero.  A low-degree tree can grow exponentially
for many levels.

### Duplicate ratio predicts next width

The same number of duplicate occurrences can coexist with different counts of
already-visited unique candidates.  Only after both exact dedup and subtraction
from `B_d` is `w_(d+1)` known.

### Peak frontier occurs halfway through diameter

It does for the symmetric Mahonian `S_n` profile, but a bottleneck followed by a
dense region or an asymmetric directed graph can peak almost anywhere.

### Lower diameter means less total work

Adding edges can shorten all distances while increasing degree enough to raise
the total number of generated transitions.  REF-004's 3-cycle pair is a finite
exact counterexample.

### Vertex-transitive means tree-like

Cayley graphs are vertex-transitive, but group relations can produce
polynomial, intermediate, or exponential growth and extensive convergence.
Every vertex looking locally the same does not determine global ball growth.

## A growth-profile audit

For each level, preserve:

1. `w_d`, `V_d`, and `e_d`;
2. unique candidate count `c_d`;
3. unique earlier-level and same-level hits separately;
4. batch duplicate occurrences `e_d-c_d`;
5. accepted `w_(d+1)` and all conservation identities;
6. parent multiplicity or shortest-path multiplicity if relevant;
7. owner-local and cross-owner versions for distributed search;
8. state/key/record bytes attached to each count;
9. graph properties used in interpretation: directedness, bipartiteness,
   regularity, generator relations, finite diameter;
10. whether a symmetry in the profile is proved or merely observed.

## Sources and evidence

- Clara Löh, *Geometric Group Theory: An Introduction*,
  [lecture notes](https://loeh.app.uni-regensburg.de/teaching/ggt_ss22/lecture_notes.pdf),
  for Cayley balls, sphere/ball growth, and polynomial versus exponential
  examples.
- MIT combinatorial analysis notes on
  [permutation inversions and q-binomials](https://math.mit.edu/~fgotti/docs/Courses/C.%20Combinatorial%20Analysis/5.%20Permutation%20Inversions%20and%20q-Binomials/Permutation%20Inversions%20and%20q-Binomials.pdf),
  for inversion tables and the q-factorial/Mahonian enumeration.
- Local exact evidence: REF-002/003 for the `S_8` layer and candidate profile,
  REF-004 for generator-set changes, REF-007 for bidirectional ball work, and
  REF-016/017 for ordering/locality effects at fixed frontier sets.

## Current synthesis

The frontier is a geometric boundary, not a branching-factor estimate.  Degree
controls transition occurrences; relations and visited history control how
those occurrences collapse; the outer boundary alone controls new states.
This explains why BFS can have abundant GPU work but little progress, why a
regular Cayley graph can exhibit intense duplicate convergence, and why
diameter, peak frontier, total transitions, visited capacity, and communication
must be measured as different properties.

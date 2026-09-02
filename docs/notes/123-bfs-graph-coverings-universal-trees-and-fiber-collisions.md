# BFS under graph coverings: universal trees and fiber collisions

Note 17 distinguishes homomorphisms, automorphism quotients, and graph covers,
and proves unique path lifting for a cover. This note asks a narrower BFS
question: how do metric balls and layers in a covering graph project to the
base, and exactly when does the universal-cover tree stop looking like the base
state graph? It adds no implementation, optimizer, benchmark, or GPU code.

## 1. Covering contract

Let

```text
p: G_tilde -> G
```

be a covering projection of connected undirected graphs. At every lifted vertex
`x`, the incident edge directions map bijectively to the incident directions at
`p(x)`. For simple loopless graphs this can be read as a bijection between
neighbor sets; loops and parallel edges require explicit edge/dart identity.

The fiber over `v` is

```text
p^(-1)(v) = {x : p(x)=v}.
```

A base path has exactly one lift after its initial lifted vertex is fixed. A
general homomorphism lacks this guarantee, and an orbit quotient may permit
several lifts as note 17 explains.

## 2. Projection cannot increase distance

Every lifted path projects to a base walk of the same length. Therefore

```text
d_G(p(x),p(y)) <= d_G_tilde(x,y).
```

The inequality can be strict because different lifted vertices in one fiber
represent the same base vertex. Covering is locally exact but not globally
distance-preserving between arbitrary fixed lifts.

For fixed `x` over source `s` and any base target `t`, unique lifting of a base
shortest path gives

```text
d_G(s,t) = min_(y in p^(-1)(t)) d_G_tilde(x,y).
```

Thus a cover preserves distance to a target fiber, not distance to every named
member of that fiber. This parallels note 17's orbit-target distinction, now
with unique rather than merely existent lifts.

## 3. Exact projection of BFS balls

For every integer radius `r>=0`,

```text
p(B_r_G_tilde(x)) = B_r_G(p(x)).
```

The two inclusions are direct:

- a lifted path of length at most `r` projects to a base walk of that length;
- every base shortest path of length at most `r` lifts from `x` to a vertex in
  the appropriate fiber at the same distance.

Consequently

```text
|B_r_G| <= |B_r_G_tilde|.
```

Equality of cardinalities holds exactly when the projection is injective on the
lifted ball. Beyond that point, extra lifted vertices are distinct histories or
fiber representatives that base-state visited semantics must merge.

## 4. Layers project less cleanly than balls

Every base vertex at distance exactly `i` has a lift at lifted distance exactly
`i`: a lifted base geodesic cannot be shortened, because its shorter projection
would contradict base distance. Hence

```text
F_i_G(s) subset p(F_i_G_tilde(x)).
```

The reverse inclusion can fail. A lifted vertex `y` at distance `i` may project
to a base vertex reachable by a shorter path whose lift ends at a different
member of the same fiber. Therefore `p(y)` can belong to an earlier base layer.

So balls project exactly as sets, while one lifted sphere can mix several base
distances. This is why depth in a walk/universal-cover tree is path-history
length, not automatically state distance.

## 5. The universal covering tree

The universal cover of a connected graph can be constructed from finite
non-backtracking walks starting at a root:

- one tree vertex per walk;
- parent obtained by deleting the last edge;
- children obtained by extending without immediately reversing the last edge;
- projection sends a walk to its final base vertex.

This graph is a tree and locally covers the base. Its BFS depth is exactly the
length of the represented reduced walk. Distinct reduced walks remain distinct
tree vertices even when they end at the same base state.

Therefore universal-cover BFS counts non-backtracking histories. Base BFS counts
minimum-distance states after fiber projection and visited deduplication. They
coincide only within a proved injective region or globally when the base itself
is a tree.

## 6. Fiber collision and closed reduced walks

Suppose two universal-cover vertices `a,b` in the rooted ball map to the same
base vertex. Their root paths project to two reduced walks with the same
endpoint. Following one and reversing the other gives a closed walk; after
cancellation it contains a nontrivial reduced closed witness unless the two
histories were identical.

In a simple graph, a nontrivial reduced closed walk contains a simple cycle.
Thus girth controls the first possible fiber collision. If base girth is `g`
and

```text
2r < g,
```

then the universal-cover projection is injective on the radius-`r` vertex ball.
This is the covering form of note 27's unique-geodesic threshold.

Injectivity of ball vertices is weaker than an induced-ball isomorphism. To
exclude an additional base edge between two projected ball vertices requires

```text
2r+1 < g.
```

The one-step difference separates state collisions from boundary edge closure.

## 7. Cycle calibrations

The universal cover of `C_n` is the infinite integer line, projected modulo
`n`.

### Odd cycle

For `C_5` and `r=2`, the five lifted vertices `-2,-1,0,1,2` project bijectively
to all five base vertices because `2r=4<5`. But the endpoints `-2` and `2` are
adjacent after projection, closing the base cycle. The lifted induced ball is a
path; the base induced ball is `C_5`.

### Even cycle

For `C_6` and `r=3`, lifted vertices `-3` and `3` project to the same antipode.
The equality `2r=g` is exactly the first vertex collision and the first pair of
equal-length geodesics from the root.

### Triangle

For `C_3` at radius one, the three lifted ball vertices are distinct, but the
two depth-one base neighbors are adjacent. This is the smallest demonstration
that local neighbor bijection does not make a radius-one induced ball a tree.

## 8. A finite-cover fixed-lift counterexample

Map `C_6` to `C_3` modulo three. This is a two-sheeted cover. The lifted
vertices `0` and `3` lie in the same fiber over base vertex zero.

```text
d_C3(0,0)=0,
d_C6(0,3)=3.
```

The distance-to-fiber theorem is still exact because the minimum over fiber
`{0,3}` is zero. What fails is the unjustified claim that every fixed lift pair
has the base distance.

## 9. Finite sheets and saturation

For a connected finite `k`-sheeted cover of a connected base, each base vertex
has `k` lifted representatives. A growing lifted BFS can therefore contain up
to `k` copies of each projected state, reached at different depths and through
different monodromy histories.

The base frontier cannot be recovered by merely dividing lifted layer counts by
`k`: one lifted sphere can project to several base layers, and different fiber
members need not occur at the same lifted depth from the chosen start.

Sheet count is a global fiber fact, not a per-layer multiplicity theorem.

## 10. Cayley presentations as tree quotients

Let a free group on a symmetric basis map onto a generated group `G`. Its Cayley
tree projects to the Cayley graph of `G` by evaluating each reduced word. Under
clean distinct nonidentity generator and labeled-edge conventions, this is the
universal covering picture:

- free reduced words are tree vertices;
- group relations identify fibers;
- shortest group word length is minimum tree depth over a fiber;
- a reduced identity word is a closed projection/deck witness.

If generator images collide, become identity, or edge multiplicity is silently
collapsed, local bijectivity can fail. A group homomorphism alone does not prove
that the induced simple Cayley-graph map is a cover.

For Schreier actions, distinct group elements can lie in the same state fiber
through a stabilizer. The first collision may therefore be a stabilizer word,
not an identity relation in the acting group. Notes 16, 39, and 61 supply the
required action distinction.

## 11. Quotients are not automatically coverings

A quotient or canonicalization may merge neighbor edges, create loops, or make
different representatives expose different neighbor classes. Such a map may be
a homomorphism or a transition congruence without being locally bijective.

Only a proved cover permits all of the following simultaneously:

- exact local edge multiplicity;
- unique path lift from a chosen concrete start;
- exact projection of every lifted ball onto the base ball;
- uniform sheet semantics in a connected finite cover.

Orbit quotients can still be correct for orbit distance, but their lifting and
fiber rules are those of note 17 rather than covering theory.

## 12. Local indistinguishability

Before the injectivity boundary, a base root and its universal-cover lift have
the same rooted ball. A procedure that observes only a bounded anonymous local
neighborhood cannot distinguish whether a visible branch later closes into a
cycle or continues in the tree.

This refines notes 35, 97, and 102: identical bounded views do not certify graph
order, finiteness, ends, or global topology. IDs or other global information can
break this indistinguishability, but then the evidence is no longer purely the
unlabeled local BFS view.

## 13. GPU and multi-GPU boundary

Expanding the universal-cover tree without base-state deduplication intentionally
keeps history copies. Its candidate count can follow non-backtracking tree
growth even when the base BFS frontier is small. This is a different search
state and workload.

Using a lifted path/history key for visited or ownership does not implement
base-state BFS unless fiber-equivalent endpoints are reconciled. Conversely,
deduplicating only by projected state discards cover-state identity and is
incorrect if the intended problem truly lives in the cover/product history.

In distributed execution, a fiber collision may occur across owners. Exact base
BFS requires one authoritative projected-state claim and correct shortest-depth
resolution. Report separately:

- lifted candidates or histories;
- projected base candidates;
- unique base states and old-state hits;
- fiber multiplicity by lifted depth;
- cross-owner fiber reconciliation;
- injective-radius proof versus measured collision onset;
- base and cover memory/throughput.

No universal-cover tree rate is a base BFS throughput claim.

## Sources

- F. T. Leighton,
  [*Finite Common Coverings of Graphs*](https://doi.org/10.1016/0095-8956%2882%2990042-9),
  Journal of Combinatorial Theory, Series B 33(3), 1982. Establishes the
  classical finite graph-cover setting and common-cover structure.
- B. Courcelle,
  [*Unfoldings and Coverings*](https://fi.episciences.org/11360/pdf),
  Fundamenta Informaticae 189(1), 2022. Gives a modern graph-theoretic treatment
  of unfoldings, universal covers, and locally bijective maps.
- J. Fiala and D. Paulusma,
  [*Comparing Universal Covers in Polynomial Time*](https://doi.org/10.1007/978-3-540-79709-8_18),
  CSR 2008. Uses the non-backtracking-walk construction of the universal cover.
- Notes 16, 17, 27, 35, 39, 61, 97, and 102 provide this repository's action,
  lifting, girth, local-growth, word-tree, stabilizer, end, and local-view
  distinctions.

## Takeaway

A graph cover is locally exact and lifts paths uniquely, yet it preserves
fixed-source distance only after minimizing over the target fiber. Cover BFS
balls project exactly onto base BFS balls, but cardinalities and layers diverge
when several lifted histories represent one base state. The universal cover
makes those histories into a tree; girth marks the first possible vertex
collision, one step later than the first possible induced-boundary closure.

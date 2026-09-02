# BFS distance embeddings and strong resolution

Notes 78-79 developed BFS distance fields as landmark coordinates and asked
whether their vectors identify vertices. Identification is still weaker than
preserving the graph metric. This note separates those guarantees.

The scope is a finite connected undirected graph with exact complete BFS
distances. Directed distance is asymmetric and does not fit the symmetric
`l_infinity` norm without a different contract.

## 1. Every BFS coordinate is nonexpanding

For an ordered landmark set `W=(w_1,...,w_q)`, define

```text
Phi_W(v) = (d(v,w_1), ..., d(v,w_q)).
```

Equip the coordinate space with maximum distance:

```text
||Phi_W(u)-Phi_W(v)||_infinity
    = max_i |d(u,w_i)-d(v,w_i)|.
```

The reverse triangle inequality gives, for every coordinate,

```text
|d(u,w_i)-d(v,w_i)| <= d(u,v).
```

Hence

```text
||Phi_W(u)-Phi_W(v)||_infinity <= d(u,v).
```

Landmark coordinates never exaggerate distance in this norm. They can collapse
distinct vertices to one vector or shrink their positive distance.

This is exactly the multi-landmark lower bound from note 78, now viewed as the
metric induced by the coordinate map.

## 2. All vertices give an isometric embedding

Take every vertex as a coordinate, `W=V`. The preceding inequality supplies
the upper direction. For any pair `u,v`, choose coordinate `w=u`:

```text
|d(u,u)-d(v,u)| = d(u,v).
```

Therefore

```text
||Phi_V(u)-Phi_V(v)||_infinity = d(u,v).
```

This is the finite Fréchet/Kuratowski distance embedding. In BFS language, the
entire all-pairs distance matrix, read by columns, embeds the graph metric
isometrically into `l_infinity^|V|`.

One arbitrary coordinate can even be omitted when `|V|>=2`: for a pair, at
least one endpoint remains available as a coordinate. Thus `|V|-1` singleton
coordinates always suffice for this particular isometric construction. This is
an existence statement, not a claim that constructing or storing them is
sensible for a large search space.

## 3. Resolving does not mean isometric

Consider cycle `C5` with landmarks `W=(0,1)`. The vectors are

```text
0 -> (0,1)
1 -> (1,0)
2 -> (2,1)
3 -> (2,2)
4 -> (1,2)
```

They are all distinct, so `W` resolves the graph. But for vertices `2` and `4`,

```text
d(2,4) = 2,
||Phi_W(2)-Phi_W(4)||_infinity = 1.
```

The coordinate map is injective yet not isometric. Therefore uniqueness of
landmark signatures validates state identification, not exact reconstruction
of every pair distance by maximum coordinate difference.

## 4. Equality for one pair and strong resolution

For a landmark `w`, equality

```text
|d(u,w)-d(v,w)| = d(u,v)
```

holds exactly when one of these triangle equalities holds:

```text
d(u,w) = d(u,v)+d(v,w),
d(v,w) = d(v,u)+d(u,w).
```

Geometrically, some shortest path from `u` to `w` contains `v`, or some
shortest path from `v` to `w` contains `u`. Such a landmark **strongly
resolves** the pair.

A set `W` is a strong resolving set when every pair is strongly resolved by at
least one member. Consequently,

```text
W strongly resolves G
    iff
Phi_W is an isometric embedding into l_infinity^|W|.
```

Strong resolution implies ordinary resolution, because exact preservation of
a positive pair distance prevents equal vectors. The converse is rejected by
the `C5` example.

The minimum size of a strong resolving set is the strong metric dimension. It
answers a stricter question than the metric dimension in note 79.

## 5. Set-distance coordinates and multi-source BFS

The singleton coordinate can be generalized. For a nonempty subset `S`, let

```text
f_S(v) = d(v,S) = min_(s in S) d(v,s).
```

One joint multi-source BFS from `S` computes exactly this scalar coordinate.
It is also nonexpanding:

```text
|d(u,S)-d(v,S)| <= d(u,v).
```

Thus several independent multi-source traversals from subsets
`S_1,...,S_q` form another nonexpanding Fréchet-type coordinate map. One BFS
from the union `S_1 union ... union S_q` does not preserve those separate
coordinates; it takes their pointwise minimum again.

Singleton coordinates are special because choosing an endpoint of every pair
immediately proves the universal isometric construction. Arbitrary subset
coordinates may reduce dimension or distort distances; neither injectivity nor
isometry follows merely from exact BFS execution.

## 6. Cayley interpretation

For a genuine Cayley graph,

```text
Phi_W(g)_i = d(w_i,g) = d(e,w_i^-1 g).
```

Left translation of both `W` and `g` leaves the coordinate geometry unchanged.
A complete identity-rooted table can evaluate these coordinates for known
group elements, as note 79 explains.

This algebraic reuse does not strengthen the embedding automatically:

- one identity coordinate still merges all elements of the same word length;
- a resolving basis can remain contractive for some pairs;
- exact recovery by the maximum difference requires strong resolution;
- a Schreier or symmetry quotient needs its own distance semantics.

## 7. Directed boundary

In a directed graph, `d(u,v)` need not equal `d(v,u)`, while an ordinary norm
distance is symmetric. Forward and reverse BFS coordinates still yield the
one-sided triangle bounds of note 78, but inserting them into the undirected
absolute-difference proof changes the object being represented.

Possible directed notions include ordered coordinate inequalities,
quasi-metrics, or a chosen symmetrization such as `max(d(u,v),d(v,u))`. They
are different contracts. This note makes no universal directed-isometry claim.

## 8. Evidence and storage boundary

For `q` singleton landmarks and `n` vertices, the raw coordinate material is
`q*n` exact distances. For `q=n`, this is the full all-pairs matrix. The
Fréchet proof establishes mathematical existence, not an efficient GPU or
multi-GPU representation.

A reduced coordinate experiment should separately report:

1. injectivity failures: pairs with equal vectors;
2. contraction: `d(u,v)` versus coordinate maximum;
3. the worst or distributional distortion over the explicitly tested scope;
4. complete versus bounded/unknown coordinates;
5. independent singleton/subset fields versus one unioned multi-source field;
6. graph, action, quotient, and direction conventions.

## Sources

- J. Matousek, *Lectures on Discrete Geometry*, section on embeddings of finite
  metric spaces. Presents the standard Fréchet isometric embedding into
  `l_infinity` by all point-distance coordinates.
- O. R. Oellermann and J. Peters-Fransen,
  [*The strong metric dimension of graphs and digraphs*](https://doi.org/10.1016/j.dam.2006.06.009),
  Discrete Applied Mathematics 155 (2007), 356-364. Defines strong resolution
  through shortest paths containing one member of a pair.
- A. Sebo and E. Tannier,
  [*On Metric Generators of Graphs*](https://doi.org/10.1287/moor.1030.0070),
  Mathematics of Operations Research 29 (2004), 383-393. Studies metric
  generators and their relation to isometric graph questions.
- Notes 13, 78, and 79 supply the multi-source, landmark-bound, and ordinary
  resolving-set distinctions used here.

## Takeaway

BFS distance coordinates form a nonexpanding map into `l_infinity`. All vertex
coordinates preserve the graph metric exactly. A smaller resolving set may
identify every vertex while still shrinking distances; exact preservation by
maximum coordinate difference is equivalent to strong resolution. Independent
multi-source subset coordinates obey the same contraction principle, whereas
one BFS from their union discards them by taking another minimum.

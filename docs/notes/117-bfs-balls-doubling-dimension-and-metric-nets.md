# BFS balls, doubling dimension, and metric nets

Doubling dimension measures how many half-radius balls are needed to cover any
metric ball. Unlike treewidth, a uniformly small doubling constant really does
constrain BFS ball growth. It still does not determine exact frontier profiles
or hardware cost.

No net construction, nearest-neighbor structure, or optimized BFS is added.

## 1. Doubling constant

For a metric space `(X,d)`, the doubling constant `lambda` is the smallest value
such that every ball `B(x,r)` can be covered by at most `lambda` balls of radius
`r/2`. The doubling dimension is

```text
ddim(X) = log_2(lambda).
```

Definitions differ slightly over open/closed balls and integer rounding. In a
discrete graph metric, those choices affect small constant scales, so a measured
value must state the convention.

The definition quantifies over every center and every positive scale. A good
cover of one BFS ball is not a global doubling certificate.

## 2. Iterated covers bound BFS balls

Apply the covering rule repeatedly. After `k` halvings, a radius-`R` ball is
covered by at most `lambda^k` balls of radius `R/2^k`.

In an unweighted simple graph, distinct vertices have distance at least one.
Choose

```text
k = ceil(log_2(2R)).
```

Then every final ball has radius at most one half and contains at most one
vertex. For integer `R>=1`,

```text
|B(x,R)| <= lambda^ceil(log_2(2R)).
```

Thus bounded doubling dimension implies polynomial metric-ball growth, with an
exponent controlled by `log_2(lambda)`. Since `F_R` is contained in `B_R`, it
also bounds frontier cardinality, though usually loosely.

This is a real capacity implication, unlike the false claim that bounded
treewidth bounds BFS frontiers.

## 3. Degree is already visible at the smallest scale

The unit ball around vertex `v` contains `deg(v)+1` vertices. Balls of radius
one half are singletons, so covering `B(v,1)` requires at least `deg(v)+1` of
them. Therefore

```text
lambda >= maximum_degree + 1
```

and

```text
ddim >= log_2(maximum_degree + 1).
```

Stars and complete graphs do not contradict doubling growth bounds: their large
first frontiers force large doubling dimension. The parameter accounts for the
branching at scale one rather than hiding it.

## 4. Bounds are not frontier profiles

Doubling dimension supplies a worst-case envelope over all centers and scales.
It does not determine:

- exact sphere sizes `|F_r|`;
- whether consecutive layers grow or shrink;
- duplicate successor multiplicity;
- edge density inside or between layers;
- shortest-path counts;
- queue order or owner-partition balance.

Two spaces with comparable doubling constants can have very different layer
oscillations and local relation structure. A loose polynomial upper bound may
also be far above the actual frontier at the radii that fit on a device.

## 5. Examples across scales

Paths and cycles have constant doubling dimension under standard conventions.
Integer grids `Z^d` have doubling dimension proportional to their geometric
dimension up to constants and polynomial ball growth of degree `d`.

A regular infinite tree is not a doubling metric with one scale-independent
constant: balls contain exponentially many separated branches, so the number of
half-radius balls needed grows with radius.

The `n`-dimensional hypercube has at least
`log_2(n+1)` doubling dimension from its degree alone. Therefore its binomial
middle frontier is not evidence against the doubling framework; the dimension
parameter itself grows with `n`.

## 6. Packing and nets

An `r`-separated set has pairwise distances greater than or equal to `r`. In a
doubling metric, the number of well-separated points that fit inside a bounded
ball is bounded by an iterated-cover expression of the form

```text
lambda^O(log(R/r)).
```

Exact exponents depend on strict-versus-nonstrict separation and ball
conventions.

A metric net is both separated and covering: selected centers stay apart while
every point lies near a center. Hierarchical nets repeat this construction over
scales and support approximate distance and nearest-neighbor structures.

A net is not a BFS visited set. Covering allows many original states to map near
one center; exact visited identity requires distinguishing every state unless a
separate quotient/output proof permits merging.

## 7. Finite-instance vacuity

Every finite `n`-point metric has a finite doubling constant at most `n`, hence
doubling dimension at most `log_2 n`. Saying only that one finite puzzle metric
is doubling is therefore vacuous.

Useful evidence reports the actual constant or bounds, the scale range, the
generator/action version, and behavior across a family. A local doubling bound
through radius `R` is not a global one, but can still be a clearly scoped
capacity observation for BFS restricted to that radius.

## 8. Cayley translation and generator dependence

For a fixed symmetric Cayley generating set, left translation is an isometry.
Every radius-`r` ball is a translate of the identity ball, and a cover of the
identity ball translates to every center. Thus the center quantifier reduces to
the identity, while the quantifier over all radii remains.

Changing generators changes the word metric, unit ball, degree, cover numbers,
and numerical doubling dimension. Note 116's bounded generator substitutions
can compare the two metrics multiplicatively, but they do not preserve an exact
doubling constant or BFS frontier profile.

For finite Cayley graphs, finiteness again guarantees only a trivial bound. For
families or infinite groups, a uniform scale-independent doubling constant is a
substantive polynomial-growth condition.

A transitive group action does not by itself make its fixed-generator
Schreier graph vertex-transitive by graph automorphisms. For example, `S_3`
acts transitively on `{1,2,3}`, but the generators `(12)` and `(23)` give the
undirected path `1--2--3` together with loops at its endpoints. Its radius-one
balls have sizes two at an endpoint and three at the middle vertex. Thus a
Schreier action alone does not justify reducing the center quantifier to one
state. A symmetry-based reduction needs a separate graph-automorphism
argument, as discussed in note 120; absent such symmetry or another uniform
all-centers argument, all centers remain relevant.

Directed positive alphabets yield asymmetric reachability distances and are
not metric doubling spaces without a separate symmetrization or directed
definition.

## 9. Evidence from samples

A large separated subset inside one ball can witness that many small balls are
needed and therefore provide a lower bound. A proposed cover gives an upper
bound only for that particular ball and scale after every contained vertex is
verified as covered.

To certify a global doubling constant one needs all centers and scales, or a
theorem reducing those quantifiers. Cayley translation removes centers but not
scales. Sampling balls can discover counterexamples to a claimed constant; it
cannot certify the universal upper bound by itself.

## 10. GPU and multi-GPU boundary

Covering and net computations use BFS distance information but have a different
contract from traversal:

- constructing exact balls or distance rows;
- selecting separated centers;
- proving complete cover of each ball;
- checking all required scales;
- storing hierarchical net links;
- running approximate queries;
- executing original exact frontier/visited BFS.

Multi-GPU cover validation needs globally complete membership evidence. A state
missing from all reported cover balls may be an actual counterexample or merely
an unreported owner partition. Net compression ratio, cover-check throughput,
and exact BFS throughput must be reported separately.

Low doubling dimension can justify a mathematical ball-size envelope. It does
not directly predict coalescing, hash contention, duplicate pressure,
communication locality, or achieved GPU throughput.

## Sources

- A. Gupta, R. Krauthgamer, and J. R. Lee,
  [*Bounded Geometries, Fractals, and Low-Distortion Embeddings*](https://doi.org/10.1109/SFCS.2003.1238226),
  FOCS 2003. Standard doubling constant and dimension formulation.
- S. Har-Peled and M. Mendel,
  [*Fast Construction of Nets in Low-Dimensional Metrics and Their
  Applications*](https://doi.org/10.1137/S0097539704446281), SIAM Journal on
  Computing 35, 2006. Hierarchical nets and algorithmic metric applications.
- R. Krauthgamer and J. R. Lee,
  [*The Intrinsic Dimensionality of Graphs*](https://doi.org/10.1007/s00493-007-2183-y),
  Combinatorica 27, 2007. Polynomial-growth graph metrics and intrinsic
  dimensional structure.

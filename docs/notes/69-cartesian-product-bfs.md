# Cartesian products: additive distance and frontier convolution

This note studies a mathematical BFS construction, not an implementation
strategy. It makes the relation between independent coordinates, metric
distance, frontier width, and shortest-path multiplicity explicit.

## The graph contract

For graphs `G` and `H`, the Cartesian product `G square H` has vertices
`(g,h)`. One edge changes exactly one coordinate:

```text
(g,h) -> (g',h)  when g -> g' in G
(g,h) -> (g,h')  when h -> h' in H.
```

The directed statement works as written. For undirected factors, replace each
edge by its two orientations. This is not the strong product: a simultaneous
change of both coordinates is not one Cartesian-product step.

## Distance is additive

For reachable endpoints,

```text
dist_G square H((s,t),(g,h)) = dist_G(s,g) + dist_H(t,h).
```

Upper bound: take a shortest path in `G` and a shortest path in `H`, then
interleave their edges. This gives a product path of the summed length.

Lower bound: every product edge changes only one coordinate. Any product path
therefore projects to a `G` walk containing `n_G` coordinate changes and an
`H` walk containing `n_H` coordinate changes. Its length is `n_G+n_H`, with
`n_G >= dist_G(s,g)` and `n_H >= dist_H(t,h)`.

This proof also handles infinity: the product endpoint is reachable exactly
when both factor endpoints are reachable.

## BFS spheres are a convolution

Let `a_i` and `b_j` be factor sphere sizes around `s` and `t`. A product vertex
lies at depth `d` exactly when its coordinate depths sum to `d`. Hence

```text
c_d = sum_(i+j=d) a_i b_j.
```

Equivalently, the spherical growth series multiply. This is more informative
than multiplying total state counts: it predicts how independent coordinate
depths redistribute states among BFS frontiers.

The product diameter is the sum of factor diameters when the finite factors are
connected. The widest product frontier need not occur at either factor's peak;
it is governed by the convolution of the whole two profiles.

## Shortest paths acquire shuffle multiplicity

Suppose an endpoint uses factor distances `i` and `j`, and the factors have
`sigma_G` and `sigma_H` shortest paths to their coordinate endpoints. Every
product shortest path must make exactly `i` G-steps and `j` H-steps. Therefore

```text
sigma_product = choose(i+j, i) sigma_G sigma_H.
```

The binomial term counts interleavings. Thus a product can have many shortest
words even when both coordinate paths are individually unique. A state-only
frontier collapses all these histories to one product vertex; a shortest-path
DAG or path counter must preserve their contributions.

## A sharp boundary: strong product

In the strong product, one edge may change the left coordinate, the right
coordinate, or both. A diagonal move can therefore reduce `(1,1)` coordinate
distance from two to one. For connected undirected factors its metric is the
maximum, rather than the sum, of coordinate distances.

Consequently the convolution rule is not a generic rule for every object
called a graph product. The allowed one-step transition is the decisive part
of the BFS contract.

## What this changes in the mental model

1. Independent state coordinates do not imply independent frontier records:
   BFS groups coordinate pairs by a shared sum of depths.
2. Frontier width is a geometric cross-section of the product metric ball.
3. State multiplicity and shortest-word multiplicity separate naturally:
   coordinate interleavings increase the latter without increasing the former.
4. A generator that changes two coordinates at once changes the metric, not
   merely the speed of generating successors.
5. Product structure can explain a frontier profile, but recognizing or using
   that structure is a separate problem and is not assumed here.

## Bounded observation

REF-031 contains a small Rust oracle for `P3 square C4` and the corresponding
strong product. After the initially unavailable Docker service recovered, its
read-only-mounted container passed four tests, formatting, compilation, and
execution. It observed factor spheres `[1,1,1]` and `[1,2,1]`, Cartesian spheres
`[1,3,4,3,1]`, 12 shortest paths to `(2,2)`, and the strong-product diagonal
distance reduction from two to one. These are bounded fixture observations,
not evidence that an application graph factorizes.

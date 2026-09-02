# BFS, geodesic triangles, and graph hyperbolicity

BFS exposes an unweighted graph's shortest-path metric one source at a time.
Gromov hyperbolicity asks a different, global question: how tree-like is that
metric when several shortest paths are compared? This note connects the two
without treating hyperbolicity as a BFS speed predictor.

No hyperbolicity estimator or optimized search implementation is added here.

## 1. Metric and convention

Let `G` be a connected undirected unweighted graph with shortest-path metric
`d`. A geodesic `[x,y]` is any shortest `x-y` path. A geodesic triangle consists
of a chosen geodesic for each of the pairs `x,y`, `y,z`, and `z,x`.

Under the thin-triangle convention, a triangle is `delta`-thin when every point
on each side is at distance at most `delta` from the union of the other two
sides. A geodesic space is `delta`-hyperbolic when all its geodesic triangles
are `delta`-thin.

For a combinatorial graph one must say whether only vertices are measured or
edges are realized as unit intervals. Those conventions can shift constants.
Likewise, the thin-triangle and four-point definitions are quantitatively
equivalent only up to stated constant conversions; the same symbol `delta`
must not be assumed to have exactly the same numeric value under every text's
convention.

## 2. Four-point form

For four vertices `a,b,c,d`, form

```text
S1 = d(a,b) + d(c,d)
S2 = d(a,c) + d(b,d)
S3 = d(a,d) + d(b,c).
```

In a tree, the two largest sums are equal. In the common four-point convention,
the metric is `delta`-hyperbolic when the largest two sums differ by at most
`2*delta` for every quadruple. Thus one quadruple with half-gap `h` proves only
the lower bound `delta(G) >= h`; a sample that finds no larger gap is not an
upper-bound certificate.

The four-point form makes the BFS connection explicit: exact evaluation needs
pair distances. One BFS supplies one distance row, not all six pair distances
for arbitrary quadruples. All-pairs BFS supplies enough distances on an
unweighted finite graph, but exact maximization still ranges over quadruples.

## 3. Trees calibrate the idea

Every tree is `0`-hyperbolic. Its three geodesics meet around one branch point,
and each triangle side lies in the union of the other two sides. Equivalently,
the two largest four-point sums agree.

This does not mean that a BFS tree certifies the original graph as
`0`-hyperbolic. A BFS tree preserves distances only from its root. Deleting
non-tree edges changes other pair distances and can erase the fat geodesic
triangles whose size was being measured.

It also exposes a critical independence:

- a path and a regular tree are both `0`-hyperbolic;
- the path has tiny BFS frontiers;
- the regular tree has exponentially growing spheres.

Therefore even perfect tree-like hyperbolicity does not bound frontier width,
visited size, or memory pressure.

## 4. Counterexamples to tempting performance readings

Long cycles and large grid rectangles have hyperbolicity that grows with their
scale: they contain geodesic configurations that do not stay in a uniformly
thin corridor. Exact constants depend on the selected convention, so only the
growth statement is used here.

These families separate hyperbolicity from other BFS-visible quantities:

- square grids have girth four but unbounded hyperbolicity as their dimensions
  grow, so small girth does not imply uniformly thin triangles;
- longer cycles have both growing girth and growing hyperbolicity, so large
  girth does not imply thin triangles either;
- complete graphs have bounded hyperbolicity but a first BFS frontier of
  `n-1`, so bounded hyperbolicity does not imply a small peak frontier;
- a regular tree has hyperbolicity zero and large frontier growth, while a path
  has the same hyperbolicity and small growth.

Every finite connected graph has some finite hyperbolicity constant, bounded
at a diameter scale. Consequently, calling one finite puzzle graph
"hyperbolic" without reporting `delta`, diameter, convention, or family-scale
behavior says little. The substantive asymptotic property is a uniform bound
over a graph family, or at least a declared normalization such as
`delta/diameter`.

## 5. What thinness says about shortest paths

For any vertices `s,t,x`,

```text
d(s,x) + d(x,t) = d(s,t)
```

holds exactly when `x` lies on some shortest `s-t` path. This is a general
metric fact and needs no hyperbolicity assumption.

Thin triangles add coarse geometry: different geodesic sides of a triangle
remain near the others. In a hyperbolic graph this can support theorems about
geodesic corridors, centers, or tree approximations. It does not by itself say
how many vertices occupy such a corridor, how many alternative shortest paths
exist, or how much off-corridor frontier BFS must discover before stopping.

Therefore low hyperbolicity does not automatically make bidirectional BFS
fast, force a unique meeting state, or justify pruning. Any such use needs a
separate theorem whose hypotheses include the actual stopping or pruning rule.

## 6. Gromov products and BFS evidence

With basepoint `s`, the Gromov product is

```text
(x|y)_s = (d(s,x) + d(s,y) - d(x,y)) / 2.
```

In a rooted tree it equals the length of the common initial segment of the
`s-x` and `s-y` geodesics. In a hyperbolic graph it is a coarse indicator of
how long those routes fellow-travel. A BFS rooted at `s` gives the first two
distances but not `d(x,y)`, so one root profile cannot compute the product
exactly.

## 7. Cayley and Schreier boundaries

A finitely generated group is word-hyperbolic when a Cayley graph for a finite
symmetric generating set is Gromov-hyperbolic. Changing the finite generating
set changes the word metric, numerical `delta`, degree, and BFS sphere profile,
but not the qualitative property of being a hyperbolic group.

This connects to geodesic-language structure: hyperbolic groups admit strong
automatic and regular-language descriptions. The converse is false; for
example, `Z^2` is automatic but its grid-like Cayley graph is not hyperbolic.
Thus an automaton for normal forms is not itself evidence of uniformly thin
geodesic triangles.

For finite puzzle groups, finiteness alone guarantees some finite `delta`.
Meaningful comparison requires a declared generating alphabet and either an
exact constant for the instance or scaling evidence for a family. A Schreier
quotient also changes the metric and must be analyzed directly; ambient group
hyperbolicity should not be transferred to an arbitrary action graph without
additional hypotheses.

Directed positive alphabets pose another boundary. Their reachability distance
is asymmetric, whereas the standard definitions above use a symmetric metric.
Silently adding inverse moves changes the BFS problem.

## 8. GPU and multi-GPU interpretation

Hyperbolicity is semantic geometry, not a throughput metric. Even exact low
`delta` does not establish small frontiers, low duplicate rates, compact
visited state, balanced owner partitions, or low communication.

Exact four-point hyperbolicity can require a large distance workload. Sampling
quadruples can discover witnesses and improve a lower bound, but cannot certify
an exact value or an upper bound unless paired with a proved covering argument.
Any GPU or multi-GPU probe should report:

- graph and generator/action version;
- directed or symmetric metric convention;
- vertex-only or geometric-realization convention;
- exact versus sampled distances and quadruples;
- the best witnessed lower bound separately from any proved upper bound;
- distance-work throughput separately from BFS frontier throughput.

This preserves the useful role of BFS as a distance oracle without converting
a geometric descriptor into an unsupported performance promise.

## Sources

- M. Gromov, [*Hyperbolic Groups*](https://doi.org/10.1007/978-1-4613-9586-7_3),
  in *Essays in Group Theory*, 1987. Foundational group/Cayley viewpoint.
- V. Chepoi, F. Dragan, B. Estellon, M. Habib, and Y. Vaxès,
  [*Diameters, Centers, and Approximating Trees of delta-Hyperbolic Geodesic
  Spaces and Graphs*](https://doi.org/10.1145/1377676.1377687), SoCG 2008.
  States the `2*delta` four-point convention and graph algorithmic setting.
- M. Bridson and A. Haefliger,
  [*Metric Spaces of Non-Positive Curvature*](https://doi.org/10.1007/978-3-662-12494-9),
  Springer, 1999. Standard reference for geodesic hyperbolic spaces and
  convention comparisons.


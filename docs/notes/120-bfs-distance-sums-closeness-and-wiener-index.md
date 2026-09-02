# BFS distance sums: farness, closeness, and the Wiener index

An exhaustive BFS returns more metric information than its last nonempty layer.
The whole layer histogram is a compressed source-distance distribution. From it
one can derive distance sums, averages, and several centrality scores without
retaining every vertex identity. This note studies those semantics and adds no
implementation, optimizer, benchmark, or GPU code.

## 1. The source distance histogram

For a finite connected unweighted graph and source `s`, let

```text
h_i(s) = |F_i(s)|.
```

Then

```text
sum_i h_i(s) = n,
max{i : h_i(s)>0} = ecc(s).
```

The polynomial

```text
P_s(z) = sum_i h_i(s) z^i
```

packages the same histogram. Its value at one is `n`, and its derivative at one
is the source distance sum:

```text
P_s(1)  = n,
P_s'(1) = sum_i i*h_i(s) = sum_v d(s,v).
```

Thus the layer counts alone are sufficient for scalar distance moments even
though they discard vertex identities, parents, path counts, and adjacency
inside or between layers.

## 2. Farness, mean source distance, and closeness

The **farness** or transmission of `s` is

```text
T(s) = sum_v d(s,v) = sum_i i*h_i(s).
```

For `n>1`, the mean distance from `s` to the other vertices is

```text
mu(s) = T(s)/(n-1).
```

A common normalized closeness convention is

```text
C(s) = (n-1)/T(s) = 1/mu(s).
```

Some literature omits the numerator or uses a different normalization. A
reported closeness value is therefore incomplete without its formula and
connectivity convention.

Eccentricity and farness summarize different parts of the histogram. In a
triangle `a-b-c-a` with a leaf `d` attached to `a`, vertices `b` and `d` both
have eccentricity two, but

```text
T(b)=1+1+2=4,
T(d)=1+2+2=5.
```

The same maximum depth does not imply the same average distance or closeness.

## 3. The Wiener index and mean pair distance

For a finite connected undirected graph, the Wiener index is the sum over
unordered vertex pairs:

```text
W(G) = sum_{{u,v}, u!=v} d(u,v).
```

Every unordered pair appears once in `T(u)` and once in `T(v)`, hence

```text
W(G) = (1/2) * sum_u T(u).
```

The mean distance over unordered distinct pairs is

```text
bar_d = W(G) / binom(n,2)
      = 2W(G)/(n(n-1)).
```

One exhaustive BFS gives one row sum `T(s)` of the distance matrix. On a
general graph it does not give every row sum or the Wiener index. Repeating BFS
from all sources is the direct exact baseline; alternative formulas or graph
classes need their own proof.

## 4. Harmonic centrality and disconnected graphs

In a disconnected graph, ordinary distance to an unreachable vertex is
infinite. Farness and reciprocal-farness closeness then require an explicit
component restriction or another convention.

Harmonic centrality avoids adding infinities by defining

```text
H(s) = sum_(v!=s) 1/d(s,v),
```

with `1/infinity=0`. In layer form,

```text
H(s) = sum_(i>=1) h_i(s)/i.
```

It therefore counts only reachable vertices, weighted toward nearby ones. This
does not make it equivalent to closeness on connected graphs: reciprocal of a
sum and sum of reciprocals are different aggregations.

Component-restricted closeness, reachability-scaled closeness, and harmonic
centrality can rank vertices differently. Their names are not interchangeable.

## 5. Directed graphs have two orientations

For a directed graph, outgoing distance `d(s,v)` and incoming distance `d(v,s)`
produce different histograms and centralities. An outgoing BFS from `s` computes
only the first orientation. Incoming scores require BFS in the reversed graph
or equivalent evidence.

If the digraph is not strongly connected, reachability conventions again
matter. An outgoing harmonic score can be finite while incoming reachability is
small or empty. Reporting merely "closeness" hides both orientation and
unreachable-node treatment.

Weighted graphs require a shortest-path method respecting weights; ordinary
FIFO BFS layers do not represent weighted distances.

## 6. Cayley symmetry collapses the all-source sum

In a finite connected undirected Cayley graph, left translation is an isometry.
Every root therefore has the same histogram, farness, eccentricity, closeness,
and harmonic centrality. If `e` is the identity,

```text
T(g) = T(e) for every g,
W(G) = n*T(e)/2,
bar_d = T(e)/(n-1).
```

So one complete identity-rooted BFS determines these global scalar metrics.
This is not an approximation; it is a symmetry proof.

Changing the generating set changes the word metric and therefore every term
in these formulas, even though the group order stays fixed.

## 7. Schreier graphs need a separate automorphism proof

A transitive group action on states does not automatically mean that applying
an arbitrary group element is an automorphism of the fixed generator-labeled
Schreier graph. Left/right action conventions and conjugation of generators can
break the required adjacency preservation.

Consequently one identity-like root in a Schreier graph does not automatically
determine every row of its distance matrix. Root-independent histograms require
an actual vertex-transitive automorphism action on the graph being measured,
not merely that the states form one orbit under a larger group.

If such transitivity is proved, the Cayley scalar reduction applies. If not,
one-root farness remains only a one-root statistic.

## 8. Bounds and accumulator semantics

For a connected graph of order `n` and diameter `D`,

```text
0 <= T(s) <= (n-1)D,
0 <= W(G) <= binom(n,2)D.
```

The lower endpoints need qualification for `n>1`, where every distinct-pair
distance is at least one. These coarse bounds help reason about accumulator
width, but they do not predict actual values.

Exact integer sums should remain integers until a declared final conversion.
Floating division can lose reproducibility, and reciprocal scores need a
specified precision and normalization. Comparing closeness rankings can often
compare exact farness integers in reverse order instead.

## 9. What bounded BFS can and cannot aggregate

A completed radius-`R` BFS gives the exact truncated histogram through `R` and
the exact partial sum

```text
T_<=R(s) = sum_(i=0)^R i*h_i(s).
```

If the component is not exhausted, this is a lower bound on full farness, not
the full answer. The observed vertex count is also incomplete. Treating every
unseen state as distance `R+1` can give a lower bound only when the total `n` is
known and all unseen states are known reachable; otherwise even that completion
assumption is unjustified.

Likewise, a sampled set of sources estimates global average distance but does
not certify the exact Wiener index without an exhaustive or structural
argument.

## 10. GPU and multi-GPU interpretation

Once an exact BFS produces layer cardinalities, accumulating `i*h_i` is a small
reduction compared with discovering the layers. The difficult contract remains
exact traversal and exhaustion.

For a general graph, exact all-source farness means a family of source-indexed
distance problems. Batching them may reuse graph storage or expose parallelism,
but it does not turn them into one BFS. For a Cayley graph, the symmetry theorem
can eliminate the semantic need for all-source repetition.

Report separately:

- traversal scope and exhaustion evidence;
- per-layer counts and integer distance sum;
- source count or symmetry theorem used;
- accumulator width and floating conversion;
- traversal, reduction, communication, and validation time;
- exact versus sampled/estimated global metrics.

Multi-GPU reductions of layer counts must use one consistent graph/source epoch.
A correct scalar reduction cannot repair missing states or duplicated ownership.

## 11. Research checklist

For a distance-aggregate result, record:

1. connected, component-restricted, or unreachable-aware semantics;
2. undirected, outgoing-directed, or incoming-directed distance;
3. unweighted BFS versus weighted shortest paths;
4. exact formula and normalization for closeness or harmonic centrality;
5. complete versus radius-truncated histogram;
6. one-source, all-source, sampled-source, or symmetry-reduced scope;
7. integer accumulator and floating-output policy;
8. whether identities, parents, multiplicities, or only scalar moments survive.

## Sources

- L. C. Freeman,
  [*Centrality in Social Networks: Conceptual Clarification*](https://doi.org/10.1016/0378-8733%2878%2990021-7),
  Social Networks 1(3), 1978/79. Establishes the classical distance-based
  closeness perspective and its normalization context.
- P. Boldi and S. Vigna,
  [*Axioms for Centrality*](https://doi.org/10.1080/15427951.2013.865686),
  Internet Mathematics 10(3--4), 2014. Develops harmonic centrality for general
  directed graphs and unreachable-node semantics.
- A. Abiad et al.,
  [*On the Wiener Index, Distance Cospectrality and Transmission-Regular Graphs*](https://doi.org/10.1016/j.dam.2017.07.010),
  Discrete Applied Mathematics 230, 2017. Uses distance-matrix row sums and
  transmission regularity.
- Notes 21, 35, 42, 57, 72, 78, 93, and 119 supply this repository's
  eccentricity, growth-profile, bounded-unknown, output-finalization,
  peripheral, landmark, generator-metric, and capacity distinctions.

## Takeaway

An exhaustive BFS layer histogram is a source-distance distribution, not just
a frontier log. Its first weighted moment is farness; related sums give mean
distance and harmonic centrality. Global pair distance normally needs all
sources, but finite Cayley symmetry reduces it exactly to one identity-rooted
BFS. Every such shortcut depends on graph automorphisms, exhaustion, and an
explicit connectivity and normalization contract.

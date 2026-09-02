# BFS, bipartite witnesses, and the shortest odd cycle

A BFS parity conflict is enough to prove that an undirected graph is not
bipartite.  It is not, from one arbitrary root, enough to find the globally
shortest odd cycle.  This note separates those two contracts.

This is a semantic study, not an implementation design.

## Setting

Unless stated otherwise, let `G=(V,E)` be a finite simple undirected unweighted
graph.  A BFS rooted at `s` records exact distances `d_s(v)` and one tree parent
for every reached non-root vertex.

For every edge `{x,y}`,

```text
|d_s(x) - d_s(y)| <= 1.
```

Consequently, adjacent endpoints have the same distance parity if and only if
they are in the same BFS layer.  In this setting, "same-parity conflict" and
"equal-depth edge" are not two different tests.

That equivalence fails as phrased for weighted distances, directed reachability,
or an arbitrary non-distance tree labeling.  Those are different problems.

### Directed counterexample: equal-depth arcs are not a bipartite test

Take the directed cycle

```text
s -> a -> b -> s.
```

Directed BFS from `s` assigns depths `0,1,2`. No arc has equal-depth
endpoints, yet forgetting directions leaves a triangle, which is not
bipartite. The back arc `b->s` joins two vertices of the same depth parity
while their depths differ by two.

The undirected proof needs both directions of the edge-distance inequality:
an undirected edge forces endpoint depths to differ by at most one. A directed
arc only gives `d(v) <= d(u)+1` in its forward direction; it may point back
across arbitrarily many layers. Therefore equal-depth-arc absence is not a
parity certificate for the underlying undirected graph.

## Extracting one simple odd-cycle witness

Suppose edge `{x,y}` has

```text
d_s(x) = d_s(y) = j.
```

Let `z` be the lowest common ancestor of `x` and `y` in the chosen BFS tree,
at depth `i`.  Delete the common root-to-`z` prefix from the two tree paths.
The remaining branches are internally vertex-disjoint.  Together with
`{x,y}` they form a simple cycle of length

```text
(j-i) + 1 + (j-i) = 2(j-i)+1.
```

Thus the edge, tree parents, and exact depth labels constitute a replayable
non-bipartiteness certificate.  A boolean conflict flag alone proves less: it
does not by itself identify a cycle that can be checked later.

If a complete BFS of every connected component finds no equal-depth edge, all
edges cross even/odd distance classes and those classes form a bipartition.

## One root need not expose the shortest odd cycle

Consider the six-vertex graph

```text
s--p--a
|     / \
q----b---c
```

where the intended edge list is

```text
{s,p}, {p,a}, {s,q}, {q,b}, {a,b}, {a,c}, {b,c}.
```

From root `s`, vertices `a` and `b` both have depth two, while `c` has depth
three.  The only equal-depth conflict among the triangle edges is `{a,b}`.
With parents `a<-p<-s` and `b<-q<-s`, its extracted tree cycle is

```text
s-p-a-b-q-s
```

of length five.  Yet `a-b-c-a` is a triangle, so the graph's odd girth is
three.

The BFS distances are correct and the length-five witness is valid.  The false
step would be upgrading "some odd cycle" to "the shortest odd cycle."  The
tree paths are individually shortest from `s`, but they need not be the two
arcs of the shortest odd cycle.

## Why all roots are sufficient

Let `C` be a globally shortest odd cycle of length

```text
g_odd = 2k+1,
```

and choose a root `s` on `C`.  Let `{x,y}` be the edge opposite `s`: the two
arcs on `C` from `s` to `x` and `y` each have length `k` before the closing edge
`{x,y}`.

We need to rule out a shorter path through the rest of the graph.  Suppose,
for example, there were an `s`-to-`x` path `P` of length at most `k-1`.  The
two `x`-to-`s` arcs of `C` have lengths `k` and `k+1`.  Combining `P` with
each arc gives two closed walks of opposite parity, and the odd one has length
at most `2k`.  Every odd closed walk contains a simple odd cycle no longer than
it: repeatedly remove repeated-vertex subwalks, retaining an odd closed part.
That would contradict the minimality of `C`.

Therefore

```text
d_s(x) = d_s(y) = k.
```

The opposite edge is an equal-depth conflict in the BFS rooted at `s`.  The
cycle extracted through the BFS-tree LCA has odd length at most `2k+1`.
Because no odd cycle is shorter than `C`, its length is exactly `2k+1`.

Hence:

> Running exact BFS from every vertex and minimizing the length of every
> LCA-extracted equal-depth witness returns the odd girth.

The conventional explicit-graph bound is `O(|V||E|)` time for the BFS runs,
plus witness bookkeeping within the same asymptotic budget.  This is an
all-roots statement; one arbitrary root is only a recognition/witness run.

## Detection, witness, and optimization are different outputs

The distinctions are:

| Requested output | Sufficient BFS work in an undirected component |
|---|---|
| bipartite yes/no | one complete rooted BFS plus every-edge parity checks |
| one odd-cycle witness | one complete rooted BFS with parents and one conflict |
| shortest witness generated by this root/tree | minimize its extracted conflicts |
| globally shortest odd cycle | all-root BFS minimization, or another proved algorithm |

Stopping at the first conflict is valid for recognition, but the returned
cycle depends on root, parent tie-breaking, edge order, and distributed arrival
order.  Even processing every conflict from one root does not repair the
six-vertex counterexample.

## Parent tie-breaking changes witnesses

Distances and bipartiteness do not depend on which shortest parent is stored.
The LCA and extracted cycle length can.  Thus two exact parallel BFS runs may
emit different valid odd cycles while agreeing on all depths.

For deterministic witness output, the contract must specify parent reduction
or minimize directly over a representation independent of incidental arrival
order.  For odd-girth length, all candidates must be reduced globally before
termination; an arbitrary first conflict is insufficient.

## Cayley interpretation

In an inverse-closed Cayley graph, an equal-depth generator edge closes an odd
word representing the identity after the shared BFS-tree prefix is cancelled.
It proves failure of a generator-parity homomorphism to `Z_2`.

From one identity-rooted BFS, the resulting relation is not automatically the
shortest odd relation.  However, a genuine Cayley graph is vertex-transitive:
translating any shortest odd cycle places one of its vertices at the identity.
Therefore an exhaustive identity-rooted BFS has the same root-position
advantage used in the all-roots proof and can recover odd girth by minimizing
all equal-depth witnesses.

This shortcut requires actual graph vertex transitivity under the precise move
set.  A Schreier or puzzle graph is not granted it merely because a group acts
transitively on states; note 21 records that boundary.

Labels also matter.  Parallel generator labels or an identity generator may
create length-two labeled returns or loops even when the underlying simple
graph convention suppresses them.  "Odd girth" must name whether it concerns
the simple unlabeled graph, a multigraph, or reduced generator words.

## GPU and multi-GPU meaning

The mathematical predicate is small, but its proof obligations are global:

- depth labels must be final and exact;
- every relevant undirected edge occurrence must be inspected or otherwise
  covered;
- equal-depth conflicts need a global minimum reduction for odd girth;
- parent chains and ownership metadata must remain replayable;
- termination must include in-flight conflicts and reductions;
- overflow may not silently discard the best witness.

The all-roots `O(|V||E|)` method is a correctness baseline, not a proposed GPU
strategy.  On a vertex-transitive Cayley graph, the one-root reduction is a
mathematical symmetry reduction, not a hardware optimization discovered here.

## Rejected shortcuts

- **The first same-layer edge gives the shortest odd cycle.** It gives a valid
  witness only.
- **Minimizing all conflicts from one root gives odd girth.** The six-vertex
  graph returns five although it contains a triangle.
- **Same parity requires checking more edges than equal depth.** Not for edges
  under exact unweighted undirected BFS distances; adjacent depths differ by at
  most one.
- **A non-bipartite flag is a replayable cycle certificate.** Parent/depth/edge
  evidence is still required.
- **One identity BFS works for every group-action graph.** It needs genuine
  vertex transitivity of the represented graph.

## Sources

- Purdue CS 580 lecture notes,
  [BFS layers and bipartite graphs](https://www.cs.purdue.edu/homes/jblocki/courses/580_Spring19/Lectures/CS580-Lecture3Graphs-single.pdf),
  give the equal-level-edge/LCA odd-cycle construction.
- Bjorklund, Kaski, and Williams,
  [The Shortest Even Cycle Problem Is Tractable](https://epubs.siam.org/doi/full/10.1137/22M1538260),
  summarize the classical `O(nm)` iterated-BFS bound for shortest odd cycles
  and the shortest-closed-odd-walk argument.
- Note 21 supplies the existing bipartiteness, eccentricity, and Cayley versus
  Schreier certificate boundaries; note 27 supplies girth conventions.

The `autolean` expert channel was asked to review the theorem and counterexample
but returned `fetch failed`; no claim above relies on an expert response.

## Current conclusion

One BFS converts a parity conflict into a concrete odd-cycle certificate.
It does not generally optimize that certificate.  Odd girth becomes exact by
placing a root on a shortest odd cycle—achieved generically by all-root BFS, or
by a proved vertex-transitivity reduction for a Cayley graph—and minimizing all
completed equal-depth witnesses.

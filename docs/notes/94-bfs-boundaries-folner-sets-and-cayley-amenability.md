# BFS boundaries, Følner sets, and Cayley amenability

For an undirected Cayley graph, the next exact BFS frontier is the external
vertex boundary of the completed ball. Amenability asks whether the graph has
some finite sets whose boundary is negligible relative to their volume. These
ideas touch the same boundary operator but quantify over different families of
sets.

This note makes that quantifier difference explicit and records what it can and
cannot say about BFS memory and distributed work. It implements nothing.

## 1. External boundary and BFS frontier

Let `Gamma=Cay(G,S)` for a finite symmetric generator set, and for finite
`A subset G` define the external vertex boundary

```text
partial A = {x notin A : x is adjacent to some a in A}.
```

For the exact root ball `B_r` and sphere `S_(r+1)`:

```text
partial B_r = S_(r+1).
```

Therefore the BFS-specific boundary-to-volume ratio is

```text
|S_(r+1)| / |B_r|.
```

This is distinct from the edge boundary, which counts crossing generator-edge
occurrences and may exceed the number of distinct outside endpoints.

## 2. Følner condition and amenability

An infinite finitely generated group is amenable exactly when, for every
`epsilon>0`, there exists a nonempty finite set `F` with

```text
|partial F| / |F| < epsilon.
```

A sequence of such sets with ratio tending to zero is a Følner sequence. The
property is independent of the chosen finite generating set, although the
actual sets, boundary counts, and finite-radius profile depend on it.

The quantifier is existential over finite subsets. It does **not** say that
metric balls `B_r` form a Følner sequence.

## 3. Amenable does not mean thin BFS frontier

From amenability alone one may conclude that some finite shapes have small
relative boundary. Ordinary BFS is constrained to metric balls around its
source. Those balls need not be the Følner shapes supplied by the definition.

Thus the implication

```text
amenable group => |S_(r+1)| / |B_r| -> 0 for all BFS radii
```

is not licensed by the Følner criterion alone. It requires additional evidence
about the chosen balls or growth regularity.

For `Z^d` with standard generators, familiar lattice balls do have boundary of
lower order than volume. This is an example, not the definition of amenability.

## 4. Positive isoperimetry forces wide boundaries

Define the external-vertex isoperimetric constant

```text
h = inf over finite nonempty A of |partial A| / |A|.
```

If `h>0`, every finite BFS ball satisfies

```text
|S_(r+1)| >= h |B_r|
```

and therefore

```text
|B_(r+1)| = |B_r| + |S_(r+1)| >= (1+h)|B_r|.
```

So positive vertex expansion forces at least exponential ball growth and a next
frontier proportional to accumulated visited volume. This is a semantic memory
pressure statement for an infinite bounded-degree graph, not a kernel timing
statement.

## 5. Exponential growth does not imply nonamenability

The converse fails. Amenable groups of exponential growth exist, including
lamplighter-type groups. Exponential ball volume can coexist with Følner sets
whose shapes are not the ordinary metric balls under discussion.

Hence:

```text
positive isoperimetric constant => exponential growth,
exponential growth =/=> positive isoperimetric constant.
```

Subexponential growth does imply amenability, but again does not by itself
identify every BFS ball as an optimal or monotone Følner witness.

## 6. Calibration: line and free group

For the integer line with generators `{+/-1}`:

```text
|B_r| = 2r+1,
|S_(r+1)| = 2,
|S_(r+1)|/|B_r| -> 0.
```

For the free group of rank `k>=2` with its standard `q=2k` generators, the
Cayley graph is a `q`-regular tree:

```text
|S_r| = q(q-1)^(r-1),  r>=1.
```

Its next-sphere/ball ratio stays bounded away from zero and in fact tends to
`q-2`. The accumulated visited set never becomes large relative to the next
wave. These examples expose the memory contrast between Følner-like and
uniformly expanding ball geometry.

## 7. Finite Cayley graphs saturate

Every finite group is amenable in the elementary sense that choosing the whole
group gives empty boundary. Consequently, group amenability alone says almost
nothing about a finite puzzle instance's middle BFS layers.

A finite expander can have large boundary for every set up to half the graph,
then its BFS frontier must eventually collapse to zero when the whole component
is visited. The relevant finite evidence is a scale-restricted expansion or
boundary profile, not the eventual whole-graph ratio.

Across a family of finite Cayley or Schreier graphs, uniform expansion constants
can create persistent pre-saturation memory pressure even though every member
is individually finite and amenable.

## 8. Cayley versus Schreier amenability

Amenability of a group and amenability of one group action are related but not
identical statements. An amenable group has amenable actions, while a
nonamenable group can still have a particular amenable action or finite orbit.

Therefore a Cayley-graph conclusion cannot simply be copied to a puzzle's
Schreier graph. The stabilizer and orbit can change boundary geometry
substantially. Boundary ratios must be measured or proved in the declared state
graph and generator alphabet.

## 9. GPU and multi-GPU interpretation

For exact level-synchronous BFS:

- distinct next-frontier storage follows the external vertex boundary;
- generated candidate work is closer to outgoing edge occurrences;
- duplicate convergence separates edge boundary from vertex boundary;
- owner-to-owner traffic follows partition cuts, not the graph's Følner ratio;
- a small semantic boundary may still hash almost entirely to remote owners;
- a large semantic boundary may remain mostly owner-local under a structured
  partition.

Amenability is therefore neither a low-communication theorem nor a GPU-speed
prediction. At most, proved boundary profiles constrain one component of the
workload. Representation, generator cost, partition, saturation, and hardware
remain separate.

## 10. Evidence checklist

1. External vertex boundary, internal boundary, or crossing edge occurrences.
2. Arbitrary Følner sets or root-centered BFS balls.
3. Infinite group, one finite graph, or a uniform finite family.
4. Cayley graph or a particular Schreier action.
5. Exact generator set and directed/undirected convention.
6. Boundary profile before finite saturation.
7. Distinct frontier states versus generated transitions.
8. Semantic graph boundary versus distributed owner cut.

## Sources

- T. Ceccherini-Silberstein, R. I. Grigorchuk, and P. de la Harpe,
  [*Amenability and Paradoxical Decompositions for Pseudogroups and for
  Discrete Metric Spaces*](https://people.tamu.edu/~grigorch/publications/zCGH.pdf),
  Proc. Steklov Inst. Math. 224 (1999), 57-97. Følner, isoperimetric, action,
  and metric-space characterizations.
- G. N. Arzhantseva, V. S. Guba, M. Lustig, and J.-P. Préaux,
  [*Testing Cayley Graph Densities*](https://doi.org/10.5802/ambp.249),
  Annales Mathématiques Blaise Pascal 15(2) (2008), 233-286. Cayley boundary,
  isoperimetric constant, and Følner-family formulation.
- Notes 10, 32, 35, 36, 46, 51, 71, and 93 provide frontier-boundary,
  distance-regularity, growth, representation, expansion, ownership,
  arbitrary-profile, and generator-change context.

## Takeaway

The BFS frontier is the boundary of one very special set: a metric ball.
Amenability guarantees the existence of some sets with vanishing relative
boundary, not automatically those balls. Positive isoperimetry does force every
ball frontier to remain proportional to visited volume, but neither amenability
nor expansion alone predicts duplicates, owner traffic, or elapsed GPU time.

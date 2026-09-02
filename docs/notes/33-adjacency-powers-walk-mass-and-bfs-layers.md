# Adjacency powers: walk mass is not a BFS frontier

Matrix powers, random walks, and BFS all propagate information through edges,
but they retain different information.  Arithmetic matrix multiplication counts
walk multiplicity; Boolean multiplication records exact-length existence; BFS
keeps only states whose first reachable length is the current depth.

This note aligns those objects for explicit and Cayley graphs.  It does not
propose a matrix or GPU implementation.

## Arithmetic powers count walks

Let `A` be the adjacency matrix of a finite graph.  Over ordinary nonnegative
integer arithmetic,

```text
(A^d)[u,v] = number of length-d walks from u to v.
```

The proof is the matrix-product recurrence: every length-`d+1` walk to `v`
chooses its penultimate vertex `x`, a length-`d` walk to `x`, and edge `x->v`.

A walk may repeat vertices and edges.  Thus `(A^d)[u,v]` does not generally
count simple paths and does not say that `d` is the distance.

However, at the first positive power,

```text
dist(u,v) = min {d >= 0 | (A^d)[u,v] > 0},
```

and every length-`dist(u,v)` walk is necessarily a shortest simple path.
Therefore, for a zero-one adjacency matrix, the first positive coefficient also
equals the number of shortest vertex paths.  With parallel-edge multiplicities
it instead distinguishes the corresponding edge-labeled choices.

## Three distinct supports

Fix source `s`.  Define

```text
W_d = {v | (A^d)[s,v] > 0}                 exact-d walk support
B_d = union_(0<=j<=d) W_j                  reachable within d
F_d = W_d minus union_(0<=j<d) W_j         first reached at d.
```

Then `B_d` is the BFS ball and `F_d` is the BFS sphere.  These sets need not
equal `W_d`.

On the three-vertex path `s-a-b`,

```text
W_1 = {a}
W_2 = {s,b}
B_2 = {s,a,b}
F_2 = {b}.
```

The return walk `s-a-s` puts the already visited source in `W_2`, while `a` is
absent because an exact two-step walk from `s` to `a` does not exist.  Thus an
exact matrix power is neither a cumulative ball nor an exact BFS frontier.

On a triangle, a depth-one neighbor also belongs to `W_2` through the third
vertex.  Exact-length support can contain earlier BFS layers as soon as cycles
and parity permit.

## Boolean powers remove multiplicity, not history

Over the Boolean semiring, multiplication is `AND` and addition is `OR`.
Boolean `A^d` records whether at least one length-`d` walk exists.  It collapses
multiple walks to one bit but still represents `W_d`, not `F_d`.

The cumulative Boolean support

```text
I OR A OR A^2 OR ... OR A^d
```

represents `B_d`.  BFS obtains the delta by masking out the previous ball:

```text
F_d = B_d minus B_(d-1).
```

The complemented visited mask is therefore not a low-level convenience.  It
changes exact-length reachability into first-discovery distance semantics.

## Bipartite periodicity

In a bipartite graph, every walk from `s` has endpoint parity fixed by its
length.  Even for very large `d`, `W_d` may omit half the connected component.
This is periodicity of the ordinary random walk, not failure of reachability.

Adding a positive holding probability at every vertex (a lazy transition
matrix) removes the parity period, but changes the walk alphabet: an exact
length now permits waiting. Adding unit-cost self-loops preserves ordinary
shortest hop distances and BFS frontiers: deleting waits from any walk cannot
increase its length, and every original path remains available. It does change
exact-length walk counts/support and random-walk probabilities. For example,
adding `s -> s` to `s -> a` leaves `dist(s,a)=1`, but permits a length-two walk
from `s` to `a`.

## Cayley words as convolution mass

Let a finite group `G` have an everywhere-applicable generator multiset `S` of
size `q`, under a fixed action convention.  Expanding

```text
(sum_(s in S) s)^d
```

in the group algebra collects every length-`d` generator word by its resulting
group element.  Equivalently, if `mu` counts one generator step, then

```text
mu^(*d)(g) = number of length-d words evaluating to g.
```

The total word mass is always

```text
sum_g mu^(*d)(g) = q^d.
```

The support contains elements reachable by a word of exactly length `d`.
The BFS sphere contains only elements whose shortest word length is `d`:

```text
F_d = support(mu^(*d)) minus union_(j<d) support(mu^(*j)).
```

Group relations concentrate many words on the same element.  Inverse returns
feed mass back into earlier spheres.  Same-depth alternative words contribute
to shortest-path multiplicity.  None creates another unique BFS state after
exact visited merging.

This separates three growth functions:

- `q^d`: raw length-`d` word count;
- `|support(mu^(*d))|`: exact-length reachable elements;
- `|F_d|`: elements of minimal word length exactly `d`.

They coincide only in a tree-like regime with no relevant convergence or
return, and even an undirected tree has inverse-return walks in `W_d` once
backtracking is allowed.

## Random-walk normalization

For a `q`-regular graph,

```text
P = A/q
```

is the simple random-walk transition matrix, and

```text
(P^d)[u,v] = (A^d)[u,v] / q^d.
```

Thus random-walk probability is normalized walk mass.  BFS instead applies an
existence threshold and a historical visited mask.  A state with tiny positive
probability and a state with huge probability are both one reached identity.

Rapid mixing says that walk mass approaches its stationary distribution under
the required connectivity and aperiodicity assumptions.  It does not say that
all vertices have the same minimum distance, that the current BFS frontier is
large, or that duplicate work is small. A lazy walk changes exact-length walk
semantics and mixing behavior even though its added unit-cost self-loops do not
change the shortest hop distances.

## What the spectrum sees

For a finite undirected graph with orthonormal eigenvectors,

```text
(A^d)[u,v] = sum_j lambda_j^d phi_j(u) phi_j(v).
```

Eigenvalues and eigenvectors therefore summarize walk propagation.  Also,

```text
trace(A^d)
```

counts closed length-`d` walks over all starting vertices.

Spectral gaps can bound mixing, expansion, and aggregate edge distribution.
Those are valuable geometric constraints, but they are not an exact visited
set.  Recovering `F_d` still requires the first-positive/support history for
the specified source.  Global spectral moments can count many closed walks
without identifying which individual states first appear at which depth.

The safe interpretation is:

```text
spectrum -> constraints and aggregate walk behavior
         -/-> exact source-rooted BFS layers by itself.
```

## Shortest paths versus all walks

For `v in F_d`, `(A^d)[s,v]` counts shortest paths because no shorter route
exists and a repeated-vertex walk could be shortened.  For `v in B_(d-1)`, the
same coefficient counts longer revisiting walks as well.

Hence an arithmetic frontier multiplication without a distance mask mixes two
kinds of contribution:

- first-arrival contributions that belong to the shortest-path DAG;
- later walks that should not alter BFS distance or shortest-path count.

The depth equality condition in note 11 is precisely what selects the first
kind.

## Work accounting for BFS

For a complete traversal of a finite `q`-generator Cayley component, BFS
attempts `q|R|` labeled transition occurrences: each reached state is expanded
once.  This is very different from enumerating all word prefixes through depth
`D`, whose count is

```text
1 + q + q^2 + ... + q^D.
```

Computing or reasoning about `A^d` does not imply that an exact graph BFS must
materialize all `q^d` walks.  Conversely, a naive word-tree search really can
pay that multiplicity because it has omitted state merging.

Useful measurements therefore keep separate:

- generator/edge occurrences attempted by frontier expansion;
- unique exact-length support, when measured;
- unique first-discovered states;
- same-batch convergence and earlier-visited returns;
- shortest-parent multiplicity;
- optional arithmetic walk mass.

Calling all of these "paths" makes performance results uninterpretable.

## GPU and GraphBLAS semantics

Sparse matrix-vector notation can express several different computations:

- arithmetic semiring: weighted sums or walk counts;
- Boolean semiring: reachability support;
- min-plus: distance relaxation;
- Boolean product plus complemented visited mask: BFS next frontier.

Similar-looking kernels do not establish equivalent outputs.  Overflow also
matters for arithmetic walk counts: `q^d` grows exponentially even after the
finite state space saturates.  Boolean reachability avoids numeric overflow but
still needs exact masking and complete level semantics.

On multiple GPUs, partial arithmetic sums, Boolean ORs, and exact identity
deduplication use different reduction laws and communication payloads.  This is
a semantic distinction to declare before measuring, not a recommendation for a
particular implementation.

## Rejected shortcuts

- **`A^d` is the BFS frontier at depth `d`.** It represents exact-length walk
  counts or support, including returns to old layers.
- **Boolean powers automatically perform visited deduplication.** They remove
  multiplicity within one power, not states reached at earlier powers.
- **The number of Cayley words is the number of reached group elements.** Many
  words can evaluate to one element.
- **Fast random-walk mixing means BFS has finished.** Probability convergence
  and exhaustive first discovery are different contracts.
- **A good spectral gap determines exact frontier sizes.** It supplies bounds
  on aggregate geometry, not the source-specific first-positive mask.
- **Lazy random walks preserve exact-length walk support.** Waiting can change
  support at a fixed length; this must not be confused with shortest hop
  distance, which adding unit-cost self-loops does preserve.

## Sources

- MIT OpenCourseWare,
  [Networks Lecture 2](https://ocw.mit.edu/courses/14-15-networks-spring-2022/mit14_15s22_lec2.pdf),
  derives the interpretation of adjacency powers as walk counts.
- Daniel Spielman,
  [Spectral Graph Theory, Chapter 16](https://www.cs.yale.edu/homes/spielman/PAPERS/SGTChapter.pdf),
  develops adjacency/diffusion matrices and random-walk spectral behavior.
- Notes 11, 25, 26, 27, and 29 supply shortest-path counts, Boolean frontier
  masks, exact-`k` powers, Cayley relations, and BFS work accounting.

## Current conclusion

Adjacency powers preserve walk mass; BFS preserves first discovery.  In a
Cayley graph, `q^d` words are redistributed by relations over group elements,
while the BFS frontier retains only elements whose first nonzero coefficient
occurs at `d`.  Boolean algebra, arithmetic algebra, and spectral analysis each
illuminate part of this picture, but only the visited-masked first-support
semantics is ordinary BFS.

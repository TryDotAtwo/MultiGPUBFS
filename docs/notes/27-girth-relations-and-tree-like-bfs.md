# Girth, relations, and when a BFS ball stops looking like a tree

A regular graph has the same degree everywhere, but its early BFS growth is
tree-like only while distinct non-backtracking paths cannot close a short cycle.
Girth makes that statement precise.

This note connects four objects:

- the non-backtracking path tree;
- the BFS state graph and its distance spheres;
- cycles and girth;
- generator relations in Cayley and Schreier graphs.

The purpose is to understand frontier geometry and duplicate onset, not to
design a duplicate-removal optimization.

## Graph contract

Unless stated otherwise, let `G` be a simple undirected graph and let

```text
g = girth(G)
```

be the length of its shortest simple cycle, with `g=infinity` for a forest.
Loops, parallel labeled edges, directed cycles, and immediate reversal walks
need separate conventions; they are discussed below.

For a root `s`, write `B_r(s)` for the vertices at distance at most `r` and
`F_r(s)` for the exact distance-`r` sphere.

## Unique shortest paths: the `2r < g` threshold

If

```text
2r < g,
```

then every vertex in `B_r(s)` has a unique shortest path from `s`.

Suppose instead that a vertex `v` at distance `ell<=r` had two distinct
shortest paths.  Remove their common prefix and any common suffix.  The two
remaining internally disjoint path segments form a simple cycle of length at
most `2ell<=2r`, contradicting the girth.

The strict inequality is sharp.  In the even cycle `C_(2r)`, the vertex
opposite `s` has two shortest paths of length `r`, and `2r=g`.

This theorem concerns shortest-path uniqueness.  It does not yet say that the
entire induced ball contains only tree edges.

## An induced ball is a tree: the `2r+1 < g` threshold

If

```text
2r+1 < g,
```

then the subgraph induced by `B_r(s)` is exactly its BFS tree.

The previous condition gives unique parent paths.  If there were any additional
edge `u--v` inside the ball, the two root paths plus that edge would contain a
cycle of length at most

```text
dist(s,u) + dist(s,v) + 1 <= 2r+1.
```

Again the strict inequality is sharp.  In `C_(2r+1)`, every vertex lies in
`B_r(s)` and the final edge between the two depth-`r` sides closes the whole
cycle.  Shortest paths from `s` are still unique, but the induced ball is not a
tree.

This explains an easy off-by-one trap:

```text
unique root geodesics  !=  no extra edge inside the ball.
```

An edge between two depth-`r` vertices can create an odd cycle without giving
either endpoint a second shortest root path.

## Non-backtracking words agree with BFS only before a cycle can close

In a `q`-regular tree, the non-backtracking word tree has sphere sizes

```text
w_0 = 1
w_i = q (q-1)^(i-1),  i>=1.
```

The factor is `q-1` after the first step because immediate reversal returns
along the edge just used.

In a `q`-regular graph of girth `g`, every non-backtracking path of length `i`
with `2i<g` is geodesic, and two such paths from the same root cannot share an
endpoint.  Otherwise that path plus a shorter path, or the two distinct paths,
would contain a cycle shorter than `g`.

Therefore the BFS spheres have the regular-tree counts throughout that safe
radius.  At the boundary, different failures can first appear:

- two paths can converge to one next state;
- a path can return to an earlier sphere by a nontrivial cycle;
- an edge can join vertices of one sphere;
- a boundary path can stop being geodesic.

Girth controls the earliest possible onset, not the number or spatial locality
of all later collisions.

## Immediate reversal is not a girth-two cycle

Even in an infinite undirected tree, expanding every edge from `F_d` generates
one transition per vertex back toward `F_(d-1)`.  The two-step walk

```text
u -> v -> u
```

traverses one edge twice; it is not a simple cycle in the usual simple-graph
girth definition.

Thus there are at least three separate duplicate phenomena:

1. **inverse backtracking:** a generated occurrence returns to the parent;
2. **relation convergence:** different non-backtracking paths reach one state;
3. **visited closure:** a non-backtracking path reaches some earlier ball state.

A graph can have infinite girth and still have inverse-backtracking visited
hits.  Saying "duplicates begin at the girth" is false unless immediate
reversal has already been removed from the path language and the duplicate
category is specified.

## Moore lower bounds are BFS tree counts

Let a finite simple `q`-regular graph have girth `g`, with `q>=2`.

### Odd girth `g=2r+1`

Grow BFS from one vertex through radius `r`.  All root paths at those depths are
unique, so

```text
|V| >= 1 + q sum_(i=0)^(r-1) (q-1)^i.
```

Edges may close cycles at the outer boundary, but the counted vertices cannot
have merged earlier.

### Even girth `g=2r`

Root the count at the two endpoints of an edge and grow away from that central
edge to depth `r-1` on both sides.  Any collision within or across the two trees
would create a cycle shorter than `2r`.  Hence

```text
|V| >= 2 sum_(i=0)^(r-1) (q-1)^i.
```

These are the Moore bounds.  They are lower bounds on the number of vertices
required to sustain a given degree and girth, derived from collision-free BFS
layers.  Equality is rare and highly structured.  The bound does not predict
the full sphere sequence after the first allowed cycle closure.

## Cayley cycles and reduced identity words

Let `S` be a finite symmetric collection of distinct nonidentity group
elements, with a fixed inverse convention, and form the simple undirected
Cayley graph.  A generator word labels a walk from the identity.  It closes
exactly when its product is the identity.

Under these clean assumptions, graph girth equals the length of the shortest
nonempty cyclically reduced generator word representing the identity:

- a simple cycle supplies such an identity word;
- a shortest reduced identity word traces a closed non-backtracking walk;
- if that walk repeated an internal vertex, it would contain a shorter closed
  subwalk, contradicting minimality;
- cyclic reduction removes a possible inverse pair across the word boundary.

This is the precise algebraic reason that short relations make the Cayley BFS
depart early from its free-group tree.

If two distinct reduced words `u` and `v` reach the same group element, then

```text
u * inverse(v) = identity
```

after applying the declared left/right action and reducing cancellations.  A
candidate collision therefore supplies an identity relation whose reduced
length is no greater than `|u|+|v|`.  Common prefixes and suffixes can make the
actual cycle substantially shorter.

## A presentation list is not the girth

For a presentation

```text
G = < S | R >,
```

the shortest word explicitly written in `R` need not equal Cayley girth.

- A listed relator may freely reduce to the empty word and merely encode
  immediate reversal.
- Its closed walk may repeat vertices and decompose into shorter cycles.
- Products, conjugates, and consequences of several listed relators can yield
  a shorter identity word than any relator as written.
- A different but equivalent presentation can have different relator lengths
  while defining the same group and selected Cayley graph.
- Changing the generating collection changes the graph and its girth even when
  the abstract group is unchanged.

The girth is determined by all reduced identity words over the actual generator
alphabet, not just the syntactic minimum in one chosen relator file.

## Loops, parallel labels, and involutions

The clean theorem above needs an explicit graph convention.

- An identity generator creates a labeled loop of length one, but a simple
  Cayley graph normally omits it.
- Duplicate generator labels can create parallel edges.  A multigraph may call
  two parallel edges a length-two cycle; a simple graph collapses them.
- For an involution `s=s^-1`, the word `ss` is immediate traversal and reversal
  of the same undirected edge.  It should not be mistaken for a simple cycle of
  length two.
- A directed generator alphabet has directed girth and no automatic inverse
  edge.  Free reduction and non-backtracking must follow the directed contract.
- Labeled-path multiplicity can change even when the underlying simple vertex
  graph and its girth do not.

Consequently every girth or relation claim must say whether the graph is
directed, simple, a multigraph, labeled, and inverse-closed.

## Schreier graphs: identity becomes stabilizer membership

In a Schreier action graph, two words can reach the same state without
representing the same group element.  Their quotient lies in the stabilizer of
the base state:

```text
u.x = v.x  implies  inverse(v)u in Stab(x)
```

up to the selected left/right convention.  Closed state walks therefore encode
stabilizer words, not necessarily identity words in the group.  A Cayley girth
argument cannot be copied to a puzzle action graph merely because both use the
same generator names.

## What BFS duplicate observations can and cannot reveal

An exact BFS trace can provide witnesses:

- two same-depth parent records for one child give two shortest words;
- a same-level edge plus BFS parents yields an odd cycle witness;
- a non-parent edge to an earlier level plus tree paths yields a cycle;
- in Cayley graphs, replayed path pairs produce concrete identity words;
- in Schreier graphs, they produce concrete stabilizer words.

But aggregate counts alone do not identify a presentation:

- the same sphere sequence can arise from different relation structures;
- one short relation can create many overlapping collisions;
- many relations can generate few locally co-located duplicate occurrences;
- frontier order changes where equal candidates meet physically without
  changing girth or sphere sets.

Girth predicts a collision-free radius.  It does not predict duplicate ratio,
warp locality, owner routing, or total traversal time after that radius.

## Cayley-specific sanity examples

### Free group

The Cayley graph of a free group on an inverse-closed basis is an infinite
regular tree.  Its girth is infinite.  After suppressing immediate inverse
backtracking, the word tree and state BFS agree at every depth.

### Free abelian rank two

With generators `a,A,b,B`, commutation gives

```text
a b A B = identity.
```

The Cayley graph is the square lattice with girth four.  The words `ab` and
`ba` first converge at depth two, exactly at the `2r=g` uniqueness boundary.
Sphere growth is linear rather than the exponential non-backtracking word-tree
growth.

### Adjacent transpositions

For adjacent swaps `s_i`, involution relations `s_i^2=e` encode immediate edge
reversal, while commutations `s_i s_j = s_j s_i` for `|i-j|>1` create genuine
four-cycles.  Braid relations create six-step closed words.  The four-cycles
already force state convergence at depth two; the braid relations add further
structure but do not set the girth.

## Audit checklist

1. Which simple/directed/labeled/multigraph girth is meant?
2. Is immediate inverse backtracking excluded before comparing with a tree?
3. What is the shortest independently replayed reduced closed word?
4. Is it an identity word or only a stabilizer word?
5. Does a listed relator trace a simple cycle, or a reducible/repeated walk?
6. Up to which radius are root geodesics proved unique?
7. Up to which smaller radius is the induced ball proved to be a tree?
8. Which duplicate count means backtracking, candidate convergence, or an
   earlier visited hit?
9. Are Moore-bound assumptions—finite, simple, undirected, regular—satisfied?
10. Which conclusions concern geometry, and which remain unmeasured hardware
    behavior?

## Sources and expert-channel failure

- Norman Biggs,
  [Girth, valency, and excess](https://doi.org/10.1016/0024-3795(80)90205-0),
  develops the regular-graph order lower bound from valency and girth.
- Exoo and Jajcay,
  [Dynamic Cage Survey](https://www.combinatorics.org/ojs/index.php/eljc/article/download/v4i2r13/pdf/),
  states the odd/even Moore bounds as counts in a BFS tree.
- Grimmett and Li,
  [Locality of connective constants](https://www.statslab.cam.ac.uk/~grg/papers/sawcayley-final14.pdf),
  records explicit generator/inverse/relator conventions for simple undirected
  Cayley graphs.
- The two girth thresholds and Cayley word correspondence are proved directly
  above with their graph-contract assumptions.
- A request to the `autolean` expert for an off-by-one check returned only
  `fetch failed`; no expert answer was used.

## Current conclusion

Before a cycle can close, non-backtracking generator paths behave like distinct
tree branches and BFS spheres have tree counts.  Girth gives the exact safe
radius, while relations explain how branches later merge.  Immediate inverse
returns, genuine relation collisions, and earlier visited hits are different
events and must not be combined into one vague "duplicate" statistic.

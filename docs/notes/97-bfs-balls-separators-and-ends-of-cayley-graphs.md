# BFS balls, separators, and ends of Cayley graphs

An end of an infinite graph is a direction to infinity that cannot be separated
from itself by deleting finitely many vertices. BFS balls provide a canonical
sequence of finite separators, so their complements reveal progressively more
of this large-scale branching structure.

This note connects ends to BFS frontiers and Cayley graphs. It does not attempt
to compute group splittings or implement an end detector.

## 1. Rays and ends

Work with a connected, locally finite, undirected graph. A **ray** is a one-way
infinite simple path. Two rays are equivalent when, after deleting any finite
vertex set, tails of both rays lie in the same remaining component. An **end**
is an equivalence class of rays.

Intuitively, two rays represent the same direction to infinity if no finite
obstacle can permanently separate their tails. Local finiteness ensures every
finite-radius BFS ball is finite and makes balls suitable separators.

Finite connected graphs have no rays and therefore zero ends.

## 2. Looking outside BFS balls

Fix root `s` and let `B_r` be its completed BFS ball. Let `c_r` be the number of
infinite connected components of

```text
G \ B_r.
```

Every ray eventually leaves `B_r` and has a tail in one such component. As `r`
increases, an old infinite component may split but distinct components cannot
merge. Because the newly removed shell is finite, each old infinite component
leaves at least one infinite descendant. Hence

```text
c_(r+1) >= c_r.
```

An end is not merely one component at one radius. It is a coherent nested choice
of outside components for every sufficiently large radius. If the graph has a
finite number `k` of ends, then sufficiently large balls separate them and
`c_r=k` thereafter. With infinitely many ends, finite-radius component counts
may keep refining without enumerating the full end space.

## 3. Frontier size bounds visible directions

Every component of `G\B_r` adjacent to the ball contains at least one vertex of
the next sphere `S_(r+1)`. Distinct outside components contain distinct sphere
vertices. Therefore

```text
c_r <= |S_(r+1)|.
```

This is only an upper bound. A huge frontier can belong to one connected outside
region, while a small frontier can already separate several directions.

The BFS sphere is a separator from note 48; ends ask how its outside components
continue through every larger finite separator.

## 4. Calibration examples

### A ray

The half-line has one end. Removing any root ball leaves one infinite tail, and
the next BFS sphere has one vertex.

### The integer line

`Z` with generators `{+/-1}` has two ends. Removing a finite interval leaves a
left and a right infinite component. Its next sphere has exactly two vertices.

### The square grid

`Z^2` has one end. Its BFS sphere grows linearly, but the region outside a large
diamond remains connected. Growing frontier width does not imply multiple ends.

### A regular tree

In a `q`-regular tree with `q>=3`, every vertex of `S_(r+1)` roots a distinct
infinite outside subtree. Thus

```text
c_r = |S_(r+1)|,
```

and the graph has infinitely many ends. Here frontier branching is permanent
rather than reconnecting later.

## 5. One finite BFS prefix cannot determine ends

For any chosen radius `R`, construct two rooted locally finite graphs that agree
exactly through `B_R`:

- continue the exposed path forever, producing one end;
- after depth greater than `R`, split into an infinite binary tree, producing
  infinitely many ends.

Every BFS observation through radius `R` is identical, including vertices,
edges, parents, and frontier counts. The end structure diverges only outside the
observed scope.

Likewise, a sufficiently long finite cycle agrees locally with the integer line
through a prescribed radius but has zero ends rather than two. Bounded BFS cannot
certify an infinite asymptotic invariant without external structure.

## 6. Ends of finitely generated groups

The number of ends of a finitely generated group is defined using any Cayley
graph from a finite generating set. This is well defined because ends are
preserved by quasi-isometry, and note 93 shows that finite-generator Cayley word
metrics are quasi-isometric.

The Freudenthal-Hopf theorem says a finitely generated group has

```text
0, 1, 2, or infinitely many ends.
```

Calibration:

- finite groups have zero ends;
- infinite virtually cyclic groups have two ends;
- `Z^d` for `d>=2` has one end;
- nonabelian free groups have infinitely many ends.

Stallings' theorem gives a deep algebraic characterization of groups with more
than one end via splittings over finite subgroups. BFS complement components are
the geometric shadow, not by themselves a proof of a particular splitting.

## 7. Generator changes preserve ends, not layers

Changing between finite symmetric generating sets can alter:

- exact BFS sphere sizes;
- the radius at which outside components visibly separate;
- girth, relations, parents, and frontier work.

It cannot alter the group's number of ends. Linear radius distortion may delay
or advance when a finite separator is seen, while coherent directions to
infinity remain the same.

This is an example of the distinction from note 93: ends are a coarse
quasi-isometry invariant; exact BFS traces are not.

## 8. Cayley versus Schreier and directed boundaries

A Schreier graph may have a different number of ends from the group's Cayley
graph because stabilizers and quotient fibers change connectivity at infinity.
A finite puzzle orbit always has zero graph ends even if an ambient infinite
group has one, two, or infinitely many.

The ray-equivalence definition above is for undirected locally finite graphs.
Positive-only directed Cayley graphs require a declared directed-end notion;
weak, strong, forward, and backward variants need not agree. One should not
silently apply the undirected theorem after forgetting orientation.

## 9. BFS, memory, and multi-GPU interpretation

End count describes persistent topological directions, not frontier capacity:

- one-ended `Z^2` has growing spheres;
- two-ended `Z` has constant-size spheres;
- infinitely-ended trees have exponentially growing spheres;
- finite graphs have zero ends but may have enormous middle frontiers.

Outside graph components are also unrelated to owner partitions. One end may be
sharded across every GPU, and many ends may hash to the same owner. End structure
does not predict routing bytes, duplicate convergence, or load balance.

For finite CayleyPy workloads, ends apply only to an explicitly chosen infinite
ambient graph or a proved family limit. They are not a replacement for measured
finite layer profiles.

## 10. Evidence checklist

1. Infinite connected locally finite undirected graph.
2. Ray equivalence under every finite vertex deletion.
3. Completed BFS balls and infinite components of their complements.
4. One finite prefix versus a proved asymptotic graph family.
5. Cayley graph, Schreier graph, or finite orbit.
6. Finite symmetric generator set and quasi-isometry scope.
7. End count versus exact frontier width and separation radius.
8. Graph outside-components versus physical owner partitions.

## Sources

- R. Diestel, [*Graph Theory*, Chapter 8: Infinite
  Graphs](https://doi.org/10.1007/978-3-662-53622-3_8),
  Springer, 5th edition, 2017. Rays, ends, finite separators, and locally finite
  graph topology.
- R. G. Möller, [*Graphs, Permutations and Topological
  Groups*](https://arxiv.org/abs/1008.3062),
  2010 survey. Quasi-isometry invariance, the `0,1,2,infinity` theorem, and
  Stallings context for finitely generated groups.
- Notes 09, 21, 35, 46, 48, 71, 84, 93, and 94 provide infinite BFS,
  diameter, growth, expansion, separator, arbitrary-profile, SCC, quasi-isometry,
  and boundary context.

## Takeaway

BFS balls are finite separators that progressively distinguish directions to
infinity. Their outside components approximate ends, but one frontier width or
any bounded prefix cannot determine the asymptotic end space. Ends survive
finite generator changes; exact layers, memory peaks, and multi-GPU traffic do
not.

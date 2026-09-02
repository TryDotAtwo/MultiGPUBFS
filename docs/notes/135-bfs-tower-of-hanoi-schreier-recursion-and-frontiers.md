# BFS on Tower-of-Hanoi graphs: Schreier state, recursion, and frontiers

The classical three-peg Tower of Hanoi gives a finite implicit state graph with
`3^n` vertices. Its legal-move graph is self-similar: fixing the largest disk
produces three copies of the `(n-1)`-disk graph, connected by three exceptional
largest-disk moves.

This structure determines not only the familiar corner-to-corner distance
`2^n-1`, but the entire BFS frontier profile from a perfect-stack corner. It
also exposes a precise Cayley/Schreier distinction and a loop-versus-legal-move
counting boundary. This note adds no optimizer, production implementation,
benchmark, or GPU code. A small exhaustive Rust probe checks `n=1..6`.

## 1. Classical puzzle contract

Use three pegs `{0,1,2}` and `n` distinguishable disks numbered from smallest
`0` to largest `n-1`. One move transfers the top disk of one peg to another peg
whose top disk, if any, is larger.

A state is the word

```text
p_0 p_1 ... p_(n-1),
```

where `p_i` is the peg holding disk `i`. Every word in `{0,1,2}^n` is legal:
once peg membership is known, disk size uniquely fixes the vertical order.
Hence

```text
|V(H_n)| = 3^n.
```

The complete word is the visited identity. Recording only top disks, peg
heights, or the largest-disk position merges states with different futures.

## 2. Legal successor rule

For a pair of pegs `{i,j}`, inspect the smallest-index disk present on either
peg. If one exists, it is the unique legal disk movable between that pair and
is transferred to the other peg. If both pegs are empty, no puzzle move exists.

Equivalently, a direct generator can find each peg's top disk and move a top
disk only onto an empty peg or a larger top disk. The successor depends on the
whole smaller-disk prefix, not on the coordinate being changed alone.

Thus the graph is implicit and locally generated, but it is not a Hamming graph
where any one coordinate may be replaced independently.

## 3. Simple puzzle graph versus labeled Schreier graph

The Hanoi Towers group has three involutory generators `a_01,a_02,a_12`. On a
state, `a_ij` performs the legal move between pegs `i,j`; if both are empty, it
fixes the state.

Therefore the level-`n` Schreier graph has three labeled generator occurrences
per state and includes one loop at each perfect-stack corner. The usual simple
puzzle graph deletes these loops:

- its three corner states have degree two;
- every other state has degree three;
- deleting loops preserves reachability and unweighted distances;
- it does not preserve generated-candidate, edge-occurrence, or label counts.

The simple graph has

```text
|E(H_n)| = (3/2)(3^n - 1),
```

because the three regular labeled occurrences at every state contribute
`3*3^n`, three occurrences are loops, and every remaining undirected edge is
seen twice.

## 4. Why this is Schreier, not a Cayley graph on puzzle states

The infinite Hanoi Towers group acts on length-`n` words. The puzzle positions
form one orbit, and state stabilizers can be nontrivial. Thus `H_n` is a
Schreier graph of a group action, not the Cayley graph whose vertices are the
group elements themselves.

Different generator words can induce the same puzzle state. BFS visited merges
them by configuration identity. Word length in the acting group and shortest
puzzle-state distance need not be the same quantity because stabilizer words
disappear in the orbit quotient.

This is exactly the distinction from note 16: transitive reachability of the
orbit does not turn configurations into group elements.

## 5. Three-copy recursion

Partition `H_n` by the peg of the largest disk. Inside each part the largest
disk cannot move, so smaller disks form a copy of `H_(n-1)`.

To move the largest disk from peg `i` to peg `j`, all `n-1` smaller disks must
be stacked on the third peg `k`. For each unordered peg pair there is exactly
one such inter-copy connector edge. Therefore

```text
H_n = three copies of H_(n-1) joined by three connector edges.
```

These are not graph-theoretic bridges: the connected copies and the other two
connectors give an alternate route after any one connector is removed. Already
`H_1` is a triangle, whose edges are not bridges. The three connectors together
are a sparse cut between the recursive copies.

This is graph recursion, not recursion of a particular program. Any exact BFS
schedule explores the same metric graph even if it never explicitly constructs
the three copies.

## 6. Corner-to-corner distance recurrence

Let `D_n` be the distance between two perfect-stack corners. Before the largest
disk can move, all smaller disks must reach the third peg, costing at least
`D_(n-1)`. The largest disk then moves once, and the smaller tower must move onto
it, costing another `D_(n-1)`. The standard construction attains this bound:

```text
D_0 = 0,
D_n = 2D_(n-1) + 1 = 2^n - 1.
```

For the classical three-peg graph this value is also the diameter. The theorem
does not transfer unchanged to four pegs, directed peg restrictions, cyclic
move rules, indistinguishable disks, or altered legal-placement rules.

## 7. Entire corner-root frontier recurrence

Let `f_n(k)` be the number of states at BFS distance `k` from corner `0^n`.
The three-copy recursion gives

```text
f_n(k) = f_(n-1)(k)                   for 0 <= k < 2^(n-1),
f_n(2^(n-1)+j) = 2 f_(n-1)(j)        for 0 <= j < 2^(n-1).
```

The first half lies in the starting largest-disk copy. In the second half, the
wave has crossed a largest-disk connector and appears symmetrically in the other
two copies.

With `f_0(0)=1`, this solves to

```text
f_n(k) = 2^popcount(k),    0 <= k < 2^n.
```

The binary digits of depth encode which recursive halves contributed a factor
of two. The maximum corner layer at `k=2^n-1` has `2^n` states, not merely the
two other perfect-stack corners.

## 8. The profile exhausts exactly `3^n` states

Summing the closed form over all `n`-bit depths gives

```text
sum_(k=0)^(2^n-1) 2^popcount(k)
= product_(bit=1)^n (1+2)
= 3^n.
```

Thus the frontier formula is consistent with complete exhaustion, not only
with early layers or a selected target path. It simultaneously proves the
corner eccentricity `2^n-1` because the last layer is nonempty.

This is a rare exact frontier law arising from strong self-similarity. It is not
a generic consequence of degree three or of having `3^n` states.

## 9. Small exhaustive Rust probe

`experiments/hanoi_bfs_probe.rs` enumerates all ternary words, generates the
three peg-pair actions, runs BFS, computes all-source diameter, counts loops and
simple degrees, and reports corner layers for `n=1..6`.

The largest checked graph has 729 states. Observed in Docker with Rust 1.85.1:

```text
n  states  corner distance  diameter  loops  degree-2  degree-3
1       3                1         1      3         3         0
2       9                3         3      3         3         6
3      27                7         7      3         3        24
4      81               15        15      3         3        78
5     243               31        31      3         3       240
6     729               63        63      3         3       726
```

For `n=6`, the 64 layers match `2^popcount(k)` and end in

```text
..., 8,16,16,32,16,32,32,64.
```

This is exhaustive evidence for `n<=6`; the general statements rely on the
recursive proofs above.

## 10. Frontier geometry is not a search tree

The classic recursive solution gives one shortest corner-to-corner path. BFS
instead reaches every state and deduplicates many move histories. The exact
layer profile is not the binary recursion tree's node count.

At depth `k`, `2^popcount(k)` counts distinct configurations, not move words,
recursive calls, or shortest paths. Those objects can have different
multiplicities and require different output contracts.

## 11. Separators and bottlenecks

At the top recursive scale, only three largest-disk connector edges connect three
subgraphs of size `3^(n-1)`. These sparse connections are metric and cut
bottlenecks even though almost every noncorner vertex has degree three.

This helps explain why degree alone says little about frontier evolution or
communication. It also supplies explicit separator witnesses, but recursively
contracting each copy would change internal distances and is not an exact BFS
substitute without additional weights and expansion semantics.

## 12. Symmetry boundary

Permuting the three peg names preserves the graph and maps the three corners to
one another. For `n>=2`, it does not map an arbitrary configuration to every other
configuration; the simple graph is not vertex-transitive, as its degree-two
corners and degree-three interior already prove. The `n=1` triangle is the
vertex-transitive exception.

Therefore a corner-root profile transfers to the other corners, but not to an
arbitrary root. The transitive group action defining the Schreier orbit and the
automorphism group of the fixed generator-labeled graph are different notions.

## 13. Bidirectional BFS

For the standard corner-to-corner query, bidirectional BFS has exact symmetric
endpoints and can exploit the same undirected transitions. Correct stopping
still requires the bounds from note 8; the recursive midpoint of one canonical
solution is not automatically the first or unique meeting state of two BFS
waves.

Meeting requires exact ternary-word equality. Sharing the largest-disk peg or a
smaller-disk subconfiguration is only partial agreement.

## 14. Variant boundary

Several changes produce different graphs:

- four or more pegs;
- directed or cyclic allowed peg moves;
- adjacent-peg-only movement;
- colored, equal-sized, or indistinguishable disks;
- limits on which disk sizes may touch;
- generator labels that retain forbidden attempts as loops.

The state count, recursive copies, connectors, diameter, and frontier formula all
need rederivation. Calling every version “Hanoi BFS” is not a correctness
contract.

## 15. GPU and multi-GPU boundary

The state admits a compact base-three rank and exactly three labeled peg-pair
actions. That suggests regular candidate production but proves no production
performance result. Move legality may require identifying top disks, and the
authoritative visited universe grows as `3^n`.

A partition by the largest-disk digit creates three equal top-level owners with
only three logical connector edges between recursive blocks. A hash partition has
different traffic and balance behavior. Neither observation is a universal
partition recommendation; both require measurement under the intended state
encoding and deeper recursive levels.

Measurements should separate:

- three labeled attempts per state, including fixed-point loops;
- two/three simple legal neighbors;
- exact frontier states and duplicate candidates;
- top-disk discovery or maintained metadata;
- ternary rank/key bytes and visited capacity;
- recursive-block locality and cross-owner routing;
- level completion, physical communication, and end-to-end time.

Deleting three Schreier loops preserves distances but changes attempted-edge
throughput. A fast candidate counter can therefore disagree with a simple-edge
BFS counter while both traverse the same distance metric.

## Sources

- A. M. Hinz, S. Klavzar, U. Milutinovic, and C. Petr,
  [*The Tower of Hanoi — Myths and Maths*](https://link.springer.com/book/10.1007/978-3-0348-0237-6),
  Birkhauser, 2013, for Hanoi graph definitions, recursion, distances, and
  puzzle variants.
- R. Grigorchuk and Z. Sunic,
  [*Asymptotic aspects of Schreier graphs and Hanoi Towers groups*](https://comptes-rendus.academie-sciences.fr/mathematique/item/10.1016/j.crma.2006.02.001.pdf),
  2006, for the automaton-group action, level Schreier graphs, and their
  self-similar structure.
- R. Grigorchuk and Z. Sunic,
  [*Schreier spectrum of the Hanoi Towers group on three pegs*](https://people.tamu.edu/~grigorch/publications/hanoi-spectrum.pdf),
  2008, for the three involutory peg-pair generators and the distinction between
  regular Schreier graphs and closely related puzzle graphs.
- Notes 8, 11, 16, 17, 21, 26, 29, 44, 46, 48, 55, 61, 64, and 113 supply this
  repository's bidirectional, DAG, Schreier, quotient, certificate, logical
  depth, complexity, GPU-transfer, memory, separator, successor, identity,
  multiplicity, and layered-structure boundaries.

## Takeaway

The three-peg Hanoi graph is a finite Schreier state graph of `3^n` legal
configurations, recursively built from three smaller copies. Corner BFS has the
exact frontier law `|F_k|=2^popcount(k)` through diameter `2^n-1`. Deleting the
three fixed-point generator loops preserves distances but changes work counts.
The recursive puzzle solution, the BFS state frontier, and the acting group's
word graph are related but distinct objects.

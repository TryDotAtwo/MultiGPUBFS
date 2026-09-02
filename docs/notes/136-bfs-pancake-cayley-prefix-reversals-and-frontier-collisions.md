# BFS on pancake Cayley graphs: prefix reversals and frontier collisions

The pancake graph is a genuine finite Cayley graph: vertices are permutations,
and generators reverse prefixes. Every vertex has the same labeled moves, so
one identity-root BFS profile transfers to every root. Yet the frontier is only
tree-like for a few steps. Involutions remove immediate backtracking, and a
length-six relation creates the first collision at depth three.

This gives a clean contrast with note 135's Hanoi Schreier graph. It also shows
why regularity and vertex transitivity do not supply a closed frontier or
diameter formula. This note adds no optimizer, production implementation,
benchmark, or GPU code. A small exhaustive Rust probe checks `P_2` through
`P_8`.

## 1. Unsigned pancake contract

Let `S_n` be the symmetric group on `n` distinct symbols. For every
`k in {2,...,n}`, define prefix reversal

```text
r_k(x_1 x_2 ... x_k x_(k+1) ... x_n)
  = x_k ... x_2 x_1 x_(k+1) ... x_n.
```

The unsigned pancake graph is

```text
P_n = Cay(S_n, {r_2,...,r_n}).
```

It has

```text
|V| = n!,       degree = n-1.
```

This note treats each reversal as one undirected unit-cost move and includes no
orientation or burnt side on a pancake.

## 2. The generators really generate every permutation

Each `r_k` is an involution. Prefix reversals generate adjacent transpositions
(and hence `S_n`), so the Cayley graph is connected. An exact BFS from the
identity therefore reaches all `n!` permutations.

No prefix reversal with `k>=2` fixes a permutation of distinct symbols, and two
different prefix lengths cannot produce the same successor. Thus each state has
exactly `n-1` distinct simple neighbors as well as `n-1` labeled candidates.

This differs from Hanoi's fixed-point generator loops.

## 3. Sorting distance is Cayley word distance

The distance from a permutation `pi` to the identity is the minimum number of
prefix reversals sorting `pi`. With a consistent left/right action convention,
the distance between arbitrary `pi,tau` reduces by translation to the word
length of `pi^-1 tau` (or the conventionally reversed equivalent).

The pancake number

```text
f(n) = max_(pi in S_n) d(identity,pi)
```

is exactly the graph diameter. A constructive sorting procedure supplies an
upper bound; one difficult permutation supplies only a lower bound unless its
optimality and global maximality are proved.

## 4. Cayley homogeneity

Left translation is a graph automorphism, so every root has the same
eccentricity and layer-size profile. One identity-root exhaustive BFS therefore
computes the diameter and the common sphere sizes.

This does not imply edge transitivity, distance regularity, identical path
counts for all targets at one depth, or a simple formula in `n`. Vertex
transitivity is a root-transfer theorem, not a frontier-shape theorem.

## 5. Word tree before state collisions

There are `n-1` generator words of length one. After a move, repeating the same
involution immediately returns to the parent, so a nonbacktracking word has
`n-2` next choices. The resulting history counts begin

```text
1,
n-1,
(n-1)(n-2),
(n-1)(n-2)^2, ...
```

These are path-history candidates, not automatically distinct permutation
states. Exact visited quotients the word tree by all prefix-reversal relations.

## 6. Exact first three nonzero layers

For `n>=3`, pancake graphs have no cycles of length three, four, or five and
have girth six. Consequently:

```text
|F_1| = n-1,
|F_2| = (n-1)(n-2).
```

At depth three, the relation

```text
(r_2 r_3)^3 = e
```

equivalently gives

```text
r_2 r_3 r_2 = r_3 r_2 r_3.
```

Two nonbacktracking length-three histories first collide. The exact classical
layer formula is

```text
|F_3| = (n-1)(n-2)^2 - 1.
```

The subtraction is a state collision, not a missing generator attempt.

## 7. Later layers are governed by the full relation structure

Beyond the first collision, longer cycles overlap and many word histories land
on already reached permutations. Neither fixed degree nor girth determines the
remaining frontier profile.

For example, the exact `P_8` profile is

```text
[1,7,42,251,1191,4281,10561,15011,8520,455].
```

The frontier grows through depth seven, then contracts sharply before
exhaustion at diameter nine. A local branching factor measured in early layers
cannot be extrapolated across this transition.

## 8. Small exhaustive Rust probe

`experiments/pancake_bfs_probe.rs` uses transparent permutation vectors and a
hash-based exact visited set. It exhaustively traverses `P_2,...,P_8`, checks
the factorial state count, and verifies the early-layer formulas.

Observed in Docker with Rust 1.85.1:

```text
n  states  degree  diameter  layers
2       2       1         1  [1,1]
3       6       2         3  [1,2,2,1]
4      24       3         4  [1,3,6,11,3]
5     120       4         5  [1,4,12,35,48,20]
6     720       5         7  [1,5,20,79,199,281,133,2]
7    5040       6         8  [1,6,30,149,543,1357,1903,1016,35]
8   40320       7         9  [1,7,42,251,1191,4281,10561,15011,8520,455]
```

Every tested graph reached exactly `n!` states, and every tested early profile
matched the three formulas above. This is exhaustive evidence only through
`n=8`; it is not an empirical derivation of the general pancake number.

## 9. Diameter remains a separate theorem

Gates and Papadimitriou introduced general constructive upper bounds and hard
families giving lower bounds. Later work improved bounds and computed larger
individual cases. The absence of a general exact formula is part of the
problem, not a reason to identify a heuristic sorting length with diameter.

For any reported `P_n` diameter, evidence should distinguish:

- one permutation's replayed path;
- proof that its distance is minimal;
- proof that every permutation is no farther;
- complete exhaustion or a mathematically sufficient bound.

The small probe supplies all four only for its declared `n<=8` scope.

## 10. Exact state identity and rank

A permutation vector is a complete but non-dense key. A Lehmer rank gives a
bijection between `S_n` and `[0,n!-1]`, enabling exact dense visited indexing
when the convention is validated.

A hash or fingerprint without collision resolution is not an exact rank.
Likewise, storing only adjacencies, breakpoints, inversion count, or the current
largest pancake position merges distinct permutations and cannot serve as
authoritative visited identity.

## 11. Parity is not the BFS depth

The sign of `r_k` is

```text
(-1)^(k(k-1)/2).
```

Different prefix lengths can therefore have different permutation parity. For
`n>=4`, the generator set contains both even and odd permutations. Ordinary
distance parity is not determined by permutation sign, and the graph should not
be assumed bipartite from the fact that its generators are involutions.

Involution means `r_k^2=e`; it does not mean every generator flips one common
two-coloring.

## 12. Unsigned versus burnt pancake graphs

The burnt pancake problem uses signed/oriented permutations. A prefix move both
reverses order and flips orientations, producing `2^n n!` states in the usual
model and a different generating set and diameter sequence.

Unsigned results, ranks, frontier profiles, and parity arguments do not transfer
merely by ignoring signs. Binary/ternary strings with repeated symbols are also
quotients with stabilizers and multiplicities, not the same `S_n` Cayley graph.

## 13. Bidirectional BFS

All prefix reversals are self-inverse, so forward and backward successor code
can use the same transformations. Exact bidirectional stopping still needs the
lower-bound proof from note 8.

Because the graph is vertex-transitive, a query can be translated to the
identity before search. Parents and move labels must be transformed under the
same left/right convention when reconstructing a path; merely reversing the
reported prefix lengths is not a complete convention proof.

## 14. GPU and multi-GPU boundary

The family has attractive conceptual regularity:

- exactly `n-1` distinct successors per state;
- fixed-length permutation payloads;
- exact dense rank universe of size `n!`;
- no fixed-point loops or parallel generator endpoints.

It also has factorial visited growth and increasing per-candidate reversal/rank
work. Early history branching overstates new states after relation collisions,
and the frontier eventually contracts. None of these facts alone selects an
optimal representation or kernel.

A GPU/multi-GPU study should separate:

- labeled generator attempts and permutation-copy/reversal work;
- raw histories, duplicate candidates, unique candidates, and new states;
- rank/key computation and authoritative visited operations;
- frontier profile and effective new-state branching;
- owner balance and cross-owner routing;
- level completion, communication, and end-to-end time;
- capacity scaling from strong scaling.

Vertex-transitive logical geometry does not imply balanced hash owners or
uniform physical communication. Those are properties of the selected mapping
and workload.

## Sources

- W. H. Gates and C. H. Papadimitriou,
  [*Bounds for Sorting by Prefix Reversal*](https://doi.org/10.1016/0012-365X(79)90068-2),
  *Discrete Mathematics* 27, 1979, for the pancake distance/diameter problem and
  classical lower/upper-bound methodology.
- M. H. Heydari and I. H. Sudborough,
  [*On the Diameter of the Pancake Network*](https://doi.org/10.1006/jagm.1997.0874),
  *Journal of Algorithms* 25, 1997, for the network definition, exact finite
  values available there, and improved diameter bounds.
- E. Konstantinova,
  [*On Some Structural Properties of Star and Pancake Graphs*](https://arxiv.org/abs/1201.1726),
  2012, for Cayley regularity, vertex transitivity, girth six, and the exact
  first three frontier sizes.
- Notes 6, 8, 10, 16, 27, 28, 35, 39, 46, 51, 60, 61, 64, 67, and 93 supply
  this repository's implicit Cayley, bidirectional, frontier-growth, Schreier,
  relation/girth, identity, growth-series, nonbacktracking, memory, ownership,
  short-relation, equality, multiplicity, parity, and generator-metric
  boundaries.

## Takeaway

The pancake graph is a regular vertex-transitive Cayley graph on all `n!`
permutations. Its first layers follow the nonbacktracking word tree until the
six-step relation creates a depth-three collision; later frontiers reflect the
full relation structure and eventually contract. Vertex transitivity transfers
one root's profile but does not provide a general diameter formula or erase
factorial visited growth.

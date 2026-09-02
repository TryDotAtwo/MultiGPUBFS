# BFS depth slack and the period of a strongly connected digraph

Strong connectivity says every vertex can eventually return. It does not say
that return walks exist at arbitrary lengths. The arithmetic obstruction is the
greatest common divisor of directed cycle lengths, called the graph's period.

Exact BFS labels expose this period through a simple per-arc quantity. This
connects directed layering, cyclic classes, adjacency powers, and Cayley word
length modulo an invariant.

## 1. Period and aperiodicity

Let `G` be a finite strongly connected directed graph containing at least one
arc. Its period is

```text
p = gcd { length(C) : C is a directed cycle of G }.
```

Equivalently, use lengths of all positive closed walks from any fixed vertex.
Strong connectivity makes the resulting GCD independent of the chosen vertex.

- `p=1`: the digraph is **aperiodic** or primitive.
- `p>1`: return lengths are trapped in multiples of `p`.

Strong connectivity alone does not imply aperiodicity. A directed `N`-cycle is
strongly connected and has period `N`.

## 2. Walk lengths to one vertex have one residue

Fix root `s` and vertex `v`. Let `P,Q` be any two directed walks from `s` to
`v`, and let `R` be a return walk from `v` to `s`. Then `PR` and `QR` are closed
walks, so their lengths are divisible by `p` modulo the same residue. Hence

```text
|P| = |Q| mod p.
```

Define the cyclic class

```text
c(v) = d(s,v) mod p.
```

Every walk from `s` to `v` has this residue, not only the shortest one. For
every arc `u -> v`, appending the arc to a walk to `u` gives

```text
c(v) = c(u)+1 mod p.
```

The vertices split into `p` nonempty cyclic classes, and every arc advances one
class modulo `p`.

## 3. BFS depth slack recovers the period

Run complete exact BFS from `s` and write `d(v)=d(s,v)`. For every arc
`u -> v`, define its nonnegative depth slack

```text
lambda(u,v) = d(u)+1-d(v).
```

It is nonnegative because the shortest path to `u` followed by the arc is a
candidate path to `v`.

Now let

```text
q = gcd { lambda(u,v) : (u,v) in E },
```

with zero values ignored in the usual GCD convention. Then

```text
q = p.
```

### Why `p` divides every slack

`d(u)+1` and `d(v)` are lengths of two walks from `s` to `v`. Section 2 shows
that their difference is divisible by `p`. Thus `p` divides every `lambda` and
therefore divides `q`.

### Why the slack GCD divides every cycle

For a directed cycle `v_0 -> ... -> v_(k-1) -> v_0`, sum its arc slacks:

```text
sum_i (d(v_i)+1-d(v_(i+1))) = k.
```

The distance terms telescope. Since `q` divides every summand, it divides every
cycle length and hence divides `p`. Both divisibilities give `q=p`.

This is a compact certificate only after distances and arc coverage are exact.

## 4. Directed-cycle calibration

Root directed cycle `C_N` at vertex zero. BFS depths are

```text
0,1,...,N-1.
```

Every forward arc except the closing one has slack zero. The closing arc
`N-1 -> 0` has slack `N`. Their GCD is `N`, exactly the period.

The example also illustrates the directed layer rule: a forward arc cannot
land deeper than `d(u)+1`, but a return arc may jump backward many layers. Its
backward jump amount `d(u)+1-d(v)` is not arbitrary modulo the period.

## 5. Undirected bipartiteness as the `p=2` case

Replace every undirected edge by two opposite arcs. Each edge gives a directed
closed walk of length two, so the period divides two.

- If the connected undirected graph is bipartite, every closed walk has even
  length and the period is two.
- If it is non-bipartite, an odd cycle exists; the GCD of its odd length and
  the length-two backtrack is one.

Thus the directed period generalizes the BFS parity coloring from note 21:
connected symmetric digraphs have period two exactly in the bipartite case,
and period one otherwise.

## 6. Relation to adjacency powers

For adjacency matrix `A`, `(A^k)[u,v]>0` means that a length-`k` walk exists.
In a period-`p` strongly connected graph, nonzero powers for a fixed pair can
occur only in one residue class modulo `p`.

BFS finds the first positive exponent for each target. The cyclic class says
which later exponents are arithmetically possible; it does not count their
walks or assert that every small exponent in that residue occurs. In a finite
aperiodic strongly connected graph, Perron-Frobenius primitivity supplies the
eventual stronger statement, but BFS exhaustion alone is not a mixing proof.

## 7. Cayley length modulo the period

Consider a strongly connected right-action Cayley digraph with positive
alphabet `S`. Let the identity be the root. Every positive word representing
group element `g` has length congruent modulo `p`, so define

```text
chi(g) = word_length(g) mod p.
```

This is well defined and satisfies

```text
chi(gh) = chi(g)+chi(h) mod p,
chi(s) = 1 mod p  for every s in S.
```

Hence `chi:G -> Z_p` is a surjective homomorphism, and cyclic classes are its
fibers/cosets of `ker chi`. Conversely, if a homomorphism to `Z_q` sends every
generator in `S` to one, every positive identity word has length divisible by
`q`; therefore `q` divides `p`. The period is the largest such common cyclic
length quotient.

Immediate consequences include:

- `p` divides the positive length of every identity relation;
- in a finite group, `p` divides the order of every listed generator because
  `s^ord(s)=e`;
- if an inverse-closed alphabet lists both `s` and `s^-1`, then
  `1=-1 mod p`, so `p` divides two.

For a Schreier action, cyclic classes still exist on the strongly connected
state digraph, but `chi` need not descend to a homomorphism on the whole group
without checking stabilizer words and quotient semantics.

## 8. Distributed GCD evidence

The GCD operation is associative and commutative. Once global BFS distances are
final, each owner can compute the GCD of slacks for an exact assigned subset of
arcs, followed by one global GCD reduction.

The reduction is exact only if every semantic arc is covered at least once
under one graph epoch:

- duplicate copies of a correct slack do not change the mathematical GCD;
- omitting slacks can only leave a multiple of the true period, falsely making
  the graph appear more periodic;
- stale or wrong distance labels can introduce arbitrary slack values and
  falsely reduce or alter the GCD;
- local vertex ownership does not by itself specify which rank is authoritative
  for cross-owner arc coverage.

This is a validation pattern, not a proposed optimized multi-GPU period
implementation.

## 9. Evidence checklist

1. Strongly connected directed graph or one finalized SCC.
2. Exact complete BFS distances from a declared root.
3. Complete semantic arc coverage, including loops and labeled parallels when
   they belong to the graph contract.
4. Nonnegative slack check for every arc.
5. Local and global GCDs with zero handling stated.
6. Independent cycle or adjacency-power calibration on a small scope.
7. Cayley positive alphabet, action side, inverses, and stabilizer semantics.
8. One graph/checkpoint/ownership epoch for distributed evidence.

## Sources

- E. Seneta, *Non-negative Matrices and Markov Chains*, chapters on irreducible
  matrices, period, and cyclic classes. Gives the standard GCD-of-closed-walks
  and cyclic decomposition theory.
- R. A. Brualdi and H. J. Ryser, *Combinatorial Matrix Theory*, sections on
  irreducible and primitive nonnegative matrices and their directed graphs.
- Notes 16, 21, 30, 31, 33, 41, 51, 67, 75, and 84 supply the Cayley action,
  parity, epoch, odd-cycle, adjacency-power, directed-layer, ownership,
  generator-parity, orientation, and SCC contracts used here.

## Takeaway

The period of a strongly connected digraph is visible directly in exact BFS
labels: it is the GCD of `d(u)+1-d(v)` over all arcs. Depth modulo the period
gives cyclic classes advanced by every transition. Bipartiteness is the
symmetric period-two case. In a strongly connected Cayley digraph, the classes
come from a word-length homomorphism to `Z_p`. A distributed GCD reduction is
easy algebraically, but only complete arc coverage and finalized labels make it
evidence.

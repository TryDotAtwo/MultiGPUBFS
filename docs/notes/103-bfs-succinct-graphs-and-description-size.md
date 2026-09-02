# BFS, succinct graphs, and description size

The classical linear-time statement for BFS measures the expanded graph. An
implicit transition rule can describe exponentially many vertices using only
polynomially many bits. Complexity measured in those two sizes can therefore
look completely different without any contradiction.

This note develops that boundary for implicit and Cayley-style state spaces. It
adds no solver or optimization.

## 1. Two independent size parameters

Suppose vertices are `n`-bit strings. The universe may contain

```text
N = 2^n
```

vertices even if a circuit or move oracle describing adjacency has size
polynomial in `n`.

Let

```text
L = length of the compact input description,
N = number of expanded semantic vertices,
M = number of expanded semantic arcs or move occurrences.
```

`Theta(N+M)` can be linear in the expanded graph and exponential in `L` at the
same time. The words "linear-time BFS" are incomplete until the size measure is
named.

## 2. Circuit-succinct graphs

A standard succinct representation uses a Boolean circuit `C` with two
`n`-bit vertex inputs and defines

```text
(u,v) is an arc  <->  C(u,v)=1.
```

The circuit has polynomial description size while representing an adjacency
matrix with `2^(2n)` possible entries. An adjacency query is compact, but a
successor list is not automatically cheap: naively enumerating all `v` tests
`2^n` candidates for one `u`.

A puzzle move oracle is a different, often stronger access model. If it gives
`q` named successors directly, successor enumeration costs `q` move
applications rather than scanning the universe. Even then, the number of
distinct reachable states can still be exponential.

Thus these must be declared separately:

- compact adjacency membership;
- direct successor enumeration;
- branching bound;
- total reachable-state volume.

## 3. Succinct reachability is PSPACE-complete

Directed `s-t` reachability in a circuit-succinct graph is PSPACE-complete when
complexity is measured in the compact description.

The hardness intuition is a configuration graph. A polynomial-space machine
has configurations encodable with polynomially many bits but may have
exponentially many configurations. A small circuit checks whether one valid
configuration follows another. Reaching an accepting configuration therefore
encodes the machine's computation.

Membership in PSPACE follows without materializing the graph. Recursive
reachability à la Savitch asks whether an intermediate vertex connects two
halves of a bounded path. With `N=2^n`, the recursion uses polynomial space in
`n`, although its time need not be polynomial.

This is a worst-case classification of a representation class. It does not say
that every implicit puzzle or every Cayley reachability instance is hard.

## 4. Bounded branching does not remove the state explosion

A configuration graph can have only a constant or polynomial number of legal
next moves from each state while still containing exponentially many states and
exponentially long computations. Direct `successors(state)` access avoids an
exponential adjacency-row scan but does not bound BFS depth or reached volume
by a polynomial in the description size.

Hence

```text
small generator count != small reachable graph,
cheap move application != cheap exhaustive BFS.
```

Generator relations may collapse the word tree dramatically, but a bounded
alphabet alone gives no polynomial-state guarantee.

## 5. BFS is not space-optimal for a one-bit decision

Exact graph BFS stores a growing `visited` set to obtain shortest distances and
avoid repeated expansion. In a succinct graph that set may contain
exponentially many `n`-bit vertices.

If the requested output is only the bit "is `t` reachable?", polynomial-space
recursive algorithms can use far less memory than BFS by recomputing
subproblems. They generally sacrifice time, frontier structure, and direct
shortest-path information.

This is another time-memory-output boundary:

```text
BFS:              broad stored history, exact minimum-depth layers;
recursive reach:  small space, heavy recomputation, decision-oriented output.
```

Calling the second method "better BFS" would be misleading; it solves a
narrower output problem with a different schedule and cost profile.

## 6. Output size can itself be exponential

A complete component listing, full distance table, parent tree, or exact
visited bitmap must represent information about every reached state. If the
component has exponential size, no algorithm can emit that explicit output in
polynomial time or space in the compact input length.

A single reachability bit may be compact. A shortest path can have exponential
length even though its distance value needs only `O(n)` bits. The replayable
path and its scalar length are therefore different output contracts.

Symbolic sets, circuits, decision diagrams, or formulas may compress some
frontiers and visited sets. They do not guarantee compactness for every graph;
their representation can grow exponentially, and exact membership/difference
operations still need proof under the chosen symbolic semantics.

## 7. A Cayley-state calibration

An implicit Cayley or puzzle state may use `b(n)` bits while the orbit contains
far more than polynomially many states. Examples include permutation families,
where the number of arrangements can grow factorially while one arrangement is
stored in roughly `Theta(n log n)` bits.

The group action can also provide special algorithms that decide orbit or
membership questions without enumerating every state. Such structure is a
reason to study the algebra before committing to BFS. It does not make shortest
word length, full sphere enumeration, or exact BFS tables free.

The correct statement is conditional:

```text
implicit description permits on-demand traversal;
it does not require, or promise, full materialization;
special algebra may answer some queries without BFS.
```

## 8. Capacity scaling has logarithmic reach in state bits

Suppose exact visited needs `c` bits per reachable state and available aggregate
memory is `B`. At most approximately `B/c` states fit. If the family can expose
all `2^n` encodings, the capacity condition is

```text
c 2^n <= B.
```

Multiplying memory by a factor `p` changes the largest feasible state-bit
parameter only by

```text
Delta n <= log_2(p).
```

In particular, doubling aggregate GPU memory buys about one additional bit in
this worst-case state-universe model. This is a capacity scaling identity, not
a runtime prediction: reachable fractions, compression, invalid encodings,
symmetry, and relations may change actual occupancy.

## 9. GPU and multi-GPU consequences

Parallel hardware can materially improve constants, throughput, and aggregate
capacity. It cannot turn exponential expanded output into polynomial output in
the compact description size.

Evidence should distinguish:

- state-description bits and valid-state count;
- reachable states versus all encodings;
- move occurrences versus distinct successors;
- per-state representation bytes and total visited capacity;
- throughput scaling versus capacity scaling;
- exhaustive output versus a decision or bounded-radius query;
- structure-specific reductions versus generic implicit BFS.

More GPUs may make a larger exact prefix observable. That is valuable without
being evidence that the asymptotic state explosion disappeared.

## 10. Rejected shortcuts

- **Compact input means a compact reachable set.** A small circuit can describe
  exponentially many vertices.
- **Constant branching makes reachability polynomial in state bits.** Depth and
  number of reachable states can still be exponential.
- **`O(V+E)` means polynomial in the puzzle description.** `V` and `E` may be
  exponential functions of that description.
- **Polynomial-space reachability gives polynomial-time BFS layers.** It trades
  stored frontiers for recomputation and need not produce shortest distances.
- **A symbolic representation always avoids explosion.** Some sets compress;
  worst-case symbolic size and operations can still blow up.
- **Linear multi-GPU memory scaling defeats an exponential family.** It extends
  the feasible state-bit parameter only logarithmically in device count.

## 11. Evidence checklist

1. Compact description length and per-state encoding length.
2. Maximum universe, valid states, and actually reachable states.
3. Adjacency-query versus direct-successor access.
4. Generator/move count and cost of one exact successor.
5. Decision, distance, path, ball, or full-component output.
6. Explicit, symbolic, or recomputation-based visited semantics.
7. Complexity measured in expanded graph size or compact input size.
8. Runtime, throughput, and capacity scaling reported separately.

## Sources

- H. Galperin and A. Wigderson, [*Succinct Representations of
  Graphs*](https://www.math.ias.edu/avi/node/751), Information and Control 56(3)
  (1983), 183-198. Establishes the succinct-graph representation viewpoint and
  the complexity jump caused by polylogarithmic-size descriptions.
- C. H. Papadimitriou and M. Yannakakis, [*A Note on Succinct Representations
  of Graphs*](https://doi.org/10.1016/S0019-9958(86)80009-2), Information and
  Control 71(3) (1986), 181-185. Complexity escalation for succinct versions of
  graph properties.
- W. J. Savitch, [*Relationships between nondeterministic and deterministic
  tape complexities*](https://doi.org/10.1016/S0022-0000(70)80006-X), Journal
  of Computer and System Sciences 4(2) (1970), 177-192. Recursive reachability
  and the deterministic-space simulation underlying the PSPACE upper bound.
- Notes 06, 09, 15, 23, 25, 28, 29, 35, 36, 45, 47, 54, and 93 provide
  implicit-state, termination, external-memory, recomputation, fixed-point,
  identity, complexity-accounting, growth, representation, GPU, work-span,
  mental-model, and Cayley-metric context.

## Takeaway

BFS can be linear in a graph that is exponentially larger than its implicit
description. Succinct reachability exposes the distinction sharply: exact
reachability is PSPACE-complete by compact input size, while full BFS output may
itself be exponential. GPU parallelism expands the tractable envelope, but
only algebraic structure, restricted families, bounded queries, or different
output contracts can change the underlying description-to-state-space gap.

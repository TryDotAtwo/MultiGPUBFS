# Eventual walk lengths, primitivity, and BFS first arrival

BFS records the first length at which a vertex is reachable. Note 85 records
the residue class of every possible walk length in a strongly connected
digraph. Between those facts lies a third quantity: the threshold after which
every sufficiently large compatible length actually occurs.

This distinction separates shortest-path frontiers from exact-length walk
frontiers and explains the exponent of a primitive adjacency matrix.

## 1. Eventual completeness inside one residue

Let `G` be a finite strongly connected digraph with period `p`. Fix vertices
`u,v` and one directed path from `u` to `v` of length `a`.

At `v`, the set of positive closed-walk lengths has GCD `p`. A finite subset of
those lengths already has GCD `p`: repeatedly taking GCDs can strictly decrease
only finitely many times before reaching the GCD of the whole set.

Divide that finite set by `p`. The resulting positive integers have GCD one.
Their additive numerical semigroup contains every sufficiently large integer.
Concatenating the corresponding closed walks at `v` after the fixed `u-v` path
therefore yields:

```text
there exists N_(u,v) such that
k >= N_(u,v) and k = a mod p
    implies a u-to-v walk of length k exists.
```

Note 85 proved that no other residue can occur. Thus exact-length reachability
for one ordered pair is eventually complete in exactly one residue modulo `p`.

Because a finite graph has finitely many ordered pairs, one may take a global
maximum threshold, although the smallest useful threshold can differ by pair.

## 2. Aperiodicity and primitive adjacency

When `p=1`, every ordered pair admits a walk of every sufficiently large
length. For Boolean adjacency matrix `A`, this means that there exists
`gamma` such that

```text
A^k has positive support in every entry for all k >= gamma.
```

The graph/matrix is called **primitive**. The least `gamma` with `A^gamma>0`
is its exponent. Strong connectivity corresponds to irreducibility;
aperiodicity is the additional condition that makes it primitive.

For an `n`-vertex primitive digraph, Wielandt's classical sharp general bound
is

```text
gamma <= (n-1)^2+1.
```

This quadratic worst-case bound concerns exact-length all-pairs walks, not BFS
diameter or the number of iterations needed for first discovery.

## 3. Periodic graphs never have one fully positive power

If `p>1`, each pair permits only one residue. Therefore no single `A^k` can be
positive in every entry: arcs advance cyclic class by one, so length `k` maps
each source class only to class `+k mod p`.

After ordering vertices by cyclic class, powers have an eventual block-cyclic
support pattern. Within the compatible source/target blocks, support becomes
complete after a threshold; incompatible blocks remain zero forever.

For directed `C_N`, the pattern is exact from the start: each source has one
endpoint at each length, determined by `k mod N`. Strong connectivity coexists
with maximally sparse exact-length support.

## 4. BFS depth and period do not determine the exponent

Consider two three-vertex digraphs rooted at `0`.

1. The complete directed graph with every loop has root distances `(0,1,1)`,
   period one, and exponent one.
2. The graph with arcs

   ```text
   0->1, 0->2, 1->0, 2->0, 1->2
   ```

   has the same root distances `(0,1,1)`. It is strongly connected and has
   cycles of lengths two and three, hence period one, but its first adjacency
   power is not positive because several arcs/loops are absent. Its exponent is
   greater than one.

Thus first-arrival distances plus period constrain exact-length support but do
not determine its transient threshold.

## 5. Two frontier recurrences answer different questions

Let `Post(X)` be the set of out-neighbors of `X`.

### Exact-length walk support

```text
W_0 = {s}
W_(k+1) = Post(W_k).
```

`W_k` contains endpoints of walks of exactly length `k`. Vertices may and must
reappear. In a finite strongly connected graph this sequence is eventually
periodic in support; it generally does not become empty.

### BFS first-discovery frontier

```text
F_0 = {s}
B_k = union_(i<=k) F_i
F_(k+1) = Post(F_k) \ B_k.
```

`F_k` contains vertices whose shortest distance is exactly `k`. Visited
subtraction intentionally removes later walks. On a finite reachable graph it
eventually becomes empty and certifies exhaustion.

Applying visited subtraction to an exact-length question destroys valid later
support. Omitting visited subtraction from BFS turns it into walk propagation
and destroys the ordinary exhaustion/linear-work interpretation.

## 6. Immediate padding versus eventual padding

In an inverse-closed Cayley or undirected graph, a path word of length `d` can
always be padded by `s s^-1`, giving walks of lengths

```text
d, d+2, d+4, ...
```

immediately. If the graph is bipartite, these are exactly the permitted parity
lengths.

In a non-bipartite connected graph, period is one because length-two
backtracks coexist with an odd closed walk. All sufficiently large lengths are
then possible, but small gaps may remain. Period one is an eventual statement,
not permission to add one step to every path immediately.

## 7. Cayley word-length spectrum

In a finite strongly connected positive-alphabet Cayley digraph, choose element
`g`. The lengths of positive words evaluating to `g`:

- begin at the BFS distance/shortest positive word length;
- all share residue `chi(g)` modulo period `p`;
- contain every sufficiently large integer in that residue.

These longer words are walks with inserted identity relations. They are not
new states, new shortest paths, or necessarily reduced words. A count of
exact-length words can continue growing long after the state BFS exhausted the
finite group.

For Schreier states, the analogous spectrum includes words differing by the
stabilizer. It must not be identified with the group-element word spectrum
without the action derivation.

## 8. Distributed and GPU semantics

An exact-length propagation kernel may look superficially like a BFS frontier
kernel, but its state contract differs:

- revisiting a vertex at a later step is meaningful;
- a permanent visited bitmap is invalid for exact-length support;
- quiescence is not the normal termination condition in a strong component;
- support may oscillate among `p` cyclic classes;
- Boolean support, walk multiplicity, and numeric matrix powers remain
  different outputs.

For multi-GPU execution, termination should be a requested length bound,
detected support period under a proved finite-state contract, or another
explicit criterion. A global nonempty frontier forever is not evidence of a
failed BFS termination detector when the intended object is `W_k` rather than
`F_k`.

This is a semantic warning, not a kernel design recommendation.

## 9. Evidence checklist

1. First-arrival frontier or exact-length walk frontier.
2. Boolean support, path/walk count, or weighted matrix value.
3. Strong component and its period/cyclic classes.
4. Pair residue and observed transient gaps.
5. Claimed pair threshold, global exponent, or only eventual existence.
6. Whether inverse padding is available in the actual directed alphabet.
7. State equality versus generator-word identity and stabilizer semantics.
8. Bounded-step versus quiescence termination contract.

## Sources

- H. Wielandt,
  [*Unzerlegbare, nicht negative Matrizen*](https://doi.org/10.1007/BF02230720),
  Mathematische Zeitschrift 52 (1950), 642-648. Establishes the classical sharp
  primitive-exponent bound.
- E. Seneta, *Non-negative Matrices and Markov Chains*, sections on primitive
  matrices, exponent, period, and Boolean support of powers.
- A. L. Dulmage and N. S. Mendelsohn,
  [*The Exponent of a Primitive Matrix*](https://doi.org/10.4153/CMB-1962-021-1),
  Canadian Mathematical Bulletin 5 (1962), 241-244. Develops graph-theoretic
  refinements of the exponent bound.
- Notes 25, 33, 39, 64, 67, and 85 supply the delta-frontier, adjacency-power,
  word/state, record-multiplicity, parity, and directed-period distinctions
  used here.

## Takeaway

BFS supplies the first possible walk length. Period supplies its only possible
residue class. Finite strong connectivity then guarantees that every
sufficiently large length in that residue occurs, but the transient threshold
is extra information. In the aperiodic case this becomes matrix primitivity
and its exponent. Exact-length propagation revisits states by design and must
not inherit BFS visited or quiescence semantics.

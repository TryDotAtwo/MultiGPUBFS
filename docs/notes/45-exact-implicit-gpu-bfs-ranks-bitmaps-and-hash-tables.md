# Exact implicit GPU BFS: ranks, bitmaps, and hash tables

Explicit-graph GPU BFS begins with vertex IDs. Implicit state-space search must
first earn such an ID or retain enough state to resolve equality. This note
separates three ideas that are often compressed into the phrase "GPU BFS":

1. a proved rank over a finite state universe;
2. one-/two-bit layer bookkeeping indexed by that rank;
3. an exact collision-resolving table for states generated on the fly.

This is a semantic source synthesis, not an implementation proposal.

## The missing operation is exact naming

Let `U` be the declared finite set of semantic states. A dense rank must satisfy

```text
rank : U -> {0, ..., N-1}
rank(x) = rank(y)  iff  x = y in the declared state semantics.
```

Surjectivity onto all `N` indices is useful for compact storage and enumeration,
but injectivity is the part needed for exact membership. A rank is stronger than
an ordinary hash because equality of indices is already a proof of equality.

For a puzzle state, the proof has to include every relevant component:
permutation, orientation, parity or legality constraints, quotient convention,
and any history state that changes legal successors. Ranking only the visible
permutation does not name a vertex of a history-sensitive product graph.

A reversible minimal perfect hash constructed for a completely specified state
set can serve this role. A static minimal perfect hash over some previously
stored subset cannot: an input outside that subset may still map to an occupied
slot. Note 28 gives the general membership distinction.

## What rank/unrank buys, and what it does not

An exact dense rank makes several explicit-BFS mechanisms available:

- permanent visited becomes an `N`-bit array;
- a frontier can be a bitmap, compact list of ranks, or both;
- atomic test-and-set on one bit can linearize the first discovery of a state;
- a distance table can be indexed directly;
- if `unrank(i)` is total and affordable, the whole universe can be enumerated,
  making pull-style predicates meaningful in principle.

None of these statements says that the representation is fast. The relevant
costs include:

```text
rank(state)
unrank(index)
apply_move(state) or update_rank(index, move)
validity/canonicalization
bitmap contention and memory locality.
```

An `N`-bit visited array may also be impossible even when each individual rank
fits in 64 bits. Dense addressability and feasible allocation are different
properties.

Most importantly, rankability does not materialize adjacency. A ranked implicit
graph still generates transitions, while CSR BFS loads stored endpoint IDs.

## At least three meanings of "one-bit BFS"

The literature and informal discussion use the label for different storage
contracts. They must not be merged.

### Permanent one-bit visited plus a separate frontier

If `rank` is exact, one permanent bit per state answers whether the state was
ever discovered. The current frontier still exists elsewhere: as a list, a
second bitmap, a level range, or another exact representation. This is ordinary
exact BFS with compact visited storage; the one bit does not encode the whole
algorithm.

### One bit as a reachable-set output

If the requested result is only the final reachable subset of a finite universe,
one bit per state can encode the output after exhaustive traversal. It cannot by
itself recover distances, parents, layer boundaries, shortest-path counts, or a
canonical path.

### Recycled layer/parity bits

Some state-space BFS schemes reuse a small number of bits across levels instead
of keeping a permanent visited bit. Correctness then depends on a proof that an
old state capable of reappearing cannot be confused with a new state under the
chosen move schedule.

For an undirected graph, every edge satisfies

```text
abs(dist(s,u) - dist(s,v)) <= 1.
```

This fact does justify one precise rolling-window result under a strict complete
level schedule: while expanding `F_d`, every previously reached neighbor lies
in `F_(d-1)` or `F_d`, so layers through `F_(d-2)` are no longer needed for
scalar duplicate rejection. Cycles do not invalidate that undirected distance
inequality. It does **not** justify an arbitrary bit-recycling rule: the previous,
current, and partially built next layers still need distinct exact meanings,
and directed edges can jump back by more than one layer. Bipartiteness removes
same-level edges but does not remove the other schedule, publication, and output
obligations. Note 181 gives the full forgetting contract and directed/Cayley
generalization.
Move-alternation or operator-pruning rules change which word transitions are
generated and need a separate coverage proof over semantic states.

Therefore every one-/two-bit claim must state:

- what each bit pattern means at the start and end of a level;
- which arrays or queues hold the current and next frontier;
- whether bits are permanent or recycled;
- which graph property bounds possible rediscoveries;
- whether operator pruning preserves at least one shortest representative of
  every state;
- which outputs survive the compression.

"One bit per state" is a byte count, not a complete correctness contract.

## Exact on-the-fly hashing is a different route

GPUexplore and related explicit-state model-checking work generate state vectors
on the fly and store explored states in a GPU hash table. This is closer to a
wide implicit Cayley state than CSR traversal, but it does not remove equality.

An exact table needs:

```text
bucket selection by hash
+ retained exact state key (or proved injective encoding)
+ complete collision resolution and equality comparison
+ linearizable concurrent insertion
+ explicit full-table/probe-limit failure.
```

The hash-table performance study by Cassee and Wijs compares GPU table designs
inside this state-space-exploration setting. Its useful transfer is the list of
possible bottlenecks: key width, probes, load factor, insertion contention,
table clearing/reuse, and the distribution of generated states. A reported
table throughput or winning load factor remains tied to the tested keys,
transition systems, hardware, and failure policy.

A bare fingerprint-only table is not this contract. If unequal semantic states
with equal fingerprints are treated as one, BFS can lose reachability. If a
fixed-capacity table silently calls insertion failure "already seen", capacity
pressure creates the same false-positive error.

## Direction optimization in a ranked puzzle abstraction

The external-memory direction-optimizing BFS work demonstrates that enormous
Rubik's Cube heuristic tables can be built when the chosen abstraction has a
finite, exactly indexed state universe and the requested artifact is a bounded
distance table. This closes one gap left by explicit CSR papers: adjacency can
be implicit while the vertex universe remains enumerable.

It does not show that pull is available for every implicit Cayley graph. Pull
requires all of the following:

- enumerate candidate unvisited states or ranks;
- reconstruct enough state to inspect predecessors;
- query frontier membership exactly;
- preserve the declared directed/inverse transition semantics;
- make the scan cheaper than pushing the active frontier in the current regime.

Inverse generators answer `predecessors(x)` for an already named `x`. They do
not enumerate all `x` that have not yet been reached.

## Comparison with the inspected CayleyPy path

The inspected CayleyPy outer search stores wide states and uses bare `Hash128`
identity in the traced beam and K1 paths. No proof seen in that audit makes this
fingerprint a dense injective state rank, and the outer search also prunes to a
bounded learned-score beam.

Consequently it cannot inherit exact one-bit BFS claims merely by treating
`Hash128` as an index:

- a 128-bit value is not a feasible dense bitmap address space;
- collision probability is not an injectivity proof;
- hash equality without retained-state collision resolution is not semantic
  equality;
- beam pruning violates complete-frontier retention independently of identity;
- a K1 exact-radius interpretation remains conditional on collision-free state
  identity and complete construction.

This does not condemn fingerprints or beams. It says their proper contracts are
probabilistic identity and heuristic search unless stronger evidence is added.

## A representation decision table

| Representation | Exactness obligation | Natural output | Main hidden cost |
|---|---|---|---|
| Dense rank + permanent visited bit | prove injectivity on every queried state | reachability; distances with extra storage | rank/update and `N`-bit capacity |
| Dense rank + recycled layer bits | above plus layer-reuse/coverage proof | contract-specific | rescans, phase meaning, lost metadata |
| Collision-resolving state table | retain exact key and complete probing | on-the-fly exact exploration | probes, wide-key traffic, capacity |
| Fingerprint-only table | probabilistic unless injective | approximate search | false-positive pruning |
| Static minimal perfect hash | exact only on its declared key set | lookup of that fixed set | nonmember recognition/rebuild |

## Measurements that would answer conceptual questions

If a future explicit experiment is requested, the minimum accounting should
separate:

```text
semantic-state bytes
rank/unrank or hash/equality time
generated, valid, duplicate, unique, and accepted states
bitmap words touched and contested
hash probes and load factor
overflow or incomplete-level status
frontier versus visited versus output bytes.
```

These are questions for a measurement contract, not a request to implement the
system now.

## Sources and evidence limits

- Stefan Edelkamp and Damian Sulewski,
  [Perfect Hashing for State Space Exploration on the GPU](https://ojs.aaai.org/index.php/ICAPS/article/view/13414),
  motivates reversible perfect hashing/ranking for dense GPU state-space
  representations.
- [Parallel State Space Search on the GPU](https://webdocs.cs.ualberta.ca/~nathanst/sara/papers/socs09_submission_24.pdf)
  discusses GPU exploration of permutation puzzles and one-/two-bit state-space
  representations.
- Thomas Cassee and Anton Wijs,
  [Analysing the Performance of GPU Hash Tables for State Space Exploration](https://arxiv.org/abs/1712.09494),
  evaluates GPU hash-table choices in on-the-fly explicit-state exploration.
- [Direction-Optimizing Breadth-First Search with External Memory Storage](https://www.ijcai.org/proceedings/2019/175)
  applies direction optimization to a very large Rubik's Cube heuristic-table
  construction.

The official pages and indexed metadata were available during this pass, but
the full PDFs could not be freshly downloaded in the current Windows session:
Schannel returned `SEC_E_NO_CREDENTIALS`, and Docker was not available as a
fallback. Therefore this note deliberately avoids attributing detailed bit
state machines, numeric speedups, thresholds, or table load factors that could
not be checked line by line. The algebraic qualifications above are derived
from the exact BFS contract and are stronger than an unverified paper-specific
recipe.

## Current conclusions

1. A proved rank can turn an implicit finite state universe into exact dense
   IDs, but does not make transitions free or the bitmap feasible.
2. One permanent visited bit is compatible with exact BFS only together with an
   independently represented frontier and any requested output metadata.
3. Recycled one-/two-bit schemes are separate algorithms whose layer and move
   invariants must be stated and proved.
4. Exact GPU hashing retains and compares semantic keys; fingerprint-only
   equality remains probabilistic.
5. Direction optimization becomes conceivable for an implicit puzzle only
   when its unvisited universe is exactly and affordably enumerable.
6. CayleyPy's inspected `Hash128` beam path does not currently satisfy the
   dense-rank or collision-resolving-table premises of exact bitmapped BFS.

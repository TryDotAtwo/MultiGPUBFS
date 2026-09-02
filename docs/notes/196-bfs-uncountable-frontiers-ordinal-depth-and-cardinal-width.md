# BFS with uncountable frontiers: ordinal depth and cardinal width

## Question

If one BFS layer is uncountable, do metric layers and the least fixed point stop
making sense, or does only the usual enumeration algorithm fail?

The mathematics survives. Explicit exhaustive enumeration does not.

## 1. Distance uses finite paths, not a countability premise

For an arbitrary directed graph `G=(V,E)` and source set `S`, define

```text
d(S,v) = minimum finite path length from some s in S to v,
F_d    = {v : d(S,v)=d},
B_d    = union_(i<=d) F_i.
```

These sets are meaningful whether `V`, `S`, degrees, and layers are finite,
countable, or uncountable. Every reached vertex still has a natural-number
distance because graph reachability is defined by a finite path.

The successor operator

```text
T(X) = S union Post(X)
```

preserves nonempty unions, because `Post` preserves arbitrary unions. For a
nonempty source set it does not preserve the empty union: `T(empty)=S`, whereas
the union of no sets is empty. The nonempty increasing chain of finite-depth
balls is enough to obtain its least reachable fixed point:

```text
R = union_(d<omega) B_d.
```

An uncountable reached set does not require a transfinite path length. It can
appear inside one finite stage.

## 2. Ordinal depth and cardinal width are independent

Two different notions of “large” are easy to conflate:

- **ordinal/stage depth:** how many layer iterations are needed;
- **cardinal width:** how many states occur in one layer or ball.

A graph can have diameter one and an uncountable `F_1`. Conversely, it can have
countably many singleton layers and require every finite depth before the
stage-`omega` union is complete.

Thus:

```text
small depth does not imply enumerable width,
uncountable width does not imply transfinite distance.
```

## 3. Exact uncountable star

Let the vertices be

```text
{s} union {x_r : r in R},
```

with one directed edge `s->x_r` for every real number `r`. Then

```text
F_0 = {s},
F_1 = {x_r : r in R},
F_d = empty for d>=2.
```

The entire metric partition is exact and finishes abstractly after one
successor application. But there is no sequence indexed by natural numbers that
lists every member of `F_1`.

This is stronger than the countably infinite branching issue in note 183.
Fair dovetailing can schedule countably many successor enumerators or indices.
No fairness rule over a sequential countable event stream can enumerate an
uncountable set.

## 4. The machine impossibility is cardinal, not performance-based

A conventional program emits a finite or countable sequence of records. Even
over an unbounded countable execution, it can explicitly materialize at most
countably many distinct states. The same remains true for finitely many CPU
threads, finitely many GPUs, or countably many workers each producing countably
many records.

The union of countably many countable record sets is countable. Therefore an
explicit exhaustive frontier contract is impossible for an uncountable layer,
independently of throughput, memory size, or load balancing. More GPUs cannot
promote a countable event history to an uncountable explicit output.

## 5. Symbolic BFS is a different representation contract

An uncountable frontier can sometimes be represented intensionally by an
interval, formula, constraint, predicate, decision diagram, or another finite
descriptor. A symbolic engine may compute an exact set without enumerating its
members.

That changes the required operations and output:

- equality becomes equivalence of represented sets or predicates;
- `Post` must transform descriptors exactly;
- difference from visited must remain representable;
- emptiness and target membership need decision procedures;
- one witness may require constructive extraction;
- cardinality may be infinite or unavailable rather than an integer counter.

Symbolic exactness is not approximate explicit BFS. It is an exact realization
of the same set recurrence in a representation closed under the needed
operations.

## 6. Positive target search versus layer enumeration

Given a concrete target `t`, proving `t in F_1` may be easy if adjacency
`E(s,t)` is directly decidable. That does not enumerate `F_1`. Conversely, a
symbolic descriptor may prove nonmembership without producing every alternative
successor.

Therefore distinguish:

```text
one target membership query
one witness path
complete explicit frontier
complete symbolic frontier descriptor
global unreachable certificate.
```

These outputs can have radically different computability even at depth one.

## 7. Countable generator alphabets rule this out for one orbit

Let a group or transformation system have a finite or countable generator
alphabet `S`. The set of finite words

```text
S* = union_(d<omega) S^d
```

is countable. Acting on one central state therefore reaches at most a countable
orbit. If `S` is finite, every fixed-depth word set `S^d` is finite, so each
semantic BFS layer is finite even when the orbit is infinite.

This is why ordinary finitely generated Cayley graphs and CayleyPy's finite
generator lists do not encounter uncountable frontiers. An ambient group or
state space may be uncountable while the declared finite-word orbit remains
countable.

## 8. Uncountable Cayley calibration

Take the additive group `(R,+)` with generator set

```text
S = R minus {0}.
```

From identity `0`, every nonzero real is one generator step away. The abstract
Cayley graph has diameter one and an uncountable sphere. But the symbolic rule
“add any nonzero real” is not an enumerable generator-list interface and does
not provide a finite GPU worklist of generator occurrences.

The word metric is mathematically valid; ordinary explicit BFS is the wrong
execution model for materializing the whole layer.

## 9. Visited and duplicate meaning

An explicit hash table cannot contain an uncountable `visited` set. A symbolic
descriptor may, but exact duplicate rejection then means deciding membership in
a represented set, not atomically inserting one integer key.

An uncountable candidate set minus an uncountable visited set can be empty,
countable, or uncountable. Reporting both as merely “infinite” supplies no work
or frontier-size equation. Finite duplicate counters and ratios lose their
ordinary meaning without a declared measure or cardinal contract.

## 10. GPU and multi-GPU boundary

For finite-generator implicit puzzle BFS, GPU questions remain record based:
generator applications, hashes, bytes, routing, and synchronization.

For an uncountably branching symbolic system, acceleration would instead target
descriptor operations: symbolic image, constraint propagation, set difference,
emptiness, membership, or witness extraction. Explicit frontier throughput is
not a meaningful metric until the representation contract is declared.

A report must distinguish explicit record count, mathematical cardinality,
descriptor bytes, symbolic-piece count, a declared measure/volume, and the cost
of symbolic decision operations.

## 11. Rejected implications

- An uncountable reached set needs transfinite graph distance.
- Stage-`omega` closure implies every stage is explicitly enumerable.
- Fair dovetailing handles arbitrary infinite branching.
- Countably many GPUs could enumerate an uncountable layer given enough time.
- An uncountable ambient state space implies an uncountable finite-generator
  orbit.
- A symbolic frontier is merely a compressed explicit list.
- “Infinite frontier size” is a sufficient performance coordinate.

## 12. Evidence boundary

The star and `(R,+)` examples are exact set/cardinality arguments. The machine
limit uses only that explicit program events form a countable sequence. No
claim is made about a particular symbolic package, representation language, or
hardware implementation, and no experiment could establish the universal
cardinality statement by enumeration.

## Compact conclusion

BFS distance remains natural-number valued on arbitrary graphs because reached
vertices have finite witness paths. The least reachable fixed point still
appears at stage `omega`, even when a finite layer is uncountable. What fails is
explicit exhaustive enumeration: a countable machine history cannot list an
uncountable frontier. Exact computation then requires a symbolic set contract,
not more explicit-BFS throughput.

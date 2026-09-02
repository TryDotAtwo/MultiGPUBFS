# Exact BFS identity: ranks, hashes, fingerprints, and approximate membership

BFS correctness is stated over semantic vertices.  A storage representation is
valid only if its membership decision implements the same equality relation.

For the accumulated ball `B`, the exact visited contract is

```text
Seen(x)  iff  exists y in B: y equivalent_to x.
```

The equivalence may be literal domain-state equality, a proved canonical
representation, or an explicitly requested quotient/product identity.  It is
never merely "the short codes happen to match" unless injectivity has been
proved over the relevant domain.

## Why the two error directions differ

### False positive

`Seen(x)=true` for a genuinely new state deletes `x` from the BFS graph.  That
can remove the only entrance to a reachable region.

Minimal witness:

```text
s -> a
s -> b -> t
```

Let `a` and `b` have the same stored hash.  If `a` is accepted first and a
hash-only visited set rejects `b`, exact BFS reports `t` unreachable.  This is
not merely a parent-choice difference; completeness is gone.

### False negative

`Seen(x)=false` for an already reached state admits duplicate work.  For the
narrow output "reachable set" this may be recoverable if a later exact stage
merges it and no capacity is lost.  It is not universally harmless:

- a longer rediscovery can corrupt a first-wins distance or parent;
- duplicate expansions can overflow a bounded frontier;
- path counts and all-parent DAGs can be overcounted;
- repeated messages can prevent termination;
- duplicate output records can violate the declared result directly.

Thus false positives are immediately lossy, while false negatives require an
explicit downstream recovery proof.

## Bijective or injective rank

Suppose a state encoder satisfies

```text
rank(x)=rank(y)  iff  x equivalent_to y
```

over every reachable valid state, with values in a bounded integer range.  A
dense bitmap indexed by `rank` is then an exact visited set.

Strictly, injectivity on semantic states is sufficient for membership.  A full
bijective `rank/unrank` pair is additionally useful when every index must recover
a state, when the entire universe is enumerated, or when successors are updated
in rank space.  Holes for invalid/unreachable encodings waste capacity but do
not by themselves break correctness.

Proof obligations include:

- the encoded fields contain every part of semantic identity;
- no two valid states share a rank;
- arithmetic cannot overflow or truncate high bits;
- canonical action/parity/orientation constraints match the declared domain;
- graph versions or path-history components are included when they affect the
  future transition language.

A compact integer is not a rank merely because no collision appeared in tested
examples.

## Ordinary collision-resolving hash tables can be exact

A hash table uses

```text
bucket = h(x)
```

to narrow the search, then compares the stored semantic key against `x` inside
the collision chain/probe sequence.  Two unequal states may share a bucket
without being treated as one vertex.

Exactness therefore depends on all of the following:

- the complete key or a proved injective encoding is retained;
- probing examines every location required by the table invariant;
- equality is checked, not inferred from hash equality;
- concurrent insertion linearizes duplicate claims correctly;
- deletion/tombstone rules do not break probe reachability;
- full-table or probe-limit exhaustion is reported as overflow, not `seen`;
- resizing/migration cannot temporarily hide a previously inserted key.

"Hash-based visited" says nothing by itself about exactness.  Collision
resolution and capacity behavior are the decisive parts.

## Fingerprints are probabilistic unless injective

A `b`-bit fingerprint maps a larger state universe into at most `2^b` values.
If the relevant universe has more elements, collisions exist by the pigeonhole
principle.  Randomization can make accidental collision unlikely under a model;
it cannot turn a many-to-one mapping into a proof of equality.

This applies to Zobrist keys, truncated cryptographic hashes, checksums, and
multiple independent fingerprints.  Two 64-bit fingerprints may make an
undetected mismatch extremely unlikely, but still provide probabilistic
validation rather than an exact set certificate.

Fingerprints have several safe roles:

- choose a bucket or owner before exact comparison;
- reject equality quickly when fingerprints differ;
- detect likely artifact corruption;
- compare repeated runs as a compact regression signal;
- trigger a slower exact audit on mismatch or collision candidates.

They are unsafe as the final reason to discard a BFS state when exact
completeness is claimed.

This distinction also qualifies existing experiment fingerprints in this
repository: they are useful cross-run checks paired with exact expected counts
and bounded oracles.  They are not a general proof that two arbitrarily large
frontier sets are identical.

## Perfect hashing is not automatically a state rank

A perfect hash function is collision-free on a particular key set `K`.  A
minimal perfect hash maps that fixed set into exactly `0..|K|-1`.  Important
questions are hidden in the phrase "on `K`":

- Was `K` known before the search?
- Does `K` equal the entire valid state universe or only a static stored subset?
- What happens when the function is queried with `x notin K`?
- Can new BFS discoveries be inserted without rebuilding/changing the map?
- Is the state key still checked after the hash lookup?

A static minimal perfect hash for already known keys need not recognize
nonmembers; an unseen state can still map to some valid slot.  It becomes an
exact dense BFS rank only when its domain membership and injectivity cover every
state the search may query, or when a separate exact key check resolves
nonmembers.

Perfect hashing is a collision-management construction, not a magical proof
that an unknown implicit state space has been enumerated.

## Bloom filters

An ideal Bloom filter for an inserted set has:

- no false negatives for correctly inserted items;
- possible false positives for nonmembers.

Therefore these compositions differ:

```text
Bloom says negative -> definitely not in backing set
Bloom says positive -> check exact backing set
```

is exact, while

```text
Bloom says positive -> discard candidate as visited
```

is approximate and can reproduce the `s,a,b,t` completeness failure.

The theoretical no-false-negative statement also assumes correct atomic updates
and stable storage.  Lost concurrent bit updates, stale replicas, corruption,
or resetting the filter can introduce implementation false negatives even
though they are not part of the abstract Bloom-filter model.

An approximate visited policy can be a legitimate separately named search, but
its failure probability, bias, and output contract must not be reported as
exact BFS.

## Hashing for multi-GPU ownership

A distributed BFS may define

```text
owner(x) = h(canonical_key(x)) mod P.
```

Hash collision is harmless for routing: different states merely share an
owner.  Correctness requires a different property:

```text
x equivalent_to y  implies  owner(x)=owner(y).
```

All representations of one semantic state must converge on the same
authoritative visited decision.  The owner must then perform exact equality;
hash equality alone is insufficient.

Conversely, using different canonical bytes, seeds, graph versions, or owner
counts concurrently can route equal states to different authorities.  Each may
accept the state independently.  A repartition therefore needs an epoch and
migration/forwarding contract, not just a new modulus.

Ownership hash quality affects balance and communication, but that is separate
from semantic equality.  A badly skewed hash can remain correct; an inconsistent
hash can be fast and wrong.

## Canonicalization is upstream of hashing

If several physical encodings denote one requested vertex, first define and
prove a canonical key or an equality procedure.  Hashing that key preserves its
semantics only as a bucket function.

Two opposite canonicalization bugs exist:

- **under-canonicalization:** equal states retain different keys, causing false
  negatives/duplicate authorities;
- **over-canonicalization:** distinct requested states share a key, creating a
  quotient and false-positive visited decisions.

The second can look like an excellent duplicate rate while silently shortening
distances or deleting fixed-target solutions.  Note 17 supplies the additional
automorphism, target-orbit, and path-lifting obligations for intentional
quotients.

## Hash equality and artifact fingerprints answer different questions

It is useful to separate four roles:

| Role | Required property |
|---|---|
| visited membership | exact semantic equality or collision resolution |
| owner routing | equal states always choose one authority |
| table addressing | distribution plus complete probing; collisions allowed |
| result fingerprint | compact probabilistic evidence unless independently made injective |

Reusing one numeric hash for all four roles does not make their guarantees
interchangeable.

## Validation hierarchy

Increasingly strong evidence for a visited representation includes:

1. unit examples with known equal and unequal states;
2. deliberately forced hash collisions;
3. exact comparison against a full-state oracle on an exhaustible graph;
4. full per-level frontier-set equality, not only counts;
5. replay of every retained parent path;
6. injectivity proof for a rank over the declared valid domain;
7. explicit capacity, overflow, concurrency, and repartition tests.

Equal frontier counts are weak: one missing state and one extra state cancel.
Equal fingerprints strengthen regression evidence but remain probabilistic.
Exact sorted key comparison or a proved bijection supplies deterministic set
equality within its stated scope.

Path replay alone is also insufficient.  It proves that returned paths exist,
not that a false-positive visited decision did not prune some other reachable
state.

## Cayley and puzzle implications

- A concrete permutation/orientation tuple can supply exact equality even when
  no compact rank is known.
- A Lehmer-style permutation rank is exact only after every additional puzzle
  field and legality constraint is incorporated.
- A group word hash is not group-element equality; relations make different
  words equal, while hash collisions can make unequal elements look equal.
- A Zobrist-style incrementally updated key remains a fingerprint unless the
  reachable domain injectivity is proved or the full state is checked.
- A symmetry-normalized key changes the vertex contract unless note 17's
  quotient/lifting proof applies.
- A parent move may permit reconstructing a state from an exact checkpoint, but
  it does not retroactively make a hash-only visited decision exact.

## Audit checklist

1. What is the semantic equality relation for one vertex?
2. Which exact bytes/fields determine it?
3. Is the compact key injective, collision-resolved, or probabilistic?
4. Can an invalid/nonmember input map to an occupied perfect-hash slot?
5. Does a Bloom positive cause rejection or an exact secondary lookup?
6. What happens at table capacity or maximum probe length?
7. Are concurrent claims linearizable under the promised output contract?
8. Do all equal states route to the same owner and epoch?
9. Are fingerprints being presented as regression evidence or exact proof?
10. Which adversarial collision and full-state oracle checks were run?

## Sources

- Burton Bloom,
  [Space/time trade-offs in hash coding with allowable errors](https://doi.org/10.1145/362686.362692),
  explicitly develops membership coding with an allowable false-positive
  frequency.
- Albert Zobrist,
  [A New Hashing Method with Application for Game Playing](https://research.cs.wisc.edu/techreports/1970/TR88.pdf),
  notes that two items may share a hash address and that stored keys or derived
  quantities are needed to detect/resolve clashes.
- Fredman, Komlos, and Szemeredi,
  [Storing a Sparse Table with O(1) Worst Case Access Time](https://www.cs.umd.edu/~gasarch/BLOGPAPERS/FKS.pdf),
  gives a perfect-hashing construction for a specified static set; that scope is
  why perfect hashing must not be confused with a universal implicit-state
  rank.
- The false-positive graph witness and distributed ownership implications are
  derived directly from the exact visited recurrence.

## Current conclusion

Exact BFS needs exact state identity at the final membership decision.  Hashes,
fingerprints, Bloom filters, and owner functions can safely reduce where or how
often exact comparison occurs, but probability and good distribution do not
replace equality.  A bijective rank is powerful because it proves the missing
implication; a collision-resolving table is exact because it checks it.

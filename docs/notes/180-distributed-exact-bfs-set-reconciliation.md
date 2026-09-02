# Distributed exact BFS set reconciliation

## 1. The validation question

Suppose two BFS executions claim the same frontier `F_d`, visited ball `B_d`,
distance map, or richer output while those objects remain sharded. Matching
counts and fingerprints are useful regression evidence, but exact equality asks:

```text
is the exact semantic symmetric difference empty?
```

This note studies how that statement can be proved without pretending that a
compact probabilistic digest is an exact certificate.

## 2. Equality is output-relative

Different BFS outputs require different comparison objects:

| Claimed output | Exact comparison object |
|---|---|
| frontier/reached set | canonical semantic state set |
| distances | map `state -> distance` |
| one arbitrary parent | replay-valid parent with decreasing distance; equality of parent choices is not required |
| canonical parent/word | exact canonical parent/word map |
| all-parent DAG | set of exact labeled predecessor edges |
| shortest-path counts | map `state -> exact count` under one arithmetic contract |
| occurrence statistics | multiset/map keyed by exact occurrence identity |
| ordered traversal artifact | sequence, not set |

Exact frontier-set equality cannot validate multiplicity, parent, label, count,
or order outputs that the set projection discarded.

## 3. Deterministic exact equality has an information cost

Represent a subset of an `N`-element known universe by its `N`-bit membership
vector. If Alice and Bob hold arbitrary independent vectors and must always
decide equality, deterministic communication complexity is linear in `N`
(`N+1` bits under the convention that the answer returns to both parties).

The pigeonhole intuition is direct: if fewer than `N` input bits distinguish
Alice's `2^N` possible vectors, two different vectors share one transcript;
Bob cannot answer both corresponding equality cases correctly.

Therefore a fixed small deterministic digest cannot prove equality of arbitrary
large BFS sets unless extra structure makes the digest injective on the declared
family. Randomized equality can be much cheaper by accepting a bounded error;
that is precisely the status of ordinary fingerprints.

## 4. Exact comparison without central gathering

Linear total information does not imply every item must be gathered onto one
rank. Exact comparison can remain distributed if corresponding semantic blocks
are co-located.

### 4.1 Dense injective-rank bitmap

Let `rank: state -> [0,N)` be proved bijective on the domain. Partition the rank
interval identically for result A and reference B. Each verifier compares its
bitmap words exactly and emits

```text
local_mismatch = OR_word(A_word XOR B_word).
```

A global OR of the local mismatch bits proves equality only because every
membership bit was already compared exactly at its responsible verifier. The
one-bit reduction is not a one-bit digest of unseen remote data.

### 4.2 Canonical sorted streams

Serialize each semantic state canonically, sort both sets by full exact key,
and merge-compare. This works for wide implicit states and external streams,
provided:

- serialization is injective and epoch-stable;
- multiplicity is either prohibited or compared explicitly;
- every shard is complete;
- ordering compares the full key, not a truncated prefix/hash;
- capacity/IO failure is explicit.

### 4.3 Collision-resolving maps

Both sides may shuffle full canonical states to a common verifier owner chosen
by a hash. Hash collision only co-locates unrelated states; the verifier still
compares full keys and counts. Exactness requires proof that every record is
delivered once/equivalently and every verifier completes.

## 5. Common verifier partition is part of the proof

Two executions may use different GPU counts, owner seeds, or internal layouts.
Comparing owner-local shard `i` to shard `i` is then meaningless. Validation
needs a separate immutable verifier map:

```text
verify_owner(canonical_semantic_key, validation_epoch).
```

Both results are normalized into that map. The verifier map may use hashing for
routing, but exact equality remains full-key collision resolving.

The normalization shuffle itself has proof obligations:

- no source shard omitted;
- no record silently lost or truncated;
- retries preserve set/multiset semantics;
- all in-flight verifier records are closed;
- validation capacity failure cannot become “equal.”

## 6. Exact local work plus one-bit reduction

After exact co-located comparisons, reduce

```text
global_mismatch = OR_r local_mismatch_r.
```

`global_mismatch=0` is exact if and only if:

1. the compared representation is exact for the requested output;
2. the union of verifier shards covers the complete objects;
3. corresponding records meet at one verifier;
4. each local comparison completes without hidden failure;
5. the reduction includes every verifier in one validation epoch.

This separates **comparison evidence** from **transport/completeness evidence**.
A perfect local comparator cannot notice an omitted whole source shard unless
the protocol proves that shard participated.

## 7. Why Merkle roots are not unconditional equality proofs

A Merkle tree hierarchically hashes exact leaves and can efficiently localize a
difference or authenticate membership relative to a root. Equal roots imply
equal leaf collections only under the assumed collision resistance/injectivity
of the hash construction and identical canonical tree layout.

For engineering validation, that can be extremely strong computational
evidence. Mathematically it remains conditional unless:

- the hash is proved injective over the finite declared artifact family; or
- matching branches are eventually compared by exact leaves/full keys.

Merkle layout also needs canonical ordering, domain separation, length/count
binding, and explicit empty/padding rules. Otherwise different sequences or
tree shapes can be ambiguously represented even before hash collision.

## 8. Why IBLT reconciliation is not automatically exact

An Invertible Bloom Lookup Table can list a small set difference with high
probability when its load is within the designed regime. Its appeal is that
communication can scale with the difference rather than full set size.

For an exact BFS claim, an IBLT is therefore one of:

- probabilistic reconciliation evidence with stated failure probability;
- a mismatch-finding accelerator followed by exact full-key validation;
- an optimization with an exact fallback when peeling/verification fails.

Treating successful probabilistic peeling or matching checksums as an
unconditional proof would cross the same fingerprint boundary as note 163.

## 9. Difference witnesses are stronger than “not equal”

When exact comparison finds a mismatch, retain a semantic witness:

```text
state/key,
which side contains it,
depth/distance,
parent or occurrence metadata if relevant,
source shard and verifier owner,
epoch,
replay result.
```

The first witness does not quantify total damage, but it falsifies equality and
usually locates whether the defect is generation, identity, routing,
publication, output merge, or validation transport.

## 10. Independence boundary

Exact equality between CPU and GPU artifacts proves parity, not correctness, if
both consume the same faulty move table, state encoder, or legality filter. The
stronger ladder is:

1. exact distributed artifact reconciliation;
2. replay of mismatching and sampled matching paths;
3. independent successor/identity oracle on bounded scope;
4. exhaustive declared-domain comparison where feasible.

Distributed exact comparison strengthens evidence validity but cannot create
oracle independence by itself.

## 11. GPU and multi-GPU interpretation

An exact device-resident validation path can conceptually use:

- common dense-rank bitmap shards and wordwise XOR/OR;
- full-key sort/merge on common verifier shards;
- collision-resolving verifier hash tables;
- device mismatch flags followed by a global OR.

Its correctness still requires completed kernels, visible payloads, capacity
status, exact state representation, and a consistent validation cut. A fast
checksum kernel is a regression tool, not a replacement for these obligations.
No particular implementation is selected here.

## 12. Counterexamples and rejected implications

- Equal cardinality proves set equality.
- Equal count, sum, xor, and finite fingerprint prove exact equality.
- Equal per-owner counts prove equality across different owner maps.
- A global one-bit mismatch reduction means only one bit of data was examined.
- Hash routing plus hash comparison is collision-resolving exact equality.
- Equal Merkle roots are unconditional mathematical equality of arbitrary data.
- Successful IBLT peeling is a zero-error proof without exact verification.
- Exact reached-set equality validates distances, parents, counts, or order.
- Exact CPU/GPU parity proves both implementations match the intended graph.
- Local exact comparisons prove global equality when a shard may be absent.

## 13. Evidence boundary and next gate

This note provides a conceptual exact-reconciliation contract and deterministic
information lower-bound argument. It contains no runtime measurement. A future
bounded Docker/Rust gate can compare two deliberately mismatched sharded
frontiers using an injective dense bitmap and canonical full-state sort/merge,
then demonstrate that count/sum/xor fingerprints miss a compensating mismatch.
That gate remains deferred while Docker server access is unavailable.

## Sources

- Stanford EE378C annotated notes, deterministic communication complexity of
  equality: <https://web.stanford.edu/class/ee378c/lecture2_annotated.pdf>.
- Michael T. Goodrich and Michael Mitzenmacher, *Invertible Bloom Lookup
  Tables*, Allerton 2011: <https://arxiv.org/abs/1101.2245>.
- Ralph C. Merkle, *A Digital Signature Based on a Conventional Encryption
  Function*, CRYPTO '87, DOI 10.1007/3-540-48184-2_32; bibliographic record:
  <https://dblp.org/rec/conf/crypto/Merkle87.html>.

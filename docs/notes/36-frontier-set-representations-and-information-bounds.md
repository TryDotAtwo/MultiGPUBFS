# Frontier representations: one set, different information contracts

At a completed BFS level, the mathematical frontier is a set.  A queue, sorted
ID list, bitmap, compressed bitmap, and complement can encode that same set,
but expose different operations and may discard order or multiplicity that some
outputs treat as semantic.

This note studies representation bounds and proof obligations.  It does not
choose or optimize a data structure.

## Exact finite universe assumption

Assume every semantic state has an injective rank in

```text
U = {0,1,...,N-1}.
```

Let frontier `F subset U` have cardinality `k`.  A dense bitmap sets bit `i` if
and only if rank `i` belongs to `F`.

The injective-rank premise is essential.  A bitmap indexed by a colliding hash
represents hash buckets, not exact states, and can introduce false visited or
frontier membership.  If the universe is unknown, unbounded, or only partially
ranked, the dense `N`-bit model is unavailable without another exact mapping.

## Information-theoretic lower bound

There are

```text
binomial(N,k)
```

possible size-`k` frontiers.  Any lossless uniquely decodable representation
for an arbitrary one of them needs at least

```text
ceil(log2 binomial(N,k))
```

bits in the worst case when `N` and `k` are known externally.  Otherwise the
cardinality or a self-delimiting description also needs representation.

For sparse `k`, Stirling-style estimates give the scale

```text
log2 binomial(N,k) = k log2(N/k) + O(k).
```

This is lower than storing `k` independent full-width IDs because sorting
implicitly supplies information: later ranks are chosen without replacement.
Gap, enumerative, and succinct encodings can exploit that fact, with different
query and decoding costs.

For arbitrary cardinality, there are `2^N` subsets, so a worst-case exact code
over the whole powerset needs `N` bits.  An uncompressed bitmap meets that
space bound directly.

## Raw list versus raw bitmap

For `N>=2`, with `r=ceil(log2 N)` rank bits per ID, idealized payload sizes are

```text
ID list:  k*r bits
bitmap:   N bits.
```

Ignoring headers, alignment, capacity, and conversion buffers, the list uses
less raw payload when

```text
k/N < 1/r.
```

For 32-bit ranks this rough byte crossover is density `1/32`, not one half.
It is not a runtime threshold and not a recommendation.  Sorted gaps,
clustering, word alignment, container metadata, update patterns, and required
operations can move the practical boundary substantially.

Because

```text
binomial(N,k) = binomial(N,N-k),
```

the membership set and its complement have the same combinatorial information
content.  But their operational costs differ: enumerating a small complement
may be cheap, while enumerating `F` from that complement can require scanning
the whole universe.

## A set is not an occurrence bag

Frontier set semantics forget:

- duplicate candidate occurrences;
- how many parents proposed each state;
- generator/edge labels of those proposals;
- arrival or generator order;
- a deterministic parent tie winner unless separately reduced.

For scalar distance-only BFS, exact set membership is usually the right logical
frontier.  For all shortest parents, path counts, labeled solutions, or
shortlex output, the discarded information belongs in separate metadata or a
completed reduction.  Replacing a candidate bag by a bitmap is not semantics-
preserving for those richer outputs merely because reached states agree.

## Enumeration and membership are different interfaces

Top-down push naturally requests

```text
for each u in F: enumerate successors(u).
```

A sparse list directly enumerates the `k` active ranks.  A plain bitmap may
scan words or use a separate summary/index to find set bits.

Bottom-up pull naturally requests repeated tests

```text
is predecessor u in F?
```

A bitmap provides direct rank-indexed membership.  A sparse representation can
support membership through sorting, hashing, or an auxiliary index, but then
its total representation is more than the ID payload alone.

This explains why direction-optimizing BFS literature uses queue and bitmap
views.  The two forms evaluate the same next-layer predicate under note 14's
conditions; they do not have the same access pattern.

Maintaining both views consumes space and requires a consistency point.
Converting between them is real work and must be included in measurements.

## Frontier, candidate, and visited have different densities

At level `d`, at least three sets/bags coexist conceptually:

```text
F_d                 current exact frontier
C_d                 raw or deduplicated candidate object
B_d                 cumulative visited ball.
```

Their cardinalities and representation needs differ.  `F_d` can shrink near
diameter while `B_d` remains almost full.  A candidate bag can exceed both due
to duplicate edge/generator occurrences.

Choosing one representation policy for all three because they are "sets of
states" hides these different lifetimes and operations.  A visited bitmap may
be natural for a dense ranked finite universe while the frontier remains a
sparse list; that is not a semantic inconsistency.

Monotone visited also supports operations that a frontier does not: test-and-
set/claim, durable accumulation, and checkpoint recovery.  A frontier bitmap
may be cleared or replaced every level.  Equal storage syntax does not imply
equal lifecycle.

## Compressed and hybrid bitmaps

Compressed bitmaps exploit structure beyond cardinality, such as long runs or
locally dense chunks.  Hybrid formats can choose array, bitmap, or run
containers per region of the rank universe.

Two size-`k` frontiers can compress very differently:

```text
{0,1,...,k-1}              one clustered run
{0,q,2q,...,(k-1)q}        widely spaced ranks.
```

Thus global density `k/N` does not determine compressed size.  Rank ordering is
part of the physical experiment: a semantic permutation of IDs preserves BFS
but changes gaps, runs, cache locality, and owner partitions.

Compression ratios must be reported together with operation costs.  A smaller
encoding may require more decoding for frontier enumeration or membership; the
trade-off is empirical and workload-specific.

## Dense bitmap does not mean dense state storage

The `N`-bit frontier can coexist with separate state records only for set bits,
or with a dense rank-to-state decoder.  A bit says that rank `i` is active; it
does not contain:

- the original state bytes;
- parent identity or move;
- distance when multiple levels coexist;
- owner/version metadata;
- a procedure to generate successors from `i`.

For implicit Cayley states, a bitmap is useful only if rank-to-state operations
or a parallel state array make expansion possible.  Dense membership alone is
not an implicit successor oracle.

## Distributed ownership

With exact owner partition `U = disjoint union U_p`, device `p` can encode its
local frontier in the local rank universe.  Communication choices include:

- sparse destination IDs;
- dense per-owner bitmaps;
- compressed bitmaps;
- locally reduced candidate sets plus owner-side exact claims.

The global density `k/N` does not determine per-link density or owner skew.  One
owner can receive a dense subset while another receives nothing.

Bitmap OR is idempotent and naturally merges duplicate membership bits.  It
does not merge parent labels or path-count contributions correctly without
their own reduction algebra.  Sparse messages can retain those records but may
repeat destination IDs.

A full global bitmap replicated on every GPU costs roughly `N` bits per GPU,
whereas an owner-sharded bitmap costs roughly `N` aggregate bits before
metadata.  Replication may change membership access and communication, but no
correctness or performance conclusion follows without the declared protocol.

## Conversion and phase invariants

Whenever representations change, useful checks are:

1. list cardinality after exact unique equals bitmap popcount;
2. every listed rank sets exactly its bitmap bit;
3. bitmap enumeration reconstructs the same sorted rank set;
4. no out-of-range, stale-epoch, or wrong-owner bit is accepted;
5. duplicate/parent metadata discarded during conversion is not required by
   the output contract;
6. conversion completed globally before a mode that depends on the new view;
7. capacity failure is explicit rather than truncating the set.

Equal cardinality alone is insufficient: one missing bit and one spurious bit
can cancel.

## GPU measurement model without an optimizer

For each representation and level, useful observed quantities include:

- semantic `N` and `k`;
- allocated, payload, and peak conversion bytes;
- set-bit enumeration and membership-query counts;
- bytes read/written, including zero bitmap words scanned;
- list-to-bitmap and bitmap-to-list time;
- compression/decompression time and resulting bytes;
- duplicate occurrences removed before/while constructing the representation;
- owner-local densities and communication bytes;
- exact equality checks against a reference set.

These measurements help explain behavior.  A density-only switching rule is a
hypothesis to test, not something derived from the information bound.

## Rejected shortcuts

- **A bitmap is exact for arbitrary hashed states.** Only an injective rank or a
  collision-resolving mapping makes the bit identity exact.
- **The smaller representation is automatically faster.** Enumeration,
  membership, conversion, and memory-access costs differ.
- **Density alone predicts compressed size.** Rank clustering and runs matter.
- **A frontier set preserves all shortest-path outputs.** It drops occurrence,
  label, order, and parent multiplicity unless stored separately.
- **The same format should represent frontier, candidates, and visited.** Their
  operations, densities, and lifetimes differ.
- **Bitmap OR makes distributed path counting idempotent.** It merges membership
  only; addition still needs contribution semantics.
- **Matching popcounts proves conversion correctness.** Membership equality is
  stronger than cardinality equality.

## Sources

- Beamer, Asanovic, and Patterson,
  [Direction-Optimizing Breadth-First Search](https://www.scottbeamer.net/pubs/beamer-sc2012.pdf),
  uses queue and bitmap frontier views for top-down and bottom-up access.
- Chambi, Lemire, Kaser, and Godin,
  [Better bitmap performance with Roaring bitmaps](https://r-libre.teluq.ca/602/1/RoaringBitmap.pdf),
  studies hybrid compressed integer-set containers and their operation/space
  trade-offs.
- Patrascu and Viola,
  [Bit-Probe Lower Bounds for Succinct Data Structures](https://epubs.siam.org/doi/10.1137/090766619),
  places exact set representations against the `log2 binomial(N,k)`
  information-theoretic baseline.
- Notes 4, 14, 28, 29, and 32 supply set/bag semantics, push/pull equivalence,
  exact ranking, work accounting, and per-layer structural variation.

## Current conclusion

Sparse lists, dense bitmaps, and compressed hybrids encode the same exact BFS
frontier only after an injective universe and output contract are fixed.  Their
space can be compared to `log2 binomial(N,k)`, but performance depends on the
operations they must support.  Representation choice changes physical work;
discarding order, multiplicity, or parent data can also change the requested
mathematical output.

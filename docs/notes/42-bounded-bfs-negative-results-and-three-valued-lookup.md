# Bounded BFS: negative results and three-valued lookup

An exact bounded BFS table answers a different question from a complete
reachability traversal. Its most easily mishandled output is absence.

```text
not present in a radius-R table
```

does not automatically mean unreachable, and an incomplete table does not even
prove distance greater than `R`.

## The exact bounded result

For source set `S`, define the exact ball

```text
B_R(S) = {v | dist(S,v) <= R}.
```

If every layer through `R` was generated completely with exact identity and no
loss, a query has two mathematical outcomes:

```text
v in B_R:      exact finite distance d <= R, plus optional witness
v not in B_R:  dist(S,v) > R, where infinity is also greater than R.
```

The second outcome combines two cases that bounded BFS cannot distinguish:

- `v` is reachable, but only beyond the radius;
- `v` is unreachable from `S`.

Bounded table absence alone cannot promote the latter possibility to a proof
of `UNREACHABLE`. That needs exhaustive completion of the reachable component
or an independent unreachability certificate.

## Operationally, three outcomes are required

A real lookup needs at least:

| Status | Meaning |
|---|---|
| `WITHIN_RADIUS(d,witness)` | exact membership, distance, and optional path witness |
| `NOT_WITHIN_RADIUS` | complete exact construction proves distance greater than `R` or infinity |
| `UNKNOWN` | interruption, overflow, approximation, version mismatch, or missing proof prevents either conclusion |

`UNKNOWN` is not a pessimistic spelling of `NOT_WITHIN_RADIUS`. Converting it
to a negative answer makes partial work indistinguishable from a completed
lower-bound certificate.

An additional `UNREACHABLE` status is justified only when a finite reachable
component was exhausted or another independent certificate establishes
unreachability.

## Positive and negative certificates are asymmetric

A positive record can often be validated locally:

```text
replay witness -> target
witness length <= R.
```

This proves an upper bound even if the rest of the table is incomplete.

A negative result is global over the bounded search space. To prove it, the
builder must establish all of:

- every predecessor/successor required by the declared graph was generated;
- every layer below the boundary was complete;
- identity decisions were exact;
- no frontier, candidate, table, message, or owner partition overflowed;
- no task was cancelled or skipped;
- the query uses the same graph, target/source, generator, and key version.

Thus positive replay can survive some partial construction, while a negative
distance bound generally cannot.

## Miss composition theorem

Let `B_R(t)` be an exact reverse-BFS ball around target `t`. Starting from state
`x`, enumerate every allowed forward move word of every length `0..K`. If no
enumerated word lands in `B_R(t)`, then

```text
dist(x,t) > K + R.
```

### Proof by contraposition

Assume a path from `x` to `t` has length `L <= K+R`. Choose

```text
j = max(0, L-R).
```

Then `j<=K`. After the first `j` moves of this path, the remaining distance to
`t` is at most `R`, so the reached state belongs to `B_R(t)`. Because every word
of every length through `K` was enumerated, that prefix must have produced a
hit. Therefore a complete miss contradicts `L<=K+R`.

This theorem requires lengths `0..K`, not only exactly `K`: without optional
wait/self-loop moves, a shorter useful prefix may not be extendable to exact
length `K` while preserving its endpoint.

## Hit composition is a different statement

If a word `w` of length at most `K` reaches state `y in B_R(t)`, and the table
stores an exact suffix of length `r`, replay gives a path of length

```text
|w| + r <= K + R.
```

This is an upper-bound certificate for any genuine hit. There is a stronger
first-hit theorem when the table is the exact **complete** reverse ball, its
suffixes are shortest, and every allowed word is scanned in nondecreasing
length including the empty word. As proved in note 40, for `D=dist(x,t)` the
first hit has prefix length `max(0,D-R)` and combined residual exactly `D`.
A hit exists iff `D<=K+R`. The shorter-prefix/farther-landing tradeoff cannot
produce a suboptimal first hit under these premises.

So complete bounded enumeration has a useful asymmetry:

- **any genuine hit:** proves some upper bound;
- **complete miss:** proves a lower bound beyond `K+R`;
- **first hit with complete-ball, shortest-suffix, and ordered-prefix
  premises:** proves the exact residual distance;
- **first hit without those premises:** is only an upper-bound witness if
  its full-state replay is genuine.

## Application to CayleyPy K1

Under the preconditions identified in note 40, K1 intends to represent the
complete reverse ball of radius `R=K1` around the target. A semantic query miss
would therefore mean `distance_to_target > K1`, not “unreachable.”

However, the current table uses bare deterministic `Hash128` identity.
Collisions complicate both signs:

- a query can match another state's stored hash and produce a false semantic
  hit whose suffix fails full-state replay;
- a colliding state suppressed during construction is not expanded, so its
  descendants with unrelated hashes may be absent, producing false semantic
  misses and destroying the negative radius proof.

Final replay protects accepted positive artifacts. It cannot validate a miss,
because there is no witness to replay. Therefore K1 misses are exact lower-bound
certificates only under the collision-free/injective premise, not merely
because the table build finished.

Configured K1 entry overflow and device packing failure throw rather than
silently returning a partial table. That preserves the distinction between
`UNKNOWN` and a normal completed miss at those capacity boundaries.

## Application to CayleyPy K2 plus K1

K2 enumerates all move words of lengths through its configured radius, while
the immediate child is checked against K1 before nonempty K2 suffixes. For one
generated child `x`, if:

- K1 is an exact radius-`R` target ball;
- K2 exhaustively scans every word of lengths `1..K` after the empty-word K1
  check;
- every scan completes without cancellation or omission;
- move generation matches the declared forward graph;

then a complete miss proves

```text
dist(x,target) > K + R.
```

For a retained parent, the kernel checks every configured immediate move. If
the parent is not already the target and every child completely misses, then
every path from that parent to the target is longer than

```text
1 + K + R.
```

This is a local lower bound for that retained parent. It says nothing about a
different source-to-target branch whose ancestor was previously removed from
the outer beam.

For a positive result, if K1 also stores exact shortest suffixes and K2 scans
in nondecreasing word length, the first hit establishes the minimum residual
for that generated child. Checking all shorter words, including the immediate
empty-word K1 check, is sufficient; longer words need not be scanned to prove
this scalar optimum. These are conditional guarantees, not evidence that the
current hash-only implementation satisfies exact semantic identity.

## Outer beam changes the scope of a miss

Goal inspection occurs for every immediate move of the current retained
frontier before score and next-beam selection. If an entire dispatched depth
drains with no hit and all K1/K2 checks complete, the strongest resulting claim
is:

> No generated child of any currently retained parent reaches the target
> within the checked residual horizon.

It is not:

> The original puzzle has no solution within that total depth.

Earlier beam pruning may have removed a prefix that supports such a solution.
The negative certificate is exact relative to the surviving frontier set, not
relative to every graph state at that prefix distance.

This illustrates a general rule: lower bounds compose only across exhaustive
sets. Exact local lookups cannot fill holes in an incomplete outer frontier.

## Capacity and cancellation semantics

Different capacity failures affect signs differently:

- dropping a table/frontier state can invalidate later misses;
- overflowing a positive-result buffer may still leave `found=true`, but can
  prevent selection of the best recorded upper bound;
- stopping after the first positive intentionally makes unscanned alternatives
  `UNKNOWN`, not negative; under the first-hit theorem this does not prevent
  certification of the scalar minimum residual for that candidate;
- a fail-closed exception preserves truth by refusing to issue either normal
  result;
- a timeout after zero observed hits is `UNKNOWN` unless complete coverage was
  independently established.

Logs therefore need completion evidence alongside result counts:

```text
radius requested and completed
all word lengths completed
states/words expected and checked
overflow and cancellation flags
graph/key/table versions
positive buffer overflow
scope of the outer frontier searched
```

## Distributed misses

In multi-GPU bounded BFS, no local owner can prove a global miss. A valid
`NOT_WITHIN_RADIUS` result requires:

- every owner completed every assigned state and edge through the radius;
- all generated identities reached their authoritative owner;
- no messages or spill buffers remain in flight;
- every capacity and failure flag participates in the reduction;
- the final negative decision occurs after a global completion barrier or an
  equivalent termination proof.

An all-reduce of `found=0` says only that no rank reported a hit. It is not a
coverage certificate unless completion and losslessness are reduced too.

## Minimal result schema

```text
status: WITHIN_RADIUS | NOT_WITHIN_RADIUS | UNKNOWN | UNREACHABLE
distance_or_bound:
radius and logical metric:
source/target and graph version:
identity/key contract:
construction complete:
overflow/cancellation flags:
witness and replay status:
outer-search scope:
distributed completion evidence:
```

## Current conclusions

1. Absence from an exact radius table proves only distance beyond the radius,
   with unreachable folded into that outcome.
2. Partial, approximate, overflowed, or version-ambiguous construction must
   return `UNKNOWN`, not a negative distance claim.
3. Any genuine replayed hit is an upper-bound witness; a complete miss is a
   lower-bound certificate and requires much stronger global evidence.
4. Exhaustive words through `K` missing an exact goal ball of radius `R` prove
   distance greater than `K+R`.
5. First-hit enumeration in nondecreasing word length, including the empty
   word, gives the exact residual distance for an exact complete reverse ball
   with shortest suffixes; without those premises replay alone gives an upper
   bound.
6. CayleyPy K1/K2 misses can be interpreted only conditionally on exact hash
   identity and complete scans, and only relative to retained outer-beam states.

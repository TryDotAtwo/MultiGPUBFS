# BFS level union, idempotence, and output merge algebra

Parallel BFS is possible because one completed level has a partition-independent
set equation. Candidate-set union is associative, commutative, and idempotent.
Those properties make duplicate delivery harmless for distance-only membership
under the right protocol.

They do not automatically extend to parent choice, path multiplicity, ordering,
or arbitrary cross-level scheduling. Each output has its own merge algebra.

This note studies semantics only.

## 1. Partition-independence theorem for one level

Assume `visited=B_d` and `frontier=F_d` are exact. Partition the frontier into
arbitrary disjoint producer shards:

```text
F_d = disjoint_union_i P_i.
```

Each shard generates its exact successor set

```text
C_i = Post(P_i).
```

Because relational image distributes over union,

```text
Post(F_d) = union_i C_i.
```

Therefore the exact next frontier is

```text
F_(d+1) = (union_i C_i) minus B_d.
```

This result is independent of:

- the number and sizes of shards;
- the order in which shard results arrive;
- how many times the same endpoint appears across shards;
- which producer first presents a new state.

It assumes every required successor appears at least once, no spurious state is
accepted, identity is exact, and the subtraction uses the correct completed
ball.

## 2. Why duplicate candidates are benign for a set

Set union obeys

```text
A union B = B union A,
(A union B) union C = A union (B union C),
A union A = A.
```

The last law is idempotence. Replaying a candidate or merging the same producer
set twice does not change the mathematical endpoint set.

The powerset ordered by inclusion with union as its join is a join-semilattice.
This algebra explains convergence of independently accumulated exact membership
sets: any merge order reaches the same union after every contribution arrives.

Idempotent membership does not mean duplicate work is free. Retries can still
consume compute, queues, atomics, routing, bytes, and time.

## 3. At-least-once is enough only for the complete transition

For distance-only level membership, physical exactly-once delivery is
unnecessary if:

- every logical successor obligation is delivered at least once;
- exact state identity merges repeat deliveries;
- one accepted copy creates durable pending frontier work;
- retry acknowledgement cannot orphan a visited state;
- capacity failure is explicit;
- termination includes all in-flight/retry obligations.

Making only the visited bit idempotent is insufficient. If a retry sees
`visited=true` after the first copy claimed the state but that first copy never
published frontier work, the reached set contains an orphan that will not be
expanded.

The semantic transaction is closer to

```text
new membership + durable pending expansion + required output metadata.
```

Its recovery/commit contract must be complete even when its membership field is
an idempotent set insertion.

## 4. Online claims preserve the set but expose other choices

An implementation need not materialize every `C_i` before subtraction. It can
perform an exact atomic or owner-serialized claim as candidates arrive.

Within one exact layer, all genuinely new candidates have distance `d+1`.
Exactly one winning claim can materialize each state in `F_(d+1)` while losing
same-state claims are discarded for distance-only output. The resulting
frontier set remains partition- and arrival-order independent.

The following can still vary:

- which producer wins;
- chosen arbitrary parent and move label;
- physical frontier order;
- local versus remote duplicate volume;
- number of failed claims and retries;
- timing of target notification.

Set determinism is not execution determinism.

## 5. Output merge algebra

| Output | Natural merge | Duplicate/reorder semantics |
|---|---|---|
| reached membership | set union | idempotent and order-independent |
| minimum distance | `min` | idempotent; asynchronous decreases still need propagation |
| arbitrary shortest parent | choose one valid equal-depth contender | valid but generally nondeterministic |
| canonical shortest parent | total-order `min` over contenders | associative, commutative, idempotent after complete contender closure |
| all shortest parents | set union of exact parent identities | idempotent after duplicate removal |
| move-label set per parent/child | set union | idempotent only if multiplicity is not requested |
| shortest-path count | sum one contribution per predecessor occurrence | not idempotent without stable contribution identity |
| labeled-word multiplicity | multiset/addition | retries must not create extra logical occurrences |
| frontier sequence order | concatenation/order rule | not commutative; schedule becomes observable |

The table shows why one universal “dedup” flag is not an output contract.

## 6. Minimal non-idempotent path-count fixture

Use the diamond

```text
s -> a -> t
s -> b -> t.
```

The reached set and distance of `t` are unchanged if message `(a,t)` is replayed.
But naive path-count accumulation gives

```text
1 from a + 1 retry from a + 1 from b = 3
```

instead of the correct count two.

A stable contribution identity such as

```text
(graph epoch, depth, parent state, child state, edge/label occurrence)
```

can make application of each logical summand exactly once. Alternatively a
complete level can be recomputed from a clean boundary. Plain addition itself
does not absorb retries.

## 7. Canonical parent is a reduction, not first arrival

Suppose every equal-depth parent proposal carries a deterministic total-order
key. Then

```text
parent[v] = min(all valid depth-d parent proposals for v)
```

is order-independent and duplicate-insensitive. But it can finalize only after
all relevant depth-`d` proposals have closed. Stopping at the first proposal
changes the merge operation from `min` to arrival-order selection.

Thus canonical metadata can retain parallel partition-independence, but only by
using an explicit associative/commutative/idempotent reduction and a complete
input boundary.

## 8. Why the theorem stops at the level boundary

If deeper work is allowed to race ahead and first claims are irrevocable, the
candidate distances are no longer all equal. A longer path can claim a state
before a delayed shorter path. Set union still computes reachability, but the
stored distance and descendants can be wrong.

Cross-level arbitrary scheduling therefore needs the label-correcting `min`
and reactivation contract from notes 18 and 164. Idempotent union of state names
alone does not preserve shortest-hop labels.

The fixed operand `B_d` is also important. Same-level claims can be added to the
visited structure while forming `F_(d+1)` for vertex membership, but they must
not erase required equal-depth parent/count contributions under richer outputs.

## 9. Multi-GPU consequences

For distance-only closed levels, every GPU may accumulate local candidate sets,
route records in any order, and let authoritative owners union exact state
identities. The global set is invariant if completeness and publication hold.

Protocol evidence must nevertheless distinguish:

```text
logical occurrences,
unique state membership,
durable pending frontier work,
metadata contributions,
physical retries and acknowledgements.
```

Global union tolerates duplicate membership. It does not tolerate a permanently
lost only occurrence, an orphaned claim, a missing canonical contender, or a
double-counted non-idempotent contribution.

## 10. Validation consequences

A bounded fault/reorder test should independently inject:

- duplicate delivery of a state candidate;
- reordered delivery within one layer;
- loss followed by retry;
- crash between visited claim and frontier publication;
- duplicate parent contribution to path counting;
- late smaller canonical-parent contender;
- deeper candidate arriving before a shallower one.

Expected outcomes differ. The first two may preserve distance-only sets; the
orphan, count replay, missing contender, and cross-level claim should be
detected or corrected according to the declared contract.

Parity of final visited sets cannot validate path counts or deterministic
parents.

## 11. Rejected implications

- Duplicate candidates are harmless for every BFS output.
- Idempotent visited makes the whole state transition idempotent.
- Exactly-once physical delivery is required for reached-set correctness.
- At-least-once delivery automatically proves completeness.
- Partition-independent frontier sets imply deterministic parents or order.
- Canonical parent can finalize on first arrival.
- Set union preserves shortest distances under arbitrary cross-level first
  claim.
- Matching visited sets validates path multiplicity.

## 12. Current synthesis

Within one exact BFS layer, concrete candidate membership forms an idempotent
union. This is the algebraic reason that frontier work can be partitioned,
reordered, and redundantly delivered without changing the distance-only set.

The freedom ends wherever the output merge is not the same idempotent join, or
where work from different distance levels is mixed without correction. A
parallel BFS protocol is therefore best understood as a collection of typed
reductions, each with its own completeness, retry, and finalization rule.

This note extends notes 03, 04, 11, 18, 25, 30, 37, 52, 121, 162, 164, and 171.


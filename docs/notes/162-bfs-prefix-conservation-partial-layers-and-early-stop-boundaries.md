# BFS prefix conservation: partial layers and early-stop boundaries

Full-traversal conservation laws assume every reachable state was expanded.
Bounded search, target search, and distributed execution often stop earlier.
Exact accounting still exists, but only at a clearly declared semantic cut.

The crucial distinction is between a completed layer prefix and a partially
processed current layer. A completed prefix certifies a metric ball; a partial
layer generally certifies only the positive states already produced.

No experiment is used. The identities apply to exact unit-cost vertex BFS with
declared logical successor occurrences.

## 1. Completed-layer prefix

Assume layers `F_0,...,F_d` have been expanded completely. This means every
logical successor occurrence of every parent through depth `d` has been
accounted for, including work that was routed, retried, or combined before
frontier insertion.

Then the exact next layer has been constructed and the known ball is

```text
B_(d+1) = F_0 union ... union F_(d+1).
```

Let

```text
T_i = logical successor occurrences generated from F_i,
M_d = sum_(i=0)^d T_i.
```

Exact claim-before-enqueue accepts every nonroot state in `B_(d+1)` once. Hence
the prefix conservation identity is

```text
nonaccepting prefix occurrences
  = M_d - (|B_(d+1)|-1).
```

This is the finite-prefix form of note 157's complete-traversal identity.

## 2. What the completed prefix proves

After complete expansion through `F_d`:

- every state in `B_(d+1)` has its exact distance;
- every state absent from `B_(d+1)` has distance greater than `d+1` or is
  unreachable;
- `F_(d+1)` is complete as a vertex set;
- one selected parent per discovered state can be final for one-tree output;
- all shortest predecessors into `F_(d+1)` are final only if every equal-depth
  contribution was retained rather than discarded after claim.

The prefix does not prove that a state outside the ball is unreachable. That
requires exhaustion or another global certificate.

## 3. Radius-R construction boundary

To construct exact ball `B_R`, it is sufficient to expand

```text
F_0,...,F_(R-1).
```

Expanding `F_R` is not required for membership or distances through radius `R`.
It is required to discover `F_(R+1)` or to continue toward an exhaustion
certificate.

This avoids two opposite errors:

- stopping after only discovering some of `F_R` and calling the radius table
  complete;
- requiring expansion of `F_R` before accepting already complete radius-`R`
  membership.

## 4. Partial current layer

Now suppose all depths below `d` are complete, but only a subset

```text
U subset F_d
```

has been expanded. Let `A(U)` be its generated occurrence records and
`N(U)` the distinct endpoints outside `B_d` accepted so far.

The local arithmetic identity

```text
nonaccepting records in this partial batch = |A(U)|-|N(U)|
```

may hold. It does not imply

```text
N(U)=F_(d+1).
```

Unexpanded parents in `F_d` may produce additional states or additional
shortest predecessors of already seen states. The contents and record count of
`N(U)` can depend on parent order, batch boundaries, and which rank finishes
first.

## 5. Positive target discovery mid-layer

Assume `B_d` is exact and a processed parent in `F_d` generates target `t`.
Then

```text
dist(s,t)=d+1.
```

The generated edge is an upper-bound witness of length `d+1`. Exactness of
`B_d` excludes any path of length at most `d`, so no remaining same-layer
parent can reveal a shorter path.

Thus mid-layer target discovery can finalize scalar distance and one replayable
shortest path. It does not automatically finalize:

- a canonical winner among all equal-depth parents;
- the complete predecessor set of `t`;
- the exact number of shortest paths to `t`;
- every meeting/crossing connector of that length;
- the complete next frontier `F_(d+1)`;
- negative results for other not-yet-seen states.

The stopping permission is output-specific.

## 6. Negative result needs full boundary closure

If target `t` has not appeared after only `U subset F_d`, no distance lower
bound beyond `d` follows: an unexpanded parent may reach it at depth `d+1`.

If the entire `F_d` is complete and `t` is absent from `B_(d+1)`, then

```text
dist(s,t)>d+1
```

with infinity included. This is a bounded negative certificate, not necessarily
an `UNREACHABLE` certificate.

Positive and negative stopping are asymmetric because one replayable edge can
prove discovery, while absence quantifies over every relevant parent and
successor.

## 7. Mid-layer work is order-dependent

Even when final `F_(d+1)` is order-independent, early-stop work is not. Parent
order can change:

- how many occurrences are generated before finding `t`;
- which shortest parent wins;
- candidate-buffer peak before cancellation;
- local versus remote traffic observed before stop;
- which equal-depth contributions have arrived.

Therefore an early-stop benchmark must declare parent order, batch size,
target-check location, cancellation granularity, and whether already-launched
work is included.

One favorable target order is not a complete-level throughput result.

## 8. Capacity and overflow boundary

A final next frontier can fit while an intermediate candidate or routed buffer
overflows. Conversely, an implementation that streams exact claims may never
materialize the logical `M_d` occurrences simultaneously.

Prefix correctness requires:

- no silent loss before the declared cut;
- every overflow or capacity limit represented in the outcome;
- logical occurrence accounting distinguished from resident peak records;
- accepted claims paired with durable publication obligations.

An overflowed partial layer yields `UNKNOWN/INCOMPLETE`, not a smaller exact
frontier.

## 9. Distributed completed-layer certificate

One rank finishing its local parents does not complete `F_d`. Global completion
requires, under the declared protocol:

- every owner/producer has retired its depth-`d` work;
- all generated messages are delivered or idempotently accounted for;
- authoritative visited and output metadata are settled;
- no depth-`d` retries, spills, kernels, or publications remain in flight;
- capacity and failure status are globally known.

Only then can the system promote the cut to exact `B_(d+1)` and use the prefix
negative certificate. This is the same unfinished-work discipline used by note
56's bidirectional stopping theorem.

## 10. Prefix waterfall

For every fully completed level `i`, note 160 gives

```text
G_i = L_i+R_i+V_i+D_i+|F_(i+1)|.
```

Summing only through completed depth `d` is valid. Adding partial current-level
counters to the sum requires labeling them partial; they cannot be used as if
`|F_(d+1)|` were final.

This supplies two reporting views:

- completed-prefix semantic totals, comparable across executions;
- partial/cancelled physical totals, useful for latency but dependent on order
  and cancellation behavior.

They should not share one unlabeled "edges processed" number.

## 11. Early stop and bidirectional search

In unidirectional BFS, exact `B_d` plus one target proposal at depth `d+1`
proves the target distance. Bidirectional search has more possible unfinished
connectors: either side may still expose a better meeting vertex or crossing
edge.

Arbitrary interleaving of partial layers therefore needs a valid exclusion of
shorter unseen routes, such as the global unfinished-depth bound in note 56;
first contact alone does not supply it. There is a narrower safe case: start
with disjoint exact balls, hold the opposite ball fixed, and expand one next
layer until its first intersection. Note 08 proves that this hit is already
shortest, without completing the active layer or separately waiting for the
generic minima test to pass.

Thus even a certified shortest-path result can accompany incomplete frontier
and transition totals. Prefix accounting tells what work is complete; the
chosen stopping proof tells whether the remainder can improve the requested
answer. These are separate certificates.

## 12. Applicability boundary

The prefix identity assumes exact unit-cost layers and one state insertion per
endpoint. Asynchronous relaxation can have tentative labels and reactivation;
its semantic cut must be defined through settled distances or a different
quiescence proof.

Likewise, beam pruning, probabilistic visited loss, incomplete successor
enumeration, or partial owner failure cannot be relabeled as a smaller exact
prefix. They change the outcome to bounded/incomplete under the project's
status vocabulary.

## Sources and internal dependencies

- Notes 08 and 56 provide complete- and partial-layer bidirectional stopping.
- Notes 09, 42, and 57 define completion, bounded negative results, and output
  finalization.
- Notes 30 and 52 give publication, failure, owner, and replica obligations.
- Notes 74, 157, and 160 provide occurrence, claim, and waterfall accounting.
- The prefix identity follows by counting one accepted claim for each nonroot
  state in the completely constructed ball.

## Takeaway

A completed BFS prefix is a mathematical object: an exact ball plus a closed
set of expansion obligations through its boundary. A partial layer is only a
work-in-progress view. Positive target distance can sometimes finalize early;
complete frontiers, negative results, canonical parents, path counts, and
distributed closure generally cannot.

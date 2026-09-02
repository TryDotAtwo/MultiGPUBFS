# Owner hashing, load balance, and routing in distributed BFS

Owner-computes BFS assigns every semantic state to one authoritative rank for
visited membership. A hash can make that assignment look random, but "balanced
on average" is not a complete statement about frontier skew, memory safety, or
communication.

This note develops a balls-into-bins baseline and compares it with the retained
exact `S_8` simulations. It does not propose or implement an owner selector.

## Exactness requirements come before balance

An owner function for `P` ranks has the semantic contract

```text
owner_P : exact semantic state -> {0,...,P-1}.
```

Correctness requires

```text
x = y  implies  owner_P(x)=owner_P(y)
```

for every representation of the same semantic state in one ownership epoch.
The owner then performs exact visited equality.

Different unequal states may share an owner or owner-hash value without a
correctness problem. Routing collisions affect load; they are not state
identity collisions unless the owner also mistakes hash equality for semantic
equality.

The mapping must be stable while records are in visited, frontier, routing, and
checkpoint state. Changing `P`, seed, canonicalization, rank encoding, or owner
rule without migration can send a rediscovery to a different authority and let
two ranks accept the same state independently.

### Four-state counterexample: occurrence-stable is not state-stable

Reuse the `Z_4` successor batch

```text
(parent=0,label=a,endpoint=1),
(parent=0,label=b,endpoint=1),
```

where distinct semantic labels `a` and `b` both act as `+1`. Suppose routing
uses the label or whole occurrence record, sending `a` to rank zero and `b` to
rank one. Both local visited shards can truthfully report “state 1 is absent”
and accept it, because neither is authoritative for all representations of
that state. The physical next frontier now contains state `1` twice and can
expand its successors twice.

Hashing each occurrence consistently does not solve this: the required
stability is with respect to the equality class of the endpoint state. Routing
by parent has the same failure when two different frontier parents converge on
one child.

Contributions may be temporarily partitioned by label, parent, or producer if
the output needs them, but then an explicit exact reduction keyed by endpoint
state must precede any claim of unique vertex acceptance or completed frontier.
Without that later convergence, the system owns histories/occurrences rather
than BFS vertices.

### Co-location is not conflation

Now let unequal `Z_4` frontier states `1` and `3` have the same routing hash:

```text
h(1)=h(3)=0.
```

Sending both to rank zero is safe. An exact owner table stores and compares the
full keys, accepts both vertices, and merely receives a less balanced shard.
If the table instead treats hash equality as state equality, accepting `1`
causes `3` to be rejected as “already seen,” deleting a real BFS branch.

The same hash collision therefore has two different meanings at two layers:

```text
routing:   choose one common place to compare          safe
identity:  declare the two states equal from the hash  unsafe
```

An owner function need not be injective on states. The identity relation used
inside the owner must be exact or collision-resolving.

## Independent uniform ownership baseline

Suppose one exact frontier contains `w` distinct states and each state is
assigned independently and uniformly to one of `P` owners. Let `X_r` be the
number assigned to rank `r`. Then

```text
(X_0,...,X_(P-1)) ~ Multinomial(w; 1/P,...,1/P)
X_r ~ Binomial(w,1/P)
mu = E[X_r] = w/P
Var[X_r] = w(1/P)(1-1/P).
```

Owner loads are not independent: if one receives more of the fixed total,
others collectively receive less. For distinct ranks,

```text
Cov(X_r,X_q) = -w/P^2.
```

The mean `w/P` is therefore only the center of a distribution, not a per-rank
capacity guarantee.

## Tail bounds and the maximum owner

For one rank and `delta>0`, a standard multiplicative Chernoff bound gives

```text
Pr[X_r >= (1+delta)mu]
<= exp(-mu*delta^2/(2+delta)).
```

A union bound over all ranks yields

```text
Pr[max_r X_r >= (1+delta)mu]
<= P * exp(-mu*delta^2/(2+delta)).
```

This is a useful ideal baseline:

- relative deviations shrink as `mu=w/P` becomes large;
- increasing `P` at fixed `w` decreases `mu` and can worsen relative skew;
- capacity planning needs a maximum/tail target, not only the mean.

The inequality is not evidence that a deterministic owner hash actually
produces independent uniform assignments on a structured BFS sphere. It states
what would follow **if** that model were justified.

## Small frontiers cannot balance over many ranks

If `0<w<P`, at most `w` ranks can be nonempty, so at least

```text
P-w
```

ranks are idle. Since the maximum nonempty load is at least one while the mean
is `w/P`,

```text
max_r X_r / (w/P) >= P/w.
```

This imbalance is combinatorial, not a poor hash. For a one-state frontier the
ratio is exactly `P`, and `P-1` devices have no owned state work.

Under the independent model, the expected number of empty owners is

```text
P(1-1/P)^w.
```

This explains why maximum-skew summaries dominated by tiny early/late layers
are often uninformative about the quality of a mapping. The retained REF-005
report corrected exactly this issue by separating large-frontier and peak-layer
statistics.

It also sets a strong-scaling boundary: ownership cannot create more independent
frontier states than exist. Move-level parallelism may still exist inside an
owned state, but rank-level state balance is impossible when `w<P`.

## Uniform final visited does not imply uniform layers

Suppose exact dense ranks cover an interval whose size is divisible by `P`, and
ownership is `rank mod P`. Full exhaustion can leave exactly the same number of
visited states on every owner.

An individual sphere need not be uniformly distributed across rank residues.
Group relations, ranking order, parity, and generator structure can correlate a
layer with low rank bits. Thus these claims are different:

```text
final persistent visited balance
per-level frontier balance
per-level candidate receive balance
peak scratch/buffer balance.
```

REF-005 observed perfect final balance for direct Lehmer modulo while one large
eight-rank layer reached `2.114943x` mean frontier load. Final allocation alone
would have hidden the transient bottleneck.

## A mixer trades correlation for locality

An avalanche-style mixer aims to make nearby or structured ranks look unrelated
before modulo `P`. If it approximates independent uniform ownership on the
actual frontier, it can reduce state-count skew toward the balls-into-bins
baseline.

But parent and child owners then also become nearly independent. When a source
rank expands states it owns, the ideal probability that a candidate remains on
the same owner is

```text
1/P,
```

so the remote probability is

```text
1-1/P.
```

For `M` source-local unique candidate records, the ideal expected remote count
is

```text
M(1-1/P).
```

This result requires candidate owner to be independent of source owner. A
locality-preserving rank can have a much smaller remote fraction by correlating
adjacent states, at the price of uneven frontier or receive loads.

REF-005/006 observed precisely this trade-off: mixed strategies improved
large-layer balance while remote fractions approached `1-1/P`; range/direct
rank strategies retained more move locality but could skew heavily.

## State balance is not transition-work balance

Equal frontier state counts imply equal transition counts only for a regular
implicit graph with equal-cost moves. Even a Cayley-like constant generator
count can retain imbalance through:

- generator-specific transformation cost;
- legality or canonicalization work;
- different duplicate convergence;
- unequal state/parent payload sizes;
- owner hash-table probe/load behavior;
- different local/remote peer distributions.

For an explicit graph, degree skew makes equal vertex counts an even weaker
proxy. The slowest owner is determined by its critical work/bytes, not its state
count alone.

Useful imbalance ratios therefore have named numerators:

```text
frontier states max/mean
generated transitions max/mean
received candidate records max/mean
post-owner unique states max/mean
accepted states max/mean
resident and scratch bytes max/mean
level wall time max/mean.
```

## Duplicate convergence changes with rank count

Two parent states on one source rank can generate the same child and remove the
duplicate before routing. If those parents move to different source ranks after
increasing `P` or changing ownership, their equal child records no longer meet
locally. They converge only at the authoritative destination owner.

Thus more ranks can shift work through the pipeline:

```text
source-local duplicate removal decreases
remote records increase
owner-side duplicate convergence increases
accepted semantic next frontier remains unchanged.
```

REF-010 measured this migration exactly on `S_8`. Generated transitions and
accepted semantics were fixed, while direct-ownership source pre-dedup removed
fewer occurrences and owner-side dedup removed more as `P` increased.

The independent balls-into-bins model for distinct frontier states does not
predict this effect. Duplicate records are correlated by graph relations and
shared children, so they need their own accounting.

## Candidate records, distinct states, and bytes

Communication can be counted at several stages:

```text
raw generated occurrences
source-local unique candidate records
remote records after routing decision
distinct semantic states after owner merge
previously unseen accepted states.
```

Only the last count is new BFS progress. Network bytes usually follow remote
records before global convergence and include more than identity:

```text
state/key
depth/epoch/side
parent or reconstruction token
validity/version metadata
framing/alignment.
```

Two partitions with equal remote **state** fraction can differ in bytes when
records or message fragmentation differ. A uniform all-to-all hash can also
touch many peers with small messages, making latency relevant even when total
bytes fit.

## Expected balance is not a no-overflow guarantee

If a rank provisions exactly `w/P` frontier slots, then even the ideal random
model overflows with substantial probability because some rank commonly exceeds
the mean. Exact BFS cannot silently discard that excess.

Safe execution needs one of:

- a deterministic capacity bound for the declared workload;
- a probabilistic capacity target with explicit failure detection and a
  non-lossy recovery path;
- spill/chunk/rebalance semantics that preserve every accepted state;
- an `INCOMPLETE/OVERFLOW` outcome rather than a smaller apparent frontier.

The bound must apply separately to frontier, receive records, owner-table slack,
dedup scratch, send buffers, and output metadata. Their peaks need not occur on
the same rank or level.

## Ownership epochs and changing world size

Let ownership depend on an epoch descriptor

```text
E = (P, hash/rank version, seed, canonicalization, graph version).
```

All equal states must route under one effective `E`. Changing world size from
`P` to `P'` changes modulo ownership for many states. Correct continuation then
requires an explicit transition such as:

- migrate authoritative visited/frontier records to the new owners;
- retain forwarding from old owners until migration closes;
- use a directory/indirection whose authority remains unique;
- restart from a clean checkpoint under the new epoch.

Simply launching another rank count while old visited shards remain creates two
authority maps. A state can be considered unseen at its new owner even though
the old owner already accepted it.

Failure/retry has the same issue: a replacement rank must recover the exact
owner shard and epoch, not merely reuse the numeric device index.

## The two-choice temptation

Classic balls-into-bins results show that choosing the less loaded of two random
bins can reduce maximum load dramatically. Directly applying a live-load choice
to exact BFS ownership introduces a semantic hazard:

```text
the same state observed at different times/senders may choose different owners.
```

To remain exact, any multi-choice scheme needs a deterministic state-stable
winner or an authoritative directory/forwarding protocol. Once live load and
migration participate, ownership becomes mutable distributed state rather than
a pure hash.

The mathematical load benefit is therefore not a free replacement for the
single-authority invariant. REF-006 correctly treats candidate strategies as a
Pareto study rather than selecting a universal mapping.

## Topology is absent from uniform hashing

The uniform model distinguishes local versus remote ownership but treats all
remote destinations alike. Real topologies do not:

- peer GPU versus host-staged transfer;
- same switch versus cross-switch;
- NVLink/NVSwitch versus PCIe;
- intra-node versus inter-node network;
- shared-link contention among several peer pairs.

A mapping can balance record counts and still overload one physical cut. A
topology-weighted byte/critical-link model is a later evidence layer; candidate
counts alone must not be reported as measured communication time.

## Validation and reporting checklist

1. What exact canonical bytes determine owner and semantic equality?
2. Is the owner map stable and versioned for the whole epoch?
3. Are equal states guaranteed to reach one authority despite retries?
4. What are per-level `w`, `w/P`, maximum load, empty ranks, and max/mean?
5. Is a random-uniform assumption tested on each structured frontier or merely
   asserted from the hash name?
6. Are frontier, transition, receive, accepted, byte, and wall-time skews
   separated?
7. Are remote fractions retained with integer numerator/denominator?
8. Where do duplicate records converge as `P` changes?
9. Are all per-rank capacities based on tails/peaks and fail explicitly?
10. Does a world-size/seed change migrate or restart every authoritative shard?
11. Are topology-weighted costs distinguished from simulated record counts?
12. Is any tuned salt/range evaluated outside the graph used to select it?

## Counterexamples and rejected shortcuts

### A strong hash guarantees perfectly balanced layers

Even independent uniform assignment has random tails, and a structured
deterministic frontier need not satisfy the independence model.

### More ranks always reduce maximum owner load proportionally

When `w<P`, most ranks must be empty and max/mean is at least `P/w`.

### Perfect final visited balance proves safe transient capacity

Individual frontiers, receive bags, and scratch can be much more skewed.

### Better mixing is unconditionally better

It can trade state-count balance for near-`1-1/P` remote traffic and reduced
source-local duplicate convergence.

### Owner hash collisions break exactness

They only co-locate unequal states if the owner performs exact equality.
Treating the hash itself as identity is the actual correctness failure.

### World size can change by recomputing modulo

Existing authoritative shards must migrate, forward, or restart under a new
epoch; otherwise equal states can have two authorities.

## Sources and evidence

- Michael Mitzenmacher and Eli Upfal,
  [Probability and Computing](https://www.cambridge.org/core/books/probability-and-computing/470E7E21DF4DE5A6B3D66B7A60D1C65D),
  provides balls-into-bins and Chernoff-bound background.
- Yossi Azar, Andrei Broder, Anna Karlin, and Eli Upfal,
  [Balanced Allocations](https://doi.org/10.1137/S0097539795288490),
  develops the power-of-multiple-choices load result whose ownership caveat is
  discussed above.
- Aydin Buluç and Kamesh Madduri,
  [Parallel Breadth-First Search on Distributed Memory Systems](https://arxiv.org/abs/1104.4518),
  supplies distributed BFS partition/communication context.
- Local REF-005 records rank-modulo versus mixed-owner balance, locality, and
  duplicate movement; REF-006 records the multi-objective Pareto surface; and
  REF-010 validates owner-computes bidirectional accounting on its stated finite
  corpus and measures `S_8` routing counts.

## Current conclusions

1. Owner hashing needs state-stable unique authority for correctness; uniformity
   is a separate performance property.
2. Independent uniform ownership gives multinomial loads and useful tail
   baselines, not deterministic per-layer guarantees.
3. Small frontiers have unavoidable rank-level skew, while large structured
   frontiers may violate random-hash assumptions through rank/move correlation.
4. Better mixing can improve load balance while increasing remote traffic and
   moving duplicate convergence from sources to owners.
5. Final visited balance, frontier balance, candidate receive balance, peak
   bytes, and level time are different quantities.
6. Changing world size or owner version requires an explicit ownership-epoch
   migration/forwarding/restart contract.
7. Simulated candidate counts are semantic workload evidence, not measured
   topology-aware communication performance.

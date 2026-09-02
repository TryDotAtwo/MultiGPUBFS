# BFS successor occurrences: support arcs and Cayley label multiplicity

An implicit successor function often returns records, not a mathematical set of
neighbors. Two generator labels may lead to the same state; an identity or
stabilizer move may return the current state; a distributed retry may repeat a
previous message without representing another graph transition.

Exact vertex BFS must separate this occurrence stream from the support graph on
states. This note generalizes note 156's simple-undirected accounting to finite
directed labeled successor occurrences.

No experiment is used. The identities follow from exact BFS layers and explicit
record identity.

## 1. Three objects, not one adjacency

For each state `u`, let the declared successor interface return a finite
multiset

```text
Gamma(u) = [(label, v), ...].
```

Distinguish:

1. **occurrence record** `(u,label,v)`;
2. **support arc** `(u,v)`, present when at least one occurrence reaches `v`;
3. **endpoint state** `v`.

Vertex BFS distance is computed on the directed support graph. Collapsing
labels preserves vertex reachability and hop distance only because every
occurrence has the same unit cost and endpoint-state identity is the declared
search identity.

It does not preserve labeled paths, move witnesses, or occurrence counts.

### Four-state trace: duplicate action, unchanged vertex BFS

Use `Z_4` with three declared move labels

```text
a -> +1,  b -> +1,  c -> -1 (=3 mod 4).
```

Labels `a` and `b` are distinct occurrences but induce the same group element.
From root zero, the successor stream is

```text
(a,1), (b,1), (c,3),
```

while the support-neighbor set is only `{1,3}`. Therefore vertex BFS is still
the four-cycle traversal

```text
F_0={0},  F_1={1,3},  F_2={2};
```

the duplicate `+1` action changes neither reachability nor distance.

The richer outputs differ. If `a` and `b` are semantically distinct moves,
state `1` has two labeled length-one witnesses, and state `2` has five labeled
shortest words:

```text
aa, ab, ba, bb, cc.
```

Four words travel through state `1`; one travels through state `3`. If the two
list entries are instead an accidental retry or duplicate encoding of one
semantic move, counting both would be wrong. Endpoint equality alone cannot
decide which contract applies; occurrence identity must be declared.

### Identity label is a different kind of extra occurrence

Instead add a declared label `e -> +0` to the ordinary generators `{+1,-1}`.
Every expanded state `x` now emits `(e,x)`, a self-loop occurrence ending in
its current layer. The support graph gains loops if loops are retained, while
all distinct-state frontiers and distances remain those of `C_4`.

Unlike duplicate labels for `+1`, the identity label creates no new shortest
labeled word. Any positive-length word containing `e` reaches the same endpoint
after deleting that symbol, one step sooner. This includes the root: its
shortest word is empty, not `e`.

Thus two changes that both preserve vertex distances have different effects:

| added occurrence | vertex frontiers | current/old hit work | shortest labeled words |
|---|---|---|---|
| distinct label for existing nonidentity `+1` action | unchanged | more parallel endpoint occurrences | may multiply |
| identity label `e` | unchanged | one self-loop occurrence per expanded state | unchanged |

“Does not change distance” is therefore too coarse to predict either work or a
richer output.

### When local endpoint collapse is semantics-preserving

Suppose one producer at completed depth `d` holds several exact occurrences
with the same endpoint `v`. They all propose vertex distance `d+1`, but the
safe local summary depends on the requested output:

| requested result | sufficient local summary for endpoint `v` |
|---|---|
| reached set / next-frontier set / distance only | one full exact state record |
| one arbitrary shortest path | one valid `(parent,label)` record |
| canonical parent or shortlex word | the minimum complete contender key |
| all predecessor vertices | set of distinct parent identities |
| all labeled predecessor arcs | set or multiset required by declared label identity |
| shortest labeled-path count | exact sum by stable semantic contribution identity |

For the `Z_4` root batch `[(a,1),(b,1)]`, these summaries are respectively
`{1}`, either witness, `min(a,b)` under the declared order, label set `{a,b}`,
or contribution count two. A retry of `(a,1)` must not silently change that
count to three.

This collapse is only producer-local. Retaining one state record proves that
at least one local occurrence survives; it does not prove that `v` is globally
new, that another producer has no better canonical contender, or that the
authoritative owner has not already accepted `v`. Local aggregation can reduce
an occurrence stream while global old/new membership and complete contender
closure remain separate BFS obligations.

## 2. Directed layer accounting

Let `F_d` be exact support-graph BFS layers and `B_d` the visited ball through
depth `d`. For occurrences generated by parents in `F_d`, define

```text
T_d = all outgoing successor occurrences,
X_d = occurrences whose endpoint lies in F_(d+1),
Y_d = occurrences whose endpoint lies in B_d.
```

Every outgoing arc `u->v` from `u in F_d` satisfies

```text
dist(s,v) <= d+1.
```

Its endpoint is therefore in `B_d` or `F_(d+1)`, giving the exact partition

```text
T_d = X_d + Y_d.
```

Unlike an undirected edge, a directed arc may point from `F_d` to any earlier
layer, not only `F_(d-1)`. All such occurrences belong to `Y_d`.

## 3. Two sources of next-state multiplicity

Let

```text
P_d = number of distinct support arcs (u,v)
      with u in F_d and v in F_(d+1).
```

Then

```text
X_d-|F_(d+1)|
  = (X_d-P_d) + (P_d-|F_(d+1)|).
```

The terms mean:

- `X_d-P_d`: extra labels/occurrences from the same parent to the same child;
- `P_d-|F_(d+1)|`: extra distinct parent states reaching the same child.

At one child `v`, write

```text
r(v) = number of (parent,label) occurrences from F_d,
p(v) = number of distinct parent states in F_d.
```

Then the same decomposition is local:

```text
r(v)-1 = [r(v)-p(v)] + [p(v)-1].
```

The first bracket is representation or labeled-edge multiplicity. The second
is structural convergence in the support graph.

## 4. Complete-traversal occurrence conservation

For a complete finite traversal of the reachable support component, exact
claim-before-enqueue accepts one occurrence-equivalent claim for every nonroot
state, hence `n-1` frontier insertions.

All remaining generated occurrences are nonaccepting for vertex-frontier
membership:

```text
sum_d T_d - (n-1)
  = sum_d Y_d + sum_d (X_d-|F_(d+1)|).
```

This is the occurrence-aware replacement for note 156's
`(n-1)+2 beta` identity. There is no general cycle-rank simplification because
directed back arcs, loops, and parallel labels add occurrences without changing
the simple support cycle rank in the same way.

## 5. What counts as a duplicate depends on output

For different result contracts, the same `r(v)` records have different status:

| output | required contribution at `v` |
|---|---|
| vertex distances/frontier | one state insertion |
| one replayable path | one chosen `(parent,label)` |
| predecessor-vertex DAG | every distinct parent state |
| labeled predecessor DAG | every distinct semantic `(parent,label)` arc |
| labeled shortest-path count | all distinct labeled contributions |
| operational execution | retries counted once, not as graph paths |

Calling every record after the first a duplicate is correct only for
state-frontier membership. It is wrong for richer labeled outputs.

## 6. Retries are outside graph multiplicity

Two identical delivered records may mean either:

- two declared parallel generator occurrences;
- retransmission of one occurrence after timeout or failure;
- accidental duplicate publication by the execution engine.

Only the first can be a distinct labeled graph edge. A correct distributed
protocol needs stable occurrence/message identity or idempotency metadata to
separate semantic multiplicity from delivery multiplicity.

Adding retry count to `r(v)` would corrupt shortest-path counts. Dropping all
equal endpoint records would erase legitimate parent or label alternatives.

## 7. Free Cayley action removes one multiplicity class

Consider a right Cayley graph on group elements with a collection of distinct
generator elements. For fixed `g`,

```text
g s = g t  implies  s=t
```

by left cancellation. Therefore distinct generators cannot produce the same
neighbor from one parent. If the identity is excluded, no generator produces a
self-loop.

Under this clean contract:

```text
r(v)-p(v) = 0
```

at every next-layer child. Candidate convergence comes from distinct parent
states and longer relations, not from same-parent length-one label aliases.

This fails if the generator input is a list with duplicate group elements or
contains the identity.

## 8. Schreier actions can have state-dependent aliases

For a right group action on states, distinct group elements need not act
differently. At state `x`,

```text
x s = x t  iff  s t^(-1) belongs to Stab(x),
x s = x    iff  s belongs to Stab(x).
```

Thus a nonfree Schreier action may expose:

- self-loop generator occurrences;
- several labels from one parent to one child;
- multiplicity that changes with the state through conjugate stabilizers.

These are genuine action occurrences but not additional endpoint states. They
can create `X_d-P_d>0` or add directly to `Y_d` at the same layer.

This is one precise reason a Cayley performance intuition does not transfer
unchanged to a Schreier state space even when the same generator list is used.

## 9. Generator relations versus immediate aliases

Two distinct generators agreeing at one Schreier state are a length-one action
alias. Distinct parents reaching one child are a boundary convergence that may
encode a longer relation or stabilizer word. Latent shortest-word multiplicity
may already have merged in earlier parent states, as note 64 shows.

Therefore at least four counts may differ:

```text
shortest word histories
(parent,label) occurrences
distinct predecessor states
one endpoint state.
```

No single duplicate ratio reconstructs the others.

## 10. GPU and multi-owner interpretation

The decomposition suggests two physically different dedup opportunities:

- same-parent label aliases may be removable during successor generation if
  the output allows it;
- cross-parent convergence requires coordination at a batch, frontier, or
  authoritative owner scope.

But semantic counts do not guarantee locality. Equal `(parent,child)` records
may be far apart in generator-major layout, and equal child states from
different GPUs meet only after routing.

For a rich output, "dedup" may instead mean combining contributions into one
state record while retaining a parent list or path-count reduction. Frontier
compaction and metadata reduction are different operations.

## 11. Practical audit fields

An implicit BFS trace should record per depth:

- total occurrences `T_d`;
- visited-ball occurrences `Y_d`;
- next-layer occurrences `X_d`;
- distinct support predecessor arcs `P_d`;
- unique next states `|F_(d+1)|`;
- same-parent label excess `X_d-P_d`;
- cross-parent state excess `P_d-|F_(d+1)|`;
- retries separately from graph occurrences;
- retained parent/label/count output after combination.

Without `P_d`, same-parent aliases and cross-parent convergence are
indistinguishable in aggregate.

## Sources and internal dependencies

- Notes 06 and 16 define implicit, Cayley, and Schreier graph semantics.
- Notes 36, 57, and 64 separate frontier states, candidate records, parent
  records, labels, and word histories.
- Note 74 gives claim/settlement and queue duplicate contracts.
- Note 156 supplies the simple-undirected support identity being generalized.
- Notes 51-52 supply owner authority, routing, replicas, and idempotency
  boundaries.
- The occurrence decompositions above follow from exact directed BFS layers and
  declared record identity.

## Takeaway

An implicit BFS does not expand "an edge" in only one sense. It expands labeled
occurrences, whose support arcs define state distance and whose endpoints define
frontier uniqueness. Separating same-parent label multiplicity from
cross-parent state convergence is essential for both Cayley/Schreier reasoning
and honest GPU duplicate accounting.

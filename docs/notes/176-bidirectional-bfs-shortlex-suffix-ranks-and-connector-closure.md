# Bidirectional BFS shortlex suffix ranks and connector closure

Bidirectional BFS can establish an exact shortest distance before it establishes
a canonical shortest word. Forward prefixes and goal-directed suffixes grow on
opposite sides of their strings, so they require different rank recurrences.

After both sides are canonical locally, every optimal meeting state or crossing
edge must still participate in a global full-word comparison.

This note studies output semantics, not a bidirectional implementation.

## 1. Forward prefix recurrence

For forward BFS from source `s`, note 175 assigns a canonical rank to each
length-`d` prefix. Extending parent `u` by label `ell` appends on the right:

```text
prefix(v) = prefix(u) ell.
```

Among equal-length candidates the comparison key is

```text
(prefix_rank(u), label_rank(ell)).
```

Parent-word order decides before the final label.

## 2. Reverse BFS stores forward-oriented suffixes

Reverse BFS starts at target `t` and traverses predecessor edges. If the
forward graph contains

```text
u --ell--> v,
```

and `suffix(v)` is a forward word from `v` to `t`, then

```text
suffix(u) = ell suffix(v).
```

The new label is prepended on the left. Consequently the lexicographic key is

```text
(label_rank(ell), suffix_rank(v)),
```

not `(suffix_rank(v),label_rank(ell))`.

After selecting the minimum key for each reverse-frontier state, ordering those
keys gives suffix ranks for the next reverse depth.

## 3. Why ordinary reverse discovery order can choose the wrong suffix

Let `a<b` and use

```text
u --a--> x --b--> t    giving suffix ab,
u --b--> y --a--> t    giving suffix ba.
```

Reverse traversal from `t` sees inverse edges in the orders

```text
t --b^-1--> x --a^-1--> u,
t --a^-1--> y --b^-1--> u.
```

If reverse paths are ranked by their traversal strings with
`a^-1<b^-1`, the path through `y` begins with `a^-1` and is visited first,
selecting forward suffix `ba`. But forward shortlex requires `ab`.

Reversing the inverse alphabet order is not a universal repair: lexicographic
order of strings is not preserved by reversing every string. The semantic
object to rank is the forward-oriented suffix under the prepend recurrence.

## 4. Vertex meetings

For a meeting state `m` satisfying

```text
d_f(m) + d_b(m) = D,
```

one shortest solution word is

```text
prefix(m) suffix(m).
```

For this fixed state and fixed split depth, the lexicographically least
solution through `m` uses:

1. the least shortest forward prefix to `m`;
2. then, if that prefix is fixed, the least shortest suffix from `m` to `t`.

Both halves need their own complete contender reductions. A valid arbitrary
parent on either side can preserve distance while losing canonicality.

## 5. Crossing-edge connectors

A shortest connector may be an edge

```text
u --ell--> v
```

with

```text
d_f(u) + 1 + d_b(v) = D.
```

Its solution word is

```text
prefix(u) ell suffix(v).
```

For a fixed connector, canonical forward prefix, connector label, and canonical
forward-oriented suffix determine its least word. All valid optimal connectors
remain global contenders.

Meeting-state-only logic can miss an optimal crossing edge when the two closed
balls have not yet shared a vertex under the chosen stopping schedule.

## 6. Local ranks from different split depths are not directly comparable

Suppose two optimal connectors have different forward depths. Their prefix
ranks live in different frontiers and order only equal-length prefixes. The
tuple

```text
(local prefix rank, connector label, local suffix rank)
```

is therefore not a universal full-word key across split depths.

One prefix can also be a prefix of the other connector's full solution word, so
the comparison may enter a suffix or connector before the other prefix ends.
Exact global selection needs a comparator equivalent to the complete
concatenated words, or a proven hierarchical representation that preserves
that order across variable splits.

This is an output-order fact, not a demand to materialize every full string.

## 7. Distance closure versus canonical-word closure

Let `mu` be the best known solution length. The ordinary bidirectional stopping
theorem can finalize distance when every unfinished connector has lower bound at
least `mu`.

Canonical word closure is stronger. It must also establish that:

- every connector of total length `mu` that could yield a smaller word has
  been generated and compared;
- both side-specific canonical reductions are closed for the states involved;
- no in-flight equal-length proposal can improve a prefix or suffix rank;
- quotient/frame lifting yields the requested concrete word.

Thus

```text
shorter-path exclusion
```

and

```text
equal-length lexical exclusion
```

are separate completion predicates.

## 8. First intersection is even weaker for canonical output

The first observed meeting provides an upper-bound word. Even if another theorem
proves its length is already optimal, later equal-length meetings may have a
smaller word.

Likewise, choosing the minimum meeting state ID or owner ID is deterministic
but unrelated to the lexical order of `prefix + connector + suffix` unless a
specific theorem connects those orders.

Distance-optimal, deterministic-meeting, and shortlex-optimal are three
different claims.

## 9. Multi-GPU reduction

With a common exact owner map, one owner can validate a same-state forward/reverse
intersection. Crossing edges and meetings can nevertheless be produced by many
owners and levels.

A partition-invariant canonical result requires:

- exact forward prefix contender closure;
- exact reverse prepend-suffix contender closure;
- stable connector identity and replay orientation;
- global reduction over every distance-optimal connector word;
- termination accounting for smaller/equal lexical contenders in flight.

Reducing only `(distance,meeting_state_id)` is insufficient for a shortlex
contract. Matching scalar distance across GPU counts is likewise insufficient.

## 10. Directed and Cayley conventions

Reverse traversal must use exact predecessors. For a bijective Cayley move
`ell`, the reverse operation applies `ell^-1` to find the predecessor while the
stored forward suffix still prepends visible label `ell`.

The ordered objects are therefore:

```text
reverse traversal operation: inverse transformation,
stored/replayed suffix label: forward transformation,
lexical order: declared forward label alphabet.
```

Confusing these three can yield a replay-valid shortest suffix under the wrong
lexical order.

In directed graphs, the forward alphabet and available predecessor operations
need not coincide. In Schreier/quotient search, canonical fiber suffix still
needs concrete frame/lifting proof for a fixed target.

## 11. Path-count and all-connector outputs

If the output requests all shortest connectors or path counts, selecting one
minimum word is not enough. Equal-length meeting states and crossing edges can
represent overlapping path families; contribution identities must prevent
retry double counting.

The canonical-word reduction uses `min`, while all-path aggregation uses set
union and/or non-idempotent addition. They can share distance bounds but require
different connector closure and merge algebra.

## 12. Validation fixtures

Validate at least:

- the `ab` versus `ba` reverse-prepend fixture;
- shuffled reverse predecessor arrival;
- an optimal vertex meeting and an optimal crossing-edge meeting;
- multiple optimal connectors with different split depths;
- a later equal-length lexically smaller meeting;
- inverse-operation replay with stored forward labels;
- different owner/GPU partitions;
- quotient/fixed-target lifting where applicable;
- retry/loss mutations for connector contributions.

Record:

```text
forward prefix ranks and closure,
reverse forward-oriented suffix ranks and closure,
meeting/crossing-edge identities,
full-word comparison key or equivalent proof,
best distance and best canonical word as separate fields,
unfinished lower-bound and lexical contenders,
action/alphabet/frame epoch,
replay and output-multiplicity contract.
```

## 13. Rejected implications

- Ordered reverse traversal automatically yields least forward suffixes.
- Inverting generator order universally repairs suffix order.
- Independently canonical halves automatically choose the global connector.
- Local ranks from different split depths are directly comparable.
- Exact shortest-distance stopping finalizes the shortlex word.
- First intersection is canonical once its length is optimal.
- Minimum meeting state ID means minimum solution word.
- Scalar parity across GPU counts validates canonical connectors.
- One canonical connector validates all-path counts.

## 14. Current synthesis

Forward BFS appends labels to prefixes; reverse BFS prepends forward labels to
suffixes. Their rank recurrences are mirror images, not identical code with an
inverse alphabet. Bidirectional canonical output then adds a third reduction:
compare the complete words induced by every distance-optimal vertex or edge
connector.

Shortest distance closes when shorter connectors are impossible. Shortlex
output closes only when smaller equal-length words are impossible as well.

This note extends notes 08, 16, 17, 19, 30, 40, 52, 56, 57, 75, 159, 174, and
175.


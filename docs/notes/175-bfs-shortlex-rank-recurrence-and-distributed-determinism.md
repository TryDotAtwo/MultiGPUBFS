# BFS shortlex-rank recurrence and distributed determinism

Deterministic parent choice and shortlex-least path choice are different
reductions. A total order on state IDs can make a parent tree reproducible while
selecting a non-shortlex move word.

For a deterministic labeled transition system, shortlex order can instead be
propagated level by level through ranks of canonical parent words. This note
derives that semantic recurrence and its distributed closure requirements. It
does not prescribe a sorting or communication implementation.

## 1. Minimal state-ID counterexample

Let labels satisfy `a<b`. Use depth-one states `p,q` with state IDs `p<q` and
edges

```text
s --b--> p
s --a--> q
p --a--> t
q --b--> t.
```

There are two length-two words for `t`:

```text
ba through p,
ab through q.
```

The shortlex choice is `ab`. The rule

```text
minimum (parent_state_id, label)
```

chooses parent `p` and word `ba`. It is deterministic and shortest, but not
shortlex. No correctness bug exists unless the requested output specifically
claims shortlex.

## 2. Canonical word ranks

Assume:

- one source `s`;
- unit-cost labeled transitions;
- a total order on generator/edge labels;
- deterministic transition semantics for a word;
- exact BFS layers and complete equal-depth predecessor proposals.

Give the empty source word rank

```text
r_0(s)=0.
```

At depth `d`, assume `r_d(u)` orders states in `F_d` by their already selected
canonical length-`d` words. For candidate edge

```text
u --ell--> v,
```

define its word-order key

```text
K(u,ell) = (r_d(u), rank(ell)).
```

For each new state `v`, choose

```text
K_min(v) = min { K(u,ell) : u in F_d and u --ell--> v }.
```

The corresponding parent/label produces the shortlex-least word for `v`.

## 3. Why the recurrence is correct

Every candidate word at depth `d+1` has equal length. Compare

```text
word(u) ell
word(u') ell'.
```

If the parent words differ, their first differing earlier label decides the
lexicographic order, exactly as `r_d(u)` versus `r_d(u')`. If the parent word is
the same, the final label decides. Therefore the tuple order on
`(parent-word rank,label rank)` is the lexicographic order on all length-`d+1`
candidate words.

Taking the minimum tuple among every proposal reaching `v` selects its least
word. Sorting the selected `K_min(v)` values and assigning dense ranks gives
`r_(d+1)` for the induction.

Because one deterministic word reaches one state, distinct states cannot have
the same full canonical word. If visible labels do not uniquely identify edge
occurrences or transformations, the declared path identity needs additional
tie fields and the theorem must use that richer alphabet.

## 4. Rank is not a state encoding

`r_d(v)` describes the order of the chosen source-to-`v` word among canonical
words in one frontier. It is not:

- a global semantic state ID;
- a dense rank of the whole Cayley group;
- a visited key;
- a hash/fingerprint;
- a proof that two equal ranks from different epochs mean the same state.

Exact state identity still decides which candidate words converge to one
vertex. Shortlex rank is metadata applied after that identity relation is
correct.

## 5. Parent minimum and frontier ranking are two reductions

The construction has two distinct closures:

1. for every child, reduce all equal-depth proposals by `min K`;
2. order the selected child keys globally to assign the next dense ranks.

The first chooses canonical parents. The second makes those canonical words
comparable as parents of the next layer.

A system can compute correct shortlex parents for the final requested target
without materializing dense ranks for unrelated future expansion if it has an
equivalent exact word-order key. Conversely, a reproducibly sorted frontier by
state ID does not implement either shortlex reduction.

## 6. Distributed reduction requirements

Partition independence follows because total-order minimum is associative,
commutative, and idempotent:

```text
min(min(A),min(B)) = min(A union B).
```

But every owner must see all relevant equal-depth proposals before finalizing
`K_min(v)`. Local pre-dedup must retain the local minimum key, not the first
arrival. Retry copies of the same proposal are harmless to `min`; a lost
smaller proposal changes the result.

Global rank assignment must compare selected word keys across owner boundaries.
Concatenating owner-local state-ID orders yields a deterministic sequence but
not generally the global shortlex order.

## 7. Partition-count invariance

For the same graph/action epoch, source, label alphabet/order, path identity,
and output contract, exact reduction should give the same:

```text
canonical parent,
canonical incoming label,
canonical word rank within every frontier,
reconstructed canonical word.
```

Changing GPU count may alter arrival order and physical frontier layout, but
must not alter these outputs if cross-owner proposal closure and global ranking
are complete.

This is stronger than distance/frontier-set parity. A run can pass exact set
comparison while failing canonical-word parity.

## 8. Generator order and epoch semantics

Shortlex output is relative to:

```text
generator set and visible labels,
total label/occurrence order,
left/right action convention,
source and graph epoch,
quotient/concrete target semantics.
```

Changing generator order preserves ordinary distances and frontier sets but can
change every rank and parent tie. Such a change is a new canonical-output epoch,
not nondeterminism within one epoch.

Hashes can establish equality or regression evidence under their own contract;
numeric hash order has no intrinsic relationship to declared lexical order.

## 9. Multi-source extension

For canonical multi-source output, initialize source words with an explicit
source priority and compare keys such as

```text
(source_rank, parent_word_rank, label_rank).
```

The priority order is part of the requested Voronoi/path contract. If path word
should take priority over source ID, the tuple order must be changed
accordingly. “Canonical source” and “canonical word” do not choose their own
lexicographic precedence.

Equal-distance source or word improvements may need propagation when descendant
metadata depends on them, even though hop distances themselves are already
final.

## 10. Target and early-stop boundaries

First target discovery at exact depth proves scalar distance under note 162's
conditions. It does not prove the shortlex target word.

Shortlex finalization needs proof that no unprocessed equal-depth proposal to
the target has a smaller key. Sufficient evidence can be:

- complete expansion/reduction of all relevant depth-`d` parents; or
- an ordered-prefix theorem proving every remaining parent-word/label key is no
  smaller than the current target key.

Notification that one owner found the target is not that closure proof.

## 11. Cayley and quotient boundaries

For a Cayley graph, the recurrence selects a shortlex normal-form word relative
to the declared generators. Group relations create multiple candidate words;
exact state identity merges them before the minimum word is retained.

For a Schreier or symmetry quotient, the chosen word is least for the quotient
state/fiber. It need not be a canonical word to one fixed concrete
representative unless path lifting and target-frame semantics prove that claim.

Owner-block or quotient-word order likewise cannot replace concrete-state
shortlex reduction.

## 12. Validation fixtures and fields

At minimum validate:

- the `ab` versus `ba` state-ID counterexample;
- shuffled within-level proposal arrival;
- different frontier partitions and owner counts;
- a late smaller proposal after a larger first winner;
- retry of the current minimum proposal;
- loss mutation of the minimum proposal;
- generator-order reversal as an intentional epoch change;
- exact replay of reconstructed words;
- quotient versus fixed-representative targets where applicable.

Record per level:

```text
all contender keys or an independently checked digest/set,
first-winner versus canonical-winner differences,
selected parent/label keys,
canonical word ranks,
rank/owner/device count,
generator/action/output epoch,
closure event and overflow status.
```

## 13. Rejected implications

- Deterministic parent means shortlex path.
- Minimum parent state ID gives the least move word.
- Sorting frontier states by semantic ID gives canonical word order.
- Owner-local ranks concatenate into global shortlex ranks.
- Retry-safe `min` makes lost contenders harmless.
- Exact frontier-set parity validates canonical parents.
- First target discovery finalizes its shortlex word.
- Hash order is a lexical path order.
- Quotient shortlex is automatically fixed-representative shortlex.

## 14. Current synthesis

Shortlex determinism is a dynamic order on paths, not a static order on states.
The correct recursive key is the canonical parent-word rank followed by the
edge-label rank. Exact identity says which words compete; complete minimum
reduction chooses the winner; global ranking makes winners comparable at the
next depth.

This separates three often conflated objects: deterministic state sets,
deterministic parent IDs, and deterministic canonical words. Only the last uses
the path-rank recurrence.

This note extends notes 13, 16, 17, 19, 28, 30, 57, 162, 172, and 173.


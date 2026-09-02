# REF-004: effects of generator relations on BFS work

Date: 2026-08-27.

## Question

How does modifying the generator set of the same finite group redistribute
candidate rejection among batch deduplication, earlier visited levels, and the
current frontier?

## Cases

All cases traverse all 40,320 states of `S8` from the identity.

1. `base`: seven adjacent transpositions.
2. `plus_identity`: base plus the identity permutation.
3. `plus_duplicate_s0`: base plus a second label for the first adjacent swap.
4. `plus_3cycle_pair`: base plus `(0 1 2)` and its inverse.

The identity and duplicate-generator cases preserve the graph's reachable
states and distances. The 3-cycle pair adds new edges and can change distances.

## Exact decomposition

For each level, current frontier states are inserted into visited before
classification. Exact tuple-state sets then partition generated occurrences as:

```text
generated
  = batch_duplicate_occurrences
  + unique_earlier_hits
  + unique_same_level_hits
  + accepted
```

The identity is checked for every case, and `accepted` is checked against the
next recorded frontier.

## Results

| case | generators | max depth | peak frontier | generated | batch duplicates | earlier hits | same-level hits | accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 7 | 28 | 3,836 | 282,240 | 201,602 | 40,319 | 0 | 40,319 |
| plus identity | 8 | 28 | 3,836 | 322,560 | 201,602 | 40,319 | 40,320 | 40,319 |
| plus duplicate `s0` | 8 | 28 | 3,836 | 322,560 | 241,922 | 40,319 | 0 | 40,319 |
| plus 3-cycle pair | 9 | 22 | 4,420 | 362,880 | 245,495 | 40,169 | 36,897 | 40,319 |

## Observations

**Identity generator:** The additional 40,320 generated transitions become
40,320 unique same-level hits. Batch dedup count remains unchanged because an
identity child is the current state and, in this bipartite base graph, is not
also produced by an adjacent transposition.

**Duplicate generator:** The additional 40,320 transition occurrences become
40,320 additional batch duplicates. The unique candidate set is unchanged.

**3-cycle pair:** The added even generators create odd cycles/same-level edges.
They reduce graph diameter but broaden the peak frontier and increase total
generated work. A lower diameter therefore does not imply a smaller maximum
frontier or less total work.

## Insights

1. A scalar duplicate ratio is insufficient for architecture decisions.
2. Duplicate generator outputs can be removed before visited lookup and before
   communication, if their detection costs less than carrying them.
3. Identity and same-level edges require current-frontier membership in the
   visited semantics; checking only earlier levels is wrong.
4. Generator-set preprocessing can detect exact duplicate and identity
   generators once, eliminating predictable work without changing the graph.
5. Nonredundant added generators may reduce depth while increasing degree,
   frontier peak, and total transition count. The objective determines whether
   this trade is beneficial.

## Architecture implications

Candidate rejection should be measured in at least three stages:

```text
generator-level redundancy
candidate-batch exact dedup
authoritative visited filtering (current and earlier levels)
```

For multi-GPU BFS, a fourth category is needed: duplicates generated on
different source ranks that meet only after owner routing.

## Limitations

- Counts are exact for this graph but no kernel or communication cost was
  measured.
- Python set construction represents an idealized global batch dedup, not a
  bounded-memory streaming implementation.
- Only one state space and four deliberately chosen generator sets were tested.

## Next experiment

Simulate multiple source ranks and owner routing over the same recorded levels.
Measure how many duplicates are removed:

- locally before exchange;
- only after candidates meet at the authoritative owner;
- by cumulative owner-local visited.

Vary the number of ranks and the state-to-owner hash to expose skew and the
maximum possible communication reduction from local dedup.

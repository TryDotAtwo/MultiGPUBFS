# REF-003: S8 level-wise candidate rejection breakdown

Date: 2026-08-27.

## Question

At each BFS depth, how many generated transitions can be removed by exact
candidate-batch deduplication before consulting the cumulative visited set?

## Definitions

For one complete level:

```text
candidates = every generated child occurrence
unique_candidates = exact set(candidates)
batch_duplicate_occurrences = len(candidates) - len(unique_candidates)
unique_visited_hits = len(unique_candidates intersect visited_through_this_level)
accepted = len(unique_candidates minus visited_through_this_level)
```

The conservation identity checked at every depth is:

```text
generated = batch_duplicate_occurrences + unique_visited_hits + accepted
```

`accepted` is also checked against the literal size of the next recorded
frontier.

## Graph and environment

Same adjacent-transposition `S8` graph and Python environment as REF-002.
All comparisons use full tuple states, not hashes.

## Result highlights

- Maximum frontier: 3,836 states at depth 14.
- Depth 14 generated 26,852 transitions.
- Exact batch dedup reduced these to 7,472 unique candidates.
- 19,380 generated occurrences (72.17%) were duplicate occurrences within the
  level's combined candidate batch.
- Of the unique candidates, 3,736 were already visited and 3,736 were accepted.
- At the final depth, all seven generated transitions were visited hits and the
  next frontier was empty.

The complete raw table is stored in
[`REF-003-s8-level-breakdown.csv`](REF-003-s8-level-breakdown.csv).

## Interpretation

**Fact for this graph:** Exact pre-visited batch dedup can reduce candidate count
substantially, especially around the middle levels.

**Inference:** In a distributed owner-computes BFS, local pre-network dedup may
save substantial payload bytes. This inference is weaker than a performance
result because different source ranks can still generate the same state, and
local sorting/compaction consumes time and scratch memory.

**Unknown:** The best placement may be two-stage: cheap block/rank-local dedup,
exchange by owner, then authoritative owner-side dedup and visited lookup.

**Unknown:** Adjacent transpositions give fixed degree and strong algebraic
relations. Other generator sets can have very different duplicate locality.

## Methodology correction retained

During the first ad-hoc calculation, `visited` was updated with the current
frontier after candidate classification rather than before it. Adjacent
transpositions change permutation parity, so this graph has no same-level edges
and the numbers happened to remain identical. The command was corrected and
rerun with the current frontier inserted before classification. Future graph
families must not rely on bipartiteness.

This near-miss is retained because the original ordering would misclassify
same-level edges as accepted states on non-bipartite graphs.

## Next experiment

Repeat the decomposition for generator sets with:

- an identity generator;
- redundant generators;
- odd cycles or same-level edges;
- long and short group relations.

This will test how strongly pre-dedup benefit depends on algebraic structure.

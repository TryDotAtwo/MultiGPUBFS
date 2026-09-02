# REF-032: duplicate queue semantics

Status: pass after intentional RED, one corrected expectation, and one
formatting-only failure.

## Question

Can a tiny exact Rust oracle separate unique-state BFS, duplicate-tolerant
settlement, and walk-occurrence expansion without turning the exercise into an
optimized queue implementation?

## Fixtures and contracts

1. A directed root, 100 depth-one parents, and 100 depth-two children with a
   complete bipartite parent/child boundary.
2. A depth-12 directed acyclic graph with two vertices per positive layer and
   all four edges between consecutive two-vertex layers.
3. `claim_before_enqueue`: exact state is claimed before one queue insertion.
4. `settle_on_dequeue`: duplicate records may queue, but only the first popped
   copy expands.
5. `expand_every_occurrence`: every path-prefix record expands, deliberately
   representing walks rather than graph-BFS states.

There are no timings, GPU operations, production data structures, or tuning.

## Test-first and failure log

1. Three tests were written against `unimplemented!` schedule functions. The
   Docker RED run failed all three for the intended reasons.
2. Before implementing, manual queue replay corrected one expected peak from
   four to six: two current-layer winners enqueue the next layer while two stale
   current-layer copies remain live.
3. The first GREEN gate stopped at `rustfmt --check`; no tests ran in that
   attempt. The proposed formatting-only diff was applied.
4. The final read-only-mounted Docker gate installed `rustfmt` inside the
   disposable container, passed formatting and three tests, compiled and ran
   the executable, and hashed the unchanged source.

## Observed results

### Complete bipartite boundary

| schedule | enqueued | expanded | stale pops | peak queue |
|---|---:|---:|---:|---:|
| claim before enqueue | 201 | 201 | 0 | 199 |
| settle on dequeue | 10,101 | 201 | 9,900 | 10,000 |

Both schedules produced identical distances for all 201 states. The
duplicate-tolerant schedule retained one unique expansion per state while its
queue represented all 10,000 boundary occurrences.

### Two-vertex layered DAG

Stale suppression produced 47 total enqueues, 25 unique expansions, 22 stale
pops, and peak queue six. Expanding every occurrence instead yielded

```text
[1,2,4,8,16,32,64,128,256,512,1024,2048,4096].
```

The graph has only 25 semantic vertices. The final 4,096 records are path
prefixes at depth 12, not new states.

## Interpretation and limits

The experiment validates the bounded fixtures and the counter taxonomy in note
74. It does not prove that one physical schedule is faster, predict a GPU queue,
or advocate delayed settlement. The peak-six correction is itself useful: even
with stale suppression, live stale records can overlap newly generated records
from the next distance layer.

Raw output, command, toolchain, image digest, and source hash are retained in
`REF-032-duplicate-queue-semantics.txt`.


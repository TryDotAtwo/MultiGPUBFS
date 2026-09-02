# Pattern databases: abstraction and BFS distance heuristics

A pattern database (PDB) stores exact distances in an abstract state graph and
uses them as lower bounds in a larger concrete search. Reverse BFS is often the
table builder, but the table's guarantee comes from the abstraction map, not
from the word "BFS" alone.

This note separates abstract lower bounds from exact concrete goal-neighborhood
suffixes such as CayleyPy K1. It does not implement A*, IDA*, or a PDB builder.

## Concrete and abstract graphs

Let the concrete directed unit graph be

```text
G = (V,E), with concrete target set T.
```

Let the abstract graph be

```text
A = (V_A,E_A), with abstraction pi : V -> V_A.
```

The fundamental path-projection obligation is:

```text
for every concrete edge x -> y,
the abstraction admits pi(x) -> pi(y)
with abstract cost at most 1,
possibly as a zero/no-change step.
```

Let the abstract target set `T_A` contain `pi(T)`. Define

```text
h(x) = dist_A(pi(x), T_A).
```

A reverse shortest-path traversal from `T_A` computes these abstract
distance-to-goal labels. When every abstract edge has unit cost, ordinary
multi-source reverse BFS is sufficient. If abstract edge costs include zero,
0-1 BFS or an equivalent shortest-path method is required.

## Why the heuristic is admissible

Take any concrete path

```text
x=x_0 -> x_1 -> ... -> x_L=t,  t in T.
```

Projecting each step gives an abstract walk from `pi(x)` to `pi(t) in T_A` whose
total cost is at most `L`. The shortest abstract path can only be cheaper:

```text
h(x) <= L.
```

Taking the minimum over all concrete goal paths yields

```text
h(x) <= dist_G(x,T).
```

This is admissibility. It needs every concrete path to project; it does **not**
need every abstract path to lift back to a concrete solution. Spurious abstract
shortcuts weaken the lower bound but do not make it overestimate.

This is the opposite direction from using an abstract path as a returned
solution. An upper-bound witness needs a valid concrete lift and replay.

## Why exact abstract BFS matters

Suppose the table stores `H(a)` larger than the true abstract distance of some
state `a`. Then for a concrete `x` with `pi(x)=a`, the projection proof only
gives

```text
dist_A(a,T_A) <= dist_G(x,T),
```

not `H(a) <= dist_G(x,T)`. An overestimated table entry can therefore destroy
admissibility.

Underestimation remains admissible but weak. Consequently:

- false-positive visited merging during PDB construction can suppress a short
  abstract route and create an overestimate or false infinity;
- incomplete construction cannot label missing entries as unreachable;
- a hash collision interpreted as abstract-state equality can invalidate the
  lower bound, not merely reduce heuristic quality;
- exact identity, completed layers, and explicit overflow status are required
  even though the PDB is "only a heuristic."

An independently replayed concrete solution does not validate all PDB lower
bounds used to prune other branches.

## Consistency follows from the same edge condition

For a concrete unit edge `x -> y`, the corresponding abstract edge and triangle
inequality give

```text
h(x) <= 1 + h(y).
```

This is heuristic consistency (monotonicity) for unit costs. More generally,

```text
h(x) <= c(x,y) + h(y).
```

Consistency implies admissibility when goal labels are zero and every relevant
state reaches a goal, but it also controls search scheduling: along a concrete
edge, `f=g+h` cannot decrease solely because the heuristic violates the local
triangle inequality.

Directed orientation is essential. `h(x)` is distance **from `x` to the goal**,
so the table is normally constructed by traversing predecessor edges outward
from the abstract goals. Running forward BFS from a goal in an asymmetric graph
computes the wrong directed quantity.

## Projection, quotient, and relaxation

An abstraction may forget puzzle pieces, orientations, history fields, or
constraints. Several concrete states then share one abstract state.

For a lower bound, it is safe for the abstract graph to be a relaxation:

- merge concrete vertices;
- include additional abstract transitions;
- broaden the abstract target set;
- assign no greater cost to the image of a concrete transition.

Each operation can only make the abstract target easier to reach.

For exact distance preservation, the obligations are stronger. One would need
appropriate path lifting/no-shortcut properties, as in note 17. A useful PDB is
usually intentionally **not** exact for every concrete state; its purpose is a
cheap informative lower bound.

## One PDB versus several

Let `h_1,...,h_k` be admissible heuristics for the same concrete objective.

### Maximum is safely admissible

```text
h_max(x) = max_i h_i(x)
```

satisfies

```text
h_max(x) <= dist_G(x,T)
```

because every term does. If every `h_i` is consistent for the same edge costs,
their maximum is also consistent:

```text
max_i h_i(x)
<= max_i (c(x,y)+h_i(y))
= c(x,y)+max_i h_i(y).
```

The maximum can improve informativeness without double-counting one move.

### Sum is not automatically admissible

Suppose state `x` is one move from the goal and that move simultaneously fixes
features represented in two abstractions. Each PDB can legitimately report

```text
h_1(x)=1
h_2(x)=1,
```

while the concrete distance is one. Then

```text
h_1(x)+h_2(x)=2 > 1.
```

Disjoint abstract feature sets do not alone prevent this: one concrete operator
may affect several patterns.

## Additivity needs cost partitioning

For each concrete edge `e`, assign nonnegative abstract costs `c_i(e)` such that

```text
sum_i c_i(e) <= c(e).
```

Build each abstract distance `h_i` using its assigned costs. Project any
concrete path `P`; the sum of abstract path costs is bounded by

```text
sum_i cost_i(P)
= sum_(e in P) sum_i c_i(e)
<= sum_(e in P) c(e).
```

Taking abstract shortest paths independently can only reduce the left side, so

```text
sum_i h_i(x) <= dist_G(x,T).
```

This is the core additive-PDB proof. Operator disjointness is one sufficient
way to allocate each move cost to at most one pattern, but cost partitioning is
the general obligation. If abstract moves have costs other than one, the table
builder must use the matching 0-1 or weighted shortest-path algorithm rather
than ordinary BFS by habit.

## A bounded abstract table still gives a capped lower bound

Suppose reverse BFS completes the exact abstract ball of radius `R` around
`T_A`.

- A stored abstract state has its exact abstract distance `d<=R`.
- A state absent from the completed ball has abstract distance greater than
  `R`, or infinity.

For integer unit costs, the capped heuristic

```text
h_R(x) = stored_distance(pi(x)) if present
         R+1                    if absent
```

is admissible, provided the radius construction and lookup identity are exact.
It throws away information beyond `R+1` but does not turn a bounded miss into an
unreachable claim.

If construction was incomplete, an absent entry is `UNKNOWN` and cannot safely
receive `R+1`: the true abstract distance might be smaller than or equal to `R`.

Here R must be the certified completed radius, not the requested radius.
An interrupted build can still retain a smaller complete ball of radius r<R.
If that ball and its exact membership remain available, absence from it proves
distance at least r+1 even though the requested radius-R table is unfinished.
For the directed chain `b -> a -> t`, reverse BFS from t completed through
radius one stores t:0 and a:1. A build intended to reach radius two but stopped
at that point cannot assign b the requested-radius miss bound 3: its distance
is 2. The completed-radius bound 2 is valid. Partial work beyond a certified
ball does not erase that certificate, but absent shards or lost membership
must not be mistaken for certified absence from it.

## Concrete reverse ball versus abstract PDB

These two reverse-BFS structures have different semantics:

| Property | Exact concrete goal ball | Abstract PDB |
|---|---|---|
| vertices | full concrete states | equivalence/relaxed abstract states |
| stored distance | exact concrete residual within radius | exact abstract distance |
| positive entry | lower bound and exact residual; may store concrete suffix | lower bound only in general |
| path replay | can provide a concrete upper-bound witness | abstract path need not lift |
| miss from complete radius | concrete distance `>R` | abstract distance `>R`, hence concrete distance `>R` |
| full-table coverage | may be infeasible | often designed to be enumerable |

CayleyPy K1 intends the first kind: a bounded concrete reverse neighborhood
with stored forward suffixes. Treating it as a PDB would understate a genuine
positive suffix certificate. Conversely, treating an abstract PDB entry as a
concrete suffix would overstate its guarantee.

Both still require exact table identity. Bare fingerprints without collision
resolution can corrupt either concrete suffix membership or abstract lower
bounds.

## Interaction with learned scores and beam search

A learned score may correlate with remaining distance without satisfying

```text
score(x) <= true_remaining_distance
```

or the consistency inequality. It is therefore not an admissible heuristic
unless separately calibrated and proved under the exact graph/target contract.

Even an exactly computed abstract PDB does not generally make fixed-width beam
search complete or optimal for the concrete graph. Its entries may underestimate
concrete distance. If the heuristic instead equals the concrete target distance
everywhere, note 24's separate exact-descent proof applies under its stated
successor/filter conditions; that still does not recover full BFS layers.
Admissibility justifies lower-bound pruning rules that explicitly use
an incumbent upper bound, such as discarding states with certified
`g+h >= incumbent` under the requested tie/output convention. Top-k removal is
not such a proof merely because states were ranked by `h`.

Likewise, an exact K1 suffix provides an incumbent upper bound for a retained
state but cannot certify branches removed earlier by beam width.

## GPU and multi-GPU lookup semantics

PDB lookup can be physically attractive when abstract states have a dense exact
rank and the table is read-only. The semantic operation is still

```text
concrete state
-> exact abstraction
-> exact abstract rank/key
-> distance entry under matching versions.
```

Relevant conceptual costs include abstraction, ranking, random table access,
entry width, cache locality, and any decompression. Lookup throughput alone does
not validate the heuristic values.

For multiple GPUs, a PDB may be replicated, sharded, or cached. These choices
should preserve one immutable table contract:

- identical abstraction, generator, cost-partition, target, and version;
- no missing shard interpreted as infinity or a large bound;
- exact routing of each abstract key;
- globally visible construction/validation completion before search;
- explicit handling of communication or cache misses as latency, not semantic
  absence.

Replication can reduce communication at a memory cost. Sharding can increase
capacity while adding remote lookup to the critical path. Neither choice changes
admissibility if every lookup returns the same exact entry.

## Validation checklist

1. What concrete state fields are forgotten or merged by `pi`?
2. Does every concrete edge project to an abstract edge of no greater cost?
3. Does the abstract target contain every projected concrete goal?
4. Is the directed reverse orientation correct?
5. Were exact abstract shortest distances computed under the assigned costs?
6. Can table collision, overflow, truncation, or missing shards overestimate?
7. Is a miss `>R`, `UNKNOWN`, or truly unreachable?
8. Are multiple heuristics combined by `max`, or is additivity proved by cost
   partitioning?
9. Is an abstract lower bound being mistaken for a concrete replayable suffix?
10. Are table, abstraction, moves, target, and cost allocation versioned
    together?

## Counterexamples and rejected shortcuts

### Any projection produces an admissible PDB

If a concrete edge has no abstract image or is assigned greater cost, a concrete
short path may look longer abstractly and the heuristic can overestimate.

### Exact BFS construction makes the PDB exact concretely

BFS is exact in the abstract graph. Relaxation/merging may make its distance
strictly smaller than the concrete one.

### Two admissible PDBs can always be summed

One concrete move may pay both abstract distances, producing a sum larger than
the real cost.

### Disjoint pieces automatically imply additive heuristics

The relevant condition is operator cost allocation, not only disjoint state
features.

### An abstract path is a concrete solution

Projection is sufficient for a lower bound; an upper-bound witness additionally
needs path lifting and concrete replay.

### PDB-guided beam becomes exact

An admissible ranking value does not certify arbitrary top-k deletion.

## Sources

- Joseph Culberson and Jonathan Schaeffer,
  [Pattern Databases](https://webdocs.cs.ualberta.ca/~jonathan/publications/ai_publications/pattern.pdf),
  develops abstraction-based stored distance heuristics.
- Richard Korf,
  [Finding Optimal Solutions to Rubik's Cube Using Pattern Databases](https://cdn.aaai.org/AAAI/1997/AAAI97-109.pdf),
  applies PDB lower bounds to optimal cube search.
- Ariel Felner, Richard Korf, and Sarit Hanan,
  [Additive Pattern Database Heuristics](https://www.jair.org/index.php/jair/article/view/10315),
  develops conditions under which multiple pattern costs can be added.
- Notes 12, 17, 24, 28, 40, 42, and 45 provide the weighted-search boundary,
  quotient/lifting direction, beam distinction, identity contract, concrete
  reverse-ball semantics, bounded misses, and dense-rank context used here.

## Current conclusions

1. A PDB is exact about distances in its abstract graph and generally supplies
   only lower bounds for the concrete graph.
2. Admissibility follows because every concrete goal path projects to an
   abstract walk of no greater cost; abstract-to-concrete lifting is unnecessary
   for that direction.
3. Exact abstract construction and identity remain mandatory because an
   overestimated entry can invalidate optimal pruning.
4. The maximum of admissible/consistent PDBs preserves those properties, while
   sums require per-operator cost partitioning.
5. A completed bounded PDB can safely return the capped lower bound `R+1` on a
   miss; an incomplete table cannot use its requested radius for that bound.
   A smaller certified complete ball can still supply its own miss bound.
6. CayleyPy K1 is intended as a concrete suffix neighborhood, not merely an
   abstract heuristic, while its hash-only identity remains a conditional
   correctness premise.
7. GPU replication or sharding changes lookup cost and capacity, not the
   abstraction proof, provided every lookup preserves one exact table version.

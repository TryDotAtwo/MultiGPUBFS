# Stabilizer-aware BFS frontier work waterfall

In a fixed-generator implicit traversal, the raw level work begins with
`|S||F_d|` labeled occurrences. Only `|F_(d+1)|` endpoint states enter the next
frontier. The difference is not one homogeneous duplicate bucket: loops,
same-parent aliases, visited support arcs, and cross-parent convergence disappear
at different semantic boundaries.

This note derives an exact per-level waterfall. It is an accounting model, not
an implementation prescription or performance optimizer.

## 1. Per-parent quantities

For frontier state `x`, let

```text
K_x = Stab(x),
g(x) = |S|,
l(x) = |S intersect K_x|,
u(x) = |{K_x s : s in S}|.
```

Here `l(x)` is the number of labeled self-loop occurrences and `u(x)` is the
number of distinct support endpoints including `x` when a loop exists. Define

```text
z(x) = 1 if l(x)>0, else 0,
q(x) = u(x)-z(x).
```

Then `q(x)` is the number of distinct nonloop support neighbors. Nonloop raw
occurrences number `|S|-l(x)`, so same-parent nonloop label excess is

```text
r(x) = |S|-l(x)-q(x) >= 0.
```

This collapses every nonloop alias class to one parent-child support arc. Loop
labels are kept as their own category because none can create a new state.

## 2. Level aggregates

For complete frontier `F_d`, define

```text
G_d = |S| |F_d|                       raw labeled occurrences,
L_d = sum_(x in F_d) l(x)             loop occurrences,
R_d = sum_(x in F_d) r(x)             nonloop same-parent label excess,
P_d = sum_(x in F_d) q(x)             distinct nonloop support arcs.
```

The parent-local identity is

```text
G_d = L_d + R_d + P_d.
```

No visited information has been used yet. This is solely generator-to-support
projection under the state stabilizers.

## 3. Support arcs split by BFS status

Every nonloop support arc from `F_d` ends either in visited ball `B_d` or in
next frontier `F_(d+1)`. Let

```text
V_d = distinct nonloop parent-child support arcs ending in B_d,
C_d = distinct parent-child support arcs ending in F_(d+1).
```

Then

```text
P_d = V_d + C_d.
```

Every state in `F_(d+1)` has at least one support parent in `F_d`. Therefore

```text
D_d = C_d-|F_(d+1)| >= 0
```

is cross-parent support convergence beyond one parent arc per new state.

## 4. Exact waterfall identity

Substituting the two partitions gives

```text
G_d = L_d + R_d + V_d + D_d + |F_(d+1)|.
```

For vertex-frontier membership:

- `L_d`: loops, immediately old;
- `R_d`: extra labels for a nonloop support arc from the same parent;
- `V_d`: one representative of each distinct support arc to an old state;
- `D_d`: extra distinct parents of new states;
- `|F_(d+1)|`: accepted unique endpoint states.

Thus total nonaccepting occurrence-equivalent mass is

```text
G_d-|F_(d+1)| = L_d+R_d+V_d+D_d.
```

This identity is exact even though an actual implementation may combine these
classes in another order or never materialize some intermediate objects.

## 5. Relation to note 157 counters

Note 157 used raw occurrences `T_d`, visited occurrences `Y_d`, next-layer
occurrences `X_d`, and distinct next support arcs. The new waterfall refines
the same-parent part across both visited and next destinations.

At aggregate level:

```text
T_d = G_d,
distinct nonloop support arcs = P_d,
distinct next support arcs = C_d,
cross-parent next excess = D_d.
```

`L_d+R_d` is the occurrence-to-nonloop-support loss. `V_d` is the
support-to-unvisited loss. `D_d` is the parent-arc-to-endpoint loss.

The word "loss" refers only to vertex-frontier membership. Rich outputs may
retain metadata from any class.

## 6. The three-point fixture under the waterfall

For the `S_3` action of notes 158-159, one parent always has `G=3`.

```text
state 1: endpoints 2,3,2 -> L=0, R=1, P=2
state 2: endpoints 1,2,3 -> L=1, R=0, P=2
state 3: endpoints 3,1,1 -> L=1, R=1, P=1
```

States one and three both have two total support endpoints when loops are
included, but their nonloop support-arc counts are two and one. Aggregate
support degree alone therefore cannot reconstruct loop and alias work.

`V`, `D`, and accepted-state terms still depend on the current BFS ball and the
other parents in the frontier. Stabilizer data determines only the first
projection stages.

## 7. Free Cayley simplification

For a free Cayley action with a set of distinct nonidentity generators,

```text
L_d=0,
R_d=0.
```

The waterfall reduces to

```text
|S||F_d| = V_d + D_d + |F_(d+1)|.
```

All raw generator occurrences are distinct nonloop support arcs from their own
parent. They can still hit visited states or converge across different parents
because of group relations. Free action removes length-one aliases, not general
BFS duplicates.

## 8. Algebraic interpretation

The categories correspond to different relation witnesses:

- loop label: `s in Stab(x)`;
- same-parent alias: `s t^-1 in Stab(x)`;
- visited nonloop arc: a generated word closes to an earlier or same-depth
  state;
- cross-parent convergence: two distinct predecessor states reach one new
  endpoint at equal depth;
- accepted endpoint: first state-level representative of a new metric layer.

These categories do not count independent group relators. They classify where
occurrence histories become redundant for one specific BFS output.

## 9. Output contracts change what survives

For vertex distance, only the final endpoint survives. For richer outputs:

- loop labels may be retained as action/cycle witnesses;
- same-parent labels may be distinct replayable moves;
- visited arcs may be relevant to cycle or relation analysis but not the
  shortest-predecessor DAG;
- all `D_d` parent arcs are required by the complete shortest-path DAG;
- path counts require weighted predecessor contribution, not only `D_d`.

A physical compaction may reduce state records while separately accumulating
these metadata. "Removed candidate" must name which output object was removed.

## 10. GPU pipeline interpretation without prescribing one

The waterfall suggests questions for any existing implementation:

- Are generator applications and state transforms paid before loops are known?
- Can same-parent aliases meet locally in the chosen expansion layout?
- Is visited tested before or after owner routing?
- At what scope do distinct parents for one child converge?
- Does metadata combination require the records that frontier compaction drops?

It does not imply that the order

```text
loops -> parent aliases -> visited -> cross-parent dedup
```

is fastest. Earlier filtering may require extra indexing, branching, storage,
or synchronization. The identity describes semantic volumes, not optimal
kernel structure.

## 11. Multi-owner matrices

Each scalar stage can be refined by source and destination owner:

```text
raw occurrence matrix,
distinct support-arc matrix,
unvisited support-arc matrix,
accepted unique-state matrix.
```

Collapsing aliases before routing changes wire records but may discard labeled
output. Testing visited before routing requires a trustworthy local replica or
owner query. Only the authoritative owner can finalize global cross-parent
convergence under the owner-computes model.

Equal `G_d` per owner does not prove equal `P_d`, `C_d`, accepted states, bytes,
or transformation cost.

## 12. Measurement boundaries

A useful trace records counts and bytes at every retained boundary, together
with:

- frontier stabilizer/alias histograms;
- generator transformation time;
- local versus owner-side combination;
- retry and idempotency traffic separately;
- retained parent/label/count metadata;
- capacity and overflow outcomes.

Ratios such as `R_d/G_d` or `D_d/C_d` describe composition. They are not speedup
predictions unless an actual implementation removes the corresponding work and
the remaining overhead is measured.

## Sources and internal dependencies

- Notes 10, 36, 57, 64, and 74 define candidate, frontier, metadata, and output
  objects.
- Notes 157-159 provide occurrence, stabilizer-coset, and direction-specific
  alias accounting.
- Notes 47, 51-52 provide work/span, owner authority, replicas, routing, and
  idempotency boundaries.
- The waterfall follows by two exact partitions: occurrences to nonloop support
  arcs, then support arcs to visited versus next endpoints.

## Takeaway

The gap between `|S||F_d|` and `|F_(d+1)|` has four distinct causes:

```text
loops + same-parent aliases + visited support arcs + cross-parent convergence.
```

Keeping them separate makes Schreier/Cayley duplicate telemetry interpretable
without pretending that a semantic count already determines an optimal GPU
implementation.

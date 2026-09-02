# Source-set updates move states between BFS layers

Adding a source can only improve scalar distances and enlarge every bounded
ball. It does not monotonically enlarge each exact-distance frontier: states
can leave a layer for an earlier one while other states enter from a later one.
This distinction matters for incremental BFS, validation, and checkpoint
identity.

## Monotone quantities under source insertion

Let source sets satisfy `S subset T`. For every vertex,

```text
d_T(v) <= d_S(v).
```

The old sources remain available in the larger minimization, so distance cannot
increase. Consequently, at every radius,

```text
B_d(S) subset B_d(T).
```

The reachable set can only grow, and an unreachable vertex may become finite.
These are genuine monotonicity laws.

## Exact frontiers are not monotone

Frontier `F_d` is a level set of the distance function, not a sublevel set.
When distance decreases:

- a former `F_d` state may move to `F_j` for `j<d` and leave the layer;
- a former deeper state may move exactly to distance `d` and enter it;
- a state may jump over `d` entirely;
- an unreachable state may enter any finite layer.

Therefore neither inclusion direction generally holds between `F_d(S)` and
`F_d(T)`.

## Equal-cardinality, different-membership witness

Take path

```text
0 -- 1 -- 2 -- 3 -- 4.
```

With `S={0}`, `F_2(S)={2}`. Add vertex `2` as a source, so `T={0,2}`.
Vertex `2` moves to `F_0`, while vertex `4` improves from old distance four to
new distance two:

```text
F_2(T)={4}.
```

The frontier cardinality remains one, but its membership is disjoint from the
old layer. A count-only incremental validator would accept a completely wrong
depth-two state set.

The same path also shows different behavior by depth: `F_1` grows from `{1}` to
`{1,3}`, while `F_4` becomes empty. There is no single layerwise monotonicity
direction.

## Source deletion reverses the scalar monotonicity

Removing sources can only increase distances or make vertices unreachable, and
balls can only shrink. Exact layers still migrate in both membership directions
relative to one fixed index. Reversing the path example from source set `{0,2}`
to `{0}` changes depth two from `{4}` back to `{2}`.

Deletion is not validated by checking that the final reached count decreased:
some vertices remain reachable but move later, while others may disappear
entirely.

## Parent and path-count consequences

An old parent edge can remain a valid graph edge after source insertion while
ceasing to be shortest under the new source set. Every state whose distance
decreases needs a parent chain rooted in an appropriate new nearest source.
States whose distance stays equal may gain additional tied-source predecessors.

Shortest-path counts are not monotone in general:

- if distance stays unchanged, newly tied nearest-source paths can add to the
  old minimum-length contributions;
- if distance decreases, all formerly shortest longer paths cease to belong to
  the new shortest-path sample space, and the new count may be smaller or
  larger;
- deletion can expose a longer alternative distance whose path family was not
  represented in the old shortest DAG.

Thus a distance field, one parent forest, predecessor DAG, and path counts have
different update requirements.

## Why ordinary irrevocable visited is insufficient for an update

A completed BFS visited set says that states were reached under the old source
epoch. On source insertion, membership often remains true while distance and
parent metadata improve. Treating every old visited state as closed prevents
those improvements from propagating to descendants.

An incremental method therefore needs relaxation/reactivation semantics for
decreased labels, or it must recompute. This observation specifies a proof
obligation; it does not choose an incremental algorithm.

Source deletion is more difficult semantically because a removed source may
have supported the only stored shortest parent tree. Alternative sources and
longer replacement paths cannot be recovered from Boolean visited membership
alone.

## Epoch and checkpoint identity

The source set belongs to the BFS problem identity. A checkpoint tied to `S`
cannot be resumed as if it were a completed-level checkpoint for `T` merely by
appending new sources to a queue:

- earlier distance labels may need improvement;
- already expanded states may need reactivation;
- parent/source tie choices may change;
- a previously empty frontier no longer proves closure for the new source set;
- termination and in-flight-work evidence refers to the old epoch.

Likewise, distributed owner placement may stay physically unchanged while the
semantic source/distance epoch changes. Ownership identity and search identity
are related metadata, not the same concept.

## GPU and multi-GPU interpretation

Adding sources is not merely increasing initial parallelism. It can reshape all
later frontiers, duplicate convergence, active-owner balance, and target-stop
depth. Comparing runs with different source sets is therefore a workload change,
not a pure scaling experiment.

Useful update evidence includes:

```text
distance decreases by old/new depth
states entering and leaving each exact frontier
newly reachable and newly unreachable states
reactivated expansions
changed parent/source ties
invalidated and rebuilt path-count contributions
global completion under the new source epoch.
```

No performance conclusion follows without such a concrete run and a declared
output contract.


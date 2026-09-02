# Incremental BFS: one-edge distance and output change cones

## Question

When one unit-cost edge is inserted, which source distances decrease, which
vertices merely gain equal-length shortest paths, and why is this different
from inserting a Cayley generator family?

Note 22 records insertion monotonicity and the usual local relaxation trigger.
This note gives the exact all-target formula for one insertion. It is a
sensitivity theorem, not a proposed dynamic-BFS implementation.

## 1. Directed single-edge formula

Let `G=(V,E)` be a finite directed unit-edge graph with source `s`. Insert one
new semantic arc

```text
e = (a,b)
```

and write old distances as `d_G`. For every target `v`, define

```text
c_e(v) = d_G(s,a) + 1 + d_G(b,v),
```

using infinity arithmetic. Then

```text
d_(G+e)(s,v) = min(d_G(s,v), c_e(v)).
```

### Proof

Every old path survives, giving the first candidate. A shortest new path that
uses `e` can be chosen simple because all edge costs are positive. It therefore
uses `e` exactly once and splits into:

1. an old-graph path from `s` to `a`;
2. the inserted arc `(a,b)`;
3. an old-graph path from `b` to `v`.

The best such path has length `c_e(v)`. Conversely, concatenating shortest old
paths for the prefix and suffix with `e` supplies that candidate whenever both
parts are finite. Taking the smaller old/new candidate proves the formula.

A self-loop has `c_e(v)>=d_G(s,v)` by the triangle inequality and cannot improve
scalar distances.

## 2. Three exact regions

The formula partitions targets into:

```text
decrease cone Q_e = {v : c_e(v) < d_G(s,v)}
equality cone Z_e = {v : c_e(v) = d_G(s,v) < infinity}
irrelevant region  = {v : c_e(v) > d_G(s,v)} plus unreachable infinities
```

- In `Q_e`, the scalar distance strictly decreases and every new shortest path
  must use `e`.
- In `Z_e`, scalar distance is unchanged but `e` supplies at least one new
  shortest path under semantic edge identity.
- In the irrelevant region, `e` appears in no shortest `s -> v` path.

Thus “affected” is output-dependent. A distance-only report sees `Q_e`; a
shortest-DAG or path-count report also sees `Z_e`.

## 3. The local trigger is necessary and sufficient for any decrease

At the inserted head,

```text
c_e(b) = d_G(s,a)+1.
```

If

```text
d_G(s,a)+1 >= d_G(s,b),
```

then for every `v`, the triangle inequality gives

```text
d_G(s,v) <= d_G(s,b)+d_G(b,v) <= c_e(v),
```

so `Q_e` is empty. Conversely, strict improvement at `b` puts `b` in `Q_e`.
Therefore

```text
Q_e is nonempty iff d_G(s,a)+1 < d_G(s,b).
```

This proves the scope of note 22's local relaxation test. It decides whether
any scalar distance can decrease, not how large the whole decrease cone is.

## 4. Equal-head insertion changes richer outputs

Suppose

```text
d_G(s,a)+1 = d_G(s,b).
```

No scalar distance decreases. The equality cone is exactly the old forward
shortest-path cone of `b`:

```text
Z_e = {v : d_G(s,v)=d_G(s,b)+d_G(b,v)}.
```

These are precisely the targets for which some old shortest path passes through
`b`. The new arc supplies another shortest prefix to `b` and therefore new
shortest paths to every such target, despite a completely unchanged distance
map.

If `d_G(s,a)+1>d_G(s,b)`, even the equality cone is empty: a path using `e` is
strictly longer than the old route through `b` for every target.

## 5. Shortest-path counts

Let `sigma_s(a)` be the number of old shortest paths from `s` to `a`, and
`sigma_b(v)` the number from `b` to `v`, under a declared finite path identity
and exact nonoverflow arithmetic.

For `v in Q_e union Z_e`, concatenation through the new semantic edge creates

```text
sigma_s(a) * sigma_b(v)
```

new paths of length `c_e(v)`. Positive unit costs and minimality exclude a
repeated-vertex cycle in such a concatenation. In `Z_e`, this quantity is added
to the old shortest-path count. In `Q_e`, old shortest paths are now too long,
so the new shortest count is the through-`e` product rather than old count plus
that product.

Parallel endpoint-identical edges make the path-identity contract material:
they preserve scalar distance but can add labeled or edge-occurrence paths.

## 6. New predecessor DAG

For a target in `Q_e`, the new shortest-path DAG can include:

- old shortest-prefix arcs on routes from `s` to `a`;
- the inserted arc;
- old shortest-suffix arcs on routes from `b` to the target.

Old source-shortest DAG membership alone is insufficient for the suffix. An
arc can be shortest relative to root `b` without belonging to the old
source-`s` predecessor DAG. The exact formula uses the two-point distance
`d_G(b,v)`, not merely descendants of `b` in the old source DAG.

This contrasts with decremental note 186: deletion invalidation is classified
inside the old source DAG, whereas insertion sensitivity can require a new
rooted distance row from the inserted head.

## 7. Undirected insertion

For one new undirected edge `{a,b}`, a shortest path can use it in either
orientation but, being simple, at most once. Hence

```text
d_(G+{a,b})(s,v) = min(
    d_G(s,v),
    d_G(s,a)+1+d_G(b,v),
    d_G(s,b)+1+d_G(a,v)
).
```

The two oriented candidates need separate equality/decrease accounting. They
cannot both be blindly added to path counts without checking whether they
represent distinct shortest paths under the declared edge orientation and
identity.

## 8. Several insertions do not decompose independently

With a set of inserted arcs, a new shortest path may use several of them. The
minimum of independent one-edge formulas can therefore miss the answer.

Counterexample: start with three isolated vertices `s,x,t` and insert

```text
(s,x), (x,t).
```

Neither insertion alone makes `t` reachable. Together they give distance two.
Batch insertion requires closure under paths alternating old subpaths and
several new arcs, or an exact traversal/relaxation on the updated graph.

The one-edge theorem remains a calibration oracle for a declared single
update, not a universal batch formula.

## 9. Cayley and Schreier generator insertion

Adding a generator label is not one edge insertion. It adds a translated action
arc from every state where the move applies. A new shortest word may use the
new generator several times, so no single `(a,b)` formula captures the changed
word metric.

Special cases still separate cleanly:

- an identity generator adds loops and changes no scalar distance;
- a duplicate transformation changes labeled multiplicity but not support
  distance;
- a generator already expressible by an old word can shorten distances if its
  new unit cost beats that word length;
- a genuinely new generated direction can merge components or change the
  reachable orbit.

Application of the one-edge theorem to one sampled state transition is local
evidence only; it cannot certify the global generator update.

## 10. GPU and multi-GPU interpretation

The formula identifies semantic work coordinates without selecting an
implementation:

- `d_G(s,.)` is the old source label row;
- `d_G(b,.)` is a head-rooted sensitivity row;
- `Q_e` counts scalar relabeling;
- `Z_e` counts unchanged labels with richer-output additions;
- path-count products require exact prefix/suffix counts;
- an incremental relaxation wave seeded at `b` must close globally before
  exact updated labels are published.

For distributed execution, owner-local absence of improvements does not prove
`Q_e` empty elsewhere. A complete result needs global closure of all proposals
whose candidate depth can beat an old label. Separate counters should report
strict label decreases, equal-label DAG contributions, routed proposals,
duplicate proposals, and final accepted changes.

Evaluating the formula on bounded fixtures would validate semantics, not GPU
speed. A real performance claim would additionally need the cost of obtaining
or avoiding the full `b`-rooted distance row.

## 11. Counterclaims rejected

- **If the head label does not decrease, the insertion changes nothing.** It
  may add equal-length parents and shortest paths throughout `Z_e`.
- **Every graph descendant of an improved head decreases.** Only vertices
  satisfying the exact inequality belong to `Q_e`.
- **The old source shortest DAG contains the whole insertion cone.** New
  suffixes are shortest from `b`, not necessarily from `s` before insertion.
- **Independent single-edge sensitivities solve a batch.** New paths can chain
  several inserted edges.
- **One Cayley generator insertion is one-edge sensitivity by symmetry.** It is
  a global edge-family update and words can reuse the new label.

## Sources and dependencies

- Note 11 supplies shortest-DAG and path-count output contracts.
- Note 22 supplies dynamic-graph versioning, insertion monotonicity, and the
  local head-relaxation rule.
- Note 77 distinguishes source-set label migration from graph-edge updates.
- Note 186 gives the complementary deletion invalidation theorem.
- The single-edge formulas and cones above follow directly from positive-cost
  path decomposition and the triangle inequality.

## Compact conclusion

One directed edge insertion has an exact all-target formula: compare every old
distance with the best old prefix to its tail, the new edge, and the best old
suffix from its head. Strict inequality changes scalar labels; equality changes
richer shortest-path outputs only. Several inserted edges and generator-family
updates require closure beyond this one-edge decomposition.

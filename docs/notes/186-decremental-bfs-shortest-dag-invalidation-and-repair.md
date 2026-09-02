# Decremental BFS: shortest-DAG invalidation versus distance repair

## Question

After deleting edges or vertices from an unweighted graph, which old BFS
distances are still exact, and what can the old shortest-path DAG prove before
any replacement-distance search runs?

Notes 22 and 118 establish deletion monotonicity and explain why neither one
parent tree nor the old shortest-path DAG contains every longer detour. This
note isolates a stronger positive fact: the old DAG exactly classifies which
old scalar distances survive. It still does not compute the increased values.

No executable experiment, dynamic data structure, or optimized implementation
is introduced.

## 1. Old distance labels and shortest-path DAG

Fix a source `s` in a finite directed or undirected unit-edge graph `G`. Let

```text
d(v) = dist_G(s,v)
```

and orient the complete source shortest-path DAG `D` by retaining every
semantic arc `(u,v)` satisfying

```text
d(u) + 1 = d(v).
```

Every directed `s -> v` path in `D` has length `d(v)`, and every shortest
`s -> v` path in `G` appears in `D` under the declared edge/label identity.

Let a deletion set `F` remove graph edges and/or vertices, and write `D-F` for
the surviving part of the old DAG. Define

```text
R = vertices reachable from s in D-F.
```

Failed vertices themselves are outside the query universe.

## 2. Exact preservation theorem

For every surviving vertex `v` that was reachable before the deletion,

```text
dist_(G-F)(s,v) = d(v)    iff    v is in R.
```

### Forward direction

If `v` is reachable in `D-F`, the surviving DAG path has length `d(v)`, so

```text
dist_(G-F)(s,v) <= d(v).
```

Deletion cannot shorten distances, giving the reverse inequality.

### Reverse direction

If the post-deletion distance still equals `d(v)`, a surviving path of that
length is also an old shortest path. By completeness of `D`, all its arcs lie
in `D`; because the path survives `F`, it lies in `D-F`, hence `v in R`.

Therefore the old shortest DAG is an exact certificate for preservation of the
old scalar labels. Its limitation begins only after a label is known to be
invalid: a longer replacement path may leave `D`.

## 3. Invalidation and repair are different phases

Let

```text
A = old reachable surviving vertices minus R.
```

Every vertex in `A` must strictly increase its distance or become unreachable.
This is the exact **old-label invalidation set**.

The old DAG answers:

```text
which labels changed?       exactly, through reachability in D-F
what are their new values?  not in general
```

Computing new values requires surviving graph edges outside the old DAG. A
repair process may discover longer incoming detours into `A`, build new layers,
and admit arcs that were not shortest before the deletion.

This distinction prevents a common overclaim: exact invalidation is not exact
dynamic BFS repair.

## 4. Support-count propagation

Because every DAG arc increases old depth by one, `R` can be characterized
layer by layer. The source survives by definition. A nonfailed vertex `v` at
old depth `k>0` belongs to `R` exactly when it has at least one surviving old
shortest predecessor in `R`:

```text
support_F(v) = |{u in R : (u,v) in D-F}| > 0.
```

Starting from directly damaged predecessor arcs, a vertex becomes invalid when
its last surviving reachable support disappears; that invalidation can remove
support from the next old layer. This is a monotone wave over the old DAG.

The counter is not the new distance. Zero says only that no old-length witness
survived. A longer predecessor from the full graph can still repair the vertex
at a greater depth.

## 5. Why one selected parent subtree is too large

Choose any BFS parent tree `T`. For deletion of a selected tree edge `(a,b)`,
every vertex whose old distance changes must lie in the tree subtree rooted at
`b`: its selected shortest path used the deleted edge.

But the converse fails. In the diamond

```text
    a
   / \
  s   t
   \ /
    b
```

choose parents `parent(a)=s`, `parent(b)=s`, and `parent(t)=a`. Deleting
`(a,t)` places `t` in the selected damaged subtree, yet `s-b-t` preserves its
distance two. The tree subtree is therefore an overapproximation of `A`.

For deletion of a non-tree edge, the selected tree itself supplies a surviving
old shortest path to every vertex, so no scalar distance changes. Other output
contracts can still change: the deleted edge may belong to the complete
shortest DAG or contribute shortest paths.

## 6. Dominance interpretation

For a single old-DAG edge `e`, a vertex `v` lies in the invalidation set exactly
when every `s -> v` path in the old shortest DAG uses `e`. In other words, `e`
dominates `v` in `D` under edge-dominance semantics.

For a failed vertex `z`, the analogous statement uses vertex dominance, with
`z` itself removed. This is dominance inside the shortest-path DAG, not
dominance in the full graph. The distinction is appropriate here because the
question is whether an old shortest path survives, not whether any path of any
length survives.

With several simultaneous failures, individual dominator tests are
insufficient: different old shortest paths may be killed by different members
of `F` even though no one failure dominates the target alone. Reachability in
the combined `D-F` remains the exact criterion.

## 7. Output-specific change regions

Scalar distance is only one output.

### One arbitrary parent tree

Deleting a retained parent edge invalidates the selected witness for its tree
subtree, even where an equal-length alternative preserves scalar distance. A
new arbitrary parent may be chosen without changing the label.

### Complete predecessor DAG

Every failed DAG arc is removed immediately. If some labels increase, the new
predecessor DAG may also contain formerly nonshortest full-graph arcs and cannot
be reconstructed from `D-F` alone.

### Shortest-path counts

For one failed old-DAG arc `(a,b)`, every old-DAG descendant of `b` loses at
least the shortest paths using that arc. If its scalar distance remains, its
old shortest-path count strictly decreases under edge-occurrence path identity.
If its scalar distance increases, the old count belongs to the wrong depth and
must be replaced rather than merely decremented.

Thus the path-count change cone can be strictly larger than the scalar
invalidation set. In the diamond, deleting `(a,t)` preserves `d(t)=2` but
changes the number of shortest paths from two to one.

### Canonical parent or word

An equal-distance alternative can preserve the old canonical choice only if
the failed contribution was not the selected minimum. Canonical-output
invalidation depends on the declared ordering, not only reachability in `D-F`.

## 8. Local edge versus global generator deletion

For a local graph-edge deletion, `F` may contain one arc. Removing a Cayley or
Schreier generator label removes a translated family of arcs, often one per
state occurrence. The preservation theorem still applies to the whole failure
set:

```text
old distance survives iff an old shortest word avoiding every failed-label
occurrence survives.
```

This can invalidate several disconnected-looking regions of a chosen parent
tree. Algebraic redundancy may preserve all scalar distances at some radii
while shortest-word multiplicities and canonical words change widely.

## 9. GPU and multi-GPU interpretation

The support wave has a natural parallel vocabulary but no performance claim:

- old-DAG arcs are logical support contributions;
- a deletion removes contributions;
- a vertex crosses from preserved to invalid only when its reachable support
  count reaches zero;
- invalidation messages flow only toward greater old depth;
- global completion requires all support removals and newly triggered
  invalidations to be closed;
- distance repair is a separate workload over surviving full-graph adjacency.

Mixing invalidation and repair counters can report false completion: a vertex
with zero old support may later receive a valid longer label, but that does not
restore its old-depth membership. Reports should separate:

```text
preserved old labels
invalidated labels
repaired finite labels
newly unreachable vertices
parent/DAG/count changes
```

An exact parallel support wave is bounded semantic evidence only after it is
compared with a fresh BFS on `G-F`. Throughput of support propagation is not
throughput of full decremental repair.

## 10. Counterclaims rejected

- **Only the selected parent subtree can change, and all of it changes.** The
  true scalar invalidation set is a subset determined by all old shortest
  predecessors.
- **If a non-tree edge is deleted, nothing changes.** Scalar distances retain
  the selected tree witnesses, but DAG, counts, or canonical outputs may
  change.
- **Reachability in the old DAG computes replacement distances.** It computes
  exactly which old distances survive; longer detours require full-graph
  evidence.
- **One surviving immediate predecessor with an old label is enough.** That
  predecessor must itself remain reachable at its old depth after the whole
  deletion cascade.
- **Testing each failure separately handles a batch.** A set can hit all old
  shortest paths without any individual member doing so.

## Sources and dependencies

- Note 11 defines the complete shortest-path predecessor DAG.
- Note 22 gives insertion/deletion distance monotonicity and dynamic graph
  version boundaries.
- Note 89 distinguishes shortest-DAG dominance from full-graph dominance.
- Note 118 defines replacement paths and proves why longer replacements may
  leave the old DAG.
- Even and Shiloach, *An On-Line Edge-Deletion Problem*, Journal of the ACM
  28(1), 1981, is the classical decremental BFS-tree foundation cited in note
  22. This note uses only the direct finite-DAG preservation proof above, not a
  claimed implementation of the ES data structure.

## Compact conclusion

After deletions, the surviving old shortest-path DAG draws an exact boundary:
inside its source-reachable part, old distances remain exact; outside it, old
labels must increase or become infinite. That boundary solves invalidation,
not repair. Parent trees overapproximate scalar damage, longer replacements
need the full graph, and richer BFS outputs have larger or differently ordered
change regions.

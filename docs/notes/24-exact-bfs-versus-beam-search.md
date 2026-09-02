# Exact BFS versus beam search: a full layer is not a top-k layer

This note makes an important naming boundary explicit.  Ordinary exact BFS and
beam search may both advance by depth, but they make different promises.

For a fixed unweighted graph, source `s`, reached ball `B_d`, and exact frontier
`F_d`, graph BFS constructs

```text
F_(d+1) = unique(successors(F_d)) \ B_d.
```

Every newly discovered state is retained.  This completeness of the next layer
is not an implementation detail: it is the reason the distance labels and
shortest-path claims apply to the original graph.

A width-`k` beam instead has a recurrence of the form

```text
C_(d+1) = successors(K_d)
K_(d+1) = top_k(score, eligible_unique_states(C_(d+1))).
```

The exact position of deduplication, visited filtering, scoring, and selection
must be specified; changing it changes the algorithm.  Whatever the convention,
if an eligible unique state is discarded solely to meet a width, threshold, or
capacity limit, the retained layer is not the BFS frontier of the original
graph.

## Three different operations often hidden under "ranking"

1. **Ordering only.** Rank all states inside a complete BFS layer but eventually
   process every one of them.  This preserves distance and completeness.  It
   can change parent choice, shortlex order, cancellation work, and physical
   locality.
2. **Early target inspection inside a complete layer.** A target may be returned
   before the rest of its layer is expanded when only one minimum-hop target
   path is requested.  The unexpanded work cannot simultaneously be claimed as
   a completed BFS ball or exhaustive layer certificate.
3. **Pruning.** Retain only a selected subset.  Completeness and shortest-path
   guarantees no longer transfer from the retained search to the original
   graph.

A timeout or capacity overflow can turn ordering into effective pruning.  If
unprocessed records are dropped, the semantic boundary is determined by the
drop, not by whether a heuristic score was present.

## Minimal counterexample: beam can discard the only shortest branch

Let the directed unit graph contain

```text
s -> a -> t
s -> b -> c -> d -> t
```

Give `b` a better heuristic score than `a` and use width one.  Exact BFS keeps
both depth-one states and finds distance two.  The beam discards `a` and returns
the length-four route.  If the surviving branch were a dead end, the same beam
would report failure despite a reachable target.

Therefore "the first target found at beam depth `d`" means shortest among the
paths that survived that particular pruning history.  It does not prove
distance `d` in the original graph.

An infinite width degenerates to BFS only when all other semantics also match.
More operationally, a finite width is sufficient for one run if it is at least
the number of every eligible unique exact frontier state before selection and
no other pruning occurs.  That is a property of the traversed instance and all
its layers, not a guarantee supplied merely by choosing a seemingly large `k`.

## Exact distance is stronger than admissibility

The absence of a general beam guarantee does not mean every pruning rule
destroys optimality. On a fixed unit-edge graph with reachable target `t`,
suppose `h(v)=dist(v,t)` exactly, every outgoing successor of the retained
state is considered, and no extra filter removes its shortest continuation.
Width one selecting minimum h then suffices for one shortest path. If
`h(v)=r>0`, a shortest path supplies a successor with h=r-1, and no successor
can have h<r-1, since that would give a shorter path from v. Repeating decreases
h to zero in exactly h(s) steps. This is following an exact distance policy,
not reconstructing complete BFS layers or all shortest paths. Computing that
exact h table may itself require substantial search; it is not free evidence.

Admissibility, even with consistency, is insufficient for this width-one rule.
In the earlier graph `s->a->t`, `s->b->c->d->t`, set h(a)=1 and h=0 at every
other vertex. All values are nonnegative lower bounds and every edge satisfies
`h(u)<=1+h(v)`. Nevertheless minimum-h width one selects b over a and returns
the four-edge route instead of the two-edge route. At equal depth, ranking by
g+h gives the same selection. Thus a lower bound is not an exact ranking of
remaining distances, and A* guarantees cannot be transferred to beam deletion.

## Top-k records and top-k states are not the same

Suppose four candidate records sorted best first denote states

```text
(x, parent=p1), (x, parent=p2), (x, parent=p3), (y, parent=p4).
```

With width three:

- `top-k records -> state dedup` retains only `{x}`;
- `state dedup -> top-k states` can retain `{x,y}`.

Thus duplicate transitions can consume record width.  The same issue arises if
already visited states are filtered after rather than before selection.  A log
field named only `beam_width=3` does not identify which contract was executed.

For Cayley graphs this distinction is structural, not rare: relations make many
move words and parent-generator records converge to the same group element.
If score is attached to a path rather than solely to its endpoint state, merging
also requires a dominance proof or an expanded product state; otherwise two
records for the same visible state need not have interchangeable futures.

## Local top-k is generally not global top-k

Partition candidates across two devices:

```text
GPU 0 scores: 100, 99
GPU 1 scores:  10,  9
```

If larger is better, retaining local top-one on each device yields `{100,10}`;
the global top-two is `{100,99}`.  Unioning equal local quotas therefore defines
a partition-dependent beam.  An exact global merge/selection can recover the
declared global beam, but that still recovers beam semantics, not exact BFS.

Tie rules and ownership matter too: without one global comparison key, changing
the number of devices or hash partition can change which equal-score states
survive.

## A BFS lookup inside a beam pipeline

A precomputed BFS table may exactly describe a limited subproblem, for example:

- distances and parents inside a goal-centered ball;
- a pattern-database abstraction;
- a small quotient or local neighborhood;
- an exact suffix once the global search reaches the table's covered set.

If a retained beam state enters a goal ball, replaying its beam prefix followed
by the table suffix can produce a valid path.  The table certifies its own
covered metric and suffix.  It does **not** restore prefixes discarded earlier,
prove the beam frontier complete, or show that the combined path is globally
shortest.  The overall method remains beam/hybrid search unless its global
exploration independently satisfies an exact-search proof.

This also prevents a terminology leak: "BFS lookup" names one component.  It
does not make the surrounding multi-GPU beam search an instance of ordinary
BFS.

## Exactness is an output contract

A run should not be labeled exact BFS merely because its kernel expands a
frontier level by level.  At minimum its evidence should identify:

- algorithm and requested output contract;
- exhaustive frontier versus top-k/threshold policy;
- generated-record, post-visited, unique-state, and retained counts per level;
- the stage at which visited filtering and deduplication occur;
- configured width/capacity and actual dropped count;
- overflow, truncation, timeout, cancellation, and early-stop reason;
- local versus global selection and deterministic tie key;
- whether a full layer, only a target path, or only a surviving beam was proved;
- lookup-table domain/radius/version and whether the final path replayed;
- explicit `completeness_guaranteed` and `shortest_path_guaranteed` claims.

Any positive dropped/overflow/truncation count invalidates an exact-frontier
claim for that run.  Reporting the loss honestly is useful evidence; silently
calling the result BFS is not.

## Relation to other best-first searches

"Best-first search" is a family name, not another spelling of breadth-first
search.  Uniform-cost search prioritizes path cost `g`; on unit-cost edges it
has a BFS-compatible distance order.  A* prioritizes `g+h` and needs its own
conditions for optimality.  Greedy best-first prioritizes a heuristic such as
`h`.  Beam search adds a width/pruning rule.  Sharing a queue, score, or the
letters "BFS" does not transfer guarantees between these contracts.

## Sources and independent check

- Meister et al., [Best-First Beam Search](https://aclanthology.org/2020.tacl-1.51/),
  explicitly describes ordinary beam search as a pruned breadth-first search
  used to approximate exact search.
- Zhou and Hansen,
  [Beam-Stack Search](https://m.aaai.org/Library/ICAPS/2005/icaps05-010.php),
  add systematic backtracking to transform ordinary beam search into a complete
  algorithm converging to an optimal solution.  The added machinery is evidence
  that fixed-width beam alone does not have those guarantees.
- Hart, Nilsson, and Raphael,
  [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://www.cs.auckland.ac.nz/courses/compsci709s2c/resources/Mike.d/astarNilsson.pdf),
  gives a distinct formal contract for heuristic minimum-cost search; it should
  not be conflated with either level-complete BFS or bounded beam search.
- The `multigpu_beam` expert independently confirmed the practical distinctions
  among complete frontiers, beam selection, dedup order, local/global top-k, and
  lookup-table scope.  Those recommendations were accepted only where the set
  recurrences and counterexamples above verify them.

## Current conclusion

The decisive question is not "does the program advance by depth?" but:

> Did it retain every eligible newly reached state required by the declared
> exact graph-search contract?

If yes, heuristic ordering can coexist with exact BFS.  If no, the run is a
pruned, bounded, approximate, or otherwise qualified search and must be named
accordingly.

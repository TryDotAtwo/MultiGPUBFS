# Bidirectional BFS: meeting is an upper bound, stopping needs a lower bound

Bidirectional BFS is often summarized as "search from both ends until the
frontiers meet."  That sentence hides four different choices:

1. what each side has *discovered* versus completely *expanded*;
2. whether a meeting is a shared vertex or a crossing edge;
3. whether processing is vertex-at-a-time, layer-at-a-time, or asynchronous;
4. whether the required output is one distance/path or every shortest path.

The clean principle is:

```text
meeting candidate -> feasible path -> upper bound mu
unfinished search state -> lower bound L
stop for one shortest distance when L >= mu
```

A meeting alone supplies the first half, not automatically the second.

## Directed graph contract

Let the original directed graph be `G=(V,E)`, start `s`, and target `t`.

- Forward BFS follows `(u,v) in E` and stores exact `d_f(v)=dist_G(s,v)`.
- Backward BFS starts at `t`, follows edges of `G^R`, and stores exact
  `d_b(v)=dist_G(v,t)`.

The backward transition API must therefore return an original-graph
predecessor and enough information to replay the original forward move.  Merely
applying the same transition function from `t` is correct only when the graph
and labels have the required reversibility.

Two ways to form a feasible path are:

```text
shared vertex x:       d_f(x) + d_b(x)
crossing edge (u,v):   d_f(u) + 1 + d_b(v)
```

The smallest candidate seen so far is `mu`.  Because each term describes
actual path pieces, `mu` is always an upper bound on the true distance `D`:
`D <= mu`.  This remains true before `mu` is known to be optimal.

## Complete-level state

The local reference keeps these invariants at every loop boundary:

```text
forward discovered = B_f(a) = {v | d_f(v) <= a}
forward frontier   = F_f(a) = {v | d_f(v) = a}, still unexpanded

backward discovered = B_b(b) = {v | d_b(v) <= b}
backward frontier   = F_b(b) = {v | d_b(v) = b}, still unexpanded
```

Here `a` and `b` are the minimum unexpanded depths on their respective sides.
Expanding either side means processing its **entire** current frontier and then
advancing only that side's depth.

Side selection—alternating, smaller frontier, or estimated edge work—does not
appear in the invariant.  It affects work but not the proof, provided every
chosen expansion completes one exact layer.

## The stopping theorem used locally

For the complete-level state above, once a feasible path of length `mu` is
known, it is safe to stop when

```text
a + b >= mu.
```

### Proof

Assume for contradiction that a shorter `s -> t` path `P` of length
`D < mu` exists.  The stopping condition gives `D < mu <= a+b`, hence
`D <= a+b`.

Walk `i=min(a,D)` edges from `s` along `P` to a vertex `x`.  Then

```text
dist(s,x) = i <= a
dist(x,t) = D-i <= b.
```

Thus `x` belongs to both discovered balls.  Both stored distances are exact, so
the shared-vertex candidate through `x` has length `D`.  The maintained best
candidate must satisfy `mu <= D`, contradicting `D < mu`.

Therefore no shorter path exists and `mu=D`.

This test is sufficient, not tight in its integer threshold. With the same
complete-ball and persistent-intersection premises, `a+b+1>=mu` also suffices:
any shorter integer length satisfies `D<=mu-1<=a+b`, hence already intersects
the known balls. Note 56 records the exact scope of this one-unit refinement.
The local implementation/evidence below still refers to its original,
conservative test; this mathematical observation is not a code change.

This is the unit-weight specialization of the familiar bidirectional Dijkstra
rule in which the minimum unsettled forward and backward keys are compared with
the best feasible path.

## Why the code's depth variables have this meaning

In `multigpubfs/bidirectional.py`, a state is inserted into a side's distance
map only while expanding the preceding complete frontier.  After that expansion
finishes:

- the newly formed frontier contains exactly the next depth;
- the depth variable is incremented once;
- no state beyond that depth has been discovered on that side.

Consequently `forward_frontier_depth` and `reverse_frontier_depth` really are
the radii `a` and `b` used in the proof.  The check

```text
forward_frontier_depth + reverse_frontier_depth >= best_distance
```

is not an intuitive frontier-size heuristic; it is the lower-bound certificate.
REF-007 exhaustively found no mismatch over all 49,152 ordered distinct pairs
in all loop-free four-vertex directed graphs, providing finite validation of
the implementation in addition to the proof of its abstract invariant.

## When the first intersection actually is safe

There is a useful special theorem that is often blurred with unsafe variants.

Assume before an expansion:

- the two exact discovered balls `B_f(a)` and `B_b(b)` are disjoint;
- expansion of one next layer begins, say forward depth `a+1`, while the
  opposite discovered set remains exactly `B_b(b)`;
- a newly discovered vertex `x` lies in `B_b(b)`.

Then `d_b(x)` must equal `b`.  If `d_b(x) <= b-1`, let `p` be the predecessor
of `x` on its newly found forward shortest path.  Since `(p,x)` is an original
edge,

```text
d_b(p) <= 1 + d_b(x) <= b.
```

But `p` is already in `B_f(a)`, which would mean the balls intersected before
this expansion—a contradiction. Therefore every first intersection produced
under this fixed opposite-ball condition has the same length

```text
(a+1) + b.
```

No shorter path existed without causing an earlier ball intersection.  In this
strict setting, the first intersection within the newly generated layer is
enough for **one shortest distance/path**.

The unprocessed remainder of that layer need not be generated for this output:
the proof uses the two complete balls *before* expansion and the fixed exact
opposite ball, not completion of the new layer. Its frontier set and alternative
shortest-path metadata remain incomplete if the search stops here.

This theorem explains why many simple layer-by-layer bidirectional BFS
implementations are correct.  It does not validate every program described as
"stop when the searches touch."

## Unsafe meanings of "first meeting"

### Partial layers with no lower-bound accounting

If both sides are allowed to extend partial layers, an intersection can use
vertices beyond both complete-ball radii. The fixed opposite-ball premise
above no longer holds. Stopping on an arbitrary contact then lacks a proof,
even if every individual distance is already exact. Note 56 gives a
length-four first contact despite a remaining length-three route. A partial
layer on just the active side is not itself a counterexample to the safe
fixed-opposite-ball theorem.

One can still design a correct partial/asynchronous algorithm, but it must track
the minimum outstanding depth—including queued, executing, buffered, and
in-flight work—and derive `L` from that state.  It may not reuse `a+b` from the
complete-level proof without showing that the variables retain the same
meaning.

### Weighted edges

For weighted search, hop layers are not cost balls.  A first shared vertex or
crossing edge can give a feasible but suboptimal route.  Bidirectional Dijkstra
maintains tentative/settled cost labels, updates `mu` through crossing edges,
and stops from minimum priority-queue keys.  The unweighted complete-layer
theorem cannot simply be copied.

### Wrong backward graph

On a directed graph, searching outward from `t` in `G` can meet the forward
search even though its suffix runs away from the target in the original graph.
The meeting is not a feasible `s -> t` path, so it is not even a valid upper
bound.

### Distributed notification before convergence

A rank can report an intersection while other ranks still hold candidates from
the same or shallower epochs.  "Target/meeting found" is a fact about `mu`;
termination additionally needs globally consistent knowledge of the lower
bound and completion of all work that could beat it.

### Heuristic bidirectional search

With A*-like keys, front-to-front estimates, inconsistent heuristics, or
reopening, frontier depth is no longer the relevant lower bound.  Such
algorithms require their own key/potential and termination proof.  Results for
ordinary BFS should not be generalized by resemblance.

## Vertex meeting versus edge meeting

In unit-cost BFS, discovering `v` from a forward vertex `u` and finding `v` in
the backward visited set simultaneously gives

```text
d_f(u) + 1 + d_b(v) = d_f(v) + d_b(v),
```

because first discovery assigns `d_f(v)=d_f(u)+1`.  Thus a crossing-edge check
can be represented as a newly shared vertex.

Checking crossing edges explicitly is still useful when:

- the algorithm distinguishes settled from merely discovered labels;
- edges have nonunit weights;
- frontiers are partitioned and a compact edge/endpoint join is cheaper;
- reconstruction metadata naturally belongs to `(u,v)` rather than one shared
  state.

The proof must match which labels are final at the time a candidate is formed.

## Empty frontiers and unreachable targets

If either exact side exhausts its reachable set before any intersection, no
`s -> t` path exists.  For example, if forward BFS has closed the entire set
reachable from `s` and none of it belongs to the reverse-reachable set of `t`,
no path can connect them.

In a distributed implementation, "frontier empty" must mean globally empty
after all messages for the relevant epoch have been delivered and processed.
A locally empty rank is not a reachability certificate.

## Output contract changes the stopping point

The condition `a+b >= mu` proves the **distance** and permits reconstruction of
one stored shortest path.  It need not enumerate:

- every meeting vertex on a shortest path;
- every crossing edge satisfying `d_f(u)+1+d_b(v)=mu`;
- every shortest predecessor on either side;
- every source owner in a multi-source tie.

If all shortest paths or all shortest meeting structures are required, equality
cases at the boundary may still contain required output.  The traversal and
metadata retention contract must explicitly say which layers/edges are
completed after the distance becomes known.  Distance-optimal termination is
not enumeration-complete termination.

## Choosing which side to expand

Under the complete-level invariant, side selection is free with respect to
correctness but not cost.

- **Alternating** avoids a selection reduction and keeps depth counts close,
  but may expand a huge frontier instead of a small one.
- **Smaller frontier** predicts vertex work, not necessarily edge or move work.
- **Estimated work** can better predict irregular expansion, but acquiring and
  globally reducing the estimate has a cost.
- **Expand one side many times** remains correct if complete levels and the
  stopping bound are maintained, but can eliminate the expected meet-in-the-
  middle saving.

REF-009 is a useful counterexample to universal policy claims: exact edge-work
selection won its tiny irregular corpus, while all three policies were
identical on regular symmetric `S_8`.  Correctness does not select the cheapest
schedule.

For a hand example, let s have edges to a and b, and let a, b and 98 other
vertices each have one edge to t, with no other edges. After forward expansion
of s, the forward frontier {a,b} has two outgoing entries in total. The
backward frontier {t} has only one vertex but 100 original incoming entries
to enumerate. Complete expansion therefore costs two adjacency inspections
on the larger frontier and 100 on the smaller one. Either expansion can
establish a length-two meeting. This counts complete-layer work as specified
here; an early-hit stopping variant may inspect fewer entries and its cost
depends on ordering and its own stopping certificate.

For a Cayley graph with q total generator occurrences, full forward expansion
attempts `q*|F_f|` transitions and full reverse expansion using the q inverse
actions attempts `q*|F_b|`. In this regular occurrence model, smaller frontier
does mean fewer attempted moves for that next step. It still does not prove
smaller wall time or smaller total search work: representation costs,
duplicates, future layer growth and stopping depth remain separate quantities.

## Implicit and Cayley graphs

Bidirectional search is natural when predecessor generation is available.  For
a Cayley graph with right action `g -> g*s`, a backward step must correspond to
the inverse action needed to recover an original forward move.  If generators
are involutions, forward and reverse state transformations may look identical,
but replay orientation is still part of the contract.

Predecessor enumeration is more general than an invertible forward operation.
Consider the finite state set {0,1,2,3} with the single move
`f(x)=floor(x/2)`. The predecessors of 0 are {0,1}, those of 1 are {2,3},
and those of 2 or 3 are empty. There is no single-valued inverse f, but reverse
BFS from target 0 is well-defined: F0={0}, F1={1}, F2={2,3}. It correctly
records the forward path 3->1->0. Merely applying f from 0 would stay at 0
and miss those predecessors.

Conversely, in a right Cayley graph the backward transform `x -> x*s^-1`
enumerates the predecessor whose allowed forward move s reaches x. The symbol
s^-1 need not itself belong to the forward move alphabet. The backward edge
stores the original move s for forward replay; adding s^-1 to the forward
alphabet would instead change the metric being solved. Efficient reverse
enumeration is an access-model question, separate from its mathematical
existence and from the bidirectional stopping proof.

Potential savings depend on actual ball growth, not only solution depth.  The
tree intuition `b^d` versus roughly `2b^(d/2)` assumes sustained branching and
little convergence.  Finite groups have relations and saturating balls.  In
REF-007, the relative work reduction on the chosen `S_8` targets fell to 9.51%
at diameter 28 because the two searches covered much of the finite group before
the lower bound closed.

## Distributed complete-level proof

REF-010 deliberately uses bulk-synchronous supersteps.  A round is not complete
until:

1. every owner has expanded its assigned frontier for the chosen side;
2. source-local optional dedup has finished;
3. all routed candidates have reached their authoritative owners;
4. owner-side exact dedup/visited and opposite-side intersection checks have
   finished;
5. the new frontier size and best `mu` are globally visible.

Only then is that side's minimum unexpanded depth advanced and `a+b>=mu`
evaluated.  Sharing the same owner function for forward and reverse visited
makes an intersection a local owner lookup after routing, but the stopping
decision remains global.

An asynchronous version would need explicit epochs or a distributed snapshot/
termination protocol capable of bounding every queued and in-flight item.  It
cannot inherit the superstep proof merely because messages carry depth fields.

## Audit checklist

For any bidirectional implementation, ask:

1. Does backward expansion traverse `G^R` and retain forward-replay labels?
2. Are distance labels exact when used in `mu`?
3. Is a candidate formed by a shared vertex, crossing edge, or both?
4. What exactly are the minimum unfinished forward and backward depths/keys?
5. Do those minima include partially processed and in-flight work?
6. What theorem converts them into a lower bound `L`?
7. Is termination `L>=mu` evaluated from a consistent state?
8. Does side selection preserve complete layers or require another proof?
9. Is the output one distance, one path, deterministic path, or all shortest
   paths?
10. On no-path results, how is global exhaustion established?

## Sources and local evidence

- Ira Pohl, *Bi-Directional Search*, Machine Intelligence 6 (1971),
  [paper copy](https://aitopics.org/download/aiclassics%3A630E1F02), for the
  classical meet-in-the-middle formulation and its bookkeeping problem.
- Hermann Kaindl and Gerhard Kainz,
  *Bidirectional Heuristic Search Reconsidered*, JAIR 7 (1997),
  [doi:10.1613/jair.460](https://doi.org/10.1613/jair.460), for the warning that
  heuristic bidirectional search has distinct frontier and termination issues.
- A bidirectional Dijkstra proof is given in the shortest-path lecture material
  summarized by the [MPI Informatics chapter](https://people.mpi-inf.mpg.de/~mehlhorn/ftp/NewToolbox/spath.pdf);
  its minimum-key-plus-minimum-key rule motivates the unit-weight specialization
  proved above.
- REF-007: exhaustive four-vertex directed validation and `S_4/S_8` work.
- REF-009: complete-level side-selection policies.
- REF-010: owner-computes bulk-synchronous intersection and stopping model.

## Current synthesis

Bidirectional BFS is not defined by two queues or by contact between colored
regions.  It is an upper/lower-bound algorithm.  Exact forward and reverse
labels make meetings feasible upper bounds; exact knowledge of unfinished
depths makes stopping possible.  Complete layers give an especially simple
proof and, under initially disjoint metric balls, make the first new
intersection safe.  Partial, weighted, heuristic, or distributed schedules are
not automatically wrong, but each must rebuild the lower-bound argument for
the state it actually maintains.

# Partial-layer bidirectional stopping and global bounds

Bidirectional BFS does not need physically synchronized whole layers in order
to prove an optimal distance. It does need a truthful lower bound on every
piece of work that can still create a better connection. Partial layers change
the bookkeeping behind that bound, not the underlying shortest-path argument.

This note formalizes that distinction. It develops no implementation or
performance policy.

## State and output contract

Let `G=(V,E)` be a directed unit-cost graph, with source `s` and target `t`.
Forward work traverses `G`; reverse work traverses `G^R`. Exact labels are

```text
d_f(v) = dist_G(s,v)
d_b(v) = dist_G(v,t).
```

A shared vertex or an original-direction crossing edge gives a feasible path:

```text
shared vertex x:       d_f(x) + d_b(x)
crossing edge (u,v):   d_f(u) + 1 + d_b(v).
```

Let `mu` be the shortest replay-valid candidate known globally. Then `mu` is
an upper bound on the true distance. The stopping result below promises the
optimal distance and enough retained metadata for one shortest path. It does
not promise every shortest connector, parent, label sequence, or path.

## What the unfinished-depth minima mean

For each direction, classify every logical expansion item that can still
produce a new vertex or connector as unfinished. This includes work that is:

- ready in a frontier or task queue;
- only partly expanded;
- executing on a CPU or GPU;
- buffered between kernels or host/device stages;
- routed but not yet accepted by its authoritative owner;
- sent, received, retried, spilled, or otherwise in flight.

Assign each unfinished item the depth of the not-yet-retired expansion that
created its obligation. In particular, a child record generated from a
depth-`d` parent but still awaiting routing, authoritative acceptance, or
opposite-side checking continues to hold the minimum at `d`; it cannot be
relabelled as harmless depth-`d+1` work before the parent expansion is
semantically complete.

Define

```text
a = minimum forward depth among all unfinished forward expansion items
b = minimum reverse depth among all unfinished reverse expansion items.
```

The useful invariant is not that each physical visited set equals one clean
ball. It is:

```text
every forward vertex with distance <= a has been discovered exactly enough
to participate in a persistent opposite-side connector check;

every reverse vertex with distance <= b has been discovered exactly enough
to participate in a persistent opposite-side connector check.
```

It follows from complete, exact retirement of every expansion depth strictly
below the minimum unfinished depth, with generated obligations charged to
their parent depth until settlement. Speculatively discovered deeper vertices
may also be present; they do not weaken the invariant.

If a depth-`d` vertex has ten outgoing moves and nine have been evaluated, the
last move is still unfinished depth-`d` work. Ordinary BFS has no stronger
per-edge lower bound that permits silently promoting it to depth `d+1`.

## Partial-layer stopping theorem

Assume:

1. forward and reverse labels used in candidates are exact;
2. every vertex through forward depth `a` and reverse depth `b` has been
   discovered under the invariant above;
3. a state discovered on either side is checked against persistent exact state
   from the other side, so chronology cannot hide an intersection;
4. every unfinished item is represented in the global minima `a` and `b`;
5. `mu` is the globally smallest replay-valid candidate known at the same
   consistent observation cut.

Then it is safe to stop for one shortest distance/path when

```text
a + b >= mu.
```

### Proof

Suppose instead that an `s -> t` path `P` of length `D < mu` exists. The stop
condition implies `D < a+b`.

Walk `i=min(a,D)` edges along `P` from `s` to a vertex `x`.

- If `D<a`, then `x=t`, so `d_f(x)=D<=a` and `d_b(x)=0<=b`.
- Otherwise `i=a`, and `D-a<b`, so `d_f(x)=a` and `d_b(x)<b`.

In either case `x` belongs to both completely discovered balls guaranteed by
the invariant. Persistent exact intersection checking must therefore already
have produced a candidate of length at most `D`. Hence `mu<=D`, contradicting
`D<mu`.

Thus no shorter path exists and `mu` is optimal.

The proof does not require `a=b`, alternation, equal frontier sizes, or a clean
physical layer boundary. It requires the semantic balls below the two global
unfinished minima to be complete.

## A chunk-completion counterexample

### Integer-depth refinement of the sufficient bound

With finite nonnegative integer `a,b`, a finite integer incumbent `mu`, and all
five assumptions of the theorem above unchanged, the weaker condition

```text
a + b + 1 >= mu
```

also suffices for one shortest distance/path. If a shorter path existed, its
integer length would satisfy `D <= mu-1 <= a+b`. Choosing the vertex at
`min(a,D)` on a shortest such path places it inside both guaranteed balls.
Completed persistent intersection checks would already have supplied a
candidate of length `D`, contradicting the incumbent.

Equivalently, every route with length at most `a+b` is covered by the known
balls. A still-undetected improvement must have length at least `a+b+1`.
Thus the earlier `a+b>=mu` test is safe but one unit conservative under these
particular conventions. This is not a correction to that theorem's validity.

For example, with disjoint exact radius-one balls and a newly found path of
length three, the old minima may still be `a=b=1`. The refined test permits
stopping (`1+1+1=3`) without pretending that either pending layer has completed.
The length-four first contact below still fails the refined test (`3<4`).

This relies on unit-cost integer path lengths, coverage of both balls including
their boundary vertices, and all relevant intersection checks having reached
the same globally consistent incumbent. Do not transplant the `+1` to weighted
search, differently defined depth variables, or outstanding connector checks.
It does not certify all shortest paths, complete frontiers, or full work totals.
No existing stopping implementation was changed or re-executed for this proof.

### Counterexample trace

Consider the directed graph

```text
s -> a -> x -> t
s -> b -> y -> c -> t
```

The first route has length three and the second has length four. There are no
other edges. At depth one, the forward frontier is `{b,a}` and the reverse
frontier is `{c,x}`. The discovered balls are disjoint.

Suppose a scheduler processes forward `b` before `a`, discovering `y` at depth
two, and then reverse `c` before `x`, also discovering `y` at depth two. Every
discovery is checked against the opposite visited set; the first intersection
is genuinely `y`, giving `mu=4`. Both per-side orders can be ordinary FIFO
orders. If the scheduler now
increments both minima from one to two merely because one chunk on each side
finished, it obtains

```text
2 + 2 >= 4
```

and stops incorrectly. Expanding forward `a` would discover already
reverse-known `x`, or expanding reverse `x` would discover forward-known `a`;
either yields the true length-three path. The known endpoints of edge `a->x`
do not imply that this unexpanded edge has already been inspected.

The correct minima remain `a=b=1`, because depth-one work is still pending.
Then `1+1<4`, so the theorem does not permit stopping.

This rejects the universal rule:

```text
completed some work from depth d  =>  minimum unfinished depth is d+1.
```

Only global retirement of all relevant depth-`d` work can justify that
advance.

**Correction history (2026-08-31):** The previous example used a length-two
route `s->b->t`, so `b` already lay in both complete depth-one balls. Exact
persistent discovery checking would have found that shorter route before the
claimed failure. It therefore mixed a missed-intersection defect with a
minimum-depth defect. The replacement above isolates premature advancement
and unsafe first contact while retaining correct discovery-time checking.

## Chronology and connector coverage

It is insufficient to compare only the two batches that happen to be active at
the same instant. A forward vertex may be discovered long before the reverse
search reaches it. The later discovery must query persistent opposite-side
state, or an equivalent exact connector index.

For directed graphs, shared-vertex checks and crossing-edge checks are distinct
forms of evidence. A system that uses crossing edges must preserve original
orientation:

```text
forward u --original edge--> reverse v.
```

Missing either a required connector class or an earlier opposite-side record
invalidates the premise that every shorter path would already have updated
`mu`.

## Asymmetric schedules

Frontier size and unfinished depth answer different questions:

```text
frontier size       -> possible scheduling cost
minimum depth a,b   -> correctness lower bound.
```

One side may advance several depths while the other remains at one large or
slow partial layer. The rule still uses the actual minima. Expanding the
smaller frontier, estimated edge work, or whichever device becomes ready does
not enter the proof unless it changes which work is genuinely complete.

Fairness is a separate liveness obligation. A scheduler can preserve the
distance theorem yet starve one depth forever and never reach a stopping state.

## Distributed and multi-GPU observation cuts

A local queue minimum is not a global minimum. A forward depth-`d` item in any
of the following places constrains the global value to `a<=d`:

```text
device queue
running kernel or persistent worker
host staging buffer
send buffer or transport
receive buffer
owner-side pending/dedup queue
spill/recovery/retry state.
```

Therefore a global reduction over currently visible local queues is sufficient
only if the protocol also proves which sends, executions, and ownership
transitions belong to the same consistent cut. Bulk-synchronous completion is
one simple realization. Epoch accounting, acknowledgements, credit/termination
detection, or a distributed snapshot may support other realizations; naming a
mechanism is not itself a proof that all work is covered.

Staleness is one-sided:

- a stale **smaller** `a` or `b` delays termination and is conservative;
- a stale **larger** minimum can permit early termination and is unsafe;
- a stale **larger** replay-valid `mu` is conservative only when compared with
  valid minima from a compatible cut; otherwise the mixed snapshot has no
  stopping proof;
- an incorrectly small or nonvalidated `mu` is unsafe because it is not a
  feasible-path upper bound.

The minima and incumbent must also describe one compatible search epoch:
graph/move version, identity convention, source/target, direction, and
ownership generation cannot be mixed.

## Empty work is not automatically infinity

A locally empty rank contributes no local candidate to a minimum reduction; it
does not prove that the direction is globally exhausted. Even global queues
appearing empty are insufficient while messages or executions can create more
work.

Only after global closure may an exhausted direction be treated separately:

- if no candidate exists and the exact forward reachable set or reverse
  co-reachable set is closed, the target is unreachable;
- if a candidate exists, closure can certify it without inventing an
  operational `infinity` that ignores in-flight work.

This is the same distinction as BFS exhaustion generally: absence of visible
work is an observation; successor closure is the certificate.

## Equality depends on requested output

For one optimal distance/path, `a+b>=mu` is sufficient under the theorem. At
equality, unfinished work can still expose other connectors whose total length
is exactly `mu`.

Consequently the same stop may be incomplete for:

- every shortest meeting vertex or crossing edge;
- all shortest predecessors or exact path counts;
- every shortest labeled generator word;
- deterministic tie-breaking whose preferred witness has not been processed;
- every nearest source in a multi-source tie.

A strict `a+b>mu` is not by itself a universal enumeration theorem. The system
must state which equality-boundary vertices and edges constitute the output
and prove that each has been finalized exactly once or with appropriate retry
deduplication.

## Relation to A* and best-first stopping

The structure is the same upper/lower-bound pattern used more generally:

```text
best feasible solution mu
versus
minimum bound among every open item.
```

In A*, an open record cannot disappear from the minimum-key calculation merely
because another chunk with the same key completed. Here a depth-`d` expansion
item cannot disappear from `a` or `b` for the same reason. This analogy helps
identify the bookkeeping obligation, but it does not turn hop-layer BFS into
heuristic search.

## Audit questions

1. What exact object is an unfinished item: vertex, edge range, move tile, or
   routed candidate?
2. Which depth does a partly processed item contribute to the minimum?
3. Can any queue, kernel, buffer, message, retry, or spill contain unreported
   lower-depth work?
4. What consistent cut makes local minima and the global `mu` comparable?
5. Are opposite-side records persistent and exact when a later discovery
   arrives?
6. Are shared vertices, crossing edges, and directed orientation handled under
   the declared connector contract?
7. Can overflow, cancellation, loss, or epoch mismatch falsely retire work?
8. Does the result require one optimum or complete equality-boundary output?
9. How is global closure proved when one side appears empty?
10. What liveness/fairness condition ensures low-depth work eventually retires?

## Sources and relation to existing notes

- Note 08 proves the clean complete-layer specialization and distinguishes a
  meeting upper bound from a stopping lower bound.
- Note 18 develops asynchronous relaxation and distributed-termination
  obligations.
- Note 30 treats partial distributed checkpoints as consistent cuts rather
  than collections of local files.
- Notes 48, 50, and 52 provide the closure-certificate, global-bound, and
  authoritative/advisory-state distinctions used here.
- The bidirectional Dijkstra minimum-key stopping pattern is summarized in
  Kurt Mehlhorn and Peter Sanders, *Algorithms and Data Structures: The Basic
  Toolbox*, shortest-path chapter:
  <https://people.mpi-inf.mpg.de/~mehlhorn/Toolbox.html>.

## Current conclusions

1. Partial layers are compatible with exact bidirectional stopping when global
   unfinished-depth minima truthfully cover all work.
2. The same condition `a+b>=mu` remains sufficient for one optimum, but `a`
   and `b` now mean semantic global minima, not loop counters or completed
   chunks.
3. One residual depth-`d` item prevents advancing that side's minimum beyond
   `d`, regardless of how much other work has completed.
4. Persistent connector detection is essential because the two discoveries
   need not occur concurrently.
5. Multi-GPU correctness requires a consistent global cut including executing,
   buffered, routed, owner-pending, retry, and spill work.
6. Smaller stale minima are conservative; larger stale minima can be unsafe.
7. Distance-optimal termination does not imply equality-boundary enumeration
   completeness.
8. The result formalizes a correctness contract, not an optimal scheduling or
   synchronization design.

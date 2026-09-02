# BFS as metric balls, invariants, and schedules

Status: conceptual study pass, 2026-08-28.

## The object BFS computes

Let `G = (V, E)` be a directed or undirected graph, let every edge have unit
length, and let `s` be the source. Define the graph distance `delta(s, v)` as
the minimum number of edges in a directed path from `s` to `v`, or infinity if
no such path exists.

The cleanest meaning of BFS is not a queue traversal. It is construction of
successive metric balls and spheres:

```text
B_0 = {s}
B_{d+1} = B_d union N+(B_d)
F_d = B_d minus B_{d-1}
```

Here `B_d` is every vertex at distance at most `d`, and `F_d` is the sphere of
vertices at distance exactly `d`. Equivalently, with
`T(X) = X union N+(X)`, BFS computes the ascending chain
`{s}, T({s}), T^2({s}), ...` until the least reachability fixed point is reached.

This viewpoint explains several facts at once:

- the algorithm is a wave, not an enumeration of complete paths;
- a frontier is a difference between two consecutive balls;
- visited is accumulated knowledge of a ball, not merely a loop-avoidance flag;
- queue order inside one sphere is semantically irrelevant to distances;
- the last nonempty distance index is the source's eccentricity D in its
  finite reachable component; constructing that ball takes D expansions from
  F0, while expanding F_D to observe an empty next frontier takes one more.
  Iteration counts therefore need a declared stopping convention.

## Why the layer recurrence is exact

Claim: after completing depth `d`, `B_d` contains exactly the vertices with
distance at most `d`.

There are two independent obligations.

**Soundness.** Every newly inserted vertex `v` has an edge from some
`u in F_d`. By induction there is a length-`d` path to `u`, so appending `(u,v)`
gives a length-`d+1` witness path to `v`. Therefore BFS never assigns a label
smaller than the true distance.

**Completeness.** If `delta(s,v) <= d+1`, take a shortest path ending in
`u -> v`. Its prefix reaches `u` in at most `d` edges. By induction `u` is in
`B_d`, so expansion of the completed ball exposes `v`. Therefore BFS never
misses a vertex whose true distance belongs to the completed ball.

The two inclusions give equality. Notice what the proof uses:

1. every required predecessor in the current layer is eventually expanded;
2. an irreversible distance acceptance cannot overtake an unresolved proposal
   that could give that vertex a smaller distance;
3. visited membership is exact;
4. an accepted edge really exists in the graph.

It does **not** require a particular order among vertices of the same frontier.

## Queue, frontier, and schedule

A FIFO queue is a compact sequential mechanism for preserving the layer order.
Its useful invariant is that queued distance labels contain at most two
consecutive values, with all smaller labels before larger ones. A level-array
implementation states the same rule explicitly by separating `frontier` and
`next_frontier`.

Thus:

- **semantic requirement:** exclude smaller distances before irreversible
  acceptance; ordinary FIFO BFS does so through nondecreasing expansion depths;
- **one implementation:** FIFO queue;
- **another implementation:** complete level barriers;
- **another possibility:** asynchronous relaxation that permits later distance
  decreases, which is no longer the simple first-discovery BFS discipline.

Distance acceptance is not expansion completion. When expanding `F_d`, a new
child can receive its final distance `d+1` before the remaining vertices of
`F_d` are expanded: the exact known ball `B_d` already excludes a shorter
distance. Remaining depth-`d` producers can still supply equal-distance parents
or additional next-layer vertices. Complete next-layer membership and rich
metadata therefore have later completion boundaries than an individual label.

Saying “visited is the core and the queue is only a schedule” is useful but
incomplete. A visited set combined with the wrong schedule can freeze a
non-shortest discovery. Schedule and finalization rule form one correctness
contract.

## The useful invariants

At the boundary after depth `d`:

1. **Exact ball:** `visited = B_d`.
2. **Exact sphere:** `frontier = F_d`.
3. **Disjointness:** no vertex belongs to two frontiers.
4. **Witness:** every discovered non-source vertex has a real parent edge from
   depth one less.
5. **Closure of completed work:** every outgoing neighbor of every vertex at
   depth less than `d` is already visited.
6. **No overflow/loss:** every first accepted vertex is represented in the next
   frontier or in an equivalent complete work structure.

The parent invariant proves existence of a path of the recorded length. It does
not by itself prove minimality. A depth-consistent but incomplete or incorrectly
scheduled tree can still contain valid edges and wrong distances.

For an edge `(u,v)` between two reachable vertices in an undirected graph,
`|delta(s,u) - delta(s,v)| <= 1`. For a directed edge only the one-sided bound
`delta(s,v) <= delta(s,u) + 1` follows. Validators must not silently apply the
undirected absolute-value condition to directed graphs.

## When to mark visited

Several superficially similar policies have different meanings.

### Mark on first enqueue/accept

The ordinary policy. A single exact check-and-set chooses one shortest parent
and prevents duplicate frontier entries. In parallel code, the check-and-set
must be logically atomic even if its physical implementation is batched.

### Mark on dequeue

With a strict sequential FIFO queue, the first dequeued copy still has a
shortest label, but the same vertex may be enqueued many times. Correctness can
survive while the work and memory bound collapse. In a bounded parallel frontier
this distinction becomes semantic: duplicate pressure can cause overflow and
then reachable vertices are silently lost.

### Deduplicate once per completed level

Also correct if the candidate set is exact and the entire next frontier is
formed as `N(F_d) minus B_d`. Individual candidates need not mutate visited
immediately. This is the set recurrence written literally.

### Forget to include the current frontier in visited

Incorrect on graphs with same-level edges. In a triangle `s-a-b-s`, after
forming `F_1={a,b}`, expanding against only `B_0={s}` makes `a` and `b` appear
to rediscover one another for `F_2`. Bipartite graphs can hide this bug because
they have no same-layer edges; REF-003 encountered exactly this near-miss.

## Counterexamples that expose the contract

### First discovery plus a stack is not BFS

Use edges:

```text
s -> a, s -> b, a -> x, b -> c, c -> x
```

If a stack explores `b,c,x` before `a`, and first discovery permanently marks
`x`, it records distance 3 although the path `s,a,x` has length 2. Exact visited
membership cannot repair an invalid schedule.

### Unit-edge BFS is not weighted shortest path

Let `s -> t` have weight 10 and `s -> a -> t` have weights 1 and 1. BFS prefers
the one-edge route while minimum total weight is 2. BFS minimizes edge count;
it does not inspect arbitrary weights.

### A hash collision is not a duplicate

If distinct children `a` and `b` share a hash and the first suppresses the
second, an entire subtree reachable only through `b` can disappear. Exact BFS
may index by hashes, but equality needs an injective encoding or a full-state
collision check.

### Non-atomic parallel discovery can become incorrect through capacity

Two threads that both observe “unvisited” may append the same vertex twice.
With unbounded storage, a later exact dedup can recover the set. With a fixed
frontier, duplicates can consume capacity before another unique vertex is
written. Overflow is therefore not merely a performance event.

## Early stopping changes the returned object

If all expanded parents are in `F_d`, the first generated target has distance
`d+1`; stopping immediately can return one shortest path. But it does not finish
`F_{d+1}`, discover every vertex at that distance, or collect all shortest
parents. Candidate-stop, parent-batch-stop and complete-level-stop can agree on
the target distance while computing different amounts of the BFS object. This
was measured in REF-008.

The required result must therefore be named:

- one shortest path to one target;
- exact target distance;
- all targets at minimum distance;
- all shortest parents/path counts;
- complete ball through a depth;
- complete reachable component.

“Run BFS” is underspecified without this termination contract.

## Parent nondeterminism and distance determinism

When several vertices in `F_d` reach the same child, any of them certifies a
shortest path. Parallel first-winner selection can therefore change the parent
tree without changing distances or frontier sets.

If deterministic parents are required, “first atomic winner” is not enough.
One may need to finish the layer and reduce all valid candidates by a canonical
rule such as minimum `(parent_key, generator_id)`. Determinism has an algorithmic
cost and is separate from shortest-distance correctness.

## Cayley interpretation

For a group `G` with generator set `S`, BFS from the identity computes the word
metric:

```text
F_d = {g in G : the shortest word over S representing g has length d}
```

Different generator words can represent the same element because of group
relations. Those are not accidental implementation duplicates; they are the
reason visited/dedup is central to Cayley BFS. Adjacent transpositions in `S_n`
make distance equal inversion count, so Mahonian coefficients give an
independent frontier-size oracle.

Changing the generator set changes the metric itself. Adding an identity adds
same-level self transitions; duplicate generators add predictable occurrences;
new nonredundant generators can reduce diameter while increasing degree and
total work. REF-004 records all three effects.

Quotienting by symmetry is also not “just a faster visited key.” It changes the
vertex identity unless canonical representatives and lifted paths are proved to
preserve the problem being solved.

## Working mental model

BFS is best pictured as three coupled objects:

```text
metric semantics:   B_0 subset B_1 subset ... subset reachable(G)
schedule:           expose every edge needed for the next sphere before advancing
representation:     store exact membership and enough witnesses/work to continue
```

Most subtle bugs come from proving only one of the three:

- a correct queue with lossy hash equality;
- an exact set with a depth-first schedule;
- correct distances with an incomplete frontier due to overflow;
- a valid parent tree without proof that labels are minimal;
- correct target distance while claiming a complete layer.

## Sources

- E. F. Moore, *The Shortest Path Through a Maze*, Proceedings of the
  International Symposium on the Theory of Switching, Part II, Harvard
  University Press, 1959, pp. 285-292. Bibliographic record:
  [BibBase](https://bibbase.org/network/publication/moore-theshortestpaththroughamaze-1959).
- C. Y. Lee, *An Algorithm for Path Connections and Its Applications*, IRE
  Transactions on Electronic Computers 10(3), 1961, pp. 346-365,
  [doi:10.1109/TEC.1961.5219222](https://doi.org/10.1109/TEC.1961.5219222).
- E. D. Demaine and C. E. Leiserson, MIT 6.046J lecture notes,
  [Shortest Paths I](https://ocw.mit.edu/courses/6-046j-introduction-to-algorithms-sma-5503-fall-2005/c0da168ddb4a6f7f252ca4461080bf71_lec17.pdf),
  for the two-consecutive-distance FIFO invariant.
- Local evidence: REF-001 (validator counterexamples), REF-003 (current-layer
  visited near-miss), REF-004 (generator-set effects), and REF-008 (termination
  granularity).

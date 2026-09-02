# A mental model of BFS: from metric balls to hardware

Breadth-first search is easiest to understand neither as a queue nor as a GPU
frontier loop, but as a proof-producing construction of distance balls. The
queue, bitmap, hash table, sort, owner routing, and kernels are mechanisms for
maintaining that construction.

This note synthesizes the preceding focused notes into one mental model. It
adds no implementation, benchmark, or optimizer.

## The one-sentence model

For a fixed unit-cost directed graph, exact vertex identity, and source set,
BFS repeatedly takes the complete outgoing boundary of the reached metric ball,
removes states already in the ball, and commits every remaining state as the
next exact distance layer.

In symbols,

```text
B_0 = sources
F_0 = sources

F_(d+1) = N_out(F_d) minus B_d
B_(d+1) = B_d union F_(d+1).
```

Equivalently,

```text
B_(d+1) = B_d union N_out(B_d)
F_d = B_d minus B_(d-1).
```

`B_d` is the set of vertices at distance at most `d`; `F_d` is the sphere at
distance exactly `d`. This is the semantic object. Any physical structure is
correct only if it represents the same sets under the requested output contract.

### Why expanding only the newest frontier is exact

Since the ball is the union of its shells,

```text
B_d = F_0 union F_1 union ... union F_d,
```

and relational successor image distributes over union,

```text
N_out(B_d) = union_(i=0)^d N_out(F_i).
```

Every successor previously generated from `F_i`, `i<d`, has distance at most
`i+1<=d` and is already inside `B_d`. After subtracting visited, those rescans
can contribute nothing new:

```text
(N_out(B_d) minus B_d) = (N_out(F_d) minus B_d) = F_(d+1).
```

Thus frontier-only expansion is not an approximation and does not assume that
older vertices have no outgoing edges. Their consequences were already closed
when their layers were processed. The frontier is the delta of newly added
facts, and BFS applies the recursive edge relation only to that delta—exactly
the semi-naive principle used in fixed-point evaluation.

The argument uses more than monotonicity: it uses complete prior layer closure
and the fact that graph relational image preserves unions. Dropped edges,
partial layers, a changing graph epoch, or an operator that does not distribute
over union invalidate this particular delta recurrence.

### Why finite BFS rounds reach the least fixed point at omega

Define

```text
T(X) = sources union N_out(X),
R = union_(d<omega) B_d.
```

Every member of `R` has a finite path witness and therefore appears in some
finite-depth ball (not necessarily a finite-cardinality ball). Conversely,
successor image preserves arbitrary unions:

```text
N_out(union_i X_i) = union_i N_out(X_i).
```

So applying `T` to `R` adds nothing beyond the union of the next finite-depth balls;
`T(R)=R`. Any other fixed point containing the sources must contain `B_0`, then
`B_1`, and inductively every `B_d`, hence must contain `R`. Reachability is
therefore the least source-containing successor-closed set.

Monotonicity alone does not imply convergence after all finite rounds. On the
universe `N union {infinity}`, consider

```text
U(X) = {0}
       union {n+1 : n in X intersect N}
       union ({infinity} if N subset X else empty).
```

`U` is monotone. Starting from empty, every finite iteration contains only a
finite initial segment of naturals. Their omega-union is all of `N`, but only
the *next* application notices that every natural is present and adds
`infinity`. Thus the finite approximants' union is not yet a fixed point.

Ordinary graph reachability has no such “wait until infinitely many premises
exist” transition: one edge and one already reached predecessor witness each
new vertex. That finitary, union-preserving structure is why BFS covers every
finite-path reachable vertex by some finite depth even when the traversal as a
whole never terminates.

### One tiny wave by hand

Use the undirected graph (each line represents both edge directions):

```text
    a -- c
   /     |
  s      b -- d
   \----/
```

Equivalently, the edges are `s--a`, `s--b`, `a--c`, `b--c`, and `b--d`.
Starting from `s`:

```text
F_0 = {s}              B_0 = {s}
F_1 = {a,b}            B_1 = {s,a,b}

outgoing occurrences from F_1 = [s,c, s,c,d]
distinct candidates            = {s,c,d}
minus B_1                      = {c,d} = F_2
B_2                            = {s,a,b,c,d}
```

This separates three objects that are often all called “the queue.” The
frontier `{a,b}` is only the current exact-distance layer. Expansion produces
edge occurrences, including two occurrences of `c` and two returns to `s`.
Exact identity collapses the two `c` occurrences; visited removes the old state
`s`; the survivors `{c,d}` form the next frontier. Visited is the accumulated
ball, not another name for the current frontier.

### Three objects called the boundary

For the reached ball `B_d`, distinguish:

```text
metric sphere F_d
crossing edge occurrences {(u,v): u in B_d, v not in B_d}
external vertex boundary {v not in B_d: some u in B_d has u -> v}.
```

In a unit-edge graph the external vertex boundary is exactly `F_(d+1)`. Also,
every crossing edge must start in `F_d`: if `u` had depth at most `d-1`, then
the edge would give `dist(v)<=d`, placing `v` inside `B_d`. This is why BFS only
needs to expand the current sphere rather than rescan the whole ball.

The three cardinalities still need not agree. First, a frontier vertex can be a
radial dead end:

```text
s -> a -> x       s -> b       F_1={a,b}, but only a has an edge outside B_1.
```

Thus `F_d` is the complete distance sphere, not merely the subset that makes
outward progress. Second, different crossing edges can converge:

```text
s -> a -> x
s -> b -> x
```

At depth one there are two crossing edge occurrences but the external vertex
boundary and next frontier are both `{x}`. Parallel labels can increase the
occurrence count further without changing either vertex set. Consequently,
frontier width, boundary-edge work, and next-frontier growth are related by the
BFS recurrence but are not interchangeable workload measurements.

### Loops and parallel labels change work/output before distance

Use a labeled multigraph with one self-loop and two parallel edges:

```text
s -e-> s
s -p-> a
s -q-> a
```

Expanding `F_0={s}` emits three edge occurrences `[s,a,a]`. Exact identity gives
two candidate vertices `{s,a}`; visited subtraction removes `s`; the next
vertex frontier is only `F_1={a}`.

Deleting the self-loop or merging the parallel edges does not change the vertex
distance `dist(s,a)=1`. It does change physical expansion work. Merging `p` and
`q` also changes richer output: the multigraph has two labeled shortest
one-edge paths to `a`, while its simple support graph has one support edge.
Therefore loops/parallel edges can be irrelevant to vertex distances without
being irrelevant to occurrence counts, labels, path counts, or capacity.

### How far one edge can move across BFS layers

For an undirected edge `{u,v}`, a shortest path to `u` followed by the edge
gives `d(v)<=d(u)+1`. Reversing the same edge gives `d(u)<=d(v)+1`. Hence

```text
|d(u)-d(v)| <= 1.
```

So expanding an undirected frontier `F_d` can touch only depths `d-1`, `d`, and
`d+1`. Same-layer edges may exist; bipartite graphs are the important case
where parity excludes them.

For a directed arc `u->v`, only the first inequality survives:

```text
d(v) <= d(u)+1.
```

The arc cannot jump forward by more than one BFS level, but it may point
arbitrarily far backward. A directed chain

```text
s -> v1 -> v2 -> ... -> vk
^                       |
|_______________________|
```

has `d(vk)=k` and an arc `vk->s` spanning back to depth zero. Therefore a small
rolling window of visited layers follows automatically from undirected edge
symmetry, but not from directed BFS layering alone. Forgetting old directed
layers needs an additional bounded-backward-span theorem for the actual graph.

For exact scalar distance/reachability traversal of a static undirected graph,
the resulting rolling state is concrete:

```text
previous = F_(d-1)
current  = F_d
next     = the exact set being built for F_(d+1).
```

While expanding `current`, membership in `previous` rejects inward neighbors,
membership in `current` rejects same-layer edges and loops, and membership in
`next` merges multiple current parents reaching the same new state. No neighbor
can lie in `F_0,...,F_(d-2)`, so those layers are irrelevant to future novelty
decisions. Once `next` is completely constructed and durably published, rotate
the roles and reclaim the old `previous` layer.

All three roles have small witnesses. In a diamond, skipping one selected
parent does not reject the child's other depth-`d-1` predecessor, so
`previous` is needed. In a triangle, one current vertex reaches another, so
`current` is needed. In the same diamond, two current parents propose one child,
so building-`next` dedup is needed.

This is safe forgetting relative to a narrow output contract, not deletion of
all BFS knowledge. It does not retain the complete reached set, old distance
table, parent chains, shortest-path DAG, or a restart image that can answer
arbitrary old queries. Distributed execution must also finish routing,
deduplication, and publication for the layer before reclaiming it. For an
inverse-closed Cayley generator set the same bidirectional-edge argument
applies; a nonsymmetric directed generator set needs a separately proved bound
on how far one move can jump backward in the chosen word metric.

That bound has a direct Cayley form. Use right edges `x -> x*g`, `g in S`, and
suppose every inverse `g^-1` has a positive `S`-word of length at most `L_g`.
A shortest word reaching `x*g`, followed by that inverse word, reaches `x`, so

```text
d(x) <= d(x*g) + L_g
d(x*g) >= d(x) - L_g.
```

With `L=max_g L_g`, expanding `F_d` can therefore hit old states only in

```text
F_(max(0,d-L)), ..., F_d.
```

Under strict complete-layer scheduling, retaining exactly those old layers plus
the building `F_(d+1)` is sufficient for scalar duplicate rejection. The
symmetric case has `L=1` and recovers the three-role undirected window. A
nonsymmetric generating alphabet may give a larger but still finite window.

For the full directed Cayley graph this bound is exact when every `L_g` is the
shortest positive-word length of `g^-1`. The edge

```text
g^-1 -> e
```

falls into the root by exactly `L_g` layers, so

```text
backward span = max_(g in S) d_S(e,g^-1).
```

It is therefore the smallest uniform window for duplicate rejection by retained
layers alone, not merely a triangle-inequality estimate. A Schreier quotient
can contract this witness because several group elements become one state. In
that setting the useful hierarchy is

```text
root-specific radial drop
    <= worst shortest return xs -> x in the action
    <= group inverse-word length.
```

The first inequality becomes equality when the directed support graph itself is
vertex-transitive. Transitivity of the underlying group action alone is weaker:
a fixed non-conjugation-invariant generator set can give state-dependent support
geometry.

A fixed window also hides individual layer lifetimes. Let `tau_j` be the last
source depth of any edge ending in `F_j`, with at least `tau_j=j` so the layer
survives its own expansion. Then `F_j` is membership-dead after depth `tau_j`
is completely expanded, and

```text
backward span = sup_j (tau_j-j).
```

Thus the window is a uniform envelope over exact last-reference depths. Current
frontier observations cannot prove an old layer dead because a still-later
frontier may refer back to it. Structural predecessor bounds or another exact
certificate must exclude those future references. On multiple GPUs, logical
last use still precedes physical reclamation until delayed candidates, retries,
and publications from that depth are globally retired.

None of these identities automatically makes rolling storage a memory win. If
some `g^-1` is not expressible in the positive generator monoid, the group
argument supplies no finite window. The proof must always use the actual
semantic state graph and requested output: quotienting, parents, path counts,
restart evidence, and canonicalization can retain obligations after scalar
membership is dead.

Even when the window is safe, its number of layers does not determine its
memory saving. Just before reclaiming the previous layer, three retained layers
on a long endpoint-rooted path contain only three states out of `d+2`. On a
rooted binary tree with geometric layer growth, the same three-layer snapshot
contains asymptotically 87.5 percent of the entire ball. Rolling storage removes
old volume; it does not remove much when almost all volume is near the current
boundary. This compares distinct membership states at a matching phase, not
actual allocation peaks or the best tree-specific algorithm (note 181).

### Skipping the parent is not a replacement for visited except on a tree

In a guaranteed undirected tree, every non-root vertex has one neighbor toward
the root and all other neighbors lie one level farther away. When expanding a
vertex, skipping that single incoming parent edge therefore leaves exactly its
previously unseen children. Global visited is redundant because unique simple
paths already provide unique discovery.

The smallest cyclic counterexample is a triangle:

```text
  a
 / \
s---b
```

After `s`, the frontier is `{a,b}`. While expanding `a`, skipping its parent
`s` still leaves `b`, but `b` is not a child: it is an already discovered
same-layer vertex. Expanding `b` symmetrically proposes `a`. Parent-skip removes
only immediate reversal; it cannot recognize same-layer convergence, longer
returns, or two different parents reaching one child.

There is a precise nuance. A finitely branching, level-ordered search over
*walk occurrences* may omit global visited and still find a shallowest target:
the first target occurrence has minimum word length. But it is then searching a
path/walk tree, not enumerating each graph state once. On a finite cyclic graph
the occurrence tree can be infinite, frontier sizes count repeated histories,
and an empty frontier no longer certifies exhaustion of the finite component.
Exact visited is what turns that occurrence search into BFS over distinct graph
states.

## Why first discovery is shortest

The proof has two halves.

### Soundness: no assigned depth is too small

Every state placed in `F_(d+1)` is reached by a real edge from a state with a
real length-`d` witness. Appending that edge gives a path of length `d+1`.

```text
true_distance(v) <= assigned_depth(v).
```

### Completeness/minimality: no true shorter path was missed

If a state has a path of length `d+1`, its penultimate state has a path of
length at most `d` and belongs to `B_d`. Complete expansion exposes the final
edge, and exact visited can remove the state only if it was already reached at
an equal or smaller depth.

```text
assigned_depth(v) <= true_distance(v).
```

Together the labels are exact. The proof uses complete successor enumeration,
exact identity, and nondecreasing finalization. A FIFO queue is one way to
enforce the schedule; it is not the theorem.

### What FIFO contributes

FIFO gives a very concrete version of nondecreasing finalization. When a
depth-`d` vertex is removed, every vertex it appends has depth `d+1` and goes
behind everything already waiting. Therefore all still-waiting depth-`d`
vertices are removed before any newly appended depth-`d+1` vertex. By induction,
the queue contains at most two consecutive depths, with all entries of the
smaller depth first.

The smallest useful contrast is:

```text
s -> a -> x
 \-> b -> c -> x
```

The true distance to `x` is two through `a`. A LIFO stack can follow
`s,b,c,x` first and, if first discovery is frozen by `visited`, incorrectly
label `x` with depth three before it ever expands `a`. FIFO may choose either
`a` or `b` first, but it must finish both depth-one vertices before `c`, so the
edge `a -> x` is exposed before any length-three claim can become final.

This isolates the roles: `visited` prevents duplicate commitment, while FIFO
makes that first commitment safe. Replacing FIFO is allowed only when another
mechanism preserves nondecreasing finalization or permits later relaxation.

### A layer is a set for distance, but an order for some outputs

Consider the diamond

```text
    a
  /   \
s       x
  \   /
    b
```

After expanding `s`, the exact layer is the set `F_1={a,b}`. Expanding `a`
before `b` or `b` before `a` exposes the same distinct next-layer set
`F_2={x}`. Both schedules therefore assign `dist(s,x)=2`: every producer is at
depth one, so every proposal for `x` has the same candidate depth two.

The order is nevertheless observable if the output retains more than distance.
A mark-on-first-discovery implementation selects `a` as parent in the first
schedule and `b` in the second. The schedules may also perform different
amounts of work under immediate target stopping. Neither result is a canonical
parent, the complete predecessor set `{a,b}`, or a shortest-path count of two.

Thus within-layer permutation is safe only relative to a declared contract.
It is confluent for exact frontier sets and scalar distances after complete
layer closure. It is not automatically confluent for parents, discovery order,
early-stop work, labeled paths, or reproducible byte-for-byte output. Parallel
BFS exploits the first fact; richer outputs need an explicit reduction over all
relevant equal-depth proposals.

### Four different meanings of "finished" on that same diamond

Assume exact unit edges, mark-on-discovery FIFO BFS, and adjacency order `a`
before `b`. The same graph separates the events without another example:

| Event | What is already exact | What is not yet complete |
|---|---|---|
| `s` fully expanded | `B_1={s,a,b}` and `F_1={a,b}` | Outgoing work from `a,b` |
| `a -> x` first accepted | `dist(x)=2`, one shortest witness through `a` | Predecessor set/count of `x`; closure of depth-one outgoing work |
| All outgoing work from `a,b` retired, including metadata | `F_2={x}`, predecessors `{a,b}`, count two | Expansion of `x` |
| `x` fully expanded | Its outgoing work is complete | Any downstream or global obligations, if the graph has more edges |

The second row is already a distance proof: `x` is outside the exact known
ball `B_1`, and `s,a,x` is a length-two witness. Waiting for `b` cannot shorten
that distance, but can change the required metadata. In a graph not yet fully
inspected, the remaining producer may also reveal another depth-two vertex.

Thus distance finality, metadata finality, complete-layer membership, and
vertex expansion are separate events. Gray vertices can have final distances;
black means fully expanded, not "became shortest only now." An accepted record
must also remain scheduled or have a live publication obligation if traversal
continues (note 178). Correctness of its scalar distance alone does not prove
that its descendants will ever be processed.

An explicit GPU/global barrier is not what makes the distance theorem true.
FIFO achieves the requisite ordering sequentially; other executions need an
equivalent exclusion of shorter unresolved paths or corrective relaxation.
Closing a layer additionally requires retiring all its relevant in-flight
work. A barrier call without that coverage is not a closure certificate.

## The semantic pipeline

One BFS level can be read as a funnel:

```text
frontier semantic states
  -> generated transition occurrences
  -> valid candidate occurrences
  -> distinct semantic candidate states
  -> candidates not in the completed ball
  -> every state in the exact next frontier
  -> optional distance/parent/count/path output.
```

Each arrow answers a different question:

| Stage | Core question | Typical failure |
|---|---|---|
| expansion | Were all required edges generated? | missing/incorrect move |
| validity/canonicalization | Is this the declared graph/state? | wrong graph or quotient |
| identity | Which occurrences denote one vertex? | hash collision or undermerge |
| visited subtraction | Is the state already in the ball? | false positive deletes branch |
| commitment | Was every new state retained once? | overflow, race, top-k drop |
| output | Which parents/labels/counts are required? | distance passes, richer output wrong |
| completion | Is the whole logical boundary settled? | local empty mistaken for global empty |

Performance mechanisms may fuse arrows, but correctness still needs every
logical question answered.

### False `seen` and false `unseen` are asymmetric

On the path

```text
s -> a -> t
```

a visited false positive on `a` says “already reached” when `a` is actually
new. BFS drops the only gateway, never expands `a`, and can falsely report `t`
unreachable. One mistaken positive can therefore delete semantic reachability.

A false negative on an actually visited `a` says “new” and normally creates a
duplicate record or re-expansion. That need not change the final reached set if
an exact authoritative check later merges it, repeated processing is
idempotent for the requested output, capacity is lossless, and termination
accounts for the extra work. Without those conditions, even a false negative
can become a correctness failure through overflow, duplicate counting, or
premature completion.

The safe summary is not “false negatives are harmless.” It is: false positives
directly remove possible paths; false negatives first add physical work and are
semantically tolerable only behind an exact, lossless fallback contract.

## Five contracts define the algorithm

### 1. Graph and identity

Declare:

```text
semantic vertex
equality/canonicalization
directed/labeled/multiedge convention
source and target sets
edge costs and graph version.
```

A hash, rank, byte layout, or symmetry normalization is not automatically the
semantic identity. A state key is exact only through injectivity or collision
resolution. A quotient changes the graph unless it already expresses true state
equality or has a distance/lifting proof.

Graph version belongs to the same semantic contract. Initially let `s -> a`
exist while `a -> t` is absent. BFS expands `s` and retains `a`. Then update the
graph by deleting `s -> a` and inserting `a -> t`; later expansion of the old
`a` record discovers `t` and reports

```text
s -> a -> t.
```

Every edge was real when individually observed, but no snapshot contained both
edges. The returned chain is therefore not a path in either static graph
version. Per-edge validity and race-free reads do not imply snapshot-path
validity.

Ordinary BFS proofs require one fixed relation `E_k` for successor generation,
visited identity, parents, and replay. Immutability, copy-on-write adjacency,
versioned reads, or an update barrier are different mechanisms for providing
that one snapshot. In multi-GPU execution, all ranks and in-flight records need
the same graph/action/legality epoch or an explicit conversion/repair rule.

Other questions are legitimate but different algorithms:

```text
snapshot BFS: exact distances in one named G_k
dynamic BFS: maintain/repair exact distances after each graph update
temporal search: paths whose edges are usable in an allowed time order.
```

The mixed chain above might be a valid temporal journey if its timestamps obey
that model, but it is not evidence for static BFS. A mutable generator set,
legality predicate, quotient, or canonicalization version creates the same
epoch obligation in an implicit/Cayley graph.

Arbitrary canonicalization can even invent a quotient path. Take concrete
states and edges

```text
s -> a       b -> t
```

with no other edges, and declare `a ~ b`. If quotient edges are added whenever
*some* representatives have an edge, the quotient contains

```text
[s] -> [a,b] -> [t].
```

But the first quotient edge reaches concrete representative `a`, while the
second is witnessed only by incompatible representative `b`. There is no
concrete path from `s` to `t` to lift. Exact BFS on this quotient is therefore
exact for an invented quotient graph, not for the original reachability query.

A safe quotient needs representative-compatible transitions/path lifting, such
as an appropriate automorphism-orbit or bisimulation contract. Even then, an
orbit quotient naturally answers distance to a target orbit; preserving one
fixed concrete target is an additional requirement.

The ordinary visited rule has a simple replacement argument. If two prefixes
reach the same semantic vertex in `p <= q` steps, any fixed legal suffix of
length `r` available to the second is also available to the first. Replacing
the second prefix gives `p+r <= q+r`, so the later arrival cannot improve
minimum hop distance. This argument concerns walks and a fixed state-based
goal; it does not preserve counts or all path witnesses. It also explains why
FIFO/nondecreasing discovery matters: the retained prefix must be no longer.

History can force the opposite change: one base configuration must split into
several semantic vertices. Use labeled edges

```text
s -a-> x
s -b-> x
x -a-> t
```

and require the complete label word to be exactly `ba`. If base-only visited
processes `s-a->x` first, it marks `x` and discards the second arrival after
label `b`. The only accepted continuation `b,a` is then lost.

The correct product graph distinguishes

```text
(x, after_a) != (x, after_b).
```

They have the same base state and depth, but different legal future languages.
The semantic BFS vertex is therefore `(base_state, memory_state)`, where the
memory may be a previous move, automaton state, resource phase, or other finite
history sufficient to make future transitions depend only on the current
product state.

This split is required only when history is part of legality or the requested
output. A “do not immediately undo/repeat” rule used solely as proved-safe
pruning for ordinary unconstrained distance does not automatically redefine
the vertex. The same filter syntax can mean either a semantic product graph or
an optimization of a base-graph search; its proof obligation decides which.

### 2. Expansion

Declare how every outgoing transition is obtained:

- stored adjacency for an explicit graph;
- `successors(state)` for an implicit graph;
- generator action for a Cayley/Schreier graph;
- product transitions when history affects legality.

In an implicit graph, the expansion procedure **is** the edge table. Missing a
move silently changes the graph even if the BFS schedule is perfect.

### 3. Schedule/finalization

Ordinary BFS finalizes by complete nondecreasing hop layers. Other valid
schedules need other proofs:

- 0-1/Dijkstra: relaxation and minimum-cost settling;
- asynchronous search: decreasing labels, reactivation, and quiescence;
- A*: global lower-bound versus incumbent closure;
- bidirectional BFS: two completed-distance lower bounds and a meeting upper
  bound.

Atomic first claim resolves concurrency, not distance order.

#### A level barrier is a proof mechanism, not the mathematical definition

Reuse the two routes

```text
s -> a -> x
s -> b -> c -> x
```

Suppose distributed or asynchronous delivery lets the length-three proposal
`s,b,c,x` reach the owner of `x` before the length-two proposal `s,a,x`. If the
owner treats its first atomic claim as an immutable visited decision, it freezes
the wrong label `3`. The atomic operation proves that only one writer won; it
does not prove that no smaller-distance message is still in flight.

A level-synchronous barrier prevents the failure by closing every depth-one
expansion and its communication before any depth-two vertex can produce a
depth-three claim. It preserves the ordinary BFS theorem that first discovery
is already final.

But the barrier itself is not mandatory. A label-correcting execution may store
`dist[x]=min(dist[x], proposal)`, accept the later decrease `3 -> 2`, and
reactivate `x` if its stale expansion could have propagated the old label. Its
completion condition must establish both quiescence and absence of in-flight
smaller proposals. That algorithm can converge to the same unit-edge distances,
but it has exchanged simple layer closure for relaxation, reprocessing, and a
stronger distributed termination proof.

Therefore there are two distinct safe contracts:

```text
ordered complete layers + irreversible first discovery
or
out-of-order proposals + corrective minima/reactivation/quiescence.
```

Mixing out-of-order delivery with irreversible first-winner visited is the
unsafe third combination.

### 4. Output

Increasingly rich outputs include:

```text
one target distance
one replayable path
all reached distances
one parent tree
canonical/shortlex parent tree
all shortest predecessor edges
shortest-path counts
uniform shortest-path sampling
explicit all-path enumeration.
```

A race or dedup rule can be benign for the first rows and destructive for later
ones. "Exact" must name the output, not only the visited set.

#### A compact BFS result may represent exponentially many paths

Chain `k` diamonds. For each stage `i`, connect

```text
v_(i-1) -> a_i -> v_i
v_(i-1) -> b_i -> v_i.
```

The graph and its complete shortest-predecessor DAG use only `3k+1` vertices
and `4k` edges. At every diamond a shortest path independently chooses the
upper or lower branch, so

```text
sigma(v_i) = sigma(a_i) + sigma(b_i) = 2*sigma(v_(i-1)),
sigma(v_k) = 2^k.
```

One parent tree keeps one of those paths. The predecessor DAG keeps all choices
implicitly in linear space. A count table keeps only their number, subject to
an explicit overflow/arithmetic policy. Actually emitting every vertex or label
sequence must write `2^k` separate outputs and therefore takes at least
output-proportional time and bytes even though BFS itself saw only `O(k)` graph
structure.

This is not cured by GPU parallelism: hardware may enumerate many paths at
once, but cannot make the required output records disappear. “Compute distance,”
“compute a compact all-shortest representation,” “count,” “sample,” and
“enumerate all” are genuinely different computational contracts.

#### Count overflow changes the question, not the distance

At `k=64`, the diamond chain has exactly `2^64` shortest paths. An unsigned
64-bit counter cannot represent that value. Common arithmetic policies now
produce different semantics:

```text
exact integer:       18446744073709551616
wrapping u64:        0
saturating u64:      18446744073709551615
modulo M:            2^64 mod M.
```

If counts are only side metadata, the BFS distances and predecessor DAG can
remain perfectly correct while the reported path count is wrong. Wrapping is
especially deceptive because zero normally means “no path.” Saturation can
truthfully mean “at least the maximum representable value” only if the
saturation flag/contract is retained. Modular arithmetic is exact for a
declared modulo-`M` question, not for the original integer count.

Exact-uniform shortest-path sampling needs exact predecessor weights. At a
vertex `v`, choosing predecessor `u` with probability

```text
sigma(u) / sigma(v)
```

makes every complete shortest path equiprobable. Wrapped, saturated, modular,
or imprecisely rounded counts generally change those ratios and can make some
branches impossible or overweighted. Thus a count representation must declare
both its arithmetic and every consumer that relies on it; a distance-only pass
does not validate count or sampling correctness.

#### Why the BFS predecessor structure is a DAG

For unit-edge BFS, retain exactly the shortest-predecessor edges

```text
E_sp = {(u,v) in E : d(u)<infinity and d(u)+1=d(v)}.
```

Every retained edge strictly increases depth by one. A directed cycle of `k>0`
retained edges would therefore require returning to the starting vertex with
depth increased by `k`, an impossibility. The original graph may have
self-loops, same-layer edges, reverse edges, and arbitrary cycles; none enters
`E_sp` unless it advances exactly one BFS layer.

The finite-depth guard matters when scanning a full graph rather than actual
frontier layers: `infinity` is not a final layer. Under extended arithmetic,
`infinity+1=infinity` would otherwise admit unreachable cycles. Their
source-path counts must be zero, not values sustained by a circular recurrence
(note 41's unreachable two-cycle counterexample).

This monotonicity is why path counts can be computed forward by depth and why
backward sampling can move to smaller depths without cycling. It also explains
why one parent tree is merely a selected arborescence inside a larger acyclic
shortest-path structure.

The word `DAG` should not be transferred blindly to every weighted shortest
subgraph. With zero-cost edges, two vertices can have equal optimal distance
and zero-cost shortest-compatible edges in both directions. For example,

```text
s -1-> a,   a -0-> b,   b -0-> a
```

gives `d(a)=d(b)=1`, while `a<->b` is a zero-cost cycle. Simple shortest paths,
shortest walks, their counts, and a condensed zero-cost-component DAG are then
different objects. Unit positive edge cost makes the ordinary BFS predecessor
DAG theorem automatic.

The same distinction affects validation. Tight chosen parent edges in the
zero-cost case may form a cycle, and even false finite labels on an unreachable
zero-cycle can satisfy every local weighted edge inequality. The weighted
certificate therefore needs parent chains that explicitly terminate at the
source. Conversely, a vertex reached by zero-cost edges can legitimately share
the source's zero distance. The unit-only unique-zero and strict-depth proofs
must not be copied unchanged (notes 12 and 41).

### 5. Completion and failure

Keep distinct:

```text
FOUND       proved target result
EXHAUSTED   reached component is successor-closed
BOUNDED     declared radius completed
INCOMPLETE  timeout, OOM, overflow, cancellation, loss, mismatch, unknown.
```

An empty completed next frontier proves closure. A temporarily empty queue or
zero local count does not. Negative results require stronger global coverage
than positive replay witnesses.

The full reached ball need not remain resident for this proof. Suppose every
earlier layer was exactly completed before its rolling-window records were
reclaimed, and complete expansion of `F_d` now constructs
`F_(d+1)={}`. If a reachable vertex existed outside `B_d`, take a shortest path
to it and its first vertex outside `B_d`; its predecessor would lie in `F_d`,
so complete expansion would have placed that vertex in `F_(d+1)`, a
contradiction. The retained chain of layer-closure facts, not the current bytes
of every old visited entry, supplies the induction.

There is an operational boundary. A continuously trusted run may carry that
proof as control state, but a crash/restart or external audit cannot infer it
from an isolated empty buffer. It needs a checkpoint/certificate stating which
layers were completely expanded, that no overflow or loss occurred, and—under
distribution—that no routed, staged, retryable, or in-flight record remains.
Reclaiming old membership and reclaiming evidence of complete processing are
different actions.

For a concrete contrast, take two disconnected paths:

```text
s -- a -- b       x -- y
```

Starting at `s`, the exact frontiers are `{s}`, `{a}`, `{b}`, then `{}`. The
last empty frontier is meaningful because `b` was completely expanded and all
of its neighbors were already in the reached ball `{s,a,b}`. That ball has no
outgoing edge to a new vertex, so `x` and `y` are unreachable from `s` in the
declared graph.

If the run instead stops immediately after completing depth one, it knows only
`B_1={s,a}`. The absent vertex `b` is reachable just beyond the cutoff, so the
same absence means `NOT_FOUND_WITHIN_RADIUS`, not `UNREACHABLE`. Empty after
complete closure and stopped-before-next-expansion are different evidence even
when neither returned the target.

### Infinite BFS can be complete without ever being exhausted

Take the infinite undirected ray

```text
0 -- 1 -- 2 -- 3 -- ...
```

started at `0`. Its exact frontiers are `F_d={d}`. Every individual layer is
finite and is completed after finite work; every particular reachable target
`k` is therefore discovered after `k` layers. Yet the next frontier is never
empty, so the full traversal never returns `EXHAUSTED` and visited memory grows
without bound.

This is the useful completeness notion for a locally finite infinite graph:
every reachable vertex eventually appears. With finitely many sources and
finite degree at each vertex, each finite-radius ball is finite by induction,
so layer-by-layer progress is operationally meaningful. It does not imply that
the reachable component is finite or that a negative reachability query will
terminate.

If local finiteness is removed, even this operational story can fail. A source
with infinitely many immediate successors has an infinite `F_1`; an ordinary
algorithm cannot finish that layer before advancing. Thus these are separate
claims:

```text
each finite-depth layer can be completed
every reachable target is eventually discovered
the whole reachable component is exhausted in finite time.
```

The infinite ray satisfies the first two and not the third. A finite reachable
component satisfies all three. Consequently, on an infinite graph “target not
yet found” is generally only a bounded negative statement, not an
unreachability certificate.

## What variants preserve and what they change

| Method | Graph/metric | Complete frontier? | Finalization proof | Characteristic output |
|---|---|---:|---|---|
| sequential/parallel BFS | same unit graph | yes | nondecreasing complete layers | hop distances |
| push/pull BFS | same unit graph | yes | same next-layer predicate | hop distances |
| multi-source BFS | source set changes | yes | all sources at depth zero | distance to set/Voronoi ties |
| bidirectional BFS | same metric, reverse graph backward | partial balls | global meeting lower bound | one/all target paths by contract |
| bounded reverse BFS | restricted radius | yes through radius | completed boundary | exact local table/lower miss bound |
| PDB BFS | abstract relaxed graph | yes abstractly | exact abstract distances | concrete lower heuristic |
| 0-1/Dijkstra | weighted metric | not hop spheres | relaxation/settling | weighted distances |
| A* | same weighted objective | best-first open set | minimum `g+h` versus incumbent | optimal target under conditions |
| beam search | original graph is pruned | no | no original-graph lower-bound closure | heuristic survivor/path |
| IDDFS | path/word tree iterations | no persistent BFS frontier | complete increasing depth limits | shallowest target under conditions |
| quotient search | quotient graph | yes in quotient | quotient metric | needs lifting for original path |
| product-state BFS | enlarged state graph | yes in product | ordinary BFS on full state | constrained paths |

### Multi-source distance is canonical; a single Voronoi coloring is extra

Seeding all sources at depth zero canonically computes

```text
D(v) = min_(s in S) dist(s,v).
```

This minimum is a lossy projection, not simultaneous storage of every source
distance. On the path

```text
A -- v -- B
```

joint BFS with sources `{A,B}` stores `D(A)=0`, `D(v)=1`, and `D(B)=0`. It no
longer contains `dist(A,B)=2`, nor separate values
`dist(A,v)=dist(B,v)=1` as two source dimensions. Once the waves share one
visited/distance field, a farther source contribution is intentionally absorbed
by the minimum.

Therefore “many sources in one BFS” and “many independent BFS queries sharing
hardware” are different semantics. The latter may batch expansion machinery,
but it must preserve source-specific information if the full matrix
`{dist(s,v)}` is required. Generic independent BFS tracks novelty of the pair
`(source,vertex)`; an old claim from another source cannot reject a new pair.
A single scalar plus one winning label cannot reconstruct all losing distances.

This is not a lower bound requiring separate traversals or explicit pair
storage. In a full right Cayley graph, left translation preserves labeled edges,
so `dist(s,v)=dist(e,s^-1 v)`. One identity table plus coordinate transforms can
represent many source rows; the source remains in the lookup argument instead
of being absorbed into a minimum. Arbitrary transitive state actions need a
separate graph-automorphism proof before such reuse is justified.

Direction remains essential: an outward identity table gives distance *to*
identity at key `x^-1`, not necessarily at `x`. On the directed six-cycle with
only `+1`, distance `0->1` is one while `1->0` is five. An allowed word to
`x^-1` replays unchanged from `x` to identity; blindly reversing a word to `x`
can require inverse moves absent from the alphabet. Table reuse, nearest-source
minimum, and reversal of the graph are different operations (note 13).

It does not canonically choose one minimizing source when several tie. Consider

```text
A -- u -- v
     |
     B
```

with sources `A` and `B`. Both `u` and `v` are tied:

```text
dist(A,u)=dist(B,u)=1,
dist(A,v)=dist(B,v)=2.
```

Coloring `u` by `B` and `v` by `A` is pointwise valid—each label names a
nearest source—but the `A` cell `{A,v}` is disconnected. Every shortest parent
of `v` is `u`, whose selected label is `B`.

If the requested output is a source-rooted connected forest, pointwise nearest
labels are therefore insufficient. Each non-source vertex must retain a
shortest parent satisfying

```text
D(parent(v))+1 = D(v)
label(parent(v)) = label(v).
```

Following parents then gives an internal shortest path from every vertex to its
selected source, proving cell connectivity. Arbitrary first-winner labels,
canonical minimum-source labels, and the complete set of tied sources are three
different outputs. Canonical labels may require finishing all equal-depth label
proposals before expansion or later propagating a better tie label through
descendants; distance finality alone does not imply label finality.

Semantic source labels also must not be confused with physical GPU owners. The
former describe proximity/ties in the graph, while the latter partition work
and may be chosen for balance or locality.

### Directed multi-source BFS has two opposite nearest-set questions

Forward BFS seeded with `S` computes

```text
D_from(v) = min_(s in S) dist(s,v): which source can reach v?
```

Many facility/goal queries instead ask

```text
D_to(v) = min_(s in S) dist(v,s): which facility can v reach?
```

On a directed graph these are different fields. The path

```text
x -> t -> y
```

with facility set `{t}` is the smallest witness. Forward BFS from `t` reaches
`y` and not `x`, so it reports `D_from(y)=1`. But the to-facility query has
`D_to(x)=1` and no finite value for `y`. Seeding `t` in the reversed graph
produces exactly the latter field: reverse frontiers `{t}`, then `{x}`.

Thus “put every target in the initial queue” is insufficient until edge
orientation is declared. Undirected or fully symmetric transition systems hide
the distinction. In an implicit directed system, computing `D_to` requires
complete predecessor generation, and replay metadata must reconstruct the
original forward path from the query state toward its selected facility.

### IDDFS buys memory by recomputing shallow prefixes

Iterative deepening runs complete depth-limited DFS iterations with limits

```text
0, 1, 2, ..., d.
```

If every smaller iteration finished without a goal and the first goal appears
in the complete limit-`d` search, then no goal has a path shorter than `d` and
the returned depth is BFS-optimal. The proof comes from exhaustive increasing
depth bounds, not from retaining BFS metric spheres simultaneously. Finite
branching and fair complete enumeration within each bound are necessary.

On a regular tree with branching `b`, BFS may retain `Theta(b^d)` frontier
records, while depth-first iteration retains a path and pending siblings,
roughly linear in depth for fixed `b`. IDDFS pays by regenerating every shallow
prefix in later iterations:

```text
total tree-node visits = sum_(i=0)^d (d-i+1) b^i.
```

For fixed `b>1`, the deepest level dominates and this is only a constant-factor
increase over the exponential tree traversal. That slogan has sharp limits. On
a chain (`b=1`), BFS performs `Theta(d)` visits while IDDFS performs
`Theta(d^2)`. In a Cayley/implicit graph, many generator words may converge to
one semantic state, so the bounded word tree can be exponentially larger than
the exact visited-quotiented state graph.

Finding one shallowest target also does not mean IDDFS materialized the BFS ball
`B_d`, exact frontier `F_d`, all shortest parents, or component exhaustion.
Current-path cycle checking only prevents repetition within one recursion path;
it does not merge transpositions reached through different words. Thus IDDFS
and BFS can share the same scalar target-depth guarantee while exposing very
different work, memory, deduplication, output, and GPU-parallelism contracts.

### A level-shaped beam is still a pruned search

Consider the directed unit graph

```text
s -> a -> t
s -> b -> c -> d -> t.
```

Exact BFS keeps both depth-one states and returns distance two. A width-one beam
whose score prefers `b` discards `a`; it later returns the length-four route. If
the `b` branch ended at `d`, it would report failure despite the reachable
target through `a`.

Advancing surviving records from depth `d` to `d+1` therefore does not preserve
the BFS theorem. The missing obligation is complete frontier retention:

```text
exact BFS: every distinct eligible state in F_(d+1) survives
beam:      only a selected subset survives.
```

Heuristic **ordering** of a complete layer is compatible with exact distances;
heuristic **deletion** needs a separate proof. The first target at beam depth `k` is shortest
only among paths surviving that pruning history, not necessarily in the
original graph.

For example, an exact target-distance oracle permits width-one descent:
from h=r, some successor has h=r-1 and none has a smaller value. With complete
successor generation and no filter excluding that continuation, following a
minimum-h successor reaches the target in h(s) steps. This proves one optimal
path, not complete BFS layers. Merely admissible, even consistent, h is weaker:
in the graph above h(a)=1 and zero elsewhere makes width one choose b and lose
the shortest route. Note 24 gives the explicit inequalities.

An effectively infinite width behaves as BFS only if it actually covers every
eligible unique state of every encountered layer and no capacity overflow,
threshold, local top-k, or other drop occurs. Merely choosing a large numeric
width supplies no such proof. This is why a layered GPU loop, frontier arrays,
and depth counters can look BFS-like while implementing a different search
contract.

### Unit edge cost is part of the BFS question

Consider two directed routes to `t`:

```text
s -1-> a -1-> t             2 hops, total cost 2
s -0-> b -0-> c -0-> t      3 hops, total cost 0
```

Ordinary BFS reaches `t` through `a` at hop depth two before the zero-cost route
can finish at depth three. This is not a failure of BFS: it correctly minimizes
the number of edges. It is a mismatch between the hop metric BFS computes and
the weighted-cost metric the question asked for.

Once edge costs differ, first discovery by hop layer is not weighted finality.
0-1 BFS or Dijkstra permits a state's tentative cost to improve and orders
settlement by cost rather than by raw edge count. Merely attaching weights to a
FIFO BFS result does not turn its hop spheres into cost spheres.

The smallest 0-1 counterexample is

```text
s -1-> a
s -0-> b -0-> a.
```

If adjacency enumeration sees `a` first, ordinary mark-on-discovery stores
`dist(a)=1` and a Boolean visited bit can reject the later route through `b`.
But that route has total cost zero, so the correct relaxation is

```text
dist(a): 1 -> 0.
```

The 0-1 deque is a specialized monotone priority queue rather than FIFO with a
cosmetic rule. A successful zero-cost relaxation goes to the front and stays in
the current cost bucket `D`; a successful unit-cost relaxation goes to the back
in bucket `D+1`. Finishing cost `D` includes the whole reachable zero-cost
closure before `D+1` can settle.

Consequently, “discovered,” “active,” and “final” separate again. Improved
vertices may need reactivation, and older queued copies may become stale.
Ordinary BFS collapses tentative and final at first discovery only because every
edge adds exactly one and FIFO/layer order proves no smaller proposal can arrive.
The reusable principle is sound distance-key finalization, not the superficial
choice of queue versus deque.

### Pull changes enumeration, whereas reverse BFS changes the root and field

On the directed path

```text
s -> a -> b
```

suppose the current source-rooted frontier is `F_0={s}`. Push enumerates
outgoing edges of `s` and proposes `a`. Pull instead enumerates unvisited
candidates `{a,b}` and asks whether each candidate has a predecessor in
`F_0`; only `a` does. Both therefore compute the same forward next frontier
`F_1={a}` and the same source distance `dist(s,a)=1`. They merely enumerate the
predicate

```text
v not in B_d  and  exists u in F_d such that u -> v
```

from opposite sides.

Reverse BFS is a different computation. It starts a second traversal at a
target, follows predecessor edges repeatedly, and computes distances *to that
target*. Starting at `b` above gives reverse frontiers `{b}`, `{a}`, `{s}`.
Thus “inspect predecessors” does not by itself mean “search backward”: in pull,
predecessors are witnesses for the next layer of the original source-rooted
field; in reverse BFS, they are the expansion relation of a target-rooted
field.

This also exposes an implicit-graph boundary. Inverse moves can enumerate
predecessors of an already named state, but pull additionally needs a cheap
outer enumeration of candidate unvisited states. An invertible Cayley move set
therefore makes reverse traversal possible without necessarily making pull
available or useful.

### Reverse BFS changes edge orientation, not the distance question

Take the directed path

```text
s -> a -> t
```

Ordinary forward BFS started at target `t` reaches only `{t}` when `t` has no
outgoing edges. It answers “where can `t` go?”, which is not the requested
question “which states can reach `t`?” Traversing predecessor edges instead
gives the reversed path

```text
t -> a -> s
```

and reverse frontiers `{t}`, `{a}`, `{s}`. Their depths are exactly the original
forward distances to the target: `dist(t,t)=0`, `dist(a,t)=1`, and
`dist(s,t)=2`.

The operation used during backward expansion follows an inverse/predecessor,
but a replayable suffix stores the original forward move from predecessor to
current state. Undirected graphs hide this distinction because every edge is
available in both directions; directed and non-inverse-closed move sets do not.

### A bidirectional meeting is an upper bound; stopping closes a lower bound

Forward and reverse records can form a feasible path either at a shared vertex
or across a connecting edge. Its length `mu` is immediately useful: replay
proves a real path, so the true distance is at most `mu`. A meeting alone does
not state what shorter unfinished work may still exist.

Let `a` and `b` be the minimum unexpanded exact depths on the forward and
reverse sides in a complete-layer unit-cost search. Once a best feasible path
`mu` is known, the standard closure rule is

```text
a + b >= mu.
```

If a shorter path existed, some vertex on it would already lie in both exact
balls, producing an equally short or shorter meeting candidate. Thus the lower
bound from unfinished depths has caught the replayable upper bound.

Under these exact complete-ball/intersection assumptions, integer unit costs
give one extra unit: `a+b+1>=mu` is already sufficient. Any shorter path has
integer length at most `mu-1<=a+b` and must already have been detected through
the two balls. The original test is conservative, not incorrect. This
refinement concerns the proof, not measured or modified implementation behavior,
and cannot be copied unchanged to weighted costs or pending intersection checks.

There is a safe special case: if the two exact balls were disjoint, one side
begins its next layer, and the opposite visited set remains that fixed exact
ball, the first intersection already supplies one shortest distance/path.
The active layer need not finish for that narrow output. In contrast,
interleaving partial expansions on both sides can first connect two depth-two
vertices along a length-four route while a length-three route remains
unexposed (note 56's corrected example). The warning about first meeting is
therefore about missing lower-bound premises, not any partial layer by itself.
Early stopping also does not collect all equal-length connectors or close the
whole next frontier.

The useful question is not "does it look breadth-first?" but "which row's proof
and output contract does it actually satisfy?"

## Explicit, implicit, and Cayley are presentations

### Explicit graph

A dense integer ID may simultaneously index adjacency, visited, frontier, and
parents. This convenience enables CSR, bitmaps, and pull over an enumerable
unvisited universe.

### Implicit graph

The state can be wide, the universe unknown, and a move expensive. Identity,
successor generation, and frontier payload may require different
representations. `O(V+E)` hides move, canonicalization, hashing, equality, and
materialization costs unless those primitives are stated.

It also measures the expanded graph rather than the compact description. Let a
state be an `n`-bit integer and the only move be

```text
x -> x+1 mod 2^n.
```

An `O(n)` increment circuit describes this directed Cayley cycle, but BFS from
zero reaches all `2^n` states and assigns distance `2^n-1` to the predecessor
of zero. The traversal is still linear in its reached vertices and generated
arcs while exponential in the state-description parameter `n`. A small move
set and cheap successor oracle remove the need to store explicit adjacency;
they do not make exhaustive reachability volume small.

The same cycle separates answer size from witness size. The scalar distance
`2^n-1` fits in `n` bits, but an explicit replayable shortest word from zero to
that state contains `2^n-1` applications of `+1`. Returning the number, one
compressed rule such as "repeat `+1`", and an arbitrary labeled path are
different output contracts; only special regularity makes that particular
word compressible. A generic succinct graph need not admit a comparably small
path description.

Nor does the example have a wide wave. Every exact frontier of the directed
cycle is a singleton, yet an exhaustive BFS must distinguish up to `2^n`
visited states. Hence peak frontier width is one while ordinary exact visited
space is exponential in `n`. Frontier width measures current parallelism and
live wave payload; reached-ball size measures persistent duplicate history.
Neither is a safe proxy for the other.

The omission rule is semantic rather than syntactic:

```text
retained frontier payload + immutable graph context
must determine the complete exact successor set.
```

In an explicit CSR graph, integer ID `i` is enough because the offsets indexed
by `i` recover its entire adjacency list. A frontier bitmap can therefore be
both a membership representation and sufficient expansion input.

For an implicit permutation state `x`, an exact dense rank `r=rank(x)` gives an
excellent visited/bitmap address. It is also sufficient frontier payload only
if `unrank(r)` or another declared reconstruction yields enough of `x` to apply
every move exactly. A one-way or prohibitively expensive rank can identify the
active vertex without making it expandable. In that case the traversal must
retain the full state or some other reconstructible expansion representation
alongside the compact identity.

The same separation applies to output. Enough data to generate successors may
still omit the parent move, source label, automaton history, or concrete lift
needed by the requested result. There is therefore no universally minimal BFS
record: sufficiency is relative to expansion, exact equality, replay, and
presentation contracts.

### Cayley/Schreier graph

One generator application supplies an implicit edge, but three objects must not
be confused:

```text
generator word occurrence
group element
concrete orbit/configuration state.
```

Relations make many words one element. Stabilizers make many elements one
configuration. The chosen generator collection and left/right action define
the directed metric and replay order.

For the common right-edge convention

```text
g --s--> g*s,
```

left multiplication by any `a` preserves every labeled edge:

```text
g --s--> g*s    becomes    a*g --s--> a*g*s.
```

Choosing `a=g^-1` translates a query from `g` to `h` into one from the identity
`e` to `g^-1*h`. Hence, with extended distance allowed in a directed graph,

```text
dist(g,h) = dist(e,g^-1*h).
```

A complete identity-rooted distance table therefore contains every pairwise
distance by relative-element lookup; a bounded table contains only the
corresponding bounded claims. The result remains valid without inverse-closed
generators, but then it is a directed positive-word distance and may be
infinite.

For a finite strongly connected Cayley graph, this also makes the maximum
identity-rooted distance the graph diameter: translating the source preserves
all distances, so every root has the same maximum. A general exhaustive BFS
only gives its root's eccentricity. On the path `1--2--3`, BFS from `2` has
last depth one although the diameter is two. Note 21 realizes this very path
as a transitive S3 action with fixed generators `(12),(23)` and endpoint
self-loops: transitivity of the state action alone is not vertex transitivity
of its fixed-generator graph. Exhaustion and a name such as `diameter()` do
not supply the missing symmetry premise.

The selected identity-rooted **tree path** between two arbitrary stored
vertices is not the same shortcut. In the undirected Cayley cycle `C_6` with
generators `+1,-1`, choose the valid BFS tree

```text
0-1-2-3       and       0-5-4.
```

The tree route from `2` to `4` is `2-1-0-5-4`, of length four, while the graph
geodesic is `2-3-4`, of length two. A BFS tree preserves exact distances from
its root, not arbitrary-pair graph distances.

The identity tree is still enough to construct one pairwise Cayley geodesic in
this setting, but by a different operation. Compute the relative element
`r=g^-1*h`, take the stored identity-to-`r` generator word, and apply the same
right-edge labels starting at `g`. For `g=2,h=4`, `r=2`; translating the stored
word `0 -> 1 -> 2` gives `2 -> 3 -> 4`. Thus

```text
tree route between stored endpoints       may be nonshortest
translated root-to-relative-element word  is shortest.
```

This requires retaining a replayable generator word (or parents plus labels),
not merely scalar identity distances.

Translation actually preserves more than one chosen geodesic. For right edges,
left multiplication by `g^-1` keeps every generator label, giving a bijection

```text
shortest labeled paths g -> h
<-> shortest labeled paths e -> g^-1*h.
```

Therefore, under one fixed path-identity convention,

```text
sigma_g(h) = sigma_e(g^-1*h),
```

and a complete identity-rooted shortest-predecessor DAG or exact count table
determines the corresponding structure/count for every pair by translation.
Inverse closure is unnecessary; reachability and direction are preserved by
the same automorphism.

What was retained still matters. A scalar distance table gives no count. One
BFS tree retains one word even when several shortest words exist. In the Klein
four example below, both `ab` and `ba` reach `c` at depth two; a first-parent
tree keeps only one, whereas a labeled predecessor DAG/count recurrence keeps
two. Likewise, merging parallel generator labels changes labeled-word counts
without changing vertex distances. “One identity BFS answers all pairs” is
therefore true separately for each output contract only if that contract was
computed at the identity in the first place.

The multiplication side is not cosmetic. Under left edges `g --s--> s*g`, the
edge-preserving symmetry is right translation and normalization instead uses
`h*g^-1` (with path-label composition interpreted consistently with that
action). Mixing these formulas silently asks for the distance to a different
group element when the group is noncommutative.

This all-pairs reduction is a Cayley translation fact, not a generic BFS fact.
An arbitrary implicit graph has no such automorphism. A Schreier graph stores
orbit/coset states rather than uniquely storing group elements, so normalization
must be derived from the declared action and stabilizer equivalence; blindly
looking up `g^-1*h` in a Cayley table can answer the cover rather than the
actual orbit graph.

Visited performs semantic convergence of words into states. It is not merely a
memory optimization.

The Klein four group makes this visible without puzzle notation. Let

```text
G = {e,a,b,c},   a^2=b^2=e,   ab=ba=c,
S = [a,b].
```

BFS from identity first has `F_0={e}` and `F_1={a,b}`. Expanding both depth-one
states emits four labeled occurrences:

```text
a*a = e          word aa returns to visited root
a*b = c          word ab reaches c
b*a = c          word ba reaches the same c
b*b = e          word bb returns to visited root
```

Thus the length-two word tree contains four occurrences, the distinct candidate
set is `{e,c}`, and subtracting `B_1={e,a,b}` leaves only `F_2={c}`. The two
words `ab` and `ba` are different labeled shortest paths but one semantic
vertex; `aa` and `bb` add work without adding a new state.

This is the Cayley funnel in miniature:

```text
generator words -> endpoint elements -> minus visited -> next frontier
       4                   2                 1
```

Counting words answers a path/occurrence question. Counting the frontier answers
a state-distance question. Exact visited is where those word histories are
merged according to group equality.

## Geometry explains work but does not choose hardware

At depth `d`, distinguish:

```text
w_d = |F_d|                       state parallelism/frontier bytes
e_d = generated occurrences      expansion work
c_d = distinct candidates        identity convergence
h_d = candidates already visited low-yield probes
|F_(d+1)|                         semantic progress
|B_d|                             persistent visited/output capacity.
```

Degree controls occurrence work, not frontier growth. Relations, cycles,
bottlenecks, expansion, and saturation control how occurrences collapse into
new states.

The frontier cardinality alone does not even determine the current layer's
edge work. Compare two directed depth-`d` frontiers, both `F_d={u,v}`:

```text
case A: u -> x,  v -> y                         e_d = 2
case B: u -> x_1,...,x_100,  v has no edge      e_d = 100
```

Both expose two state-level work items, but one contains fifty times as many
edge occurrences. Assigning one thread or one owner per frontier state can
therefore be perfectly balanced in vertex count while badly imbalanced in
transition work. `w_d` describes available state records; `e_d` describes the
inner expansion volume.

Even the complete frontier-size sequence and distance map do not determine
total BFS work. On vertices `{s,v_1,...,v_k}`, compare:

```text
star:       s is joined to every v_i
filled star: add every edge v_i--v_j between leaves
```

Rooted at `s`, both undirected graphs have exactly

```text
F_0={s}, F_1={v_1,...,v_k}, F_2={}
```

and therefore identical distance labels and semantic progress. But scanning
the stored adjacency occurrences gives `2k` for the star and `k(k+1)` for the
filled star. At depth one the star emits only `k` returns to the root; the
filled star emits those plus `k(k-1)` same-layer occurrences, all of which
produce no new state. The work ratio grows linearly with `k` while the entire
frontier profile remains unchanged.

So the wave profile records how many vertices become newly certified at each
distance. It does not record the density of edges inside or back into the
already certified ball. Predicting traversal work needs at least the layer-to-
layer edge/occurrence profile in addition to frontier cardinalities.

Adding the total edge count still does not recover when that work occurs. Use
five vertices with base edges

```text
s--a, s--b, a--x, b--y,
```

and compare one extra edge `a--b` with one extra edge `x--y`. Both graphs have
five vertices, five undirected edges, and the same rooted layers

```text
F_0={s}, F_1={a,b}, F_2={x,y}, F_3={}
```

but their per-level adjacency-occurrence vectors are

```text
extra a--b: (2,6,2)
extra x--y: (2,4,4).
```

The total scan is ten in both cases; one graph concentrates more of it at depth
one and the other at depth two. Therefore equal `V`, equal `E`, and equal
frontier sizes determine neither per-level work, scratch pressure, nor which
barrier carries the expensive phase. The missing coordinate is where edges sit
relative to the BFS layering.

A fixed generator set gives a useful special case. If every one of `q`
generators is attempted for every Cayley frontier state, then the raw labeled
attempt count is exactly `q|F_d|`. This removes raw-degree skew, but it does not
make the rest of the pipeline uniform: legality checks, generator cost,
canonicalization, endpoint aliases, visited hits, accepted states, and output
metadata can still differ. Regular algebraic degree is one conserved work
budget, not a complete GPU load model.

### A faster occurrence rate can mean slower semantic progress

Graph500 TEPS has a specific benchmark contract: input edges in the traversed
component divided by timed BFS duration. It is not necessarily the number of
adjacency entries physically inspected, and its numerator does not include
duplicate, synchronization, or communication work. An implicit Cayley run
should call its direct counter `generated transitions/s`, not Graph500 TEPS,
unless it actually adopts the Graph500 graph and normalization contract.

Even the direct occurrence rate is not a useful-progress rate by itself:

```text
run A:  100 generated transitions in 1 s, 100 accepted new states
run B: 1000 generated transitions in 1 s,   1 accepted new state.
```

Run B has ten times the transition throughput but one hundredth the semantic
discovery throughput. Its extra work may be self-loops, relations, repeated
parents, visited hits, or other collapsing occurrences. This is not necessarily
an implementation defect—the declared graph may require those checks—but it is
not evidence of faster BFS progress.

A useful measurement therefore preserves the waterfall and its numerators:

```text
frontier states
-> generated labeled occurrences
-> valid occurrences
-> distinct candidate states
-> previously unseen accepted states
-> requested output records.
```

Even identical frontier sets and identical generated-occurrence counts do not
determine where the waterfall collapses. Start with directed edges

```text
s->a, s->b
```

and compare the depth-one expansions

```text
old-return case:  a->x, a->s, b->y, b->s
new-merge case:   a->x, a->y, b->x, b->y.
```

Both cases have

```text
F_1={a,b}, generated occurrences=4, F_2={x,y}.
```

But the old-return case has three distinct candidate identities `{s,x,y}` and
two occurrence hits on visited `s`. The new-merge case has only two distinct
candidates `{x,y}`: all four proposals point outward, then cross-parent
convergence removes two copies. Equal input work and equal semantic progress
therefore conceal different rejection mechanisms.

That distinction survives into physical placement. Old-state rejection needs
an authoritative or safely replicated membership fact; cross-parent merging
needs equal new candidates to meet at some combine scope. The same scalar
duplicate ratio can consequently imply different local atomics, sort volume,
routed bytes, or owner pressure.

Report rates and bytes at the relevant stages plus end-to-end latency. A change
can improve one stage while worsening acceptance yield, communication, memory,
or closure time; one headline edge/transition rate cannot identify which.

Examples anchor intuition:

- a tree has large growth and little convergence;
- `Z^m` has exponential word choices but polynomial state growth;
- a clique has huge degree and one discovery layer;
- a cycle is vertex-transitive with constant-width frontier;
- a bounded-degree expander has logarithmic depth but a forced linear-width
  layer;
- adjacent-transposition `S_n` has exact Mahonian spheres.

The path and star make the time profile concrete. Both the `n`-vertex path
`P_n` and the `n`-vertex star `K_(1,n-1)` have `n-1` undirected edges, so a full
adjacency scan performs `2(n-1)` directed edge occurrences in either graph.
Root the path at an endpoint and the star at its center:

```text
path frontier sizes: 1,1,1,...,1        n logical levels
star frontier sizes: 1,n-1              2 logical levels
```

Total edge work therefore does not determine instantaneous parallelism or
causal depth. The path exposes very little state-level work at once and forces
a long chain of dependent levels. The star exposes almost all remaining states
in one burst, then ends. Equal `|V|`, equal `|E|`, and equal full-scan occurrence
work can coexist with opposite GPU-utilization profiles.

Geometry supplies workload. Representation turns workload into bytes and
operations. Scheduling turns those operations into locality and dependencies.
Hardware determines elapsed time. No arrow is reversible without evidence.

## The performance stack

```text
semantic graph and requested output
        ↓
frontier/visited/candidate counts and path multiplicity
        ↓
state, key, parent, wire, and scratch representations
        ↓
expansion, identity, compaction, routing, synchronization schedule
        ↓
memory traffic, atomics, locality, communication, critical path
        ↓
measured end-to-end time, capacity, and energy on named hardware.
```

Moving upward from a timing result is unsafe without controlled evidence. A
fast primitive does not prove a fast traversal; a fast traversal on CSR does
not establish performance for wide implicit states; a simulated byte count does
not establish interconnect time.

## Single-GPU mental model

A GPU supplies parallel work over frontier states, edges/moves, candidate
records, bitmap words, hash probes, or compaction elements. The exact next layer
still requires:

```text
complete expansion
exact identity/visited decision
lossless commitment
level/target completion proof.
```

Narrow frontiers underfill hardware; wide frontiers increase memory pressure.
Duplicates may be close within a warp or scattered globally. Frontier order is
semantically optional for distance sets but physically relevant to locality.

REF-014 makes the distinction concrete on one exact bitmap kernel and one
RTX 3070 Laptop GPU. At `2^24` candidates, putting every occurrence on one
co-located key let warp equal-key aggregation reduce median isolated kernel
time from `10.154 ms` to `0.373 ms`. A different batch still had four
occurrences per key globally, but permuted equal keys apart; aggregation then
changed `0.606 ms` to `0.622 ms`. The BFS-level intuition is not "many
duplicates help aggregation," but "duplicates help only the mechanism whose
physical visibility scope actually contains them." These are retained
synthetic kernel measurements, not an end-to-end traversal or universal GPU
threshold.

The work-span lower bound

```text
T_P >= max(W/P, S)
```

explains why more throughput resources cannot remove the causal depth and
per-level reductions. A persistent kernel can hide launches without erasing
logical dependencies.

REF-017 exposes this distinction in one complete exact `S_9` Cayley traversal.
All four measured layouts/bitmap variants expanded the same 37 levels and
`2,903,040` transitions, but median host-observed traversal time was
`3.462--4.361 ms` while the corresponding sum of fused-kernel intervals was
only `0.515--0.606 ms`: roughly a `6.7--7.3x` gap. The host interval included
per-level event synchronization, count/overflow copies, launches, and FFI
overhead, per-step counter resets, and Rust count checks. It excluded setup and
full oracle copies. The gap does not identify which individual non-kernel cost
dominates; that requires evidence beyond these two timers. Thus a faster level kernel
can have limited traversal impact when the wave has many short levels. This is
a small `S_9` result with unlocked laptop clocks, not evidence that a
persistent device loop is automatically correct or faster; termination and
the next launch size still depend on the newly closed frontier.

There is also a distinction between proving exhaustion and implementing an
unknown-depth stopping rule. REF-017's Rust driver executes a predetermined
number of steps from the Mahonian oracle and checks each next-layer count,
including the final zero. It validates the complete S9 traversal, not a
generic stop-on-empty driver. For a hand example `s -> a`, the two nonempty
frontiers are `F_0={s}` and `F_1={a}`: expanding the latter produces `F_2={}`.
Knowing beforehand to perform two expansions and discovering when to stop
can yield the same trace, but they are different control contracts.

## Multi-GPU mental model

A robust conceptual pipeline is

```text
source rank expands owned frontier
-> optional source-local exact pre-dedup
-> candidates route by stable owner(state)
-> one owner performs authoritative exact visited claim
-> owner stores accepted next frontier/output
-> global completion proves the level/target bound.
```

The union across owners is the semantic frontier. Local shards are physical.

A two-owner trace shows why local emptiness is not global completion. Let the
only edge be `a -> b`, with owner 0 responsible for `a` and owner 1 for `b`:

```text
time  owner 0                    transport             owner 1
t0    frontier {a}                                    frontier {}
t1    expand a; local {}         message(b) in flight frontier {}
t2    local {}                   delivered            frontier {b}
```

At `t1`, both visible local frontiers are empty, yet the logical next frontier
is not empty: responsibility for `b` exists in transport. Declaring completion
there would lose a reachable state.

Distributed termination therefore needs one consistent global cut proving that
all owners are passive **and** no accepted, staged, in-flight, retryable, or
publication work can create another frontier record. Adding local empty counts
observed at unrelated times does not establish that cut.

Stable ownership solves a different problem: global identity convergence. Let
depth-`d` parents `p` and `q` live on different GPUs and both generate the same
child:

```text
GPU 0: p -> x
GPU 1: q -> x
```

If each GPU treats only its local visited table as final authority, both can
accept `x` as new. The physical next frontier then contains two records for one
semantic vertex, and both may expand the same outgoing edges at depth `d+1`.

Routing both candidates by a stable function of the exact identity of `x`
makes them meet at one authority. That owner commits `x` once for frontier
membership and distance. It must not blindly discard the losing occurrence when
the output asks for all shortest parents, labeled paths, or path-count
contributions: one state commitment and many valid incoming contributions are
different contracts.

More GPUs introduce:

- ownership skew and unavoidable idle ranks on small frontiers;
- record traffic and message fragmentation;
- duplicate convergence moving from sources to owners;
- topology and slowest-rank critical paths;
- ownership epochs for restart/repartition;
- global termination with work in flight.

A stale exact visited replica can safely have missing updates: positive is sound,
negative is unknown. A Bloom positive cannot delete an exact candidate. Advisory
caches never replace authoritative novelty or termination.

### Three multi-GPU scaling claims that must not be merged

**Strong scaling** fixes one semantic BFS workload: same graph/action version,
source set, direction, stopping rule, exactness, and output. If `T_p` is its
end-to-end time on `p` GPUs, then

```text
speedup S_p = T_1/T_p,       efficiency E_p = S_p/p.
```

The semantic frontier, accepted-state, and logical-transition profiles should
match across GPU counts; routing, retries, synchronization, and physical bytes
may change and must be separated. If the exact workload does not fit on one
GPU, `T_1` does not exist and a one-to-`p` speedup cannot be claimed. Narrow
levels, the slowest owner, communication, and per-level closure bound the fixed
work even when middle frontiers scale well.

For a level-synchronous run, let `W_(d,i)` be expansion/owner work on GPU `i`
at depth `d`. Even with equal per-record cost and no communication, the layer
cannot finish before its busiest owner:

```text
T_d >= max_i W_(d,i) / rate_i.
```

Two GPUs with work `[50,50]` and `[100,0]` have the same total `100` and the
same average `50`, but the second layer has twice the ideal critical-path time
on equal devices. Its idle GPU cannot advance to depth `d+1` while the exact
barrier still waits for unfinished depth-`d` proposals.

The exact `S_8` owner simulations in REF-005 make the transient part visible.
With eight owners, direct Lehmer-rank modulo finished with perfectly balanced
total visited capacity (`max/mean=1`) yet reached `2.114943` frontier
`max/mean` on a large level. SplitMix-style ownership reduced that recorded
large-level maximum to `1.287356`, but raised the remote fraction from
`0.635901` to `0.875729` and moved more duplicate convergence to owners.
REF-006 broadened the same trade-off: among 35 deterministic mappings, 20 were
non-dominated across frontier/receive imbalance, remote fraction, cross-rank
duplicates, and final capacity. These are ideal exact-set simulation counts,
not GPU or network timings; they show why neither final balance nor one cut
metric determines the slowest-rank critical path.

Real `T_d` is additionally constrained by routing, receive processing,
deduplication, publication, and collectives that remain on the critical path.
Summing across layers makes temporal imbalance visible:

```text
T_total is bounded by the per-level maxima, not by one whole-run average.
```

Balancing frontier vertex counts is also weaker than balancing work: degrees,
generator costs, duplicate destinations, owner receives, and output payloads
can differ. Useful scaling evidence therefore retains per-level sum and maximum
owner work, bytes, and idle time instead of reporting only global TEPS or mean
utilization.

**Weak scaling** grows the workload with `p` under a declared rule and asks
whether time or rate stays controlled. For BFS, “fixed work per GPU” cannot be
replaced casually by “fixed vertices per GPU”: diameter, reachable fraction,
frontier widths, degrees/move costs, duplicate convergence, partition cuts, and
output volume may all change. The growth rule must name what is held roughly
constant—logical occurrences, reachable states, bytes, or independent queries—
and report the resulting per-level semantic profile.

**Capacity scaling** asks whether aggregate memory makes a larger exact
traversal feasible. Reaching a deeper ball, retaining a larger visited set, or
avoiding overflow is success even if runtime increases. Nominal summed VRAM is
not usable capacity: replicated data, imbalance, candidate/sort scratch,
communication buffers, allocator headroom, and the maximum-loaded owner all
reduce it.

Thus these observations do not imply one another:

```text
larger exact problem fits       != fixed problem became faster
time stayed flat as work grew   != one fixed query sped up linearly
more GPUs reached a deeper BFS  != a valid one-to-many speedup ratio.
```

Independent-query throughput is a fourth useful regime: queries per second can
rise by running several waves concurrently even when latency of one wave does
not improve. Combining those sources into one multi-source BFS would change the
semantic problem rather than merely batch it.

## Certificates: upper bounds, lower bounds, and closure

Many search guarantees become clearer as bound closure:

| Evidence | Direction |
|---|---|
| replayable path/suffix | upper bound on distance |
| completed BFS layers below `d` | no solution at smaller hop depth |
| exact radius-table miss | lower bound beyond radius |
| PDB abstract distance | admissible concrete lower bound |
| `g+h` open record | lower bound for solutions through record |
| incumbent concrete path `U` | global upper bound candidate |
| empty complete frontier | component closure/unreachability |
| parent chain only | upper witness, not minimality proof |
| complete edge label inequalities | complementary lower certificate |

A parent-only counterexample is

```text
s -> v
s -> a -> v.
```

Reporting `L(s)=0`, `L(a)=1`, `L(v)=2` with parents `s->a->v` gives a real path
of the recorded length and every parent edge decreases the label by one. Yet
the direct edge proves the true distance to `v` is one. Replay establishes

```text
true_dist(v) <= L(v),
```

not the reverse inequality required for minimality.

For a complete single-source directed result, a compact local certificate has
three parts:

```text
1. L(s)=0 and no other vertex has label zero;
2. every finite v!=s has a real predecessor u->v with L(u)=L(v)-1;
3. every edge u->v from finite u satisfies L(v)<=L(u)+1.
```

Condition 2 chains to a real length-`L(v)` path, giving the upper witness.
Condition 3 propagates along every possible source path and gives
`L(v)<=true_dist(v)`; it also forbids a finite-labeled vertex from pointing to
an infinity-labeled reachable omission. Together they prove equality and
reachable-set closure. Neither half alone is sufficient, and checking only the
selected parent edges does not inspect the shortcut that falsifies the example.

For an implicit graph, “scan every edge” means the validator must have an
independently trusted complete successor relation for the same graph epoch. A
validator that calls the same buggy move generator as the search can reproduce
the same omitted edge and provide correlated, not independent, evidence.

### Matching counters and fingerprints are tripwires, not set proofs

Suppose the expected frontier is `{0,3}` and the produced frontier is `{1,2}`.
Both have

```text
cardinality = 2
sum         = 3
xor         = 3.
```

One missing identity and one spurious identity can therefore cancel in several
aggregate checks at once. Per-level and per-owner counters localize many losses,
duplicates, retries, or overflows, but balanced errors can still survive their
reductions.

Finite commutative fingerprints add strong regression sensitivity, not
deterministic arbitrary-set equality. Unless the encoding is proved injective
over the entire declared set domain, some unequal sets collide. More hash bits
can make accidental collision negligible for an experiment without changing
that logical status.

Useful evidence forms a ladder:

```text
counts/conservation identities
< per-level and per-owner accounting
< independent fingerprints
< replay plus complete local edge inequalities
< exact canonical frontier-set comparison
< exhaustive tiny-domain independent successor oracle.
```

A mismatch at a lower rung decisively falsifies some assumption. A match says
only that this rung did not expose the bug. Exact bounded comparison requires
full-state collision resolution—such as canonical sorted states or an injective
domain bitmap—and should avoid sharing the same move table/encoder omission as
the implementation under test. One-, two-, and many-GPU parity can reveal
routing bugs while all configurations still share one semantic common-mode bug.

BFS proves shortestness by making upper and lower bounds meet at a distance
layer. A* closes `min_open(g+h) >= U`. Bidirectional BFS closes a bound between
two completed regions. The schedules differ; the proof pattern is shared.

## Information that cannot be silently discarded

For frontier membership, equal child records may collapse to one state. They
may still carry nonredundant:

- alternative shortest parents;
- move labels and generator-word identity;
- shortest-path count contributions;
- source labels/Voronoi ties;
- history/automaton context;
- deterministic tie keys.

A representation is not "more compact" if it has changed the requested output.
Bitmaps retain membership, not order, parent multiplicity, or expandable state
unless those are separately recoverable.

Even “one parent per state” contains two possible contracts. A post-layer
reduction can select a deterministic shortest-valid parent for every vertex yet
produce a tree that no serial FIFO first-discovery history can realize: parent
choices sharing competing predecessors impose coupled layer-order constraints.
Thus shortest-path-tree validity, deterministic reduction, and first-in
BFS-tree realizability are separate properties. Parallel execution commonly
wants the first two without promising the third. This obstruction already
occurs in the highly symmetric all-transposition Cayley graph of `S_3`: its
three depth-one transpositions all precede both depth-two 3-cycles, so one FIFO
run must assign both cycles to the same first expanded parent even though a
child-dependent post-layer reduction may choose different shortest parents.
A common fixed parent-ID minimum cannot do so in this fixture, because both
children have the same candidate-parent set.

More generally, in a right Cayley graph the raw common-successor count of
parents `u,v` is the generator autocorrelation

```text
|uS intersection vS| = |S intersection (u^-1 v)S|.
```

Its next-layer restriction counts shared shortest children. Two such children
already permit contradictory first-in parent choices. The same incidence also
drives duplicate proposals, but pair intersections are not rejected-occurrence
counts: a child with `k` shortest parents contributes `C(k,2)` parent pairs and
only `k-1` excess occurrences.

## False equalities to resist

```text
queue                    != BFS semantics
generated transitions    != unique candidates
unique candidates         != new frontier states
hash/fingerprint          != exact identity
dense rank                != cheap expandable state
inverse moves             != enumerable pull universe
one parent                != shortest-path DAG
correct path replay       != shortestness proof
empty local queue         != global exhaustion
level-shaped top-k        != exact BFS
admissible ranking        != fixed-width pruning proof
more GPUs                 != proportional latency reduction
bounded miss              != unreachable
minimum hops              != minimum heterogeneous cost
union of source frontiers != joint multi-source frontier
one vertex commitment     != one shortest-path contribution
equal total work          != equal parallelism or time profile.
```

Most subtle BFS bugs are one of these substitutions hidden behind a familiar
name.

## A practical reading checklist

When reading a paper, code path, or run artifact, ask in this order:

1. What are the semantic vertices and exact equality?
2. What directed/labeled/costed edges exist?
3. What source, target, quotient, and graph version are declared?
4. What output is promised?
5. What makes a label final?
6. Are all needed successors and frontier states retained?
7. What negative/completion certificate is available?
8. Which structures are authoritative versus advisory?
9. What happens on collision, overflow, timeout, retry, and restart?
10. Which counts describe occurrences, distinct states, accepted states, and
    output records?
11. Which bytes and critical-path stages correspond to those counts?
12. What evidence proves semantics before performance is compared?

If the first nine answers are unclear, throughput is premature.

## Boundaries exposed by later variations

Several later questions do not replace the core model; they reveal which of
its boundaries stop coinciding.

**Traversal order and live memory are different widths.** Metric layer width,
FIFO queue peak, processed/unprocessed vertex separation, edge cut, and
semantic output liveness can disagree even on the same graph. A BFS order fixes
the nondecreasing-distance constraint but often leaves within-layer order free.
Pathwidth describes the best vertex separation over all linear orders, not the
memory of a particular BFS schedule, and neither vertex count is automatically
a byte lower bound. On multiple owners, temporal liveness and local/remote
ownership form independent cuts.

**Finite branching is part of ordinary BFS finality.** With an effectively
countable but infinite successor stream, a finite-depth target may be hidden
behind an unfinished shallower layer. Fair dovetailing can eventually expose
every finite witness, but first discovery need not be shortest and pointwise
convergence of tentative distances need not have a detectable finalization
time. Exact visited merges identities; it cannot make an infinite layer or a
nonterminating redundant successor presentation finite.

There is a stronger cardinal boundary beyond countable branching. Metric layers
are defined by finite paths on any graph, so even an uncountable reachable set
still appears in the stage-`omega` union of finite-depth balls. But one finite
layer can itself be uncountable. No sequential or countably parallel explicit
record stream can enumerate it; fair dovetailing applies only to countable
choices. Exact work then needs a symbolic frontier descriptor with exact image,
union, difference, emptiness, and membership operations. Ordinal depth and
cardinal width are independent.

Symbolic image computation can still be exact BFS. With predicates for
frontier `F_d`, reached ball `R_d`, and transition relation `T`, existential
relational image followed by `not R_d` implements the same next-layer set
difference. Iterating only an accumulated reachable predicate also produces the
balls, but retaining just its final fixed point loses first-entry depth. A
saturation schedule may traverse several semantic edges per implementation
round, so its round number is not automatically BFS distance. Classic Boolean
decision diagrams represent huge finite valuation sets; “symbolic” alone does
not solve uncountable domains.

**Equivalent-looking action graphs need an explicit transport map.** Equal
generator orders and shallow sphere counts are fingerprints, not proof that two
puzzle implementations use the same labeled action. Transferring distances
needs an orbit graph isomorphism; transferring move words needs one simultaneous
position conjugacy and one signed-label map for every generator. Canonical-word
transfer additionally depends on label order. This is why CayleyPy/DeepCubeA
agreement remains conditional until their full action conventions are matched.

**Dynamic BFS separates label survival from repair.** After deletions, reach in
the surviving old shortest-path DAG says exactly which old scalar labels remain
valid, but vertices outside that region may require longer paths through the
full graph. For one inserted directed edge, every new distance is the minimum
of the old distance and an old prefix plus the new edge plus an old suffix;
equal candidates can change DAG/count outputs without changing scalar labels.
A batch requires min-plus closure over paths that may alternate old metric
segments with several inserted edges, so independent one-edge formulas are not
enough.

**Distributed layouts factor the same semantic wave differently.** A 2D
top-down layout separates frontier expansion from candidate folding; correctness
still requires complete delivery to adjacency shards and authoritative merging
at destination owners. Distributed bottom-up search moves candidate
responsibility through predecessor shards so one-parent early exit requires an
exact completed-state protocol and a closed snapshot. These schedules can move
or reduce communication in a declared regime, but they do not weaken the BFS
closure obligation. For implicit Cayley graphs, a matrix checkerboard or full
unvisited pull universe exists only after separately proving a suitable state,
generator, ranking, and ownership representation.

The historical Moore/Lee wavefront view reinforces the common thread: the
search labels first arrival by a closed wave, while an optional backtrace
chooses a witness afterward. The FIFO queue, a dynamic repair wave, and a
distributed expand/fold schedule are mechanisms for maintaining or restoring
that boundary, not alternative definitions of shortest distance.

## Applying the map to inspected CayleyPy

The retained Python library also contains an ordinary `BfsAlgorithm`, separate
from the production beam pipeline discussed below. Its source-level recurrence
in `external/cayleypy-installed-source/cayleypy/algo/bfs_algo.py` is a bulk
layer BFS:

```text
current layer
-> graph.get_neighbors
-> graph.get_unique_states
-> remove hashes seen in retained old layers
-> next layer.
```

Batching changes where candidates first combine: every batch is uniqued, tested
against old layers, and tested against already accepted earlier batches before
the batches are stacked. It does not intentionally prune by score or width.
When the declared generators are inverse-closed, `seen_states_hashes` is
truncated to the last two layers. At the next expansion those are exactly the
previous and current layers, so the code is a direct implementation of the
undirected rolling-window theorem rather than permanent visited storage.

The exactness boundary lies in identity. If an encoded state is one `int64`,
`StateHasher` uses that value as an identity hash. For wider encoded states it
produces one 64-bit hash, and `CayleyGraph.get_unique_states` retains the first
state for each distinct hash without comparing colliding full states. Old-layer
subtraction also tests only those hashes. Therefore the layer recurrence is
mathematically exact only under an additional injectivity/no-semantic-collision
assumption for the reached domain; the inspected wider-state path is not a
collision-resolving exact visited implementation by itself.

Completion is nevertheless kept distinct from bounded stopping. The result's
`bfs_completed` flag becomes true only when a fully constructed next layer is
empty. Diameter, layer-size, and callback stops return the computed prefix with
`bfs_completed=False`. Stored layer contents are also an output choice:
large middle layers may be represented only by their sizes unless explicitly
requested, while the traversal still uses their encoded states internally.

The specialized `bfs_numpy` makes another theorem visible. It requires an
inverse-closed permutation generator collection and one-int64 exact encoding.
Instead of storing one undifferentiated frontier, it assigns every frontier
state to one generator bucket. If several generators produced the same state,
`_make_states_unique` leaves it in one bucket; that bucket is a selected
incoming label, not every possible parent.

When applying outgoing generator `i1`, the code skips the bucket whose selected
incoming generator is `inverse(i1)`. For a state reached as

```text
parent --i2--> state,
```

the omitted `inverse(i2)` transition returns exactly to that selected parent in
the previous layer. Removing it cannot remove a new child. But the code still
subtracts every bucket of both `layer0` and `layer1`, and it still uniques all
new generator buckets. Other previous-layer parents, same-layer neighbors, and
multiple proposals for a new state remain possible.

So this is not “last move replaces visited.” It is:

```text
one retained parent label removes one guaranteed backtrack;
rolling exact layer membership handles every other old-state occurrence;
next-layer uniquing handles convergence.
```

The retained tests compare resulting growth sequences for several declared
permutation/Coset fixtures, including a bounded `lrx(16)` prefix. They are
regression evidence for those layer-size outputs, not a proof of canonical
parents or of a general history-constrained search: `bfs_numpy` returns only
layer sizes, and its generator bucket choice is an internal tie outcome.

CayleyPy also demonstrates that one shortest path does not require parent
pointers when every exact layer membership remains available. With
`return_all_hashes=True`, `restore_path` starts from the concrete target,
generates its predecessors through the inverse-generator graph, and selects the
first predecessor whose hash belongs to the preceding BFS layer. Repeating this
step walks from depth `D` through `D-1,...,0`; reversing the collected labels
gives a length-`D` path.

Under complete layers and exact identity, every selected transition is real and
every membership step decreases exact distance by one, so the result is a
shortest path even though the traversal stored no selected parent. The trade is
different storage and query work:

```text
parent tree:       one predecessor choice stored at discovery time
layer backtracking: all layer membership retained, predecessors regenerated
                    during every reconstruction query.
```

This also explains why traversal-safe rolling reclamation and later path
reconstruction are separate output contracts. Keeping only the previous/current
layers is enough for future scalar novelty, but not for this backward walk;
`return_all_hashes` deliberately retains every layer hash.

The same hash boundary remains. A collision can make a nonmember look like a
valid preceding-layer state or make the target appear at the wrong layer.
`restore_path` chooses the first hash match and does not itself assert that the
final reconstructed start state equals the declared central state. The retained
path tests call `validate_path`, which supplies positive replay evidence for
their fixtures, but does not make the generic hash-only reconstruction
collision-resolving. Generator order also selects among tied predecessors, so
the path is arbitrary shortest under the premises, not declared canonical.

The inspected CayleyPy production outer loop is a layered learned-score beam
over wide implicit puzzle states. It generates moves from retained parents,
checks goals, deduplicates by bare `Hash128`, and retains a bounded global beam.
This is not exact complete-frontier BFS.

Its K1 component intends a bounded concrete reverse-BFS goal neighborhood with
suffix witnesses; K2 enumerates short residual words. Those components have
useful local upper/lower-bound interpretations under exact identity and complete
construction. With an exact complete radius-`R` K1 ball, shortest suffixes,
and all K2 words through `K` in nondecreasing length including the empty word,
the first hit returns the candidate's exact residual distance; it exists iff
that distance is at most `R+K` (note 40). These conditional local guarantees do
not restore prefixes removed by the outer beam.

The map therefore labels the system without devaluing it:

```text
outer search: heuristic beam/hybrid
K1: conditional exact bounded reverse neighborhood
K2: bounded word enumeration/first-hit residual search
final replay: positive concrete path validity
global shortestness/completeness: not supplied by those facts alone.
```

This is the kind of separation needed before any single- or multi-GPU
performance comparison.

The retained `CayleyGraph` source also fixes an occurrence-layout contract.
`get_neighbors` emits generator-major blocks: one generator over all input
states, then the next. Frontier batching changes the global order to
batch-major/generator-major and deduplicates earlier batches before later ones.
For a non-identity hasher, accepted state batches remain vertically stacked
while their hashes are globally sorted, so state and hash rows are not promised
to align unless batching is disabled. The scalar traversal recomputes hashes
where needed and can still preserve the frontier set; hooks and first-winner
interpretations have a stronger ordering contract.

Generator definitions also permit duplicate permutation entries. Expansion
preserves both occurrences, scalar dedup collapses their endpoint, and the
permutation inverse map collapses equal transformations to the last dictionary
index. Thus transformation-valid inverse replay does not preserve duplicate
label identity. This is separate from stabilizer aliases and hash collisions.

The retained `BfsDistributed` path applies the same scalar layer recurrence
through hash ownership. Producers locally unique batches, route records by
`hash mod workers`, and owners perform the authoritative union against prior
layers and earlier accepted batches. Under torchrun, a global maximum batch
count makes every rank participate in the same ordered collective rounds; a
global next-layer sum is evaluated only after those rounds. This is an ordinary
level barrier, not asynchronous label correction.

Distributed execution does not strengthen identity: unequal states sharing a
hash meet at one owner and collapse there. It also narrows output. The
distributed result has no edge list; in torchrun dispatch, `return_all_edges`
and `disable_batching` are removed rather than honored. Full layers are gathered
to every rank only for storage, returned hashes, or stop hooks, and the hook runs
on every rank before a global-any stop decision.

`BfsResult` adds further export-specific semantics. Its `diameter()` is always
the last returned depth; it becomes a graph diameter only under additional
completion/root/symmetry premises. On an incomplete `return_all_edges` run, the
single-device BFS appends the reverse of the last generated edge block to make
the boundary symmetric. That is justified as support completion for an
inverse-closed graph, but can fabricate arcs in a directed non-inverse-closed
export. NetworkX conversion then collapses endpoint multiplicity and chooses
the first generator that replays each support edge, rather than retaining the
original generator occurrence.

Partial-order reduction exposes another graph-contract boundary. Commuting
actions justify swapping valid adjacent occurrences, but do not by themselves
justify deleting successors. A stubborn-set-style proof can retain, for every
goal path, a permutation of the same unit actions; this preserves the length of
one shortest goal path because permutation preserves length and edge deletion
cannot make a shorter one. That goal-optimality theorem is weaker than
preserving the complete BFS metric: discarded linearizations can contain
different intermediate states, other shortest words, different canonical
words, and different frontier/duplicate geometry. Exact all-state BFS needs an
all-target preservation theorem, not merely goal reachability or planning
optimality.

The quantifier matters: if equal-length, same-endpoint representatives of
every finite source path are retained, then all source distances and every
distinct-state BFS frontier are preserved. Apply the premise to a shortest
path to each vertex and combine it with the fact that edge deletion cannot
shorten paths. Even the singleton traces `a` and `b` must then be represented.
The loss of intermediate states above applies to goal-only coverage, not that
stronger contract. Shortest-word counts and generated occurrence work may
still change even when every frontier cardinality is unchanged (corrected
note 198 and SEM-2053).

## Where understanding is strong and weak

Current evidence is strong on:

- metric-ball and least-fixed-point semantics;
- exact identity/visited failure directions;
- output and stopping distinctions;
- Cayley/Schreier/action conventions;
- bounded reverse lookup and beam boundaries;
- conceptual GPU work/representation/ownership accounting;
- small exact reference and retained synthetic/local experiments.

Current evidence remains weak or intentionally absent on:

- application-scale exact implicit BFS traces;
- real multi-GPU topology/timeline scaling;
- injective ranks for the intended full puzzle domains;
- independent validation of implicit successor completeness at large scale;
- cross-family prediction of relation/duplicate/frontier geometry;
- current CayleyPy K1/K2 full-path integration under forced collisions and
  complete retained production fixtures.

These are research gaps, not an instruction to build a production system.

## Source map

This synthesis is derived from the detailed proofs and evidence in:

- notes 01, 03-06, 25, and 37 for the semantic/contract core;
- notes 08-13, 18-24, 41-43, and 48-50 for stopping, bounds, outputs, variants,
  and certificates;
- notes 10, 16-17, 27, 32-35, 39-40, 46, and 53 for Cayley geometry, actions,
  relations, abstractions, and path multiplicity;
- notes 07, 14-15, 26, 28-30, 36, 44-47, and 51-52 for representation,
  hardware, distributed authority, and failure semantics;
- notes 57, 60, 73-74, 158, 161, 165-166, and 173-181 for output finality,
  relation duplicates, queue records, Schreier support, directed back depth,
  scaling coordinates, distributed obligations, communication, and safe
  forgetting;
- notes 182-190 for traversal/live-boundary width, infinite-branching
  finality, labeled-action conjugacy, the Moore/Lee wavefront view, dynamic
  insertion/deletion closure, and distributed 1D/2D top-down and systolic
  bottom-up semantics;
- note 191 for the distinction between a shortest-path parent tree and a tree
  realizable by one serial first-in FIFO BFS history;
- note 192 for Cayley generator autocorrelation as common-successor geometry,
  its next-layer parent coupling, and the pair-count versus duplicate-count
  boundary;
- note 193 for the inspected CayleyPy generator-major neighbor layout, batching
  order/alignment boundary, duplicate generator occurrences, and inverse-label
  collapse;
- note 194 for the inspected CayleyPy distributed BFS owner routing, global
  layer closure, rolling seen state, gathering, and backend-specific output
  limits;
- note 195 for CayleyPy `BfsResult` last-depth/diameter semantics, incomplete
  edge symmetrization, retained-hash checks, simple support export, and label
  loss;
- note 196 for the boundary between abstract BFS on uncountable layers,
  stage-`omega` reachability, countable explicit enumeration, and symbolic set
  execution;
- note 197 for symbolic image/visited recurrences, layer retention, fixed-point
  distance loss, saturation schedules, BDD representation work, and witness
  extraction;
- note 198 for the distinction between commuting-path equivalence,
  action-preserving goal optimality, full BFS-metric preservation, and loss of
  intermediate states and shortest-word multiplicity under partial-order
  reduction;
- REF-001 through REF-044 for bounded validation/measurement evidence, each only
  within its recorded scope; REF-045 remains a preserved `not run`
  infrastructure outcome; REF-046 later completed as a bounded Rust
  discovery/publication interleaving model after Docker returned naturally.
  Its five passing integration tests distinguish blind drop from helpable/logged
  publication in six local and eighteen three-edge single-stop schedules, but
  do not validate a runtime memory model, GPU protocol, or multi-GPU recovery.

## Current synthesis

BFS is a distance proof carried by an evolving exact boundary. The graph and
identity determine what the states mean; complete expansion and visited define
the next metric layer; the schedule determines when claims become final; the
output determines which duplicate information is disposable; completion and
failure semantics determine what negative statements are justified.

GPU and multi-GPU execution do not change that core. They change where the
obligations are paid—in move work, bytes, atomics, sorting, routing, skew,
capacity, and synchronization. Cayley structure does not change BFS either; it
makes the distinction between words, elements, states, generators, and
relations impossible to ignore.

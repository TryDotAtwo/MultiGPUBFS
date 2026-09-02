# What BFS complexity means: input, generated work, output, and memory

The phrase "BFS is `O(V+E)`" is correct under a particular representation and
output contract.  It is not a complete cost model for every algorithm called
BFS, and it does not transfer unchanged to an implicit Cayley state space.

This note separates:

- input size and adjacency inspections;
- logical transition generation;
- unique-state discovery;
- parallel work and dependency depth;
- output size;
- peak live memory;
- distributed records and bytes;
- benchmark normalization.

No implementation or optimization is proposed.

## Explicit adjacency-list traversal

Let `R` be the component reachable from the source in a fixed finite directed
graph.  Let `A_R` be the number of stored outgoing adjacency entries of vertices
in `R`.

A standard exact graph BFS with a constant-time visited operation performs

```text
Theta(|R| + A_R)
```

work:

- each reached vertex is accepted/enqueued once;
- each reached vertex's adjacency list is opened once;
- every stored outgoing adjacency entry in those lists is inspected once.

For an undirected CSR that stores both orientations, `A_R` is approximately
twice the number of non-loop logical edges in the component.  Parallel edges
and loops are separate adjacency entries even when they add no new vertex.

The familiar notation `O(|V|+|E|)` commonly uses the whole input graph as a
worst-case bound. The traversal loop pays for its reached component, but
per-query initialization is a separate cost. Clearing a dense visited or
distance array over all vertices takes `Theta(|V|)` even if most vertices are
unreachable. Thus a run with that initialization costs
`Theta(|V| + A_R)` overall, while its traversal phase costs
`Theta(|R| + A_R)`. A sparse/on-demand record model can avoid touching the
unreachable vertices during query preparation; that is an additional
representation assumption, not a consequence of BFS itself.

For a hand example, take `N` isolated vertices and one source. The traversal
accepts one vertex and scans no adjacency entries: `|R|=1`, `A_R=0`. Preparing
an `N`-entry visited array still writes `N` entries. Constant traversal work
and linear whole-query work are consistent because their boundaries differ.
Explicitly returning a distance entry for every vertex, including unreachable
ones, likewise requires `N` outputs.

### Why this is also a worst-case lower bound for full exhaustion

If the requested output certifies the entire reachable component, an algorithm
must in the worst case inspect every outgoing entry of every reached vertex.
An uninspected entry could lead to one additional unseen vertex.  It must also
represent or emit every reached vertex required by the output.

Thus `Theta(|R|+A_R)` is not merely an artifact of a simple queue
implementation for this explicit full-exhaustion model.  The statement is
qualified by the access model and output: a target search may stop early, a
compressed/symbolic graph can expose many edges at once, and prior indexes may
move work into preprocessing.

## Adjacency matrices use another input size

If adjacency is supplied as a dense `|V| x |V|` matrix and a row is scanned
entry by entry, BFS can require `Theta(|V|^2)` inspections even when few entries
are true.  Word-parallel bit operations change the machine-operation count but
do not make a sparse matrix representation appear automatically.

This illustrates a general rule:

```text
algorithm complexity is relative to how the graph is presented.
```

CSR, dense matrix, compressed relation, successor oracle, and generator action
have different primitive operations and input sizes.

## Implicit graphs have no pre-existing `E` bill

An implicit graph exposes something like

```text
successors(state)
```

rather than stored adjacency. Let `C` count generated legal transition
occurrences from a complete frontier, `U` be their distinct endpoint set, and
`B_d` be the visited ball including that frontier. A disjoint work decomposition
is

```text
C = (C - |U|) + |U intersect B_d| + |U minus B_d|
  = within-batch duplicate occurrences
  + unique endpoints already visited
  + accepted unique new states.
```

The old-endpoint term counts unique endpoints, not all occurrences reaching
old states. Otherwise duplicates of an old state are counted twice: two
occurrences ending at an already visited `s` must give `2=1+1+0`.
Invalid attempts for partial moves are a separate count, not part of `C`.

Wall work also includes:

```text
move application
legality checking
state normalization/canonicalization
rank or hash computation
exact equality and collision resolution
candidate materialization/deduplication
parent metadata and replay data.
```

Two implicit graphs with equal reached-state and logical-edge counts can have
very different costs when one move is a table lookup and another transforms a
wide structured state.  Writing `O(V+E)` hides that difference by treating an
edge-generation oracle call as constant without stating it.

A more honest parameterized expression is

```text
sum over expanded states x
    sum over attempted moves m  cost_apply_and_validate(x,m)
+ cost_exact_identity(all candidates)
+ cost_output_and_control.
```

It need not be evaluated symbolically to be useful: it identifies which events
must be counted before a throughput claim can transfer between domains.

## Cayley occurrence work is especially explicit

For an ordered collection of `q` total generator occurrences, every fully
expanded state attempts `q` labeled moves.  A complete finite-component
traversal therefore generates

```text
q * |R|
```

occurrences if every generator is total and the final frontier is expanded to
prove exhaustion.

This number is not the count of distinct simple edges:

- inverse labels may describe opposite orientations of one undirected edge;
- duplicate generators repeat an occurrence;
- identity generators add loops;
- relations make different words converge;
- a Schreier action can make distinct generators coincide at a state;
- partial/illegal moves reduce or separately charge attempted transitions.

The raw word tree can grow exponentially while state spheres grow only
polynomially. Do not assign that word-tree growth to ordinary graph-BFS
generation work: with fixed q and one expansion per accepted state, layer d
generates exactly `q*|F_d|` occurrences, and expanding all of B_d generates
`q*|B_d|`. Thus polynomial ball growth also gives polynomial occurrence work
under these assumptions. Notes 10 and 27 explain the geometric distinction.

For a hand example, use integers with moves +1 and -1, starting at zero.
There are `2^d` raw words of length d, but `F_d={-d,d}` for d>=1 and
`|B_d|=2d+1`. Each noninitial BFS layer expansion attempts only four moves.
Constructing the complete ball B_d for d>=1 expands B_(d-1), costing
`2*(2d-1)=4d-2` move applications. Expanding B_d as well costs `4d+2` and
constructs the next layer; it does not prove exhaustion of this infinite graph.
For d=2, six move applications build the five-state ball, whereas the four
length-two words end at -2, 0, 0, 2 rather than four new distance-two states.

## `O(b^d)` is a tree-search model, not a graph identity

For branching bound `b` and shallowest target depth `d`, the size of the
unmerged path tree through depth `d` is bounded by

```text
1 + b + ... + b^d.
```

This describes worst-case tree search or a graph with no convergence in that
radius.  It is not an equality for graph BFS:

- visited merges paths reaching one state;
- relations can create polynomial sphere growth;
- bottlenecks can shrink then regrow frontiers;
- a high-degree clique saturates after one step;
- generator occurrences and unique out-neighbors can differ.

`b`, distance `d`, reached volume, and generated occurrences should be reported
separately rather than collapsed into one branching-factor estimate.

## Full traversal and target search have different parameters

Full exhaustion computes all of `R` and proves successor closure.  A target
search need inspect only enough work to prove a minimum target distance.

For a target at distance `d`, exact level-synchronous BFS must completely settle
all smaller depths.  Work at the target boundary depends on the output:

- candidate-stop may return one shortest path after the first valid target
  candidate, under a correct level invariant;
- complete-level stop processes the entire relevant layer;
- all shortest parents/path counts require the equality boundary;
- a full ball through `d` requires every state at that distance;
- an unreachable result requires component exhaustion or a separate
  nonreachability certificate.

Neighbor order and cancellation granularity can change actual target-search
work without changing the distance result.  Therefore target depth alone is not
an exact operation count.

## Output size is a lower bound

Different BFS outputs have different unavoidable sizes:

| Output | Minimum representation scale |
|---|---|
| one target distance | scalar after search |
| one replayable target path | `Omega(d)` labels/vertices |
| distance for every reached vertex | `Omega(|R|)` entries |
| one parent per reached non-source vertex | `Omega(|R|)` entries |
| full shortest-predecessor DAG | `Omega(number of retained predecessor edges)` |
| explicit enumeration of all shortest paths | potentially exponential in graph size |

No internal data structure can asymptotically beat the size of an output it must
materialize.  Conversely, an algorithm asked only for one target path need not
store the all-parent DAG.  Complexity claims must include the output contract.

## Queue space is not total BFS space

Let

```text
W = max_d |F_d|
N = |R|.
```

A simple frontier queue may require `O(W)` live entries, but exact graph BFS
also retains `O(N)` visited identity in the usual model.  If parents or all
distances are requested, they add `O(N)` records.  A bulk implementation may
temporarily hold a raw candidate bag as large as the outgoing occurrence count
of one frontier.

Here `O(W)` does not mean a capacity of exactly `W` entries. For ordinary FIFO
BFS marking on enqueue, the queue can contain an unprocessed part of one
layer and a discovered part of the next. Note 73 proves a peak bound of
`2W-1` for a nonempty full traversal under that convention. Its example with
layers of sizes `[1,100,100]` has queue peak 199 or 100 depending on whether
the parent of all last-layer vertices is processed first or last. The same
asymptotic bound therefore permits different exact capacity requirements;
allocated storage and visited/output bytes are separate again.

Useful peak categories are therefore:

```text
current frontier
next frontier
raw candidates
visited identity
distance/parent output
dedup/sort/routing scratch
communication buffers.
```

The textbook `O(V)` space statement normally assumes compact constant-size
vertex records.  For an implicit state of `s` bytes plus metadata, byte capacity
is the relevant quantity.  A `128`-byte state and a 32-bit dense rank do not
share the same practical `O(N)` memory.

## Parallel work, span, and elapsed time

An asymptotically work-efficient parallel BFS can keep total operations within
`O(|R|+A_R)`.  This does not imply elapsed time equal to work divided by the
number of processors.

For strict level-synchronous traversal, a vertex at distance `D` creates a
causal chain of at least `D` successive edge discoveries.  The traversal also
has per-level coordination, load imbalance, memory latency, atomics, and
communication.  Useful complexity axes are:

- **work:** total primitive operations across all workers;
- **span/dependency depth:** critical causal path;
- **communication:** messages and bytes crossing owners;
- **synchronization:** global or local completion events;
- **capacity:** peak resident state and scratch;
- **throughput:** completed work per wall-clock time under a declared numerator.

Two algorithms can have the same asymptotic work and very different span,
traffic, or memory.  A method can reduce wall time by doing more total work, or
reduce work while losing to synchronization overhead.

## Multi-GPU accounting

Distributed ownership adds counts not present in `V+E`:

```text
locally generated occurrences
locally removed duplicates
remote candidate records
remote bytes
owner-side duplicate convergence
replicated visited/frontier bytes
per-owner skew
messages or collectives per logical level
in-flight work at termination.
```

The sum of unique reached states is still a semantic property.  Network records
are execution-dependent: one state may be sent several times by different
parents before the owner accepts it once.  Communication volume is therefore
not determined by `|R|` alone.

Strong scaling fixes the graph and ideally reduces elapsed time.  Weak scaling
grows the graph with resources.  Capacity scaling asks whether more aggregate
memory permits a larger exact traversal.  Reporting "multi-GPU speedup" without
which parameter is fixed confounds three different claims.

## TEPS is a benchmark-normalized rate

The Graph500 specification defines TEPS using the number of input edges in the
traversed component divided by timed BFS duration.  It validates a BFS parent
tree but does not require one internal algorithm.

Consequently TEPS is not necessarily a hardware counter of actual adjacency
tests:

- direction-optimizing traversal may inspect a different number of entries;
- duplicate, synchronization, and communication work is not in the numerator;
- graph construction is timed separately;
- the numerator assumes the benchmark's explicit undirected input semantics.

For an implicit Cayley graph, generated transitions/s is a natural occurrence
rate, while accepted unique states/s measures discovery yield.  Neither should
be labeled Graph500 TEPS without adopting its exact numerator and graph
contract.

Rates with different numerators can each be useful; they are not directly
comparable merely because all contain the word "edge" or "transition."

## Complexity does not select an implementation

`Theta(V+E)` identifies asymptotic work in one model.  It does not say whether
the limiting resource is:

- memory bandwidth;
- random visited access;
- wide-state transformation;
- duplicate convergence;
- frontier materialization;
- synchronization latency;
- interconnect bandwidth;
- memory capacity.

These require measurements on a declared graph and representation.  Conversely,
a fast kernel timing does not change the mathematical work/output contract.

## Audit checklist

1. Is the graph explicit, dense-matrix, compressed, or implicit?
2. Does `E` mean logical edges, stored adjacency entries, or generated move
   occurrences?
3. Is the run a full component traversal or an early target query?
4. Which target/equality boundary must be completed?
5. What output is materialized and what is its size?
6. Are state-generation and exact-identity costs treated as constant or
   measured separately?
7. What are total work, critical depth, and peak bytes?
8. Which communication records and bytes are additional to local work?
9. Is a rate's numerator measured operations or benchmark-normalized graph
   volume?
10. Are strong, weak, and capacity scaling claims separated?

## Sources

- Merrill, Garland, and Grimshaw,
  [Scalable GPU Graph Traversal](https://research.nvidia.com/sites/default/files/pubs/2012-02_Scalable-GPU-Graph/ppo213s-merrill.pdf),
  explicitly targets asymptotically optimal `O(|V|+|E|)` work for stored sparse
  graph traversal.
- The [Graph500 benchmark specification](https://graph500.org/?page_id=12)
  defines its BFS parent-tree output, timing boundary, validation, traversed
  component edge count, and TEPS normalization.
- Notes 06, 07, 10, 11, 15, and 28 provide the implicit-state, hardware-work,
  geometry, output, external-memory, and exact-identity distinctions assembled
  here.

## Current conclusion

`O(V+E)` is a representation-relative work statement for explicit graph
traversal, not a universal performance prediction.  To understand BFS across
CPU, GPU, multi-GPU, and Cayley spaces, count the events that actually exist in
each model: stored adjacency inspections, generated move occurrences, exact
identity decisions, unique outputs, peak bytes, dependency levels, and remote
records.

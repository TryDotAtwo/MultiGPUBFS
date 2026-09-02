# BFS versus union-find and connectivity state

BFS and disjoint-set union can both answer connectivity questions in an
undirected graph. They retain very different information. BFS constructs
source-relative metric layers and paths; union-find maintains only a partition
into connected components under edge insertions.

This note studies that information boundary. It adds no connectivity
implementation.

## 1. The common static answer

Let `G=(V,E)` be a finite undirected graph.

- Running BFS from an unvisited root of each component labels every vertex by
  its connected component.
- Starting with singleton disjoint sets and applying `UNION(u,v)` for every
  edge `{u,v}` yields exactly the same component partition.

For vertices `x,y`, after all edges have been processed,

```text
FIND(x)=FIND(y)  <->  x and y are connected in G.
```

The equality is about the final equivalence relation. It does not make the two
internal data structures interchangeable.

## 2. What BFS retains beyond connectivity

A complete BFS from root `s` can retain:

- exact hop distance `d(s,v)`;
- frontier membership by depth;
- one replayable parent edge per reached non-root vertex;
- all shortest-parent edges or path counts if requested;
- an exhaustion certificate for the source component.

Plain union-find retains a representative for each set. It does not retain
minimum distances, BFS layers, shortest parents, or adjacency.

The representative name is an implementation choice. Union by rank/size and
path compression may change parent pointers without changing the represented
component.

## 3. A union-find parent forest is not a graph path forest

Disjoint-set parent pointers connect set representatives used by the data
structure. A parent pointer need not correspond to an input graph edge. When an
edge joins two components, union may attach one component root to the other
even though the processed edge connected two non-root members.

Path compression makes the distinction sharper: a vertex may point directly
to a representative many graph hops away, or to one not adjacent to it at all.

Therefore

```text
DSU tree depth != graph distance,
DSU parent chain != replayable graph path.
```

A separate witness forest can be retained from the actual union-causing edges,
but that is additional output and still need not be a shortest-path tree.

## 4. Insertions preserve connectivity but can rewrite distances

Union-find naturally supports incremental undirected connectivity. On insertion
of `{u,v}`, union their sets; old connected pairs remain connected and
components can only merge.

BFS distances are not equally stable. Start with

```text
0--1--2--3.
```

Inserting edge `{0,3}` leaves the component partition unchanged, so DSU state
has no semantic connectivity change. Yet

```text
d_old(0,3)=3,
d_new(0,3)=1.
```

Many shortest-path labels and parents may change after one insertion. An
incremental connectivity certificate is not an incremental BFS certificate.

## 5. Deletions expose the missing split operation

Deleting a redundant cycle edge may preserve a component; deleting a bridge
splits it. Plain union-find has no inverse operation that reconstructs the two
sets from its compressed forest, because it deliberately forgot which graph
edges supplied alternative connectivity.

Fully dynamic or decremental connectivity requires additional structures and
different proofs. Rebuilding by BFS/DFS after a deletion is one correct option,
but not the only possible dynamic-connectivity strategy.

This separates monotonicities:

```text
edge insertion: connectivity classes only merge;
edge deletion:  connectivity classes may split;
BFS distance:    can decrease on insertion and increase/become infinite on deletion.
```

## 6. Directed graphs change the question

Unioning endpoints of a directed arc discards orientation and computes weak
connectivity in the underlying undirected graph. It does not compute directed
reachability.

For the one-arc graph

```text
a -> b,
```

DSU places `a` and `b` together, while `b` cannot reach `a`. Strongly connected
components also require directed cycle/reachability structure; ordinary DSU on
arc endpoints is not an SCC algorithm.

Thus union-find's clean equivalence relation matches undirected connectivity,
not arbitrary one-way reachability.

## 7. Query workload changes the trade-off

For one source-target question, BFS can stop after proving the minimum target
distance and may avoid unrelated components. A DSU built from scratch usually
processes the relevant edge stream before answering connectivity queries but
then answers many pair queries cheaply.

For an incremental edge stream, union-find amortizes updates and repeated
connectivity queries. If queries ask for paths, distances, eccentricity,
frontier sizes, or shortest-path counts, the DSU partition is insufficient no
matter how fast `FIND` becomes.

Algorithm choice follows the requested output and update model, not the fact
that both algorithms sometimes answer "connected."

## 8. Completeness of the processed edge set

`FIND(x)!=FIND(y)` proves disconnection only relative to the edges actually
processed. In an implicit graph, unseen legal moves may later merge the sets.
Union-find does not discover vertices or successors by itself.

To use DSU for exact component claims over an implicit Cayley/puzzle graph, one
still needs evidence that:

- every relevant semantic state is represented;
- every required move occurrence was generated or otherwise accounted for;
- endpoint identity is exact;
- all cross-owner unions completed;
- the graph is interpreted as undirected when union semantics require it.

DSU can organize discovered connectivity; it cannot certify completeness of an
unknown transition oracle.

## 9. Cayley and Schreier interpretation

For an inverse-closed generator set, an undirected Cayley/Schreier component is
an orbit under the generated moves. Union-find over a fully enumerated finite
state graph can recover the orbit partition. It still does not recover minimum
word lengths or move sequences.

If the intended graph is the Cayley graph of the generated group from the
identity, connectedness may be true by construction: every represented group
element is a generator word. The interesting BFS information is then sphere
growth and word distance, precisely what DSU omits.

For positive directed alphabets, symmetrically unioning endpoints changes
forward reachability into weak connectivity and can merge states that are not
mutually reachable under allowed moves.

## 10. Parallel connectivity is not level-synchronous BFS

Parallel connectivity algorithms can use hooking, pointer jumping, and repeated
representative compression. Their rounds need not correspond to graph
distances. Shiloach-Vishkin is a classical example of logarithmic-time PRAM
connectivity without building BFS layers from one source.

On GPU or multiple GPUs, a union-based method may expose:

- concurrent root discovery and hooking conflicts;
- atomic representative changes;
- pointer-compression rounds;
- cross-partition unions and eventual representative agreement;
- load depending on edge order and component evolution.

These costs differ from frontier expansion, visited claims, and one logical
barrier per BFS depth. Faster component labeling is not evidence for faster
shortest-path BFS.

## 11. Exact identity remains prior

Union-find assumes each element name denotes one exact semantic vertex. If two
states collide under an unsafe hash key, union-find can irreversibly merge their
components. If one state has several uncanonicalized names, it can remain split
into several DSU elements.

Path compression accelerates representative lookup; it does not validate the
mapping from encoded records to semantic states. The exact-key and collision
rules of note 28 remain prerequisites.

## 12. Evidence checklist

1. Undirected connectivity, weak connectivity, reachability, or SCC output.
2. Static, insertion-only, deletion-only, or fully dynamic edge model.
3. Complete explicit edge set or partial implicit discovery.
4. Component label, connectivity bit, path, or shortest-distance output.
5. DSU parent versus separately retained graph-edge witness.
6. Exact endpoint/state identity and canonicalization.
7. Global completion of remote unions.
8. Connectivity work versus BFS layer/transition work.

## Sources

- R. E. Tarjan, [*Efficiency of a Good But Not Linear Set Union
  Algorithm*](https://doi.org/10.1145/321879.321884), Journal of the ACM 22(2)
  (1975), 215-225. Classical analysis of intermixed `UNION` and `FIND`
  operations and inverse-Ackermann behavior.
- Y. Shiloach and U. Vishkin, [*An `O(log n)` Parallel Connectivity
  Algorithm*](https://doi.org/10.1016/0196-6774(82)90008-6), Journal of
  Algorithms 3(1) (1982), 57-67. Parallel connectivity outside the BFS-layer
  schedule.
- Notes 04, 09, 11, 18, 21, 22, 28, 51, 52, 57, 74, 77, and 84 provide
  frontier, completeness, shortest paths, asynchronous, component certificate,
  dynamic, identity, distributed ownership, replica, output, discovery,
  source-update, and directed-component context.

## Takeaway

Union-find stores the equivalence relation "in the same undirected component."
BFS stores source-relative metric discovery. They agree on a static component
partition after complete input processing, but DSU representatives are not
paths, insertions can rewrite all BFS distances without changing connectivity,
and directed or implicit semantics require additional proof. Connectivity
throughput and shortest-path throughput are different achievements.

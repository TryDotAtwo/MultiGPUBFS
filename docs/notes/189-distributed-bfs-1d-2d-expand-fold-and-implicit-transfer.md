# Distributed BFS: 1D/2D expand-fold semantics and implicit transfer

## Question

What does a two-dimensional distributed BFS decomposition actually change,
which correctness obligations remain unchanged, and which parts fail to
transfer automatically from an explicit sparse matrix to an implicit Cayley
graph?

Notes 07, 44, and 179 mention 1D/2D partitioning and reject a universal winner.
This note makes the two communication phases semantically explicit. It does not
design or tune a multi-GPU implementation.

## 1. Fixed adjacency convention

Let a directed explicit graph have adjacency matrix

```text
A[u,v] = 1 iff u -> v.
```

Rows therefore name outgoing source adjacency and columns name incoming
destination adjacency. This convention matters: transposing `A` swaps which
processor dimension needs the frontier and which dimension combines candidate
destinations.

The source papers sometimes express a BFS step as `A^T F_k` because they use a
column frontier. That algebra is compatible with the source-row convention
above; it must not be copied without also copying vector orientation.

## 2. One-dimensional ownership

In a standard 1D source partition, process `p` owns a vertex subset `V_p`, its
outgoing adjacency, and authoritative metadata for those vertices. One exact
level has the conceptual phases:

```text
local frontier expansion
-> route candidate destinations to their vertex owners
-> owner exact visited decision and next-frontier publication
-> global level closure
```

Candidate routing can involve all processes. The graph edge cut influences
remote destinations, but actual records and bytes also depend on local
deduplication, metadata, batching, retry, and output contract.

## 3. Two-dimensional checkerboard

Arrange `p=p_r p_c` processes as `P(i,j)` and split the sparse matrix into
blocks `A_(i,j)`. A block owns only the edges whose source rows and destination
columns fall in its two index groups. Outgoing adjacency for one source group
is therefore distributed across a processor row rather than stored at one
process.

The level-synchronous top-down algorithm has two communication phases named in
the literature:

1. **Expand:** distribute the appropriate frontier pieces along one processor
   dimension so every matrix block holding relevant outgoing edges sees its
   active sources.
2. **Fold:** combine or route the locally generated destination candidates
   along the orthogonal dimension to the process or process slice responsible
   for authoritative vertex update.

Under the Buluç--Madduri layout, expand is an all-gather along processor columns
and fold is an all-to-all along processor rows. These row/column names belong
to that declared data/vector distribution; another matrix orientation may swap
them without changing the abstract obligations.

## 4. The two independent correctness obligations

### Expand completeness

For every `u` in exact frontier `F_d`, every block containing an outgoing edge
of `u` must receive enough identity to inspect that edge exactly once or under
a declared idempotent/retry contract. Missing one required block is successor
incompleteness and can delete reachable states.

### Fold authority

Every generated candidate for destination `v` must reach an authority that can
decide exact membership against the complete visited state for `v`'s epoch.
Partial per-block visited decisions are insufficient unless their merge is
proved equivalent to the authoritative decision.

These obligations meet only after all expanded source pieces, local edge
visits, candidate folds, and authoritative publications for the level are
closed. Empty local buffers inside either phase do not prove level or traversal
termination.

## 5. Why 2D can alter communication structure

The 2D scheme replaces one potentially process-wide candidate exchange with
two collectives on processor-grid slices. For a roughly square grid, each
collective may involve about `sqrt(p)` participants rather than `p`.

This is not a byte-free improvement:

- expand replicates frontier identities to blocks sharing their adjacency;
- fold transports partial destination discoveries to their authority;
- a frontier vertex whose adjacency spans many blocks has a large expand fanout;
- a destination discovered by many blocks has convergence work in fold;
- local/current-level/persistent aggregation policies change fold volume;
- communicator startup, topology, congestion, and collective algorithms affect
  time independently of logical word count.

The source partitioning study explicitly separates communication volume from
collective scaling. It models expand through adjacency connectivity and treats
plain edge cut as only a bound for fold, because fold traffic depends on the
space-time grouping of discoveries and aggregation.

## 6. Edge cut, adjacency connectivity, and traffic differ

For frontier vertex `u`, let `lambda_out(u)` be the number of matrix blocks
that collectively store its active outgoing adjacency. Expand must make `u`
available to those blocks, giving a fanout coordinate related to
`lambda_out(u)-1` under the declared owner placement.

For destination `v`, several incoming edges can be visited:

- on one block and aggregated once;
- on several blocks in the same level and folded as several records;
- in different levels after `v` is already visited;
- with or without persistent local suppression.

Consequently, one static cut number does not determine fold messages. The same
edge partition can produce different traffic under different BFS roots,
frontiers, aggregation stages, and metadata contracts. This refines note 179's
cut/information/protocol distinction for the 2D case.

## 7. Exact output contract survives the decomposition

The matrix layout does not weaken BFS semantics:

- distance requires exact level and visited closure;
- one arbitrary parent needs one valid depth-`d` proposal retained for each new
  depth-`d+1` vertex;
- a canonical parent needs every potentially smaller equal-depth proposal;
- a complete predecessor DAG retains every semantic shortest predecessor;
- path counts combine every logical contribution exactly once;
- retry-safe set membership does not make additive metadata idempotent.

Graph500 permits a correct BFS tree, not every richer output. A 2D
implementation validated only against that contract cannot be cited as evidence
for complete DAG, path-count, or canonical-word support.

## 8. Memory and replication coordinates

“Vertex ownership” is no longer one scalar concept in a checkerboard layout.
Record separately:

- where adjacency blocks live;
- where frontier identity is initially authoritative;
- along which slice it is replicated for expand;
- where visited/distance/parent metadata is authoritative;
- where advisory or partial metadata is replicated;
- where candidate buffers exist before fold.

Moving from 1D to 2D may reduce adjacency concentration while increasing
frontier or metadata replication. Capacity claims need peak simultaneous bytes
per process/device, not only total graph bytes divided by `p`.

## 9. Why explicit 2D does not automatically transfer to implicit BFS

An implicit graph has no stored nonzero `A[u,v]` to checkerboard. The successor
oracle computes `v=succ(u,g)` from a state and generator. To imitate a 2D
decomposition one must introduce a real second axis, for example:

- partition generator labels among expansion shards;
- partition transformation components;
- route generated endpoints by destination owner;
- replicate or fetch parent state representations to the shards that need them.

If generator shards are used, expand completeness becomes:

```text
every frontier state reaches every shard holding a required generator.
```

Fold authority remains:

```text
every generated full endpoint reaches its exact visited owner.
```

This can replicate wide implicit states rather than compact integer vertex IDs.
It may also repeat ranking, hashing, or canonicalization work. The regular
degree of a Cayley graph makes generator occurrence counts predictable but does
not prove that state replication, transformation cost, destination skew, or
owner traffic improves.

## 10. Cayley/Schreier-specific aliases

Several generator labels can induce the same endpoint from one state because of
stabilizers or duplicate transformations. A generator-sharded 2D analogue can
therefore create:

- same-parent endpoint aliases across expansion shards;
- different-parent convergence at fold;
- already-visited destinations;
- distinct labels required by a labeled predecessor output.

Producer aggregation can remove records only under the declared output merge
algebra. Exact state set union may collapse them; labeled DAG or count output
may not. Quotient/coset routing can supply a second structural axis only when
transition congruence and concrete endpoint lifting are proved, as notes
167--172 require.

## 11. Logical grid versus physical topology

A `p_r x p_c` process grid is a communication schedule, not a hardware fact.
Mapping its row and column communicators onto:

- GPUs behind one PCIe switch;
- NVLink or NVSwitch fabrics;
- NUMA sockets;
- network interface cards;
- inter-node links

can change which phase is local, oversubscribed, or serialized. A square grid
does not automatically match a hierarchical physical topology. Report expand
and fold bytes, messages, collective duration, overlap, and path separately.

The published CPU/MPI advantage of one 2D implementation on particular systems
is evidence for those graph, partition, collective, and machine regimes. It is
not a universal 2D-over-1D theorem and not direct multi-GPU Cayley evidence.

## 12. Minimum evidence ladder

Before a 1D/2D comparison supports a claim, retain:

1. adjacency orientation and processor-grid mapping;
2. exact source/destination block ownership;
3. frontier expand fanout and bytes;
4. local edges inspected and local aggregation stage;
5. fold records/messages/bytes before and after aggregation;
6. authoritative visited and output merge location;
7. per-level closure and independent frontier/distance parity;
8. peak adjacency/frontier/metadata/buffer bytes;
9. row/column communicator topology and timings;
10. explicit versus implicit state width and successor cost;
11. declared output contract;
12. strong/weak/capacity scaling regime.

Without these coordinates, “2D reduced communication” is underspecified.

## Sources

- Aydın Buluç and Kamesh Madduri,
  [*Parallel Breadth-First Search on Distributed Memory Systems*](https://arxiv.org/abs/1104.4518),
  SC 2011. Defines and measures the 1D and 2D distributed BFS approaches.
- Aydın Buluç and Kamesh Madduri,
  [*Graph Partitioning for Scalable Distributed Graph Computations*](https://people.eecs.berkeley.edu/~aydin/DIMACSprocfinal.pdf),
  DIMACS proceedings. States the checkerboard layout, column expand, row fold,
  aggregation variants, and communication models.
- Scott Beamer, Aydın Buluç, Krste Asanović, and David Patterson,
  [*Distributed Memory Breadth-First Search Revisited*](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2013/EECS-2013-2.html),
  UCB/EECS-2013-2. Extends the 2D setting to distributed direction-optimizing
  BFS and supplies regime-specific performance evidence.
- [Graph500 reference implementation](https://github.com/graph500/graph500)
  and [benchmark specification](https://graph500.org/?page_id=12), for the BFS
  tree contract and current reference-code boundary.

## Compact conclusion

Two-dimensional BFS splits communication into frontier expand and candidate
fold on orthogonal processor-grid dimensions. Correctness requires complete
frontier delivery to every adjacency block and authoritative folding of every
candidate. The layout can narrow collectives but introduces replication and
space-time aggregation effects. An implicit Cayley graph has no stored matrix
checkerboard; it needs a separately justified generator/transformation axis,
and wide-state replication can dominate the intended benefit.

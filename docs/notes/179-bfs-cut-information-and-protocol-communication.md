# BFS cuts, information, and protocol communication

## 1. Why “the cut is communication” is too strong

For a vertex ownership map `owner: V -> {0,...,P-1}`, define the active
cross-owner edge occurrences at depth `d`:

```text
X_d = {(u,v) in E : u in F_d and owner(u) != owner(v)}.
```

`|X_d|` is a graph-plus-frontier quantity. It is not automatically:

- the number of messages;
- the number of unique remote states;
- the minimum bytes any correct algorithm must send;
- the critical-path latency;
- the amount of useful new information;
- the communication needed for the requested output.

Those quantities coincide only under additional storage and protocol
assumptions. This note separates them rather than proposing a partitioner.

## 2. Three communication layers

### 2.1 Structural exposure

The cut occurrences `X_d` say which active transitions cross the declared
vertex partition. They predict where a conventional source-owner expansion
would create remote candidates.

### 2.2 Information obligation

A destination must learn enough to distinguish the correct next-frontier/output
state from alternatives consistent with its initial local knowledge and prior
transcript. This depends on:

- which adjacency/action data are initially local or replicated;
- which frontier/visited data are local or replicated;
- whether redundant computation is allowed;
- exact state encoding and shared dictionaries/ranks;
- the family of possible graphs/sources, not only one fixed execution;
- which output must become known at which owner.

### 2.3 Physical protocol traffic

Actual records, headers, collectives, retries, padding, acknowledgements,
replica updates, and output reductions implement the information transfer.
They may exceed the information obligation by a large factor.

The useful inequality is therefore model-relative:

```text
protocol traffic >= encoded information required by the declared knowledge model,
```

not `protocol traffic >= |X_d| * sizeof(edge)` universally.

## 3. Why no graph-only nonzero lower bound exists

Fix a finite graph and source. If every rank initially stores the complete
graph, source, deterministic BFS rules, and enough memory, every rank can
recompute the entire traversal locally. Traversal communication can be zero,
although work and storage are replicated `P` times. A final distributed output
contract may still require reduction or redistribution.

Therefore any nonzero communication lower bound must restrict at least one of:

- initial data placement;
- local memory/replication;
- redundant work;
- output placement;
- graph/source uncertainty;
- number of synchronization rounds.

This does not say replication is desirable. It says communication cannot be
proved unavoidable without declaring what information is absent locally.

## 4. Frontier information rather than crossing-edge count

Suppose an owner must learn an arbitrary `k`-element subset of a known universe
of `N` exact dense state IDs. Even with optimal encoding, distinguishing all
possibilities needs at least

```text
ceil(log2 binomial(N,k))
```

bits in the worst case, conditioned on no useful prior knowledge. A bitmap uses
`N` bits; a sorted list roughly uses `k log2 N` bits before compression; either
may be better depending on density and shared state.

This is a subset-information bound, not a claim about one retained BFS trace.
If the receiver can derive part of the subset from replicated adjacency and
frontier data, conditional information is smaller. If states are wide implicit
objects without shared injective ranks, identifying the same cardinality can
require far more bits.

## 5. Counterexample: many cut edges, one frontier fact

Let owner A hold root `r`, owner B hold leaves `v_1,...,v_n`, with edges
`r--v_i`. Then `|X_0|=n`.

Two legal data layouts give different communication:

1. A alone stores the outgoing adjacency of `r`; it may send `n` child IDs (or
   an equivalent encoded set) to B.
2. B already stores the relevant adjacency block; A can announce that `r` is
   in the frontier once, after which B derives all `n` children locally.

The graph, partition, and edge cut are identical. Initial edge placement and
allowed destination-side computation change the information obligation.
This is the conceptual difference exploited by changing between vertex-based
and two-dimensional sparse-matrix distributions, though their real collectives
and costs need measurement.

## 6. Counterexample: one cut edge, large causal effect

Join two large subgraphs by one bridge and place one subgraph per owner. Only
one structural edge crosses the partition. When the wave reaches that bridge,
one remote discovery can activate every later frontier in the second owner.

Thus a low cut can coexist with:

- a compulsory cross-owner causal handoff;
- late owner activation and poor early balance;
- a synchronization/latency event on the critical path;
- large downstream local work.

Cut volume describes neither causal amplification nor time-to-participation.
This complements the retained small-world and spatial observations in
REF-043--044.

## 7. Unique states, occurrences, and owner-pair fanout

For each depth retain at least:

```text
C_d = number of cross-owner candidate occurrences,
U_d = number of unique exact remote destination states,
A_d = number of active ordered owner pairs,
I_d = encoded payload bits before transport framing,
B_d = actual bytes sent,
R_d = communication rounds/collective phases.
```

These coordinates are independent enough to matter:

- many occurrences can collapse to one unique state;
- the same `U_d` can be concentrated on one pair or spread all-to-all;
- the same state count can mean compact ranks or wide puzzle states;
- the same bytes can arrive in one bulk phase or many latency-dominated rounds;
- zero accepted remote states can still follow heavy speculative traffic;
- output metadata can dominate the reached-state payload.

Consequently remote fraction alone is not a communication model.

## 8. Producer filtering and authority

A producer-side exact cache, Bloom filter, local sort/unique, or relation-aware
dedup can reduce `C_d` before routing. It cannot silently replace authoritative
exact membership unless its one-sided safety and epoch are proved. Conversely,
owner-side dedup may reduce accepted states but not bytes already sent.

The location of convergence matters:

```text
word duplicates
 -> producer-local duplicates
 -> cross-producer same owner
 -> owner visited hits
 -> output-specific equal-depth contributions.
```

Eliminating an earlier class changes physical traffic. Eliminating a later
class may only change insertion/output work.

## 9. Replication trades communication dimensions

Replication can replace candidate traffic with other costs:

- replicated graph/action tables consume memory and distribution time;
- replicated frontier bitmaps require synchronization;
- replicated visited state needs authority or coherent monotone updates;
- redundant expansion consumes compute and memory bandwidth;
- epoch changes require invalidation or migration;
- richer parents/counts may still need owner reductions.

Therefore “communication avoiding” often means moving bytes between phases or
trading bytes for storage/work. Report preprocessing, steady traversal,
replica maintenance, and final output separately.

## 10. Cayley and implicit graphs

In a Cayley graph, every owner may know the small generator set, so adjacency is
an action rule rather than a stored CSR row. This can make recomputation cheap
relative to sending a full child state. It still does not make the next frontier
common knowledge:

- the receiver must learn which parent/frontier facts are active, or receive
  exact child identities;
- sharded visited authority must decide novelty;
- a shared injective rank can shrink identity payload, while a wide state or
  collision-resolved key enlarges it;
- algebraic coset ownership can predict routes, but does not erase the need to
  transfer whatever frontier information is absent at the destination;
- relations can make many remote word occurrences converge to few states.

Thus Cayley regularity changes the conditional information and computation
trade, not the BFS correctness recurrence.

## 11. 1D and 2D distributions as knowledge rearrangements

Buluç and Madduri compare vertex-based 1D distribution with a 2D sparse-matrix
distribution for distributed-memory BFS. The relevant conceptual transfer is
that 2D distribution changes which processors initially know which adjacency
blocks and constrains communication to processor rows/columns in phases. It
does not prove that 2D always wins: bandwidth, latency, frontier density,
collectives, memory, and graph shape remain part of the model.

Work on sieving/compression similarly demonstrates that protocol bytes can be
reduced after structural crossings are fixed. This directly rejects treating
edge cut, logical information, and framed traffic as one number.

## 12. Rejected implications

- Each active cut edge forces one network message.
- Edge cut alone gives a universal byte lower bound for BFS.
- Zero traversal messages imply an efficient distributed algorithm.
- Equal remote fractions imply equal communication cost.
- Equal unique remote-state counts imply equal bytes.
- Few cut edges imply small critical-path communication impact.
- Destination-side recomputation eliminates visited authority.
- Compression reduces the underlying information obligation rather than its
  representation overhead.
- A 2D partition is universally better than a 1D partition.
- Algebraic routing eliminates communication because the owner is predictable.

## 13. Evidence boundary and next gate

This is a conceptual decomposition with finite counterexamples. It does not
derive a universal distributed-BFS lower bound or measure any interconnect. A
future bounded Rust model may hold graph/partition fixed while varying initial
adjacency placement and allowed recomputation, then report `C_d,U_d,A_d,I_d`
and actual serialized bytes separately. That gate remains deferred until Docker
is naturally available.

## Sources

- Aydin Buluç and Kamesh Madduri, *Parallel Breadth-First Search on Distributed
  Memory Systems*, SC 2011: <https://people.eecs.berkeley.edu/~aydin/sc11_bfs.pdf>.
- Scott Beamer, Aydin Buluç, Krste Asanović, and David Patterson,
  *Distributed Memory Breadth-First Search Revisited: Enabling Bottom-Up
  Search*: <https://crd.lbl.gov/assets/pubs_presos/mtaapbottomup2D.pdf>.
- Huiwei Lu, Guangming Tan, Mingyu Chen, and Ninghui Sun,
  *Reducing Communication in Parallel Breadth-First Search on Distributed
  Memory Systems*: <https://www.mcs.anl.gov/papers/P5226-1114.pdf>.
- Graph500 reference implementations, including the tuned MPI 2D-distribution
  implementation: <https://graph500.org/?page_id=47>.

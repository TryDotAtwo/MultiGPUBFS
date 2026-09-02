# BFS orders, live boundaries, and pathwidth

## 1. Several widths hide behind “the frontier”

For a total processing order

```text
pi = (v_1,...,v_n)
```

let `P_i={v_1,...,v_i}` and `S_i=V\P_i`. In an undirected graph define:

```text
L_i = {u in P_i : u has a neighbor in S_i},
R_i = {v in S_i : v has a neighbor in P_i},
E_i = {{u,v} in E : u in P_i, v in S_i}.
```

They are respectively the processed-side live vertex boundary, unprocessed-side
live vertex boundary, and crossing edge cut at processing position `i`.

They differ from:

- metric layer `F_d={v:dist(s,v)=d}`;
- physical BFS queue/open set;
- candidate occurrence bag;
- permanent visited ball;
- owner-partition boundary in a distributed layout.

One implementation may call several of them “frontier,” hiding different
memory and communication obligations.

## 2. Vertex separation and pathwidth

For a fixed order define

```text
vs(pi) = max_i |L_i|.
```

The graph's vertex separation number minimizes this over all vertex orders.
Kinnersley proved that vertex separation equals pathwidth under the standard
convention where pathwidth is maximum bag size minus one.

This is an ordering theorem about the smallest possible processed-side live
boundary. It does not say a BFS order attains that minimum.

The reverse order converts right boundaries into left boundaries:

```text
max_i |R_i(pi)| = vs(reverse(pi)).
```

For one fixed order, `|L_i|` and `|R_i|` can differ sharply.

## 3. BFS-constrained vertex separation

For root `s`, call an order BFS-valid when

```text
dist(s,u) < dist(s,v)  =>  u appears before v.
```

Ties inside one layer are free. Define

```text
bvs_s(G) = min over BFS-valid pi of max_i |L_i|.
```

Because BFS-valid orders are only a subset of all orders,

```text
pathwidth(G) <= bvs_s(G).
```

The inequality can be very loose. Root choice and within-layer order affect
`bvs_s`, while ordinary pathwidth is root-free.

## 4. Star: huge BFS layer, tiny left boundary

Let `K_(1,n-1)` be rooted at its center `c`. Then

```text
F_0={c}, |F_1|=n-1.
```

Immediately after processing `c`, all leaves are discovered/unprocessed, so a
conventional queue can contain `n-1` states. But

```text
L_1={c}, |L_1|=1.
```

As leaves are processed, `c` remains the only processed vertex adjacent to the
unprocessed suffix until the last leaf. Hence `vs(pi)=1` for that BFS order.

This rejects two implications:

- maximum layer/queue width is not processed-side vertex separation;
- small pathwidth/live boundary does not guarantee a small materialized Open
  list.

A streamed or implicit representation might avoid storing every leaf record,
but then its generation/output assumptions must be declared.

## 5. Complete binary tree: BFS can lose the pathwidth advantage

Take a complete binary tree of height `h`, rooted at its natural root. Its last
layer has `2^h` vertices. At the cut after all depth-`h-1` parents have been
processed but before their children are processed:

```text
|L_i| = 2^(h-1),
|R_i| = 2^h.
```

Every BFS-valid order reaches that inter-layer cut, independent of tie order.
Thus its BFS-constrained live boundary is exponential in `h`.

The same tree has pathwidth `O(h)` (a depth-oriented path decomposition gives
that upper bound). Therefore an unrestricted depth-like order can keep a narrow
live boundary while a metric-layer order exposes a wide wave.

This is not a defect in BFS: breadth ordering is exactly what certifies minimum
root distance. It is a memory consequence of the requested schedule.

## 6. Edge cut, left boundary, and right boundary

For every position with no isolated crossing endpoint:

```text
|L_i| <= |E_i|,
|R_i| <= |E_i|,
```

but edge multiplicity/degree can make `|E_i|` much larger. A single processed
hub may contribute many crossing edges and one left-boundary record. Conversely,
one cut can expose many distinct unprocessed endpoints.

Therefore:

- cutwidth counts crossing edges;
- vertex separation counts live vertices on one side;
- queue capacity counts physical records under a schedule;
- communication counts information/protocol traffic under note 179's model.

None is a universal proxy for the others.

## 7. Relation to safe forgetting

Note 181's undirected rolling scheme retains entire recent layers. It is a
simple sufficient certificate, not a minimal live-boundary theorem.

Frontier search aims to retain a solid active boundary plus used-transition
metadata and delete the interior. Its record count is closer to an Open/right
boundary object, while pathwidth characterizes the best possible left-boundary
order in the abstract graph model. Exact equivalence requires matching the
paper's node/operator semantics and pathwidth convention; the words “frontier
size equals pathwidth” are too loose.

Forgetting an interior vertex is safe only after no unresolved crossing edge or
operator can re-enter it. The boundary is therefore a certificate of separation,
not merely the latest metric layer.

## 8. Ordering and output constraints

An arbitrary low-separation order need not preserve BFS distances online. To
use it for exact shortest paths, one needs another mechanism such as:

- distance labels already known;
- a corrective relaxation algorithm;
- a decomposition dynamic program;
- repeated/divide-and-conquer search;
- an oracle stronger than local successor expansion.

Each changes the work, preprocessing, or output contract. Pathwidth alone does
not turn a depth-first traversal into BFS.

Richer outputs also extend lifetime:

- a parent record lives until path reconstruction or persistence;
- all-parent edges live until equal-depth closure;
- path counts live until predecessor contributions close;
- canonical outputs live until all smaller contenders are excluded.

Semantic output liveness may exceed graph-boundary liveness.

## 9. Multi-GPU boundaries are two-dimensional

Distributed BFS has at least two independent cuts:

1. **temporal/order boundary:** processed versus future work;
2. **ownership boundary:** local versus remote state/adjacency.

A vertex can be temporally live but owner-local, temporally dead but retained
for output, or newly exposed across several owners. Per level retain:

```text
total and maximum-owner |L|, |R|, queue/open records,
cross-owner occurrences and unique states,
boundary metadata bytes,
publication/reclamation dependencies.
```

Minimizing a static owner edge cut does not minimize BFS-constrained temporal
boundary. Conversely, a low-pathwidth order may destroy level parallelism or
require communication incompatible with the chosen owner layout.

## 10. Cayley and Schreier interpretation

For identity-root Cayley BFS, metric spheres are canonical under the chosen
generator metric, but within-sphere order remains free. Relations determine
which processed sphere states retain edges to unprocessed states in the same or
next sphere.

Vertex transitivity does not imply small live boundary. Large spheres in an
expanding Cayley graph can force large BFS waves even though every vertex has
the same local degree. Amenable/Folner behavior concerns boundary-to-volume
asymptotics of selected sets, not automatically the vertex separation of a
finite BFS order.

Schreier stabilizer aliases can reduce distinct states without proportionally
reducing transition occurrences or used-operator metadata. State boundary,
label boundary, and occurrence boundary remain separate.

## 11. Representation is not a universal bit lower bound

`|L_i|` or `|R_i|` counts semantic vertices. It does not prove that an
implementation needs one full record per vertex:

- dense ranks permit bitmaps;
- intervals/algebraic subsets may compress;
- implicit rules may regenerate boundary states;
- wide puzzle states may require much more than one bit;
- output metadata can dominate identity.

Information lower bounds require a family of possible boundaries and declared
prior knowledge, as in notes 36 and 179. Pathwidth is a combinatorial width, not
a byte formula.

## 12. Rejected implications

- Maximum BFS layer size equals pathwidth.
- Maximum queue size equals vertex separation of the processing order.
- Small pathwidth guarantees low-memory level-synchronous BFS from every root.
- A BFS order minimizes graph vertex separation.
- Left and right live boundaries have equal size at every cut.
- Low edge cut implies low live-vertex boundary or vice versa.
- A three-layer rolling window is a universal minimum-memory BFS.
- Frontier-search memory equals pathwidth without semantic qualifications.
- Vertex-transitive/Cayley graphs have uniformly small BFS live boundaries.
- Boundary vertex count directly equals GPU bytes.

## 13. Evidence boundary and next gate

The star and complete-binary-tree examples are conceptual exact families. This
note does not measure a queue or compute optimal pathwidth. A future bounded
Rust gate can enumerate small graphs/orders and record metric layer width,
queue peak, left/right separation, edge cut, and BFS-constrained separation
independently. It remains deferred until Docker is naturally available.

## Sources

- Nancy G. Kinnersley, *The Vertex Separation Number of a Graph Equals Its
  Path-Width*, Information Processing Letters 42(6), 1992, 345--350,
  DOI 10.1016/0020-0190(92)90234-M:
  <https://www.sciencedirect.com/science/article/pii/002001909290234M/pdf>.
- Fedor V. Fomin and Dimitrios M. Thilikos, *An Annotated Bibliography on
  Guaranteed Graph Searching*, including definitions of vertex separation,
  pathwidth, and search number:
  <https://citeseerx.ist.psu.edu/document?doi=f499f199eb9a3a1e099a19fd192eaa550e091be5&repid=rep1&type=pdf>.
- Richard E. Korf et al., *Frontier Search*, Journal of the ACM 52(5), 2005,
  DOI 10.1145/1089023.1089024; used with the output/metadata qualifications in
  note 181.

# BFS foundations

Status: initial study pass, 2026-08-27.

## Mathematical contract

For an unweighted graph and source `s`, define:

```text
F[0] = {s}
Visited[d] = union(F[0], ..., F[d])
F[d + 1] = N(F[d]) minus Visited[d]
```

`F[d]` is the frontier at depth `d`. A correct level-synchronous BFS maintains:

- every vertex in `F[d]` has shortest distance exactly `d` from `s`;
- frontiers at different depths are disjoint;
- after depth `d` completes, `Visited[d]` contains every reachable vertex with
  distance at most `d`;
- a parent selected when a vertex first enters a frontier belongs to the
  previous frontier and therefore defines a shortest path.

The shortest-path guarantee assumes equal-cost edges. Weighted graphs require
0-1 BFS, Dijkstra, delta stepping, or another SSSP algorithm as appropriate.

## Reference pseudocode

```text
visited = {source}
parent[source] = none
frontier = [source]
depth = 0

while frontier is not empty:
    next_frontier = []
    for u in frontier:
        for v in neighbors(u):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                distance[v] = depth + 1
                next_frontier.append(v)
    frontier = next_frontier
    depth += 1
```

Sequential `if not visited` becomes a concurrent first-discovery operation in
parallel BFS. Its implementation may be nondeterministic while distances remain
correct. Reproducible parent trees require an explicit tie-break rule.

## Complexity

For an explicit adjacency representation, ordinary BFS performs `O(V + E)`
work and uses `O(V)` auxiliary memory.

For an implicit graph, `E` is not stored. A more useful accounting is:

```text
generated transitions = sum(degree_generated(state)) over expanded states
```

This exposes costs hidden by the `O(V + E)` notation:

- materializing a neighboring state;
- canonicalizing it;
- computing routing/index keys;
- exact visited membership;
- storing enough state or metadata to continue traversal.

## Push, pull, and direction optimization

Top-down or push BFS expands outgoing edges from the current frontier. It is
natural when the frontier is small.

Bottom-up or pull BFS scans unvisited vertices and stops checking a vertex as
soon as it finds a predecessor in the current frontier. It can examine far fewer
edges on low-diameter graphs with a very large frontier. Beamer, Asanovic, and
Patterson combine push and pull and report large speedups on scale-free and
social-network graphs.

Applicability boundary: pull normally assumes an enumerable unvisited vertex
universe, compact membership structures, and cheap predecessor access. An
implicit Cayley graph may provide inverse generators, but it still lacks a cheap
way to enumerate all unvisited states. Therefore inverse generators alone do
not make bottom-up traversal practical.

## Explicit CSR versus implicit Cayley graphs

| Property | Explicit CSR graph | Implicit Cayley/state graph |
|---|---|---|
| Vertex identity | Usually dense integer | Structured state/permutation |
| Neighbors | Read from CSR | Compute by applying generators |
| Frontier bitmap | Direct | Requires a proved dense rank/encoding |
| Pull traversal | Natural with reverse CSR | Often cannot enumerate unvisited states |
| Visited check | Bit/byte/word per vertex | Full state, exact encoding, or indexed bucket |
| Dominant cost | Irregular edge reads and load balance | State generation, dedup, visited memory |

Observation: a GPU optimization demonstrated on Graph500 does not automatically
benefit Cayley BFS. First identify whether the saved work is edge reading,
neighbor generation, duplicate handling, or state storage.

## Validation principles

Correctness should not be inferred from a plausible visited count. A validator
should check:

- source depth is zero;
- every non-source discovered vertex has a valid parent edge;
- parent depth is exactly one less than child depth;
- every directed edge `u -> v` with reachable `u` satisfies
  `depth(v) <= depth(u)+1`; for an undirected graph, applying this in both
  directions gives `|depth(u)-depth(v)| <= 1`;
- every reachable vertex in the tested component was discovered;
- reconstructed target path replays and has length equal to its recorded depth.

The absolute-difference condition is not valid for a general directed graph.
For example, `s -> a -> b -> c -> s` has depths `0,1,2,3`; the edge `c -> s`
is valid even though its endpoints differ in depth by three.

Graph500 requires a valid BFS parent tree and defines performance through
traversed edges per second (TEPS). For implicit graphs, TEPS is insufficient:
generated transitions/s, unique accepted states/s, and bytes per accepted state
should be reported separately.

## Initial observations

1. The semantic distinction in graph BFS is between vertices of the chosen
   state graph and the many path descriptions that can reach them. Exact
   old-ball exclusion is what turns path-tree enumeration into unique-state
   graph frontiers; `visited` is its usual realization, not a universal BFS
   requirement. On a guaranteed tree, excluding the incoming parent edge is
   sufficient. A finitely branching tree-search BFS can also find a shallowest
   target without global visited, but on cyclic graphs it loses finite-component
   exhaustion and unique-state work bounds. The queue/frontier supplies a
   schedule, and a completed frontier can also be a requested metric layer or
   a certificate that one depth has closed.
2. Parallel first-discovery may choose different shortest parents without
   changing distances.
3. Hash equality is not state equality unless the encoding is provably
   injective.
4. A Bloom filter can accelerate an exact backing set but cannot reject a state
   by itself without risking lost vertices.
5. Frontier growth and visited memory are fundamental limits, not merely kernel
   optimization problems.
6. Bidirectional BFS is especially relevant to reversible generators, but its
   termination condition and path joining require a separate exact analysis.

## Primary sources for the next passes

- Scott Beamer, Krste Asanovic, David Patterson,
  [Direction-Optimizing Breadth-First Search](https://people.eecs.berkeley.edu/~krste/papers/beamer-sc2012.pdf).
- Duane Merrill, Michael Garland, Andrew Grimshaw,
  [Scalable GPU Graph Traversal](https://research.nvidia.com/sites/default/files/pubs/2012-02_Scalable-GPU-Graph/ppo213s-merrill.pdf).
- [Graph500 benchmark specification](https://graph500.org/?page_id=12).
- Yangzihao Wang et al.,
  [Gunrock: GPU Graph Analytics](https://arxiv.org/abs/1701.01170).

Performance numbers from these works are historical results on their stated
graphs and hardware. They are not expectations for this repository.

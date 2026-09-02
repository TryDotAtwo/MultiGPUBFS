# CayleyPy BfsResult: prefix edges, diameter, and label loss

## Question

What graph does retained CayleyPy `BfsResult` actually export when BFS is
complete or truncated, and which distance/edge/label claims are justified?

This source audit reads `bfs_algo.py`, `bfs_result.py`, and the public
`CayleyGraph.to_networkx_graph` wrapper. No graph is executed.

## 1. What `return_all_edges` records

In the non-batched single-device path, expanding a frontier with `k` generator
entries appends:

```text
source hashes = current_layer_hashes repeated k times
target hashes = generator-major get_neighbors(current_layer).
```

This records every declared generator occurrence emitted by each expanded
state, but only as endpoint hash pairs. Generator index is not stored in the
edge record.

If traversal reaches an empty next layer, every reachable state has eventually
been expanded and every recorded target lies in the reached hash universe. The
edge list is then a hash-keyed occurrence stream for the completely explored
reachable orbit, conditional on hash identity.

## 2. Truncated traversal adds a synthetic reverse boundary block

When BFS stops without exhausting the graph, `BfsAlgorithm` takes the most
recent block of recorded edges and appends the reversed endpoint pairs. Its
comment says this is done so the adjacency matrix is symmetric.

For an inverse-closed generator set, a reversed support edge is semantically
present. Adding the boundary reverse direction can compensate for not expanding
the newly returned last layer. It may also duplicate occurrences already
recorded from old or same-layer endpoints, which later simple-graph/matrix
representations collapse.

For a non-inverse-closed directed graph, however, reversal is not licensed by
the generator contract. The added pair `v->u` can be absent even though `u->v`
exists. The code adds it without checking `generators_inverse_closed`.

Therefore an incomplete `return_all_edges` result is not generally an exact
directed prefix-edge set. It contains a symmetric completion of the last
expanded block. `to_networkx_graph(directed=True)` does not remove those added
arcs.

## 3. `diameter()` is last recorded depth

`BfsResult.diameter()` returns

```text
len(layer_sizes)-1.
```

This quantity has several possible meanings:

- if the run stopped early, it is only the last returned BFS depth;
- if a finite reachable orbit was exhausted from one root, it is the root's
  outward eccentricity in that orbit;
- with several start states, it is the maximum minimum distance to the source
  set;
- it equals graph diameter only under an additional theorem, such as an
  appropriate vertex-transitive single-root metric setting.

The method does not inspect `bfs_completed`, source multiplicity, directedness,
or symmetry. Its docstring calls the value a maximal distance, and `__repr__`
prints it as `diameter`, but callers must interpret it using the run contract.

In particular:

```text
bfs_completed=False -> no diameter certificate.
```

## 4. Hash-to-index checks cannot recover traversal collisions

Explicit export requires hashes for every returned layer. The
`hashes_to_indices_dict` property inserts all hashes into a Python dictionary
and asserts that the number of distinct keys equals `num_vertices`.

This detects a duplicate hash if two colliding records both survived into the
stored layer lists. But scalar BFS deduplicates by hash before producing those
lists. If one semantic state was already lost because it collided with another,
the returned `num_vertices` is reduced too, and the dictionary can remain
internally consistent.

Thus the assertion proves uniqueness of retained hash records, not injectivity
over the intended semantic orbit. It cannot retroactively validate the hash-only
visited decision.

## 5. NetworkX export is a support graph, not a labeled multigraph

`to_networkx_graph` constructs `networkx.Graph` or `networkx.DiGraph`, not a
multi-edge graph. Endpoint pairs repeated because of:

- duplicate generator transformations;
- stabilizer aliases;
- symmetric reverse insertion;
- repeated physical edge occurrences

collapse to one support edge.

When labels are requested, `get_edge_name(i1,i2)` replays generators in list
order and returns the **first** generator whose transformation maps the stored
source state to the stored target state. The original edge occurrence carries
no generator index, so later aliases cannot be recovered.

Consequently the export preserves at most:

```text
one support adjacency + one first-matching display label.
```

It does not preserve generator multiplicity, every valid label, path-count edge
identity, or the label of the occurrence that originally generated the edge.

## 6. Directedness checks are incomplete-run asymmetric

`to_networkx_graph` correctly requires `directed=True` when the generator set is
not inverse-closed. This prevents representing a known directed action as an
undirected NetworkX graph.

But the check occurs after `BfsAlgorithm` may already have inserted synthetic
reverse pairs for an incomplete run. Choosing `DiGraph` preserves their
direction—it does not distinguish observed forward arcs from added reverse
arcs. Directed export is therefore faithful only when:

- traversal is complete; or
- every added reverse arc is independently known to exist, for example through
  inverse closure, and occurrence multiplicity is not part of the contract.

## 7. Dense and sparse matrices collapse multiplicity

`adjacency_matrix` assigns `1` for every endpoint pair into an `int8` dense
matrix. `adjacency_matrix_sparse` creates a COO entry per recorded pair; later
consumer behavior on duplicate coordinates determines whether duplicates are
summed or retained as duplicate storage entries.

The dense adjacency is unambiguously Boolean support. Its row sums therefore
count distinct target vertices, not generator occurrences. In a Schreier action
or duplicate-generator definition, this can be smaller than the declared
generator count even though raw expansion always performs every generator
entry.

`laplacian_matrix` forms `diag(row_sum)-adjacency`. For a directed export this is
an out-degree-style directed matrix, not automatically the symmetric
combinatorial Laplacian assumed by undirected spectral theorems.

## 8. Serialization boundary

Layer hashes are saved under HDF5 keys named `edges_list_hashes__i`; loading uses
the same convention, so the misleading name is internally consistent.

More materially, `load` reconstructs the graph through
`CayleyGraphDef.create`, the permutation-generator constructor. The serialized
format writes `self.graph.generators` generically but does not record a generator
type or matrix modulus and does not call `for_matrix_group` on load. A matrix
definition therefore has no demonstrated round-trip contract in this loader.

This is separate from BFS correctness: it concerns whether a persisted result
reconstructs the same implicit action definition.

## 9. Practical interpretation table

| Artifact field | Strongest direct meaning |
|---|---|
| `layer_sizes[d]` | number of retained hash identities at returned depth `d` |
| `bfs_completed` | empty next retained-hash frontier was reached |
| `diameter()` | last returned depth; diameter only with extra premises |
| `edges_list_hashes` complete run | generated endpoint-hash occurrences over exhausted orbit |
| `edges_list_hashes` incomplete run | generated occurrences plus reversed last block |
| NetworkX graph | simple support graph with first-matching display labels |
| dense adjacency | Boolean support adjacency |
| hash-index assertion | retained hashes are unique, not semantic injectivity |

## 10. Rejected implications

- `diameter()` proves graph diameter whenever it returns an integer.
- An incomplete directed edge export contains only generated arcs.
- `DiGraph` removes synthetic reverse boundary arcs.
- NetworkX export preserves parallel generator labels.
- `get_edge_name` recovers the original generator occurrence.
- Unique retained hashes prove there were no semantic collisions during BFS.
- Dense adjacency row sum equals the number of generator applications.
- `BfsResult.load` has a demonstrated matrix-generator round trip.

## 11. Evidence boundary

All claims are source-derived from the retained snapshot. No exported graph or
HDF5 file was produced, and no installed-version parity is claimed. Potential
runtime consequences remain untested; this note records the present code's
logical output boundaries rather than proposing fixes.

## Compact conclusion

`BfsResult` mixes exact scalar-layer intent with convenience export semantics.
Its complete hash-edge stream can describe the exhausted support graph under
exact identity, but incomplete runs symmetrize the last edge block, `diameter()`
is only a last-depth/eccentricity quantity without extra premises, and NetworkX
export collapses labeled multiplicity to one support edge and first matching
label. These distinctions matter before an exported artifact is used as a BFS
oracle.

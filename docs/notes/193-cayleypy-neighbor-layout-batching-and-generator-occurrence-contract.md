# CayleyPy neighbor layout, batching, and generator-occurrence contract

## Question

What exact implicit edge stream does the retained CayleyPy `CayleyGraph`
produce, and which ordering or label properties survive batching and inverse
path operations?

This is a source audit of:

- `cayleypy/cayley_graph.py`;
- `cayleypy/cayley_graph_def.py`;
- `cayleypy/algo/bfs_algo.py`.

No execution is used.

## 1. The semantic state action

For permutation generators, `apply_generator_batched(i,src,dst)` performs

```text
dst[row,j] = src[row, permutation_i[j]].
```

This is a pull/reindexing action on the entries of every state vector. It is not
safe to infer an abstract left/right group convention from the word “Cayley”
alone; path replay is defined by repeated execution of this concrete gather.

The central state is allowed to repeat values. `CayleyGraphDef.create` checks
that every generator is a permutation of positions and that central-state
values lie in range, but it does not require the central state itself to be a
permutation. Hence the same implementation naturally describes:

- a free regular action when all state labels distinguish positions;
- a non-free Schreier-like orbit when repeated values create stabilizer aliases.

For matrix generators, the action is left matrix multiplication, optionally
modular. The code explicitly does not require every matrix generator to be
invertible, so “Cayley graph” there can denote a directed transformation
semigroup orbit rather than a group Cayley graph.

## 2. `get_neighbors` is generator-major

For `m` input states and `k` declared generator entries, `get_neighbors`
allocates `mk` rows and fills:

```text
rows 0       .. m-1     : generator 0 on every state
rows m       .. 2m-1    : generator 1 on every state
...
rows (k-1)m  .. km-1    : generator k-1 on every state.
```

Thus the occurrence order is

```text
generator -> state,
```

not parent/state -> generator. The `return_all_edges` path agrees with this
layout by using `layer1_hashes.repeat(k)` for sources.

This matters for locality and arbitrary first-winner behavior, but not for the
semantic neighbor multiset: every declared `(state,generator-index)` occurrence
appears exactly once in the unbatched call.

## 3. Batching changes the global occurrence order

When a frontier exceeds `batch_size`, `BfsAlgorithm` splits the state rows and
runs `get_neighbors` separately on each batch. The effective global order is
then

```text
state batch -> generator -> state within batch,
```

rather than one global generator-major stream.

Each batch is deduplicated and filtered against previous layers. Later batches
are additionally filtered against the already accepted hashes of earlier
batches. Therefore batch order selects the first retained representative for a
hash that occurs in several batches.

For exact identity or collision-free hashes, this preserves the frontier set.
It does not preserve arbitrary first-winner metadata, occurrence order, or the
representative full state under a semantic hash collision.

The API acknowledges part of this boundary: `disable_batching` is documented
for callers that need states and hashes in the same order.

## 4. Batched non-identity states and hashes are separate orderings

At the end of the batched branch, the code:

```text
layer2        = vertical stack of accepted state batches
layer2_hashes = horizontal stack of hashes, then globally sorted.
```

For a non-identity hasher, it does not permute `layer2` by the same global hash
sort. Consequently row `i` of `layer2` is not promised to correspond to row
`i` of `layer2_hashes` after batching.

The core scalar traversal mostly tolerates this:

- future successor hashes are recomputed from state rows;
- seen membership consumes the sorted hash sets;
- stored decoded layers need only contain the right state set;
- `return_all_edges=True` disables batching.

But a `stop_condition(layer,hashes)` or external hook must not interpret the two
tensors as aligned records unless batching is disabled. This is an ordering
contract, not evidence that the scalar frontier set is wrong.

## 5. Generator list entries preserve occurrences during expansion

`CayleyGraphDef.create` validates each permutation but does not require the
generator permutations to be distinct. Therefore two list entries may:

- have identical transformations;
- have different names;
- emit duplicate labeled occurrences from every state.

`get_neighbors` applies both entries. The raw edge stream therefore preserves
declared list multiplicity. `get_unique_states`, however, reduces by state hash,
so ordinary scalar BFS intentionally collapses those occurrences to one child.

This is consistent for reachability/distance and insufficient for an output
that treats parallel generator labels as distinct shortest paths.

## 6. Inverse mapping collapses duplicate permutation labels

For permutation generators, `generators_inverse_map` first constructs

```text
permutation tuple -> generator index
```

as a Python dictionary. Duplicate permutation entries overwrite earlier ones,
so the stored index is the last occurrence of that transformation. Every
generator then maps its inverse permutation through this dictionary.

Consequences:

- inverse closure is tested at the transformation-set level;
- `revert_path` returns a valid inverse transformation sequence when the map is
  correct;
- it need not preserve which duplicate inverse label occurrence a labeled path
  contract would select;
- generator names do not participate in inverse identity.

Thus a path of generator indices is replay-valid under transformation semantics
while its inverse may not be label-involutive in the presence of duplicate
transformations. This is distinct from hash collision and from stabilizer alias:

```text
duplicate generator entries  -> same transformation, different list labels
stabilizer alias              -> different transformations, same endpoint at a state
hash collision                -> distinct semantic states, same stored identity key.
```

## 7. Inverted graphs preserve index position, not original names

`with_inverted_generators` replaces each generator at index `i` by its inverse
at the same index. This is exactly what `restore_path` needs: inverse-graph
candidate row `i` corresponds to applying the inverse of original move `i`.

The newly constructed definition does not pass the original generator names;
default names are regenerated from inverse tables. Path restoration returns
indices, so replay semantics remain index-based, but an audit must not infer
that the inverted graph preserves the original human-readable label metadata.

## 8. GPU interpretation

The source fixes several physical work coordinates before any kernel tuning:

- raw generation count is `|frontier| * number_of_generator_entries`;
- occurrence locality is generator-major inside each batch;
- batching changes which duplicates meet in one local `get_unique_states` call;
- cross-batch duplicate removal is sequential in accepted-batch order;
- sorting is by hash, not by semantic state or generator label;
- scalar frontier order after hashing is an implementation artifact.

A benchmark that changes `batch_size` changes occurrence grouping and reduction
history even when the exact frontier set is unchanged. It should not interpret
timing or first-winner changes as a change in the abstract graph without first
checking the output contract.

## 9. Rejected implications

- CayleyPy emits parent-major neighbors.
- Batching is only a memory limit and preserves every ordering property.
- A `(state,hash)` row pair remains aligned after batched non-identity hashing.
- Duplicate generator permutations are rejected by the definition.
- Transformation-level inverse closure preserves duplicate label identity.
- A repeated-value central state still gives a free Cayley action.
- Matrix mode always defines a group.

## 10. Evidence boundary

These claims follow from the retained source snapshot. No runtime fixture checks
whether another installed version differs, and no timing is claimed. The audit
establishes layout and contract consequences, not a proposal to change the
library.

## Compact conclusion

CayleyPy expands a generator-entry multiset in generator-major blocks, while
frontier BFS collapses the result by state hash. Batching changes global
occurrence order and can separate state-row order from globally sorted hash-row
order. Duplicate transformations survive as generated occurrences but collapse
in scalar BFS and in permutation inverse lookup. Exact distances, labeled paths,
and reproducible first-winner behavior therefore inhabit different contracts.

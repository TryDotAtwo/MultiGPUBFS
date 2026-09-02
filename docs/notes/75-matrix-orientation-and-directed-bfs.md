# Matrix orientation, `vxm`/`mxv`, and directed BFS

Sparse matrix-vector multiplication expresses one BFS expansion only after the
matrix indexing convention, vector side, and edge direction agree. A transpose
mistake can compute exact BFS on the reverse graph while appearing correct on
every symmetric test fixture.

## Fix the adjacency convention first

This note adopts

```text
A[u,v]=1  iff the directed edge u -> v exists.
```

Let `f` be the Boolean indicator of the current frontier. Over the Boolean
`OR-AND` semiring, treating `f` as a row vector gives

```text
(f A)[v] = OR_u (f[u] AND A[u,v]).
```

Hence `f A` is the forward successor support. In GraphBLAS terminology this is
vector-matrix multiplication, `vxm`.

Treating the same indicator as a column vector gives

```text
(A f)[u] = OR_v (A[u,v] AND f[v]).
```

This selects vertices `u` with an edge into the frontier: predecessor support.
To compute forward successors with column-vector `mxv`, use

```text
A^T f.
```

An equally valid library may instead store `A[v,u]=1` for `u->v`; then the
formulas swap. The bug is not choosing one convention, but failing to bind the
storage convention to the algebraic operation.

## Three-vertex witness

Take the directed path

```text
0 -> 1 -> 2
```

under source-row adjacency. From frontier `{0}`:

```text
row f A       = {1}   forward BFS
column A f    = {}    predecessors of 0
column A^T f  = {1}   forward BFS.
```

From frontier `{1}`, column `A f` returns `{0}`, while row `f A` returns `{2}`.
Both products are internally correct; they answer opposite reachability
questions.

After expansion, ordinary exact BFS still applies the complemented old-ball
mask:

```text
f_(d+1) = successor_support AND NOT visited.
```

Correct masking cannot repair a reversed multiplication. It only subtracts
visited states from whichever directed support was actually computed.

## Why undirected tests hide the error

For an undirected simple graph represented in both orientations,

```text
A=A^T.
```

Then forward and reverse neighbor sets coincide, so `vxm(A)`, `mxv(A)`, and an
explicit transpose can all produce the same Boolean support after adapting row
versus column shape. A test suite containing only undirected graphs cannot
validate directed orientation.

The same masking occurs in an inverse-closed Cayley graph: when every allowed
move has its inverse as an allowed unit move and labels are ignored, the simple
state adjacency is symmetric. Such a workload can validate set cardinalities
while failing to expose a reversed directed convention needed by a
positive-only generator alphabet.

A minimal orientation gate therefore needs an asymmetric directed fixture such
as `0->1->2`, not merely a larger symmetric graph.

## Reverse BFS

Backward search from a target must traverse predecessor edges of the original
graph. Under this note's convention, it can use:

- row-vector multiplication by `A^T`; or
- column-vector multiplication by `A`.

Forward search can use row `vxm(A)` or column `mxv(A^T)`. The paired formulas
are algebraically equivalent when the same convention is used consistently.

Calling an operation “backward” does not reverse edges by itself. The transpose
or predecessor oracle must be real, and bidirectional meeting checks must still
interpret one side as distance from the source and the other as distance to the
target.

## Cayley and implicit-state implications

An explicit CSR graph can materialize or view a transpose. An implicit Cayley
graph has no stored matrix; transpose means supplying the correct predecessor
operation.

For right-action edges

```text
g -> g s,
```

a predecessor under label `s` is `g s^-1`. This requires the mathematical
inverse transformation even if `s^-1` is not part of the allowed forward
alphabet. Adding inverse moves to the forward alphabet would symmetrize and
change the original directed metric; using them only inside the reverse oracle
does not.

Left-versus-right multiplication is another independent convention. A matrix
transpose reverses graph edges; it does not silently convert a right action
into the intended left-action replay semantics.

## Physical representation is a separate question

An implementation may realize the algebraic transpose through:

- a stored transposed matrix;
- a transpose view or descriptor;
- incoming adjacency;
- inverse implicit generators;
- a kernel whose indexing naturally implements the opposite orientation.

These choices can have different memory and performance costs while computing
the same support. Conversely, matching dimensions and fast execution do not
prove the orientation is correct.

For reproducible GraphBLAS-style BFS, record at least:

```text
edge-to-matrix index convention
row-vector vxm or column-vector mxv
transpose descriptor/view status
forward or predecessor semantic direction
Boolean semiring and mask semantics
directed asymmetric validation fixture
```

## Sources

- [GraphBLAS C API Specification v2.1](https://graphblas.org/docs/GraphBLAS_API_C_v2.1.0.pdf),
  which defines separate `vxm` and `mxv` operations and transpose descriptors.
- [Design of the GraphBLAS API for C](https://crd.lbl.gov/assets/Uploads/GABB17.pdf),
  which gives the mathematical forms `v^T A` and `A v`.
- [LAGraph Bellman-Ford `mxv` source](https://graphblas.org/LAGraph/experimental/algorithm/LAGraph_BF_full_mxv.c.gcov.html),
  whose interface comment explicitly requires transposed adjacency when using
  `mxv` instead of `vxm`.


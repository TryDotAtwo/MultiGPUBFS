# REF-033: graph covering and k-center boundaries

## Question

Can tiny exhaustive fixtures distinguish exact multi-source BFS evaluation
from center-selection heuristics?

## Method

- Language: Rust.
- Runtime: `rust:1.85-bookworm` Docker image.
- Graphs: a path with a high-degree endpoint fixture, `P6`, and a disconnected
  three-vertex fixture.
- Oracle: enumerate every center subset of the requested cardinality and
  evaluate it with the same transparent multi-source BFS.
- Scope: semantic counterexamples only; no benchmark or optimizer.

## Command

```powershell
docker run --rm -v "${PWD}:/work:ro" -w /tmp rust:1.85-bookworm bash -c "/usr/local/cargo/bin/rustc /work/experiments/graph_covering_bfs_probe.rs -O -o /tmp/graph_covering_bfs_probe && /tmp/graph_covering_bfs_probe"
```

## Result

```text
degree_trap degrees=[4, 2, 2, 2, 1, 1, 1, 1] eccentricities=[4, 3, 3, 4, 5, 5, 5, 5] highest_degree=0 highest_degree_radius=4 optimum_radius=3 optimum_centers=[[1], [2]]
path6 start=0 greedy_centers=[0, 5] greedy_radius=2 optimum_radius=1 optimum_centers=[[1, 4]]
path6 centers=[1, 4] radius=1 covered_at_one=true
disconnected centers=[0] covering_radius=None
```

Maximum degree failed as an exact one-center rule. Endpoint-seeded
farthest-first attained radius two while exhaustive two-center search attained
radius one. A center set missing a component returned no finite radius.

## Failure retained

The first combined format-and-run gate stopped before compilation because the
minimal `rust:1.85-bookworm` image does not contain `rustfmt`. Formatting then
passed in `multigpubfs-rust-toolchain:dev`; compilation and execution passed in
the minimal image. This was an image-component failure, not an algorithmic one.

## Status

Pass after one unavailable-`rustfmt` gate. These are exhaustive fixture
results, not general performance evidence.

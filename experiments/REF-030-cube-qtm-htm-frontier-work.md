# REF-030: Cube QTM versus HTM frontier work

Date: 2026-08-28

Status: pass after intentional RED and one formatting-only gate

## Question

How does changing only the generator set from Cube QTM to HTM change the work
that exact BFS sends toward `visited`, before any hardware optimization?

## Graphs

Both traversals use REF-029's checksum-pinned CayleyPy 54-sticker permutation
fixture and exact unique-sticker identity state.

- QTM: 12 signed quarter turns; a half turn costs two edges.
- HTM/FTM: the same 12 quarter turns plus six half turns as unit edges; degree
  18.

These are two Cayley graphs on the same generated group. They have the same
vertices but different adjacency and word metrics.

The exact sphere prefixes reproduce published counts:

```text
QTM: 1, 12, 114, 1068, 10011
HTM: 1, 18, 243, 3240, 43239
```

## Per-level partition

For every labeled transition occurrence from `F_d`, the endpoint is classified
as:

- backward in `F_(d-1)`;
- same-level in `F_d`;
- older than `F_(d-1)`;
- forward, outside the completed ball `B_d`.

Forward occurrences are then separated into unique `F_(d+1)` states and
duplicate extras. Full 54-entry state equality is used throughout.

| Metric | Expanded layer | Degree | Total | Backward | Same | Older | Forward records | Unique next | Forward extras |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QTM | F0 | 12 | 12 | 0 | 0 | 0 | 12 | 12 | 0 |
| QTM | F1 | 12 | 144 | 12 | 0 | 0 | 132 | 114 | 18 |
| QTM | F2 | 12 | 1,368 | 132 | 0 | 0 | 1,236 | 1,068 | 168 |
| QTM | F3 | 12 | 12,816 | 1,236 | 0 | 0 | 11,580 | 10,011 | 1,569 |
| HTM | F0 | 18 | 18 | 0 | 0 | 0 | 18 | 18 | 0 |
| HTM | F1 | 18 | 324 | 18 | 36 | 0 | 270 | 243 | 27 |
| HTM | F2 | 18 | 4,374 | 270 | 540 | 0 | 3,564 | 3,240 | 324 |
| HTM | F3 | 18 | 58,320 | 3,564 | 7,128 | 0 | 47,628 | 43,239 | 4,389 |

## What changed

### Degree is only the first multiplier

HTM has 1.5 times as many generators, but at expanded depth three it has about
3.03 times as many frontier states and 4.55 times as many generated transition
occurrences as QTM. This is not a throughput comparison: HTM radius three and
QTM radius three are different subsets under different metrics.

It does show why `degree * current frontier` must be measured per graph rather
than inferred from a common state representation.

### Same-level traffic appears in HTM

QTM has no same-level occurrences in the measured prefix. Quarter turns flip
the usual Cube move parity, so its Cayley graph is bipartite.

HTM adds a half turn as one edge. For one face, the solved vertex and the three
nonidentity powers form a triangle-containing local subgraph under
`g`, `g^-1`, and `g^2`. Same-level occurrences therefore appear immediately:
36 while expanding F1, then 540 and 7,128.

A visited design tested only on a bipartite QTM prefix can therefore miss the
requirement to reject the current frontier itself. HTM exposes that bug without
changing the state encoding.

### No edge reaches too far backward

`older=0` for every row. This is the general undirected distance inequality:
an edge from `F_d` can end only in `F_(d-1)`, `F_d`, or `F_(d+1)`.

### Occurrence conservation across a boundary

For both metrics,

```text
backward_occurrences(F_d)
    = forward_candidate_occurrences(expanding F_(d-1)).
```

Every labeled undirected edge occurrence crossing the layer boundary is seen
once in each direction. Unique frontier size cannot express this conservation
because multiple parent/label occurrences may share one child.

## Implication for GPU and multi-GPU study

Before hardware, the generator set has already decided:

- total transformation occurrences;
- frontier width;
- how many visited probes hit the previous versus current layer;
- how much convergence is presented to candidate dedup;
- whether a two-color/parity representation is available.

Consequently a QTM kernel timing cannot be transferred to HTM by multiplying
only by `18/12`. Conversely, the larger same-level rejection stream is not
automatically a performance loss: its locality and the representation of the
visited ball still need measurement.

For multi-GPU owner routing, the same semantic records may be filtered before
routing, at the destination owner, or after a batch sort. REF-030 measures only
their mathematical occurrence counts, not where they converge physically.

## Evidence and limits

The intentional RED compile failed on absent metric/profile functions. Six
tests passed after implementation, including the four inherited REF-029
fixture/relation tests and two new sphere/conservation tests. The first full
gate then stopped only on one `rustfmt` layout request.

```text
image: multigpubfs-rust-toolchain:dev
workspace mount: read-only
tests: 6 passed, 0 failed
GPU requested: no
```

Artifacts:

- `experiments/ref030_cube_qtm_htm_frontier_work.rs`
- `experiments/REF-030-cube-qtm-htm-frontier-work.txt`
- probe source SHA-256:
  `d25709ca2f457374de3074b5d9fcb4544d031c9fab183b12eb86b864dd7e3331`;
- raw output SHA-256:
  `cdb4d4557b3aa90266382fd8aa94ba3c1eae4ee52114bc1367fae74152660aa3`.

Scope is exhaustive expansion through F3, producing exact F4. No timing,
production implementation, GPU code, or optimization is claimed.

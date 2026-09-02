# CayleyPy output-contract audit

This note applies note 57's output matrix to one concrete CayleyPy production
path and one retained two-GPU artifact. The purpose is to identify the strongest
result actually evidenced—not to optimize or modify the implementation.

## Inspected state and evidence boundary

The source checkout was read-only inspected at:

```text
D:\100XH100
branch: main
HEAD: b5fcf6b0ee3247a1f189e6da79a1641d2304bd1c
inspection date: 2026-08-28
```

The working tree was heavily dirty, including modifications to
`tools/production_runner.cu`, runtime configuration, static memory, and HPC
wrappers. Therefore line-level observations describe this working tree, not a
clean release or necessarily the binary that produced an older artifact.

The retained artifact examined was:

```text
D:\100XH100\test_results\kaggle_cube4_public_v1\
  cube4_transformer_libtorch_logs\
  torchrun_native_transformer_10m_p0_d100_b65536.log
```

It records eight length-10 solutions found across ranks 0 and 1 at beam 65,536,
then stops after one configured extra depth and reports `best_length=10`. Its
short log does not record a source commit, container digest, generator-manifest
digest, or binary digest. The bundle contains a binary, data, weights, and
manifests, but this pass did not establish an immutable provenance chain from
every file to the current source.

No executable was rebuilt or rerun. Current runtime status remains unknown.

## What the production path constructs

The outer loop is a depth-indexed beam search, not exact graph BFS. For every
retained parent it generates configured moves, checks direct/K1/K2 target
conditions, scores and hash-deduplicates candidates, then retains a globally
bounded beam. Earlier pruned parents are not recovered by the local suffix
lookup.

The produced solution has three pieces:

```text
retained beam-history prefix
+ optional Stream2/K2 suffix
+ optional solved-neighborhood/K1 suffix.
```

Current source reconstructs the prefix by following one stored
`(parent_idx,route_packed)` record per retained state, appends suffix moves, and
applies the full move sequence on the CPU. It compares the resulting full state
with the target and throws before accepting the artifact if they differ.

Thus the natural semantic output is:

```text
one replay-validated move word found by a bounded heuristic beam pipeline.
```

It is not a shortest-distance certificate for the full puzzle graph.

## Positive witness evidence

The current source supports a strong positive-witness contract:

- `reconstruct_solution_from_history` follows one parent lineage and retains
  move IDs (`production_runner.cu:2929-2955`);
- `append_solution_suffixes` appends K2 and K1 suffix words
  (`production_runner.cu:3058-3071`);
- `apply_solution_moves` replays every move on the CPU
  (`production_runner.cu:2958-2967`);
- ordinary and distributed output compare the final full state with the target
  and throw on mismatch (`production_runner.cu:4705-4735, 4768-4793`);
- solve-bucket records are printed only after the same full-state check
  (`production_runner.cu:4588-4657`).

The retained log's eight `solve_bucket_solution=1` records occur after this
validation point under the corresponding source design. This is evidence that
those move sequences were accepted as positive witnesses by that packaged
runner. Because immutable binary/source/data linkage is absent from the short
log, the strongest wording is "historically runner-validated," not "revalidated
against the current dirty checkout."

The standalone CSV

```text
D:\100XH100\test_results\submit_p0_d60_b16777216.csv
```

contains only `initial_state_id,path`. By itself it is a candidate witness, not
a replay certificate: it carries no final state, validation flag, generator
version, target version, or producing binary identity.

## Why `solution_length` is not a BFS distance

CPU replay proves that the emitted word reaches the target:

```text
dist(initial,target) <= solution_length.
```

It does not prove the reverse inequality. The outer recurrence intentionally
discards states by score and global beam width, uses hash identity, and has no
complete old-ball visited proof. A shorter path may pass through a parent
pruned at an earlier depth.

K1 is a reverse local neighborhood and can supply an exact residual distance
within its correctly constructed table. K2 adds a bounded suffix lookup. Those
local facts do not turn the outer pruned prefix into an exhaustive ball.

Consequently:

- `solution_length=10` is a replayed witness length;
- `best_length=10` is the minimum among the records collected by the configured
  solve-bucket window;
- `puzzle_solved=0` means no witness was found under that bounded run, not that
  the target is unreachable;
- neither value is an exact original-graph shortest-distance result without an
  independent lower-bound certificate.

## What `solved_count` and bucket counts mean

The Stream2 kernel performs `atomicAdd(solved_count,1)` for every detected hit
and stores records only while the index is below
`solved_result_capacity`; otherwise it sets `solved_overflow`
(`cuda/stream2.cu:182-207`).

This counter is an operational hit-occurrence count. It is not the
shortest-path recurrence

```text
sigma(v)=sum sigma(parent).
```

It may reflect several candidate occurrences, suffix hits, or convergent beam
histories. It has no predecessor-DAG completeness or exactly-once semantic edge
contract. Likewise, the retained artifact's

```text
solve_bucket_length_count length=10 count=8
```

counts eight stored/printed solution records. It does not prove that the puzzle
has exactly eight shortest paths or even that every hit in the configured
window was semantically distinct.

Solve-bucket mode throws if the local solved-result buffer overflow flag is set
before consuming the bucket (`production_runner.cu:4578-4581`). Ordinary mode
records `solved_overflow` in artifacts but the inspected selection path does not
turn it into the same hard failure. Therefore ordinary-mode selection cannot be
called complete over all detected hits when overflow is nonzero.

## Selection and canonicality

Within one stored snapshot, the current source selects by the tuple

```text
(total_depth, parent_idx, route_packed, hash, suffix_id).
```

Distributed reconstruction compares the selected record from every rank and
adds owner rank as the final tie (`production_runner.cu:3356-3406`). This makes
the chosen record deterministic under important fixed-layout assumptions.

It is not a declared representation-independent canonical puzzle word:

- `parent_idx` depends on retained frontier order;
- `route_packed` contains routing/rank fields as well as the move;
- the hash is an implementation identity key;
- rank is a physical ownership property;
- only stored hits participate, and capacity overflow can omit competitors;
- changing world size or frontier order can change these keys.

Therefore the artifact supports one selected replayable witness. It does not
currently evidence shortlex-minimal, lexicographically minimal, rank-count
invariant, or otherwise semantic canonical output.

## K1/K2 suffix output semantics

The host solved-neighborhood builder performs reverse breadth layers from the
target and inserts only the first suffix for each `Hash128`
(`production_runner.cu:1128-1172`). Under complete generator enumeration and
collision-free/exact identity, this gives one minimum-length K1 suffix within
the radius. It does not retain every shortest suffix or specify a semantic
tie-order among equal-length words.

For an exact complete K1 ball with those shortest suffixes, exhaustive K2 words
in nondecreasing length including the empty word make first hit a shortest
combined residual for that candidate (note 40). This supplies one shortest
suffix under the premises, not all shortest suffixes or a canonical tie choice.

The current evidence remains qualified:

- table identity is full `Hash128`, not full state equality;
- CPU full-path replay protects accepted positive artifacts from a false hash
  hit but cannot repair a true state lost through a hash collision;
- existing K1/K2 unit fixtures are narrow positive component tests and bypass
  important production builders, as detailed in note 43;
- no current dirty-tree Docker rerun was performed here.

Thus K1/K2 strengthen the ability to produce a short replayable suffix, not
the outer pipeline's global distance, all-suffix, or counting contract.

## Output-contract matrix

| Note 57 contract | Current source intent/evidence | Retained artifact status |
|---|---|---|
| target reached / positive witness | full CPU replay and target equality before acceptance | **historically validated**, with incomplete immutable provenance |
| one arbitrary replayable path | one parent lineage plus K2/K1 suffix | **present**: eight printed words, one selected summary word |
| original-graph shortest distance | beam pruning and hash identity lack exact closure/lower bound | **not established**; length is an upper bound |
| canonical shortest path | implementation tuple over stored records, not semantic word order | **not established** |
| predecessor DAG | one history record per retained survivor; losing parents not retained as a DAG | **not provided** |
| exact shortest-path count | `solved_count` counts operational hits, not `sigma` | **not provided** |
| all shortest paths | bucket samples a configured depth window and finite result buffers | **not provided**; `all_solutions` filename is not a completeness proof |
| uniform shortest-path sample | no DAG/count-weighted sampling contract | **not provided** |
| unreachable / no solution | bounded pruned run only | **unknown**, never `UNREACHABLE` from this evidence |
| multi-source owner/ties | single puzzle start and target; rank owner is physical | **not applicable** |

The strongest supported result is narrower and clearer than "BFS solved the
puzzle optimally":

```text
the bounded beam pipeline produced one or more move words that its packaged
CPU replay accepted as reaching the requested target.
```

## Artifact-schema gaps exposed by the matrix

The code writes useful fields—path, final state, history bytes, hit count, and
overflow—but a standalone result would be more self-describing if it recorded:

- graph/puzzle/generator and target manifest digests;
- source commit, dirty-tree digest, binary/container digest;
- explicit `output_contract=one_replay_valid_beam_witness`;
- `optimality_status=not_proved`;
- path identity and move-label/occurrence convention;
- K1/K2 radii, suffix table/version, and beam/score pruning parameters;
- `solved_overflow` as a validity column for hit-set claims;
- `no_solution_semantics=not_found_within_bounded_beam_run`;
- whether a reported best is best-in-snapshot, best-in-window, or globally
  certified.

These are observations for future artifacts, not an implementation request.

## Distinguishing absence from contradiction

The audit found no evidence that CayleyPy promises all shortest paths, path
counts, or canonical words in this pipeline. Their absence is therefore not a
bug relative to the observed one-witness purpose. It would become a contract
failure only if a downstream consumer interprets:

- `solution_length` as a certified optimum;
- `puzzle_solved=0` as unreachable;
- `solved_count` as a number of shortest paths;
- `all_solutions` as complete enumeration;
- the selected tuple as a semantic canonical word.

The open issue is downstream interpretation, not evidence of an incorrect
positive path.

## Current conclusions

1. The inspected CayleyPy production path is designed to return replay-checked
   beam-search witnesses, not exact BFS distances.
2. The strongest positive guarantee is one full move word whose CPU replay
   reaches the target; replay does not prove global shortestness.
3. K1/K2 suffixes can shorten and validate the residual witness while leaving
   outer beam completeness unchanged.
4. `solved_count` and solve-bucket length counts are operational record counts,
   not shortest-path counts.
5. The current tie tuple is deterministic only under implementation/layout
   assumptions and is not a declared canonical generator-word order.
6. Predecessor-DAG, all-path, exact-count, uniform-sampling, and unreachable
   contracts are not provided by the inspected artifact.
7. The retained two-GPU artifact contains historically validated positive
   records but lacks an immutable provenance chain to the current dirty source.
8. No code, test, benchmark, or optimization was added or run.

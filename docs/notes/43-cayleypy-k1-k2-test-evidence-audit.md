# CayleyPy K1/K2 test evidence audit

Tests are evidence only for the properties their fixtures and assertions
exercise. A passing executable name such as `stream2_cuda_tests` does not by
itself validate the whole production Stream2, the host K1 builder, shortest
suffix selection, or outer search semantics.

This is a read-only audit of the `D:\100XH100` working tree described in note
38. No tests were run in this pass because the project's execution contract is
Docker-only and the current session received `permission denied` when opening
the Docker Desktop Linux-engine API pipe. This does not prove that Docker
Desktop itself was stopped; it proves that this session could not use the
required engine. Existing reports were inspected as historical artifacts, not
treated as current-run evidence.

## Artifacts inspected

```text
tests/stream2_cuda_tests.cu
tests/contract_tests.cpp
tests/history_tests.cpp
tests/stitched_cuda_tests.cu
CMakeLists.txt
test_results/stream2_cuda_tests_2026-05-20.md
test_results/stream2_solved_neighborhood_k1_2026-05-27.md
test_results/stream2_suffix_k2_2026-05-28.md
test_results/contract_tests_2026-05-20.md
test_results/history_tests_2026-05-22.md
```

The current test/report files above were clean relative to current `HEAD`, but
`tools/production_runner.cu` was locally modified. Clean test source means the
fixture was not edited; it does not mean it was rebuilt or rerun against the
dirty production working tree.

Git history ties the K1 report and test addition to commit `3b03fee7` on
2026-05-27, and the K2 report/test extension to `f8b17039` on 2026-05-28. The
current inspected commit was `b5fcf6b0`, with many later production-runner
commits and local changes. The retained reports do not record a current binary
hash or current working-tree fingerprint.

## What `stream2_cuda_tests` directly checks

### Direct target hit

The fixture builds a central state and a start state differing by a swap. One
generator swaps positions 0 and 1 back to the target. Assertions verify:

- GPU child hash equals the CPU Zobrist reference for that move;
- solved, stop, and count flags are set without overflow;
- the recorded score is `GOAL_SCORE_KEY`;
- the supplied depth value is recorded;
- direct hits use suffix ID zero.

This is strong component evidence for one exact direct-hit fixture. It does not
exercise multiple simultaneous hits, result overflow, `stop_on_found=0`, or
distributed propagation.

### Manually constructed K1 lookup hit

The test manually allocates a device neighborhood table and inserts one known
hash into one first-choice bucket slot. It then arranges for one immediate child
to have that hash and verifies solved flags, recorded hash, and suffix ID zero.

It directly checks:

- fingerprint screening accepts the matching fingerprint;
- full `Hash128` equality accepts the matching slot;
- Stream2 records an immediate K1 hit.

It does **not** call `build_solved_neighborhood_host`. Therefore it does not
test:

- reverse BFS frontier construction;
- inverse-generator correctness;
- K1 first-discovery distance;
- packed K1 suffix orientation or replay;
- automatic two-bucket placement and resize;
- maximum-entry failure;
- collision behavior or a mismatching full hash behind a matching fingerprint;
- lookup of a second-choice bucket entry.

The test name `solved_neighborhood_lookup` is accurate for device membership;
it should not be broadened to “K1 BFS construction passed.”

### K2 base-generator hit

The fixture starts two disjoint swaps away from the central state. The immediate
move fixes one swap and a hand-written one-move K2 suffix fixes the other. It
verifies solved flags, central hash, and suffix ID one.

This confirms one positive K2 path for the base-generator backend. The suffix
list is manually supplied with exactly two records—empty and the desired move—
so the test does not exercise the production word-list builder or competing
suffix order.

### K2 composed-permutation hit

The same semantic fixture is repeated with a manually composed permutation.
It checks parity for that one suffix and backend branch. It does not call
`build_stream2_composed_permutations`, compare every generated suffix between
backends, or test multi-move composition order.

## Historical K1/K2 reports

The K1 verification report records that Docker builds and targeted tests passed
at its May implementation point. Its production smoke ran
`production_runner 0 1 4096` with K1 radius one and observed 25 entries, 32
buckets, and 2560 device bytes without a crash.

That smoke supports initialization, table construction, upload, and one shallow
runner invocation. It does not report a K1 solution hit, suffix replay, exact
expected membership set, or shortest-distance oracle.

The K2 verification report records Docker builds and targeted tests at the May
K2 implementation point, plus notebook parsing. It does not record a production
K2 solution smoke or complete-ball oracle fixtures covering the empty-word
case `D<=K1`, the boundary `K1<D<=K1+K2`, and exact membership and word coverage.

Both reports are useful historical evidence. Neither proves current dirty-tree
runtime status.

## What the CPU contract test checks

`contract_tests.cpp` verifies several small contracts:

- configuration derivation and beam alignment;
- score quantization;
- padding storage and hash invariance to padding;
- Stream3/4 threshold and same-hash tie choice;
- one CPU reference depth detects a direct solved child and materializes a
  nonempty next frontier.

The dedup fixture deliberately gives two records the same `Hash128`, but treats
them as duplicate representations of one key. It is not a forced semantic hash
collision between two distinct `State128` values and therefore does not test
collision resolution or expose its absence.

The CPU reference depth has no accumulated visited ball, as note 38 observed.
Its pass validates its beam-depth component behavior, not exact multi-level BFS.

## What the history test checks

`history_tests.cpp` constructs a small `CpuHistoryStore`, appends two depth
arrays, and checks that reconstruction emits moves in root-to-goal order with
the expected parent indices.

This is direct evidence for the standalone history library. The production
runner currently contains its own `CpuCandidateHistory` implementation with
RAM, disk, static-hybrid, pruning, and distributed reconstruction paths. A pass
of the standalone history test cannot automatically validate those separate
paths or K1/K2 suffix append order.

## What the stitched CUDA test means

`stitched_cuda_tests.cu` invokes Stream1, Stream2, Stream3, Stream4, and final
materialization in one executable, but its data flow is only partly stitched:

- Stream1/2/3 share the synthetic frontier and generated buffers;
- Stream4 receives a new host-constructed pair of candidates rather than the
  actual Stream3 output;
- final materialization receives another manually constructed request rather
  than Stream4's selected candidate.

Therefore it is a component-integration smoke, not an end-to-end proof that one
candidate's identity, score, route, parent, selection, materialized state, and
history remain coherent through the full production pipeline.

No retained `stitched_cuda_tests` report matching the searched naming pattern
was found in this pass, so source registration in CTest must not be confused
with evidence that the current executable ran.

## Coverage matrix

| Property | Current inspected evidence | Status |
|---|---|---|
| direct child hash parity | CPU/GPU comparison in Stream2 test | covered for one fixture |
| direct goal flags/depth | Stream2 assertions | covered for one hit |
| device K1 hash lookup | manually inserted first-bucket entry | narrow positive coverage |
| host K1 reverse BFS | no direct test found | uncovered here |
| K1 suffix replay/order | no nonempty K1 suffix assertion found | uncovered here |
| K1 exact membership by layer | no independent set oracle found | uncovered here |
| forced semantic hash collision | no fixture found | uncovered here |
| K1 capacity failure | no assertion found | uncovered here |
| one K2 suffix hit | Stream2 base/composed fixtures | covered narrowly |
| production K2 word-list builder | manually supplied list bypasses builder | uncovered here |
| multi-move composed order | one one-move permutation only | uncovered here |
| exact-ball first-hit premises and residual distance | no complete-ball/ordered-prefix oracle fixture | uncovered here |
| complete K1/K2 miss lower bound | no negative oracle fixture | uncovered here |
| solved-result overflow | no overflow fixture found | uncovered here |
| full candidate pipeline lineage | stitched test replaces intermediate data | not established |
| final full-state solution replay | production code and historical workflows, not these unit assertions | separate evidence |
| current dirty-tree execution | Docker API access denied in this pass | unknown |

“Uncovered here” does not mean the property is false or that no test exists
anywhere in the repository. It means the named inspected artifacts do not prove
it. Broad repository searches were used to locate relevant test references,
but manual scripts or external workflows may carry additional evidence.

The corrected theorem in note 40 proves that the first hit is a shortest
residual for an exact complete K1 ball with shortest suffixes and exhaustive
nondecreasing K2 word lengths, including the empty word. A fixture with a
strictly better later hit cannot refute that theorem while preserving those
premises. Relevant evidence instead checks the premises, boundary distances,
and behavior when entries or words are missing or semantic hashes collide.
This correction does not claim that any additional fixture has been run.

## Shared-oracle risks

Some tests compare GPU output with CPU helpers built from the same generator,
Zobrist, packing, or action conventions. This is valuable implementation parity
but can preserve a shared semantic mistake. Examples include:

- CPU and GPU hashing the same incorrectly interpreted state bytes;
- manually composed suffix data reproducing the same composition convention;
- target fixtures using involutory swaps, which do not distinguish several
  inverse/action-order mistakes;
- padding checks validating storage consistency without validating logical
  state identity.

Stronger semantic evidence would need an independent puzzle-action oracle,
non-involutory permutations, explicit expected state sequences, and forced
collision fixtures. These are evidence gaps, not an unsolicited implementation
plan.

## Evidence ladder for this component

From weakest to strongest:

1. target is registered in CMake/CTest;
2. source contains assertions;
3. a dated report says the executable passed;
4. report binds source commit, container image, GPU, command, and artifacts;
5. positive component fixture checks an expected witness;
6. negative and failure fixtures check miss/overflow semantics;
7. independent small oracle checks every K1 member, distance, and suffix;
8. forced collisions and malformed generators test exactness preconditions;
9. end-to-end production fixture replays one lineage through all stages;
10. parity across backends, rank counts, and current working-tree revision.

The inspected evidence occupies different rungs for different claims. It must
not be summarized by one global “tests pass” Boolean.

## Current conclusions

1. Existing tests give useful positive component evidence for direct Stream2,
   one manual K1 lookup, and one K2 suffix in each backend.
2. They do not directly test the host K1 reverse-BFS builder or a nonempty K1
   suffix replay.
3. The K2 fixtures bypass production list/composition builders and do not test
   the complete-ball/ordered-prefix premises for first-hit optimality or
   negative bounds.
4. The stitched executable is not a continuous end-to-end candidate lineage.
5. Historical Docker PASS reports are not current dirty-working-tree execution
   evidence and do not record all provenance needed for replay.
6. The strongest open risks are shared action/hash conventions, forced
   collisions, negative-result semantics, overflow, and distributed parity.

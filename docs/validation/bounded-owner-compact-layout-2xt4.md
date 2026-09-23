# Compact accepted-bucket layout: physical T4 leaf gate

Scope: one owner CUDA leaf, not an integrated macro-depth BFS or a multi-rank
exchange. The existing fixed `bucket*K` API remains a wrapper. New
`compare_layout`/`commit_layout` consume immutable device-side offsets and
capacities, so a provisional future arena can reserve only the sum of its
configured bucket extents. A touched bucket is validated against the prefix
directory and total arena records before persistent mutation. Commit remains
separate from compare and requires a granted-row credit.

The fixture uses two unequal contiguous extents (capacities 5 and 3), checks
same-job duplicates, old accepted membership, stable survivor indices and
the final merged hashes. It also injects an out-of-range prefix extent and
an overfull bucket, requiring unchanged persistent counts. This tests
layout arithmetic and failure atomicity for the leaf; it does not test
cross-depth slot reuse, 2*Km history membership, StateRing credits, or
distributed owner ordering.

| Kaggle run | Pinned source | Result |
|---|---|---|
| `trydotatwo/mgbfs-bounded-owner-t4` version 3 | `e085f0a5dcd6de93e77be187e487dca868457e72` | CUB fixture passed plain + memcheck/racecheck/initcheck/synccheck on each of two physical Tesla T4s; 10/10 checks, zero reported errors/hazards/warnings. |
| `trydotatwo/mgbfs-bmma-owner-gate` version 4 | `a98fceb7bf75689ba0de89472573af50b32f427f` | BMMA fixture passed tile limits 1/8/256, each plain + four sanitizer tools (15/15 checks) on 2xT4; SASS contains `BMMA.88128.XOR.POPC`. This source predates the stricter prefix-directory/full-bucket assertions. |
| `trydotatwo/mgbfs-bmma-owner-gate` version 5 | `e085f0a5dcd6de93e77be187e487dca868457e72` | Stricter BMMA compact-layout fixture passed 15/15 checks on two physical T4s; SASS contains `BMMA.88128.XOR.POPC`, racecheck reports zero hazards/errors/warnings. |

The run's `summary.json` is `COMPLETE`; raw outputs are retained on the
private Kaggle notebook and in ignored `target/bounded-owner-layout-v3/`.
BMMA version 5 is separately verified from its terminal `COMPLETE`
`summary.json` and downloaded logs under ignored
`target/bmma-owner-layout-v5/`.

Local Windows `nvcc` compilation was unavailable because `cl.exe` is absent.
`mgbfs-cuda --features cuda` Cargo check also requires a locally built
`MGBFS_CUDA_LIB_DIR`; the Kaggle CUDA compilation and execution are the
authoritative target-device leaf evidence here.

## Multi-layer history extension pending hardware result

Source `80be3aa02544355235f99e0eabe0a57872f0333c` adds a bounded
`compare_history_layout` ABI. Its fixed `history_slots` loop compares one
incoming microbucket job against a flat, immutable `2*Km`-style history arena
and device-side `[slot][bucket]` ranges. The fixture covers three history
slots, duplicate priority, accepted-next membership, stable survivor order,
and an invalid history range that must not mutate accepted counts.

Private Kaggle CUB version 4 completed on two physical T4s: source
`80be3aa`, 10/10 plain and sanitizer checks passed, with zero reported
racecheck hazards/errors/warnings. Its raw logs are in ignored
`target/bounded-owner-history-v4/`. BMMA version 6 at the same source passed
15/15 checks over tile limits 1/8/256, including all four sanitizers, and
SASS contains `BMMA.88128.XOR.POPC`. Raw logs are in ignored
`target/bmma-owner-history-v6/`; both summaries report `COMPLETE` and two
Tesla T4 devices. The implementation is still a leaf and does not establish
history-slot rotation or full BFS.

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

The run's `summary.json` is `COMPLETE`; raw outputs are retained on the
private Kaggle notebook and in ignored `target/bounded-owner-layout-v3/`.
BMMA version 5 is pinned to the stricter `e085f0a` source and is running
separately; its result must be recorded after terminal status and output
inspection, not inferred from version 4.

Local Windows `nvcc` compilation was unavailable because `cl.exe` is absent.
`mgbfs-cuda --features cuda` Cargo check also requires a locally built
`MGBFS_CUDA_LIB_DIR`; the Kaggle CUDA compilation and execution are the
authoritative target-device leaf evidence here.

# Device next-frontier extent publication

The rank-owner DENSE path previously read owner control, ring and one extent
after every batch to build `self.next` on CPU. The new single-writer CUDA
publisher merges physically adjacent committed extents into a preallocated
two-entry device array. A wrap remains a second extent. Overflow or malformed
metadata sets sticky fatal before changing the array. The runtime now reads
the array once at `FinalizeDepth` and collectively propagates fatal before
rank-owner finalization. The two extents and count are charged to the fixed
library VRAM plan before depth zero.

Private Kaggle `trydotatwo/mgbfs-state-commit-t4` v13 at exact source
`5a3167a359a76b15bc3a7e5f6d1d34d6189ea709` failed as expected at link
with `undefined reference to mgbfs_state_publish_next_extent`. The fixture
then corrected its initial ring occupancy before implementation. V14 at
`f8ffb8b3fe79305d4ad6be1ecf084291fb91ad27` completed on two physical
T4s: state-commit and archive-pack tests passed plain, memcheck, racecheck,
initcheck and synccheck; zero reported errors and zero racecheck hazards or
warnings. Raw logs: ignored `test_results/kaggle_next_extent_red_v13/` and
`test_results/kaggle_next_extent_green_v14/`.

This validates the CUDA leaf, not the Rust full BFS integration. The rank
runtime integration is at `b15b74b4c946d7e551e991fa0b0c04087baf9c89`;
private `trydotatwo/mgbfs-cuco-dynamic-shard-refs-t4` v29 at that source
passed the 1-GPU full-state oracle but stopped on an obsolete capacity-test
error-string assertion. The test was updated at
`5f87e241f25356b0317346dcd06e245560a77aea`. V30 at that exact source
passed the full plain gate: both physical T4s passed the 1-GPU full-state
and fatal/recreation tests, the two-GPU NCCL fixture passed layer/archive
oracles, and the two-process CLI S4/U4m2 DENSE/HASH_FIRST verification passed.
Raw logs: ignored `test_results/kaggle_rank_extent_full_bfs_v29/` and
`test_results/kaggle_rank_extent_full_bfs_v30/`.

V31 on the same exact source completed with `summary.status=PASS`. The
rank-owner two-GPU fixture passed plain and all four Compute Sanitizer tools:
two full-state/archive tests passed in every mode, with zero memcheck,
initcheck and synccheck errors and zero racecheck hazards/warnings. The
one-GPU full-state/fatal-recreation suite passed on **each** T4 in all five
modes (three tests per run). Two-process CLI S4/U4m2 verification also ran,
but its scenarios cover cuDF/cuCO-indexed, not the rank owner; the latter is
covered by the two-GPU fixture. Raw logs and manifest:
`test_results/kaggle_rank_extent_full_bfs_sanitizers_v31/` (ignored local
artifact). This is a clean sanitizer gate for the integrated extent change,
not evidence of an asynchronous transport or an Nsight timeline.

Route counts, NCCL payload sizing and parent retirement still require
separate CPU-dependency work.

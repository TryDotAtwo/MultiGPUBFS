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
its full-state 1/2-rank gate and timeline are still required. Route counts,
NCCL payload sizing and parent retirement remain CPU-dependent.

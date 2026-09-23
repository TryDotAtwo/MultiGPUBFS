# cuCO GPU-selected shard references: T4 gate

Private Kaggle `trydotatwo/mgbfs-cuco-dynamic-shard-refs-t4` v1 completed at
source `e836a11bb9870a6b1b031b7258ad049ded2a89b5` with NVIDIA
cuCollections `532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4`.

The focused owner probe constructed two distinct persistent static sets,
uploaded their `contains` refs to a GPU array, and chose the ref per candidate
from the high hash bit inside a CUDA kernel. Its full-Hash128 test returned
`[hit, hit, miss]` on both physical T4s. The existing indexed owner contract
also passed. On each T4, memcheck, racecheck, initcheck and synccheck completed
with zero reported errors/hazards for that probe.

Raw logs and summary: `test_results/kaggle_cuco_ref_probe_v1/library-owner/`.
The notebook deliberately disabled its full BFS gate and load screen. This
establishes the feasibility of dynamic table-ref selection only, not an
implemented rank-batch owner, removal of host synchronization or a speed gain.

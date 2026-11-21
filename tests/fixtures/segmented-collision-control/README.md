# TEST-ONLY: forced two-shard hash collision and replay

This fixture is intentionally invasive. It changes the hash and forces a CUDA
scheduling interleaving. Never include its patch or build flags in a release.
Production sources do not enable this fixture.

To reproduce, use an isolated checkout of the commit containing this directory.
Build a matched executable and CUDA library after applying the fixture patch.
Use a CUDA toolkit, CMake, Rust, CUTLASS and NCCL compatible with the main
project. Set CUTLASS_ROOT and NCCL_ROOT to their installed locations and
CUDA_ARCH to your GPU architecture (the recorded physical run used SM86).

    git apply tests/fixtures/segmented-collision-control/segmented_collision_control.patch
    cmake -S cuda -B /tmp/mgbfs-collision-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_CUDA_FLAGS=-DMGBFS_TEST_HASH_TWO_SHARDS=1 -DCUTLASS_ROOT="$CUTLASS_ROOT" -DMGBFS_NCCL_ROOT="$NCCL_ROOT"
    cmake --build /tmp/mgbfs-collision-cuda --target mgbfs_cuda
    MGBFS_CUDA_LIB_DIR=/tmp/mgbfs-collision-cuda MGBFS_CUDART_LIB_DIR=/usr/local/cuda/lib64 cargo build --locked --release -p mgbfs-cli --features cuda
    export MGBFS_EXECUTABLE="$PWD/target/release/mgbfs"
    export LD_LIBRARY_PATH="/tmp/mgbfs-collision-cuda:$NCCL_ROOT/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    export PYTHONPATH="$PWD"
    python tests/fixtures/segmented-collision-control/collision_control_search.py

The search requires two idle physical GPUs and prints the output directory.
The test-only executable and library must both come from this patched checkout.
Use that output directory as the single argument to
tests/fixtures/segmented-collision-control/verify_collision_hit.py. It reruns
the trigger case, materializes every completed layer, compares exact CPU
multisets, and checks per-rank routing and the forced full-hash collision.
The search may report CONTROL_GATE_OPEN when a schedule is not found; that is
not a correctness pass. See the acceptance evidence for the recorded test
build commands, library/executable hashes and successful packed/wide runs.

The physical tests force a schedule and hash collision; they do not establish
that the same interleaving naturally occurred on production hardware. The
fixture writes bounded test jobs under MGBFS_COLLISION_OUTPUT, or /tmp if unset.

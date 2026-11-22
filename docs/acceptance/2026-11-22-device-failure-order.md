# Experimental GPU epoch: failure precedence

The experimental two-phase device decision makes the stop request provisional. A
rank publishes its final failure bits only after its local writers stop; the
coordinator waits for all participating ranks before publishing the result.
Fatal or unknown failures take precedence over a resource stop. A lost rank
still requires the independent supervisor to cancel the generation; the helper
alone does not implement that supervision.

On the same 2 x Tesla V100 SM70 / CUDA 12.8 / NCCL 2.30.7 machine, a test-only
fault schedule made rank 0 report a resource stop and delayed rank 1's fatal
error until that stop was observed. Before this change both ranks returned
INCOMPLETE (3). With this change both returned ERROR (4). The same schedule
passed on 4 physical GPUs; ranks 2 and 3 had no local error, and all four
reported ERROR. This is a controlled ordering test, not a claim that a real
rank-loss event was reproduced.

The existing 2-GPU tests for normal completion, exact layers and a clean
resource stop also passed. Commands (after configuring the test build with the
matching CUDA/NCCL paths and SM70):

    cmake --build /workspace/mgbfs-device-build70 -j4
    ctest --test-dir /workspace/mgbfs-device-build70 --output-on-failure \
      -R 'device-(packet-bfs|pack4-(segmented|dynamic|real-segments|mixed-failure))-experimental'

The 4-GPU mixed-order fixture was compiled with the same source and
MGBFS_EXPERIMENTAL_WORLD=4, then run on devices 0-3. Its per-rank log and the
before/after logs are in the saved evidence archive. This fixture and the
helper are not connected to the public CayleyPy entry point or the production
persistent runtime. The test-only injected faults and finite spin thresholds
must not be treated as production behaviour; actual supervisor failure handling
and public-API integration remain open.

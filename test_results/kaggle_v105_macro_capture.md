# Existing macro producer: T4 capture evidence

Worker COMPLETE, source7815d1766b4037054f52f68f67b948fb593dc51e,
root macro_producer_capture=PASS. Exact Rust lib test:
macro_native::producer_capture_tests::macro_produce_captures_and_runs_without_host_count_readback.

It constructs S4, K=2, batch=1; advances to three parents so both producer
banks are reused; captures the actual producer DAG, instantiates and launches
the graph, checks all device future counts against conservative bounds, then
settles depth2 and compares its full state set to the CPU matrix oracle.
The gate rejects cargo output with zero matching tests.

Two physical T4/P2P1 were present, but this test executes one-rank macro
code. It does not prove distributed macro dispatch, full settlement capture,
all four sanitizers, archive publication, or performance. These remain open.
Root retained at build/kaggle-v105-summary; small logs being retained at
build/kaggle-v105-observation before any notebook replacement.

Separate existing sourcecb4603c S8 timeline inspection: all12 owner/profile
rank traces have only returnValue=0 cudaEventQuery calls wholly contained
in same-thread mgbfs.batch ranges. Per-trace calls122–211 and API duration
0.806–1.473ms. This does not indicate the one-millisecond not-ready polling
branch was used in these batch ranges; do not claim speeding up that branch
would improve these recorded runs. FinalizeDepth and other calls excluded.

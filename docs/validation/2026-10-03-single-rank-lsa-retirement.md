# Single-rank LSA terminal retirement

Base `49135b2`. RED test
`one_rank_lsa_abort_retires_local_readers_without_peer_callback` runs actual
native CUB/LSA generation and owner work before explicitly aborting. It failed
with status12, not a CUDA memory error: terminal_abort required a retirement
callback even when world1 has no peer reader. Its defensive retirement_failed
path intentionally retained the communicator/window allocations.

Fix: after successful NCCL revoke and successful cudaDeviceSynchronize, world1
may acknowledge its own reader retirement without a peer callback. If a callback
is present it is still honored. World>1 still rejects missing callbacks and
requires the full peer ACK loop. No healthy-batch GPU drain/readback was added;
the existing terminal-error drain is used. No multi-rank safety relaxation is
claimed or intended.

GREEN on RTX3070Laptop/sm86: abort status0, repeated completed abort status0.
Unfiltered Compute Sanitizer2025.1.0.0 memcheck/initcheck/synccheck/racecheck
each exit0, zero errors/hazards. Separate memcheck --leak-check full exits0,
zero bytes leaked in zero allocations. Logs retained under
`build/single-rank-retirement-20261003/` (ignored).

Retested CUCO_RANK/LSA U4/F2 banks2/3/4 in DENSE/scalar HASH_FIRST: capacity
failure/latching plus full-state/hash/archive checks pass all twelve cases.
Capacity abort now prints status0 in each case. Full CPU cargo test --workspace
passed. The pre-existing sm86 rejection of the Tensor case in the complete GPU
file remains as recorded in the capacity receipt; no whole-file green claim.

Build: existing CUDA12.8/sm86 native cache, library-owner12.9/sm86, isolated
NCCL2.29.7/sm86. Native SHA256:
`89a25dd5961964f4e4898bf6603ea3b13764aea79e819f60ca0d4ab0f819a851`.
CUDA-only unit binary SHA256:
`2bd7062877b0c8934bac38ec579c8382be60f2a646db8f7455f350482715b4f6`.
Initial rebuild used the wrong source mount path and failed before compiling;
initial sanitizer command used a nonexistent path and did not run. Both were
corrected without changing the test or suppressing diagnostics.

These are world1 targeted gates, not asymmetric two-rank teardown, full BFS
sanitizer acceptance on T4, a timeline, or A/B performance evidence.

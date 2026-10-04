# Pair timing audit

Pinned diagnostic dataset revision:8977998a67d4a7b72bcc873a9ff545db1abb538e.
Run20261003-full-grid-small-capacity:147 attempted cases, all147 manifests read.
Sum of completed-layer maximum-rank durations:13.372343 seconds.
Mean per attempted case:0.0909683197 seconds; median0.083718 seconds.
Largest recorded completed-layer total:n32r30,0.260734 seconds,39 layers.
This deliberately256-row/rank diagnostic is not a large-graph benchmark.

The old ledger has no runner start/end or transition timestamps. Its exact
inter-pair overhead cannot be recovered after rental deletion. Layer sums omit
unfinished layers, capacity probes, graph calibration, process/CUDA/NCCL startup,
CPU oracle, final archival work and publication. HF commit timestamps are not
valid substitutes for execution timestamps because publication is asynchronous.

New scheduler telemetry records runner wall time, the interval between adjacent
runners (including ledger/progress/backpressure), all-rank production-search time
when reports exist, and the completed-layer total. Admission and graph calibration
wall times are recorded separately in launch configuration. These timers execute
only at case/phase boundaries and introduce no GPU synchronization. First
transition is null; incomplete/missing rank timings remain null. Final publication
and first global setup require a separate outer wall timer if comparing total job.

15 automatic-launch and13 sweep tests pass. Controlled-clock scheduler test
separates2 seconds runner wall,3 seconds inter-case progress and0.25 seconds
native search, including every rank. New timers are not yet measured on a GPU run.

# Current local full-BFS timeline

Main `f43fd21`, one RTX3070 Laptop sm86, U4(F2), six cases: DENSE/scalar
HASH_FIRST, banks2/3/4, K3, batch1, memory-backed archive.
Updated existing `/linux-build/route-banks-sm86-nvtx` from current source.
Test binary from `/linux-build/target-fixture-guard`.

Initial profile used non-NVTX library and failed before BFS with
PROFILE_NVTX_NOT_BUILT; retained `banks.nsys-rep` is NOT acceptance evidence.
Corrected profile `banks-nvtx.nsys-rep` succeeded, fixture 4.02 seconds;
SQLite export `build/nsys-current-20261004/banks-nvtx.sqlite`.
Trace cuda,nvtx; CPU sample/context-switch disabled. No CUDA_CALLCHAINS table:
callsite attribution is unavailable, not silently inferred.

NVTX:384 batch,384 archive_d2h,54 FinalizeDepth ranges. Same-thread CUDA API
intervals contained within batch:274 EventQuery,2546 EventRecord,19976 kernel
launch,384 Memcpy2DAsync,768 MemcpyAsync,1152 MemsetAsync,1344 StreamWaitEvent.
No StreamSynchronize, EventSynchronize or blocking Memcpy in those ranges.
EventQuery remains the bounded completion-credit admission poll.

Correlated copies in batch:768 D2H copies of16B, ALL within archive_d2h;
384 kind8 copies of64B. No additional D2H count/control copies observed in
batch. Range/correlation evidence is not full callchain proof and does not
cover initialization/finalization or a multi-rank NCCL issue path.

GPU recorded span3099004638ns; busy141633629ns; multi-stream4819965ns;
compute/copy overlap1211697ns;25194 GPU events. This tiny batch1 correctness
fixture is NOT saturation evidence: observed overlap is limited. These are
recorded intervals, not occupancy, FLOPS, whole-search critical path or paired
performance numbers. Real target-hardware representative workload still needed.

Original raw reports stay in ignored build directory. No target2T4/registration
initcheck, asymmetric rank failures, ideal overlap or A/B gate closed here.

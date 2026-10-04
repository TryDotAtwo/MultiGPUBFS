# v90 retained-archive recovery

Recovery notebook `trydotatwo/mgbfs-v90-artifact-recovery` v1 completed.
Original source: `749c691363007836969536c8629e9b30bc92e842`.
Verifier source: `11338625e6bda5b7d22b939bb1668e1f15c7efe0`.
Original summary SHA256: `664e545880881f022f2fe52c88858a87ecaeffa5c84ca84ce1dcee141b7d3cdc`.

Full canonical states at every depth, archive checksums and group commits verified
for three retained initcheck cases: DENSE banks2 repeats0/2 and HASH_FIRST banks4
repeat0. Each contains 24 states with layers `[1,3,5,6,5,3,1]`. Their two rank
instrumentation logs also passed the existing clean-log check. This does not
reconstruct the original process exit codes.

Both retained timeline cases, DENSE and HASH_FIRST, passed the same full-state
oracle: 64 U4/F2 states, layers `[1,3,5,8,11,13,13,8,2]`, route-bank reuses
`[11,12]`. The other 15 initcheck cases lack a valid archive chain; their original
NCCL activation failures remain open. The overall sanitizer gate is NOT PASS.

Four raw Nsight reports were downloaded without state archives, then exported
with the existing Docker Nsight Systems 2025.6.3 tool. Files remain under
`build/kaggle-v90-timelines/lsa-bfs-gate/timeline-*/healthy-None/`.
Recovery JSON/log remain under `build/kaggle-v90-recovery/`.

Whole-capture CUDA API counts (not hot-path attribution):

| Profile/rank | EventSynchronize | StreamSynchronize | synchronous Memcpy |
|---|---:|---:|---:|
| DENSE/0 | 163 | 80 | 147 |
| DENSE/1 | 161 | 77 | 147 |
| HASH_FIRST/0 | 163 | 83 | 144 |
| HASH_FIRST/1 | 161 | 80 | 144 |

The callchain table exists, but the current analyzer yields no symbols for these
calls. Table existence alone does not establish usable attribution. Startup,
FinalizeDepth, archive-worker and teardown calls must be separated from healthy
batch submission before claiming the CPU-readback requirement is satisfied.
No performance or complete-pipeline acceptance claim is made here.

Thread-matched NVTX attribution on all four databases: inside `mgbfs.batch`
there are zero runtime host synchronize calls and zero synchronous memcpy calls.
The only batch D2H copies also belong to nested `mgbfs.archive_d2h`: DENSE and
HASH_FIRST rank0 each 66 copies/1056 bytes; rank1 each 62 copies/992 bytes.
HASH_FIRST additionally has 216 device-to-device copies/9504 bytes per rank.
`mgbfs.FinalizeDepth` contains 45 StreamSynchronize and 108 synchronous Memcpy
calls per rank. These are measured phase boundaries, not evidence of a batch
readback. This attribution uses CPU thread and fully contained API intervals;
it does not yet classify inter-batch gaps or partially intersecting API calls.

Follow-up interval inspection: the 8 main-thread StreamSynchronize calls outside
annotated ranges occur after the prior FinalizeDepth and before the next depth's
first batch. The associated 24 synchronous copies are also depth-transition
control. Archive worker EventSynchronize calls run on a different CPU thread
(33 rank0 / 31 rank1). Thus whole-capture counts must not be reported as
per-batch readback. API containment evidence is still not a proof about every
possible configuration or performance at large frontier sizes.

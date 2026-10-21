# dev32: measured NCCL admission and verified startup transport

The B200 report failed in NCCL allocation after the BFS arena had consumed nearly all free memory. This release measures NCCL device memory with all selected ranks before BFS memory admission. It exercises all-peer bounded-triplet traffic and scalar collectives, verifies peer payload boundary signatures, then reserves the maximum measured per-rank amount plus 512 MiB, with a 2 GiB floor and no reduction of a larger explicit reserve. Scratch buffers are excluded from the measurement.

The RTX failure was reported by an asynchronous CUDA copy after exchange/owner processing. GDRDMA was present on the failing node and absent on the successful node's corresponding links; this is a transport hypothesis, not proof of defective hardware or the exact original CUDA999 origin.

Before autotuning, fresh disposable ranks check the inherited transport and the first four BFS layers. If that fails, host-staged IB and then local SHM are tested. A transport is accepted only after exchange readback and a successful graph prefix. Explicit operator NCCL transport settings are never overridden. If every candidate fails, launch stops with evidence; it does not substitute an untuned BFS profile. Full BFS failures are never automatically restarted. The ordinary measured autotuner still runs after this startup check.

Persistent workers reuse the verified communication result for the same executable, devices and relevant environment. The report identifies the graph used for the initial rehearsal; reuse is not proof for every later graph. First-use evidence is saved beside the result directory with the suffix `-communication-evidence`. The final report includes `communication_preflight`, selected policy, measured bytes and failed attempts.

Native CLI/CUDA/NCCL components and their hashes remain those of dev31. SM100/SM120 cached components need no download or rebuild. This is a Python orchestration and admission change.

Validation scope: cheap 2xRTX3060; synthetic B200-sized measured-memory regression. No B200 or original failing RTX node was available for acceptance. The startup calibration is bounded; production BFS remains unlimited by default. Calibration cannot guarantee that later workloads never allocate additional driver/NCCL memory or encounter hardware faults.

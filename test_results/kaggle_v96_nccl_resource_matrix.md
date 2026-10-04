# v96 NCCL resource matrix and actual instrumenter

Worker COMPLETE; report DIAGNOSTIC_COMPLETE, not BFS acceptance.
Notebook `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, source
`50a4cd42809a2376d8e4d0086563e5e6d85bd87c`.
Two independent processes on physical 2xT4, P2P admitted both directions.
Actual driver: 580.178.04; compiler: CUDA12.9 V12.9.86;
Compute Sanitizer: 2025.1.0.0, build35583870, public-release.
These versions describe this host, not previous or subsequent hosts.
Summary SHA256:
`6915b880d24328f835ebd5208fd9917a1067637e9966bf854f4f0bdbad25e6ab`.

| Mode | Plain | Memcheck | Racecheck | Initcheck | Synccheck |
|---|---|---|---|---|---|
| User window | PASS | PASS | PASS | PASS | PASS |
| Zero-resource device communicator, no user window | PASS | PASS | PASS | PASS | PASS |
| 16-barrier device communicator, no user window | PASS | PASS | PASS | FAIL [12,12] | PASS |
| User window + 16 barriers | PASS | PASS | PASS | PASS | PASS |

All 20 process pairs terminate without timeout. In the failing device-only
initcheck pair rank1 reports dev_runtime.cc:1037 unspecified launch failure
while initializing internal resources. The window-plus-device pair contains
both successful activation markers and zero exits. No BFS kernel executed.

The production-shaped window-plus-device path passed on this host, but this
does not resolve earlier activation failures or waive full-runtime initcheck.
The matrix contradicts an unconditional claim that 16 barriers always fail;
registration/order/address-sensitive instrumentation remains a hypothesis,
not a proven cause or implemented fix. Next gate must execute actual BFS,
asymmetric errors, archives and all tools on the updated runtime.

All small outputs and 40 rank logs retained in build/kaggle-v96-observation
before any subsequent submission.

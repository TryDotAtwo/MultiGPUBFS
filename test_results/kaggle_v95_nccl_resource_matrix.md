# v95 NCCL resource matrix

Worker COMPLETE; report DIAGNOSTIC_COMPLETE, not full BFS acceptance.
Notebook: `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`.
Source: `629cf36d48253b21f1ed928eda7da9f89b8d58cf`.
Two independent processes on two physical Tesla T4 GPUs, bidirectional P2P admitted.
Experimental NCCL library SHA256:
`93aefd5f5f47a897ecfbf1bcd236826f2e3044362f6db56668e54dbe3cdab4e3`.
Summary SHA256:
`9c5a3bfa48f8a0cac5197da1787b1478a04d1807114e783730d994d2eb1e9b1e`.

| Mode | Plain | Memcheck | Racecheck | Initcheck | Synccheck |
|---|---|---|---|---|---|
| User window | PASS | PASS | PASS | FAIL [7,7] | PASS |
| Device communicator, zero resources, no user window | PASS | PASS | PASS | PASS | PASS |
| Device communicator, 16 barriers, no user window | PASS | PASS | PASS | FAIL [12,12] | PASS |
| User window + 16 barriers | PASS | PASS | PASS | FAIL [7,7] before activation | PASS |

All 20 process pairs terminated without supervisor timeout. The corrected
finalize-before-destroy policy passed in the zero-resource and 16-barrier
controls outside initcheck. No BFS kernels executed.

The 16-resource initcheck rank-1 failure starts at `dev_runtime.cc:1037`,
an unspecified launch failure during internal resource initialization,
followed by cleanup failures. The user-window initcheck failure occurs earlier
during registration, unlike v94 where registration passed. Consequently this
does not establish that barriers alone cause the blocker. Zero reported
instrumentation errors do not turn a failed application into a passed gate.

The compiler is pinned CUDA 12.9; the actual host Compute Sanitizer version
was not recorded by this source and remains unknown. HEAD a01f7db adds actual
driver/compiler/instrumenter version capture for subsequent runs. Toolchain
attribution is needed before a dependency workaround; no gate is waived.

All small outputs and 40 rank logs were downloaded to
`build/kaggle-v95-observation` before any subsequent notebook submission.

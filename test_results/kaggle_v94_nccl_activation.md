# v94 reduced NCCL activation matrix

Notebook `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, v94, worker COMPLETE;
report DIAGNOSTIC_COMPLETE, not BFS acceptance. Source:
`98ccc5882bf24200533ab7e3162511e8627952a3`.
Two independent rank processes, physical 2xT4, both P2P directions admitted.
Experimental NCCL2.29.7/POSIX/sm75 library SHA256:
`8a1d53dd318b7706250a64f382e99949dd6159a8f880ba94329f6f8ca741f1e9`.
Summary SHA256:
`2369d78dc8ea30712c06f64ebc3badc7351a3d94d8e3c53e35ac303aa2d677db`.

| Stage | Plain | Memcheck | Racecheck | Initcheck | Synccheck |
|---|---|---|---|---|---|
| User window registration | PASS | PASS | PASS | PASS | PASS |
| Internal device comm without user window | activation PASS, teardown FAIL | same | same | activation FAIL | activation PASS, teardown FAIL |
| User window + device comm | PASS | PASS | PASS | activation FAIL | PASS |

All process pairs completed without supervisor timeout. Initcheck activation
pairs exited [12,12]; rank1 reports `dev_runtime.cc:1037` unspecified launch
failure (internal resource initialization stream sync), then cleanup failures.
Rank0 receives an NCCL system error after peer failure. Error summaries are
zero but do not constitute successful activation. No BFS kernels ran.

The device-only control's plain/other-tools exit [14,14] occurs *after* the
activation PASS marker. Its harness erroneously calls the progress poll on
`ncclCommDestroy` success and thus queries an already destroyed handle:
`misc/argcheck.cc:41`, corrupted comm object. Do not label this an activation
failure or waive its failed teardown. Correct finalize-before-destroy and
never query the destroyed handle before rerunning this control.

This matrix isolates a reproducible initcheck failure to NCCL internal device
resource activation on this host/build. It does not prove the general IPC
limitation is its cause, does not validate BFS, and does not close initcheck.
Next discriminating control: zero LSA barrier resources versus the production
16-resource request, preserving two processes and unfiltered instrumentation.
All 30 rank logs and available small outputs retained in
`build/kaggle-v94-observation` before any new submission.

# Production warmup v103 and direct VMM v6

Both workers COMPLETE. v103 report TYPED_WARMUP_PASS, source4490ef4:
two independent rank-process production RunConfigV1 DENSE/HASH_FIRST
warmup+measure runs with full S4 archive oracle and three source banks/
completion credits. Both nested suites pass, all36 asymmetric injected
faults pass. Warmup-setting-disagreement cases added after this source
remain unverified on hardware. All small JSON/rank logs retained at
`build/kaggle-v103-observation` before replacing the notebook.

Second notebook v6 source38a4c25: direct CUDA VMM diagnostic, no NCCL/BFS.
All20 local/import × Driver-before-import/Runtime-async-after-import ×
plain/four-tool pairs pass, zero timeouts and exit0/0. All40 rank logs
retained at `build/kaggle-import-v6-observation`. This narrows the original
initcheck hypothesis further but does not close the full BFS gate.

Next discriminating test preserves these controls and adds the symmetric
address reservation/stride used by NCCL2.29.7. No production resources,
fatal votes or sanitizer reporting are disabled.

# Kaggle v89: unsupported LSA host

Observed 2026-10-04. Source `749c691363007836969536c8629e9b30bc92e842`;
notebook `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, version 89.
Worker COMPLETE, report UNSUPPORTED_HOST. Both physical GPUs are Tesla T4,
but CUDA peer-access queries return allowed=0 in both directions (query status 0).
The preflight correctly stopped before build/search. No initcheck or timeline
acceptance evidence was produced. Do not weaken the P2P admission requirement.

Logs and JSON: `build/kaggle-v89-observation/`. Repeat the unchanged package
only after verifying this execution is terminal; no concurrent Kaggle job.

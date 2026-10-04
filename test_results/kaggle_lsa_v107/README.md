# Kaggle LSA v107, 2026-10-04

Runtime base: 7877c27086bff6dad635700f949a8bc4ba5485bc.
Notebook: trydotatwo/mgbfs-lsa-full-bfs-gate-t4, version 107.
Harness preflight environment regression reproduced (1 failure of 16), then fixed (16/16 pass). These are Python harness tests, not the full project suite.
Physical two-T4 P2P admission returned allowed=0 in both directions. Notebook correctly stopped UNSUPPORTED_HOST before CUDA/NCCL build/search. No A/B measurements, full-state GPU validation, sanitizer or timing evidence claimed for this run.
No paid instance was started. Files copied directly from Kaggle API into GitHub; no local datasets or builds.

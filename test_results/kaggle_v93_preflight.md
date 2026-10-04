# Kaggle v93: P2P unavailable

Notebook `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, v93, worker COMPLETE.
Pinned source `98ccc5882bf24200533ab7e3162511e8627952a3`.
Actual run status `UNSUPPORTED_HOST`: two Tesla T4s, but
`cudaDeviceCanAccessPeer` returned allowed=0 in both directions (API status=0).
No NCCL probe, sanitizer or BFS execution took place. No acceptance claim.

Summary SHA256:
`6c6dc7965f082f552a4c5d7d8c975b5cecae677dc1fa851e03ce9c2382d1724`.
All available logs/summary downloaded into `build/kaggle-v93-observation`.
An unchanged retry is necessary to obtain a P2P-admitted Kaggle host; this is
not a runtime fallback or a change to the requested hardware gate.

Separately, production wrapper commit `647dc13` passed the deterministic
RED/GREEN terminal-admission regression and actual nvcc sm75 compilation,
both without LSA against installed NCCL headers and with LSA against the
retained NCCL 2.29.7 wheel. Compilation is not linked hardware execution.

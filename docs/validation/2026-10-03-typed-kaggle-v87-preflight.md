# Typed production gate: Kaggle v87

Notebook: `trydotatwo/mgbfs-lsa-full-bfs-gate-t4`, version87. Kaggle worker
reported COMPLETE; the runtime gate reported **UNSUPPORTED_HOST**, not PASS.

Runtime source: `12391050694ad1b50669f2eee3490fd68cd1e0e1`.
Gate recipe: public `3a3531c` (typed_rank_gate).

Two distinct Tesla T4, each15360MiB total and14912MiB free, were admitted.
Both `cudaDeviceCanAccessPeer` calls succeeded as API calls (status0), but
returned allowed0: rank0→rank1 and rank1→rank0. LSA requires real peer access.
The gate therefore stopped before dependency builds and BFS launches.

No typed CUCO correctness, fault, sanitizer or performance result was produced.
There was no CPU/host-sized fallback and no duplicate notebook launch. This
does not establish a BFS algorithm failure and must not be counted as GPU PASS.

Small downloaded evidence is retained under `build/kaggle-typed-v87/`, including
`lsa-bfs-gate/summary.json`, `inventory.log` and the notebook log. No large
dataset was downloaded. The current typed runtime still needs an admitted
two-T4 P2P host; earlier CUCO reference-benchmark results do not prove this path.

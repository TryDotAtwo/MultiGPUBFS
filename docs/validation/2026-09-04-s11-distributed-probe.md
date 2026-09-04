# Physical 2xT4 S11 capacity probe

Kaggle kernel `trydotatwo/mgbfs-s11-distributed-probe`, version 2, exhausted
the S11 matrix Cayley graph on two physical Tesla T4 devices.

- Source: `3f66af2a8e5de3ecbac208f3c09c676b03be47f2`.
- Exact result: 39,916,800 unique states in 56 layers.
- Search completion: 4.164070 seconds.
- Durable lossless state + Hash128 archive: 47.956265 seconds.
- Peak frontier: 3,049,721 states at depth 37.
- Peak-frontier owner split: 1,524,287 / 1,525,434.
- External peak VRAM: 7,107 MiB per rank, 14,214 MiB total.
- Runtime allocation reading: 7,452,098,560 bytes per rank.
- Pinned archive allocation: 3,142,451,200 bytes per rank.
- Preallocated disk extent: 5,535,710,464 bytes per rank.

The layer sum equals `11!`; both ranks completed with matching wall time and the
owner split stayed balanced. This is one capacity/correctness run, not a
five-repeat performance comparison. The result also shows that S11 is still a
seconds-scale search workload; archive durability, not GPU search, dominates
the wall time.

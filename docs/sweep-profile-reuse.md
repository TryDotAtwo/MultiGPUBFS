# Reusing tuning across a graph sweep

`run_graph(..., autotune=True)` automatically consults the persistent compatible-family cache. The first substantial graph in a family measures admitted candidates. Later compatible `(n,r)` pairs reuse dimensionless modes, shard count and batch fractions, with fresh memory admission for each graph. Production again queries live VRAM. Absolute capacities and donor batch sizes are never imported.

A family separates action type, power-of-two element-size buckets, state encoding and packed-permutation boundaries, generator count, retained-history depth, exact matrix dimensions/moduli, exact GEMM eligibility and LRX/specialized eligibility. Profiles also bind selected GPUs, driver/topology, native binary and library dependencies, selector code, force flags and transport environment. An incompatible family or rejected admission runs the normal bounded tuner. A family profile is a starting policy, not a globally optimal target-graph measurement.

Receipts use `REUSED_COMPATIBLE_GPU_PROFILE`, `measured=false`, donor graph digest/identity and source scores. Small/large tiers are marked `REUSED_SOURCE_SIZE_PROFILE`; live chunk adaptation remains enabled. No target prefix pilots execute for a successful shared hit. Tiny graphs still take the conservative path. Separate GPU counts have separate caches.

Use `MGBFS_PROFILE_CACHE` to retain the cache on a persistent disk across invocations. Ordinary default launches reuse it automatically; a deleted machine with no restored cache must measure again. Do not claim that arbitrary state encodings or unsupported specialized modes share one measured profile.

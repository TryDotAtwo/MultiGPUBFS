# One-rank path in the distributed reference engine (GPU pending)

The same `DistributedNativeBfs` now accepts world size 1 or 2. Single-rank
mapping is `[0,0]`: both high-bit hash ranges belong to rank 0. Global bucket
and shard counts are not halved. It uses the full-prefix bucket directory,
routes all sorted records to local owner commit, and issues no nonexistent
peer send/recv or remote materialization round trip. Generation, deduplication,
StateRing, archive, failure guards and finalization remain the shared code path.
The example launcher accepts `WORLD_SIZE=1` and reports world size explicitly.

CPU topology tests verify geometry and invalid maps. CUDA-feature Rust check
passes. A new real-GPU archive fixture runs the three matrix profile/generation
choices times two owners times two pre-dedup choices, plus four compact DENSE
variants: 16 one-rank configurations. Each decodes every archived state/hash
and compares complete layers with the independent small-group oracle, as the
existing two-rank fixture does. This fixture is not yet run on hardware.

Scope limitations: not an arbitrary-N-rank router; conservative bounded peer
buffers still exist in the one-rank allocation plan. The Python paired benchmark
orchestrator still assumes two ranks and needs a separate tested update before
one-vs-two-rank performance claims. No speed or VRAM improvement is claimed
from compilation.

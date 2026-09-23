# Device-count cuCO rank-batch owner: two-T4 gate

Private Kaggle `trydotatwo/mgbfs-cuco-dynamic-shard-refs-t4` v5 completed at
`a47ba0276486b780f5fcb3f8f1c9e222bea35a3a` with pinned cuCollections
`532795b81e72e3fe4ce2b26eb0c5abc8abb1e2b4`. The full BFS gate and load
screen were deliberately disabled. The probe passed independently on both
physical T4s in plain, memcheck, racecheck, initcheck and synccheck modes;
the sanitizer logs report zero errors/hazards.

The test captures compare → GPU shard counts → all-shard StateRing reservation
→ persistent commit as one CUDA Graph and executes it with no survivor-count
host readback. It verifies deterministic first-source ordinals and accepted
Hash128 keys across two shards, then reuses the same accepted tables in a
second batch. A third batch overflows one shard's accepted capacity: both
rank-level fatal and unchanged accepted counts/StateRing tail are checked,
with no new persistent keys published. v2 was the expected RED link failure
for the missing class; v3 caught a cuCO mutable-ref compile error; v4 passed
the initial one-batch gate; v5 adds graph capture, reuse and overflow.

Raw logs: `test_results/kaggle_cuco_rank_v2_red/`,
`test_results/kaggle_cuco_rank_v3/`, `test_results/kaggle_cuco_rank_v4/` and
`test_results/kaggle_cuco_rank_v5/`.

This is a library-level candidate, not an integrated BFS runtime. It lacks a
stable C ABI, production memory-plan admission, DENSE/HASH_FIRST state
materialization, transport and retirement integration, full-layer equality,
large-graph performance and VRAM evidence. The synchronous V1 owner is still
the runtime path.

# Device-only logical-owner route window: physical 2×T4 gate

The packer publishes `owner_counts` in logical-owner order. The new
`mgbfs_owner_window_from_counts` derives the packed `begin/rows` window on
device, without a host count readback. The fixture covers all four logical
owners in reversed invocation order, an empty owner, total over-capacity and
the packer's sticky `UINT32_MAX` source-reference failure. Invalid counts
publish `rows=UINT32_MAX`; the downstream owner window must treat it as fatal.

| Private Kaggle run | Exact source | Result |
|---|---|---|
| `trydotatwo/mgbfs-device-count-owner-pack-gate` v4 | `2f0986044bb7535ce12d525b9cfdc003290144ad` | Expected RED: link failed with undefined reference to `mgbfs_owner_window_from_counts`. |
| Same notebook v5 | `064185ae2784e8e627bde930c8acddbeb35ac350` | Both physical T4s passed plain plus `memcheck`, `racecheck`, `initcheck`, `synccheck`. Memcheck/initcheck/synccheck had zero errors; racecheck reported zero hazards, errors and warnings. |
| Same notebook v6 | `f653836957fa7102fbe84fe34abc14b718608841` | Expected RED: C-linkage signature mismatch after requiring device-side `sum(owner_counts) == routed_count`. |
| Same notebook v7 | `35c6db7d668fa0f39a5bce2af5e572e81ef24992` | Both T4s passed the equality/overflow/sentinel fixture in plain and all four sanitizer modes; zero reported errors, hazards and warnings. |

Raw outputs: ignored `test_results/kaggle_owner_window_red_v4/` and
`test_results/kaggle_owner_window_green_v5/`,
`test_results/kaggle_owner_count_mismatch_red_v6/` and
`test_results/kaggle_owner_count_mismatch_green_v7/`.

The equality check is required before removing the old host-side
`packed_count` guard. The producer's `route_count` is the authoritative
device word; checking only `sum(owner_counts) <= capacity` would permit a
short partition to silently drop candidates.

The AoS→SoA window bridge returns **absolute** source ordinals. Therefore
`mgbfs_state_materialize_rank_batch` must receive the total source-row count,
not the owner-window row count, when the window begins after zero. Private
`trydotatwo/mgbfs-cuco-dynamic-shard-refs-t4` v21 at source
`9b8e28221770a91cb0d260bb197adb6031a5b9c7` passed the actual
nonzero-window materialization fixture on both physical T4s in plain mode.
Raw output: ignored `test_results/kaggle_absolute_source_materialize_v21/`.

This is a route-window leaf gate, **not** evidence that the Rust BFS scheduler
uses the device-only path. Host-sized NCCL exchange, owner control snapshots
and CPU retirement still remain in the current runtime.

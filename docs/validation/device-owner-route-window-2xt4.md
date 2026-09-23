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
The same source in v22 passed plain plus all four Compute Sanitizer tools
on both T4s: zero reported errors and zero racecheck hazards/warnings.
Raw output: ignored `test_results/kaggle_absolute_source_materialize_v21/`
and `test_results/kaggle_absolute_source_materialize_sanitizers_v22/`.

The rank-owner candidate-copy guard was tested separately. At source
`486a5a390287e8001930b535cf536a756a00c399`, private v23 reproduced
`RANK_FATAL_MUST_NOT_READ_CANDIDATES`: after a sticky capacity fatal, another
queued compare overwrote the shared candidate plane. Source
`540698c1cab3b50c8d966bb66390f66d7da71666` added a device-side early
return in `copy_candidates`. Private v24 passed the fixture in plain mode on
both T4s; v25 passed plain and all four Compute Sanitizer modes on both T4s,
with zero errors and zero racecheck hazards/warnings. Raw outputs: ignored
`test_results/kaggle_owner_fatal_scratch_red_v23/`,
`test_results/kaggle_owner_fatal_scratch_green_v24/`, and
`test_results/kaggle_owner_fatal_scratch_sanitizers_v25/`. This covers a
preexisting fatal, not an initially clean `valid_rows > capacity` input.

The clean over-capacity case was reproduced at source
`5769e7356eb9a9e014ee6fc11664a3a1bd6b86d1`: private v26 failed the
focused fixture with `RANK_OVERCAP_MUST_NOT_READ_CANDIDATES`. The copying
kernel poisoned the control from one thread but other threads still read
candidate input. Source `62bec6155edae372b244309b2a044bbb13f226a6`
returns every thread before any candidate read when `valid_rows > capacity`.
Private v27 passed the full plain owner fixture on both T4s. Raw outputs:
ignored `test_results/kaggle_owner_overcap_red_v26/` and
`test_results/kaggle_owner_overcap_green_v27/`.

This is a route-window leaf gate, **not** evidence that the Rust BFS scheduler
uses the device-only path. Host-sized NCCL exchange, owner control snapshots
and CPU retirement still remain in the current runtime.

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

Raw outputs: ignored `test_results/kaggle_owner_window_red_v4/` and
`test_results/kaggle_owner_window_green_v5/`.

This is a route-window leaf gate, **not** evidence that the Rust BFS scheduler
uses the device-only path. Host-sized NCCL exchange, owner control snapshots
and CPU retirement still remain in the current runtime.

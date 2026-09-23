# GPU-count owner window and rank teardown gates

These are bounded component checks on physical Kaggle 2xT4, not an integrated
CPU-free owner/transport/retirement BFS or a performance result.

| Private Kaggle run | Source commit | Observation |
|---|---|---|
| `trydotatwo/mgbfs-cuco-dynamic-shard-refs-t4` v16 | `c1238239a13adc12ed26faa5ceffb66d2f5eb75a` | Expected RED: after a delayed same-stream callback, rank destruction returned before the callback completed. The executable failed at `RANK_DESTROY_MUST_DRAIN_IN_FLIGHT_WORK`, not at build or unrelated tests. |
| Same notebook v17 | `50cd032993b21c8f226b7dabd820967b85a0be05` | GREEN: same probe passed on both T4s after teardown drains its creating stream. Other plain library probes passed. |
| Same notebook v18 | `64a5ae77c4efde6e5d3562e4c8c7af8be282e3b9` | Expected RED link failure: `mgbfs_library_candidates_from_aos_window_v1` was declared and used by the fixture, but not implemented. |
| Same notebook v19 | `a46fa6e958c17af1f0740c3fdd93cf21c609e746` | GREEN: fixed-capacity AoS→SoA window with device-resident begin/count passed on both T4s. The fixture checks every key word, absolute source ordinal, untouched padding, and sticky fatal with no scratch writes for a source-range overflow. Plain mode only. |
| Same notebook v20 | `a46fa6e958c17af1f0740c3fdd93cf21c609e746` | The same window fixture passed on both T4s under `memcheck`, `racecheck`, `initcheck` and `synccheck`. Each error summary was zero; racecheck reported zero hazards and warnings. This is a component gate, not BFS integration. |

Raw downloaded results: `test_results/kaggle_rank_destroy_red_v16/`,
`test_results/kaggle_rank_destroy_green_v17/`,
`test_results/kaggle_device_window_red_v18/`, and
`test_results/kaggle_device_window_green_v19/` and
`test_results/kaggle_device_window_sanitizers_v20/`. The Kaggle CLI prints a Windows
`charmap` error after downloading the files; conclusions come from each
`summary.json`, build log and fixture output.

The new window bridge is not called by the Rust BFS runtime. Its device counts
can feed the cuCO rank-batch owner, but local owner offsets, LSA exchange
epochs, GPU descriptor publication, parent/archive leases and rank-group fatal
propagation must still be integrated. The v20 sanitizer pass does not discharge
those end-to-end requirements.

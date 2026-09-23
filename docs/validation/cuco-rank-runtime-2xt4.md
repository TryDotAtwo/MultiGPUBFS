# CUCO_RANK reference BFS: physical 2xT4 gates

This is a correctness gate for the new DENSE reference owner, not evidence of
device-driven end-to-end scheduling or a speed improvement.

| Private Kaggle run | Exact Git source | Result |
|---|---|---|
| `trydotatwo/mgbfs-cuco-dynamic-shard-refs-t4` v12 | `177bcaac542ea48b828fbaa965a472ceda504876` | Expected RED: `cuco_rank_dense_layers_match_full_state_oracle` failed at `REFERENCE_CUCO_RANK_NOT_WIRED`; old library test passed. |
| Same notebook v13 | `ae3cac679f41dea15534d8d6ec0de2b7580c6e43` | Both physical T4s passed the one-rank full-state rank-owner test and old library test; two-GPU baseline passed. Sanitizers disabled. |
| `trydotatwo/mgbfs-library-owner-t4` v56 | `24c38bc8822b84c3536549d30d029d99264e9ed6` | Two-GPU full-state/archived rank-owner oracle passed with both rank maps and pre-dedup ON/OFF. Two-process torchrun CLI completed S4 (24 states) and U4(m=2) (64 states); per-rank archives verified and backend label was `library_nccl_dense_cuco_rank_v1`. Sanitizers disabled. |
| Same notebook v57 | `c16c8c7f25b39c6a0bbe0af16d998a302bab8860` | Both physical T4s passed `cuco_rank_capacity_failure_releases_pool_after_gpu_work` plus the other one-GPU full-state tests (3/3 per GPU); the two-GPU suite passed 2/2 with one ignored. Plain mode only. |
| `trydotatwo/mgbfs-cuco-dynamic-shard-refs-t4` v15 | `aabbf45fd70815aee0d0f45e822dafe92bd917d5` | The one-rank full-state cuCO-rank BFS passed on both T4s and the two-rank full-state/archive oracle passed. `memcheck`, `racecheck`, `initcheck`, `synccheck` all passed for these runtime tests; racecheck reported zero hazards, errors and warnings. The eight-rank test was correctly ignored on two GPUs. This predates the new capacity cleanup fixture. |
| `trydotatwo/mgbfs-library-owner-t4` v58 | `c16c8c7f25b39c6a0bbe0af16d998a302bab8860` | Capacity-failure/drop/recreation test passed under `memcheck` on each physical T4 (3/3 tests per GPU, zero memcheck errors); the two-GPU suite also passed 2/2, with eight-rank ignored. This is a bounded fixture, not a general asynchronous lifetime proof. |

Raw downloaded outputs are in ignored `test_results/kaggle_cuco_rank_red_v12/`,
`test_results/kaggle_cuco_rank_green_v13/` and
`test_results/kaggle_cuco_rank_cli_v56/` and
`test_results/kaggle_cuco_rank_capacity_v57/` and
`test_results/kaggle_cuco_rank_sanitizers_v15/` and
`test_results/kaggle_cuco_rank_capacity_memcheck_v58/`. The Kaggle CLI reports a Windows
`charmap` error after downloading files; conclusions above come from the
downloaded `summary.json`, BFS test logs and CLI rank records, not the CLI exit
status.

Not yet discharged: racecheck/initcheck/synccheck for the new capacity fixture,
large-graph performance/VRAM, HASH_FIRST rank owner, and removal of host
waits/readbacks across owner, transport and retirement. V58 checks one bounded
failure/recreation path under memcheck; it does not establish general
asynchronous lifetime safety. V15 covers only the existing small-graph
runtime fixtures, not these outstanding requirements.

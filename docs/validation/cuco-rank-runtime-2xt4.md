# CUCO_RANK reference BFS: physical 2xT4 gates

This is a correctness gate for the new DENSE reference owner, not evidence of
device-driven end-to-end scheduling or a speed improvement.

| Private Kaggle run | Exact Git source | Result |
|---|---|---|
| `trydotatwo/mgbfs-cuco-dynamic-shard-refs-t4` v12 | `177bcaac542ea48b828fbaa965a472ceda504876` | Expected RED: `cuco_rank_dense_layers_match_full_state_oracle` failed at `REFERENCE_CUCO_RANK_NOT_WIRED`; old library test passed. |
| Same notebook v13 | `ae3cac679f41dea15534d8d6ec0de2b7580c6e43` | Both physical T4s passed the one-rank full-state rank-owner test and old library test; two-GPU baseline passed. Sanitizers disabled. |
| `trydotatwo/mgbfs-library-owner-t4` v56 | `24c38bc8822b84c3536549d30d029d99264e9ed6` | Two-GPU full-state/archived rank-owner oracle passed with both rank maps and pre-dedup ON/OFF. Two-process torchrun CLI completed S4 (24 states) and U4(m=2) (64 states); per-rank archives verified and backend label was `library_nccl_dense_cuco_rank_v1`. Sanitizers disabled. |

Raw downloaded outputs are in ignored `test_results/kaggle_cuco_rank_red_v12/`,
`test_results/kaggle_cuco_rank_green_v13/` and
`test_results/kaggle_cuco_rank_cli_v56/`. The Kaggle CLI reports a Windows
`charmap` error after downloading files; conclusions above come from the
downloaded `summary.json`, BFS test logs and CLI rank records, not the CLI exit
status.

Not yet discharged: sanitizer gate for the runtime path, capacity-failure
cleanup, large-graph performance/VRAM, HASH_FIRST rank owner, and removal of
host waits/readbacks across owner, transport and retirement. V15 (sanitizers)
and v57 (capacity failure) were still running when this record was created.

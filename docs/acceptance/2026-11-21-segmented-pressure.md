# S13 limited-VRAM pressure tests on 2×RTX 3060

Source on the GPU host: `d067734f3e9f012e3e394ccf712f123867c00e0c`. Coxeter S13, batch strategy, exact exchange, eight shards, physical segments of 1,048,576 rows per bank, configured segment limit 2^32, no per-rank user row cap. The test imposed a **10,300,000,000-byte per-GPU reserve**, deliberately restricting the budget. These stops do not show that an unreserved 12 GB GPU was physically full.

| Backend | Preset | Status | Last completed depth | BFS seconds |
|---|---|---|---:|---:|
| permutation bitmap | balanced | INCOMPLETE RESOURCE_10 | 24 | 58.993 |
| permutation bitmap | max_capacity | INCOMPLETE RESOURCE_10 | 26 | 977.512 |
| segmented hash | balanced | INCOMPLETE RESOURCE_10 | 21 | 28.353 |
| segmented hash | max_capacity | INCOMPLETE RESOURCE_10 | 21 | 638.900 |

Every completed layer size matches an independently calculated Mahonian distribution. These incomplete results do not compare the full sets of states at the final depths; partial next-layer live rows must not be treated as completed BFS states.

For hash, the sum of completed layer times through depth 21 was 23.458 s for balanced and 494.317 s for max_capacity, one run each. The presets use different geometries: 4 owner lanes and queue capacity 24,576 for balanced versus one lane and queue capacity 2,048 for max_capacity. They reached the same completed depth, so this test does not establish a capacity advantage for max_capacity on hash.

The rank-1 hash stop was `hash_view_budget`. Although the internal storage budget still showed 21,296,916 bytes remaining for balanced and 23,788,215 for max_capacity, `cudaMemGetInfo` reported free memory below the artificial 10.3 GB reserve (10,297,999,360 and 10,295,902,208 bytes respectively). The allocation guard checks both quantities. This explains the reported stop; it is not evidence of physical VRAM exhaustion. The remaining-budget number is not a guarantee that allocations can proceed while the independent reserve guard is violated.

Artifact: `2026-11-21-segmented-pressure.json.gz`, SHA-256 `6245fbe580b4bb886ae7d3193147347bb249d51345fcf31df355456e95bf4e9d`. It contains these four summaries, six previously reported S12 results, validation logs and test scripts; it contains no graph-state sets. The sampled memory and tracked segment allocations have different scopes; see the individual receipts. No B200 execution is implied.

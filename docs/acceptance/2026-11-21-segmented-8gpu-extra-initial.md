# Segmented capacity pressure: interim evidence

This snapshot preserves completed pressure tests while the remaining measurements continue. Source on the GPU host: `d067734f3e9f012e3e394ccf712f123867c00e0c`. Two RTX 3060 GPUs, Coxeter S13, batch strategy, exact exchange, no per-rank user row cap, 1,048,576 rows per bank and segment, eight shards. An artificial 10,300,000,000-byte per-GPU reserve intentionally reduces the memory budget; this is **not** an ordinary physical out-of-memory event.

| Backend | Preset | Status | Last completed depth | BFS seconds |
|---|---|---|---:|---:|
| permutation bitmap | balanced | INCOMPLETE RESOURCE_10 | 24 | 58.993 |
| permutation bitmap | max_capacity | INCOMPLETE RESOURCE_10 | 26 | 977.510 |
| segmented hash | balanced | INCOMPLETE RESOURCE_10 | 21 | 28.35 |

All completed layer sizes match the independently computed Mahonian distribution. These incomplete runs have not compared full state sets at their final depths. Their elapsed times cover different completed prefixes and are not a speed comparison. The fourth pressure case and the eight-GPU S13 sweep are not included in this snapshot.

The compressed evidence `2026-11-21-segmented-8gpu-extra-initial.json.gz` has SHA-256 `86406fba105b836ffa9a0b5b2c312f1f98201a4203ff0bca16747ff05c04ecea`. It contains the pressure summaries, six previously reported S12 results, validation logs and test scripts; no state sets are included. GPU source objects for SM100 were compiled separately, with the hash batch kernel only in a debug build at this point. This is compile evidence and does not establish B200 runtime correctness.

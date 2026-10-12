# Pair-sweep Kaggle acceptance, 2026-10-12

Native source419fa34f; public utility version5; private pair-sweep version2; physical2×Tesla T4. Public version3 uses the same accepted run cells, omitting the private independent CPU enumeration.

| Family | Pairs n=2..5, r=1..n | Complete | CPU-verified layer rows | Resident PID groups | Sum API seconds |
|---|---:|---:|---:|---:|---:|
| LRX |14|14|53|1|3.376155611|
| Rapaport M2 |14|14|58|1|3.613858071|

This is tiny-graph API/sweep validation, not maximum-capacity performance. GPU count2, native devices[0,1], source419fa34f and CSV/report layer consistency were checked for each case. Each family reused the same two native worker PIDs.

Version1 failed at the first constructor: CayleyPy0.2.0 `lrx(2)` asserts n>=3. The exception occurred at66.6s; the platform exposed the error/logs substantially later while reporting RUNNING. Version2 explicitly extends the L/R/X formula at n=2 with three identical [1,0] permutations. M2 uses the actual CayleyPy constructor. Both degenerate n=2 cases passed the independent oracle. No GPU resource stop or native deadlock was observed in that failed version: it never entered BFS.

Outputs verified: per-pair report.json, results.csv and layers.csv. State payloads remained on Kaggle; only metadata/logs were downloaded for verification. Optional HF publication/readback code was not executed because no write secret was supplied. Public notebook readers must set their own destination/secret to use it.

Public notebook version3 subsequently completed on2×T4. All14 default LRX cases and53 layer-count rows matched the accepted private gate, and public metadata readback confirmed is_private=false. Optional HF upload remained disabled.

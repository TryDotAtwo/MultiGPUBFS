# S12 compact working state capacity probe

Kaggle s12-capacity-probe v4; source 9f440a1ceff379f1c9df57fe97baa6ff98bd21c0.
Two physical T4s, batch 262144, 32M state ring and layer capacity per rank.
Archive disabled in both runs below: these are search/capacity diagnostics,
not durable catalog artifacts. No HF publication claim.

| Working representation | Search seconds | Peak MiB per rank |
|---|---:|---:|
| Matrix, earlier v3 | 63.522871764 | 7545 |
| Compact, v4 | 40.409229307 | 2915 |

All 67 layer counts agree with matrix v3; sum is 479,001,600 (12!).
No full-set verification was performed. Each result is one run, not a repeated
controlled A/B performance claim. The global peak allocation includes scratch
and other runtime buffers, not just state payload.

Evidence: test_results/s12-compact-working-v4/s12-capacity-probe/summary.json
and test_results/s12-capacity-ring-v3/s12-capacity-probe/summary.json.

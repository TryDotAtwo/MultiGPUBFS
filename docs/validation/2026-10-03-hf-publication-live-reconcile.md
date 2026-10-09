# Live HF publication reconciliation

Read-only check against TryDotAtwo/multigpubfs-bfs-results at immutable revision
c6e23ff7fdf1868840060ce38640ac21a714db30, current HEAD222abcf.
No token used, no upload/remote rewrite, no state payload downloaded.

The actual production reconcile_publication function passes for:

| Run | Declared layer-count total | State objects |
| --- | ---: | ---: |
| s8-native-2xt4-20260905-124255 | 40320 | 2 |
| s11-native-2xt4-20260904-194733 | 39916800 | 40 |
| s13-native-2xt4-20260905-152407 | 6227020800 | 6228 |

S13 state objects total151415292851 bytes. Reconcile compares all remote object
sizes and LFS SHA256 metadata, plus matching manifest/verification Git blob or
LFS identity, on this single revision. Only small run JSON and layer Parquet
metadata were fetched; no151GB payload on Ivan's machine.

This proves the stored publication contract is reconciled, not recomputed
unique-state/BFS correctness, full-state payload verification, or acceptance
of the current owner/transport runtime. These are historical20260904/05 runs,
not replacement evidence for the still-open current2T4/sanitizer/A-B gates.

files[].rows backwards compatibility is already in commitfaead68. Current
test_promotion_reconcile.py and test_promote_hf_stream.py each pass11 tests,
including legitimate legacy manifest/verification without per-file rows,
alternative layer Parquet encodings, immutable revision checking and corrupt
state rejection. The live runs above exercise existing public legacy records;
they were not reformatted or republished.

Dataset Viewer /is-valid currently returns HTTP500 busy/not-ready. This is
not an absent/corrupt graph proof: Hub object reconciliation above succeeds.
The separate Viewer availability gate is not called complete.

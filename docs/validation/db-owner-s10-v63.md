# Two-T4 S10 three-owner screen v63

Private Kaggle `trydotatwo/mgbfs-library-owner-t4`, version 63, source
`48bc324c771128e327015c32f4ea4a8f8fab2b3f`; native baseline source
`013ed5c979f4225db273e0015fa9ed72fd230c90`. Each owner ran one
matched S10 DENSE full BFS on two Tesla T4s, 8 shards, 256 archive slots,
96 MiB fixed library pool per rank where applicable. All cases completed
46 layers and 3,628,800 states; both rank archives in every case were
reported VERIFIED for committed checksums and counts.

| Owner | Search s | Durable s | Sampled peak MiB, total (per rank) |
|---|---:|---:|---:|
| CUB_SORT_MERGE | 0.871420 | 4.076922 | 914 (457, 457) |
| CUCO_RANK | 0.491433 | 3.915824 | 1058 (529, 529) |
| CUDF_RELATIONAL | 1.920906 | 4.715612 | 1134 (567, 567) |

Memory figures are 50 ms external `nvidia-smi` samples of full-device
consumption, not exact peaks. Each row has **one** sample, so this is a
capacity/correctness screen, not a stable performance comparison. The
same-source five-repeat screen is running as private Kaggle v64. No
sanitizer ran in v63; its summary cites earlier separate sanitizer evidence.

Raw result: `test_results/kaggle_db_screen_v63/library-owner/summary.json`
and per-case `screen-summary.json` plus rank archive verifier logs.

# S13 streamed archive: complete publication

Kaggle `trydotatwo/mgbfs-s11-hf-stream` v27, source
`edb9fa4cf399036550e4f160dacd9a7a770d25e8`, real 2xT4.
Run `s13-native-2xt4-20260905-152407`.

- Native exit 0, both archive consumers COMPLETE.
- 6,227,020,800 states; 79 nonempty layers, depths 0 through 78.
- All 79 layer counts equal the completed search-only S13 v5 run.
- Search: 4012.695384772 s; native durable archive: 4090.478055409 s.
- External sampled peak: 14,005 MiB per rank, 28,010 MiB total.
- Search-only reference: 3583.067481444 s, 13,985 MiB per rank.
  These are separate single runs, not a paired repeated benchmark.
- Fixed archive ring: 1280 slots per rank; host budget preflight passed.

HF final commit returned HTTP 504 to Kaggle, so the notebook reports ERROR.
The server nevertheless committed the complete publication:
`d43c3aa640ef12935ff12f986e53d3e6fef6e92f`.
No repeat BFS or duplicate publication was necessary.

Verified the immutable HF run manifest (COMPLETE) and all 6,228 published
Parquet objects using paths-info: each object's size and LFS SHA256 match
the manifest. Total Parquet bytes: 151,415,292,851.
Only metadata was fetched locally; graph payload stayed Kaggle -> HF.
This is object metadata verification, not a fresh decode of every remote row.
Hash128 dedup remains probabilistic; cardinality and layer equality do not
constitute independent exact full-set verification.

Evidence: `test_results/hf-editor-v27/s11-hf-stream/`, including native JSON,
both consumer stdout logs, GPU archive test, host budget and promote traceback.
Public manifest:
https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/d43c3aa640ef12935ff12f986e53d3e6fef6e92f/runs/s13-native-2xt4-20260905-152407.json

Follow-up: make promotion reconcile an ambiguous commit response instead of
reporting a successfully published run as failed. Full-runtime sanitizer and
remaining backend/profile gates are still separate unfinished work.

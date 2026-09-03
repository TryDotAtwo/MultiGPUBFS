# MultiGPUBFS result datasets

Each published revision contains every unique state from a completed BFS plus
per-depth and full-run metadata in Parquet. The canonical identity of a state
row is `(run_id, rank, rank_ordinal)`; `depth` always means distance in the
original generator graph even when macro lookahead generated the candidate.

Export locally:

```text
python scripts/export_hf_dataset.py --run-id RUN --summary summary.json \
  --archive 0=rank0.mgbfsar1 --archive 1=rank1.mgbfsar1 --output dataset
```

Before upload, verify archive commits, manifest SHA256 values, global row counts,
per-depth counts, exact uniqueness of `(state)` and replayed successors. Upload
only that immutable directory as a new Hugging Face dataset revision.

`scripts/verify_hf_dataset.py` performs manifest/count validation and an exact
external merge-sort uniqueness check with bounded RAM. `verification.json` must
report `PASS`; hash recomputation and successor replay remain separate release
gates because they require the frozen graph/hash manifest.

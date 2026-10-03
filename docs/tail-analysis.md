# Analyze an archived BFS tail

Run state analysis on a GPU/server host. The automatic runner does not download
states to the workstation. `scripts/analyze_tail.py` reads HF Parquet through
range requests, with no local payload cache. Install `pyarrow`, `huggingface_hub`
and `fsspec` in the analysis environment.

Always pin a dataset commit: publication updates may replace the final open
cohort. A case manifest references every shared physical shard using `repo_path`.
Several cases share the same shard; filtering by both n and r is necessary.
The state column is fixed binary of 8 or 16 bytes, so read widths separately.

```python
from huggingface_hub import HfFileSystem
from scripts.analyze_tail import load_manifest, layer_statistics, iter_case_batches, unpack_state

fs = HfFileSystem()
repo = 'datasets/TryDotAtwo/multigpubfs-bfs-results'
revision = '8977998a67d4a7b72bcc873a9ff545db1abb538e'
prefix = f'{repo}@{revision}'
run = '20261003-full-grid-small-capacity'
manifest = load_manifest(fs, f'{prefix}/tail-runs/{run}-n8-m6/manifest.json')

# Counts, whole-layer seconds and independently sampled GPU peaks for ALL
# completed layers: no Parquet reads are needed.
for layer in layer_statistics(manifest):
    print(layer['depth'], layer['states'], layer['seconds'], layer['vram_peak_bytes'])

# Only retained depth 8, streamed in <=65,536-row batches.
rows = 0
for batch in iter_case_batches(manifest, fs, prefix, depths=[8]):
    rows += batch.num_rows
    example = unpack_state(batch['state'][0].as_py(), manifest['graph']['n'])
print(rows)
```

This documented example is for the diagnostic run; its buffer capacity was
deliberately small. Local synthetic tests verify shared-case selection, depth
selection, bounded yielded batches, width rejection and nibble order. This
helper's HF range-read path subsequently passed on an isolated Linux rental
for (8,6), (17,14), and (32,31), including case/depth filtering, 8/16-byte
decoding and batches bounded to seven rows. The retained row counts were
56, 4080 and 32. The same host passed 47 archive tests and 15 automatic tests.

Evidence: https://huggingface.co/datasets/TryDotAtwo/multigpubfs-bfs-results/blob/e597e44bfa50a744d60b7b1aaaed808fe9ec25cc/evidence/20261004-remote-analysis/report.json

`ordinal` is the position within a layer, preserved for partial tails. A missing
depth means its states were not retained, not an empty BFS layer. Determine
coverage from `retained_layers`, or each file's `layers` spans and `full_layer`.
COMPLETE keeps at least three whole layers, then whole older layers until about
10 decimal GB or the graph start. INCOMPLETE keeps at most one decimal GB from
completed layers. Parquet compression changes physical file size, not retention.

The iterator checks state width and safe paths; it does not certify SHA256.
For verification, use the publisher's pinned HF readback evidence or stream each
entire file and compare its physical byte size and SHA256 with the manifest.
GPU peak null means no sample landed inside that layer; it must not become zero.

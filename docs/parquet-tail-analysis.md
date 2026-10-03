# Reading a saved BFS tail

New automatic sweeps publish Parquet with Zstd compression. A file contains
multiple layers when they fit; a large layer spans files. Target uncompressed
column size is 512 MB, row groups at most 262144 rows. Target is approximate;
physical compressed file sizes are recorded in the manifest.

Each row has `n`, `r`, `depth`, `ordinal`, `state`. State is fixed-width binary:
8 bytes for n<=16, otherwise 16 bytes. Symbol i occupies bits 4*i, little endian.
Ordinal is within that layer, not a path, global rank, or predecessor pointer.

Download only the selected run's manifest and Parquet files on the analysis
machine. During collection the producer uploads directly from the GPU host.
Immutable manifests appear after the corresponding payload upload commits.
Check the manifest's SHA256 and byte size before opening downloaded payloads.

```python
from pathlib import Path
import pyarrow.parquet as pq

n, r = 15, 4
tables = [pq.read_table(path, columns=['depth', 'ordinal', 'state'],
                       filters=[('n', '=', n), ('r', '=', r), ('depth', '>=', 20)])
          for path in Path('downloaded-run').glob('*.parquet')]

def symbols(packed, n):
    word = int.from_bytes(packed, 'little')
    return [(word >> (4*i)) & 15 for i in range(n)]
```

For full-layer statistics use `manifest.layers`: state counts, whole-layer time,
and separately observed per-GPU VRAM peaks. Counting Parquet rows counts only
retained states. `retained_layers` records complete/partial layers and the first
retained ordinal; `files[].layers` maps row groups/shard spans to those layers.
Retention thresholds count original packed bytes (`retained_packed_bytes`),
independently of Parquet compression or analytical column overhead.

New automatic configurations use `archive_format=parquet_cohort`. Up to 20
completed cases share immutable payloads under `tail-shards/<run>/<fingerprint>/`.
Widths 8 and 16 are grouped separately. Existing configurations retain their
chosen format. Completed groups are cached by case manifest hashes; later
cases do not repack earlier full groups. An unfinished final group can be
replaced by a new content-addressed group on a resumed sweep.

Each case keeps its own manifest under `tail-runs/<run>-<case>/manifest.json`.
Use `files[].repo_path` as the complete repository filename when downloading
shared payloads. Deduplicate these paths across manifests. A shared file's
`states` counts all its rows; `case_states` and `files[].layers` describe this
case's rows. Each span has `file_row_offset`, `states`, `depth`,
`first_state_ordinal`, and `full_layer`. Filter shared Parquet by **both n and r**
before counting states or decoding them for a case. `retained_layers` and
`retained_packed_bytes` remain case-specific and preserve the original retention
contract. Per-case statistics and launch reproducibility remain in each manifest.

CPU integration gates cover sharing, widths, shard boundaries, partial offsets,
cache reuse, corruption rejection, payload-before-manifest commits and shared
HF readback. Remote publication of this cohort format remains unverified.

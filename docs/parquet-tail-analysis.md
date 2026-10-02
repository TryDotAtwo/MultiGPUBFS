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

tables = [pq.read_table(path, columns=['depth', 'ordinal', 'state'],
                       filters=[('depth', '>=', 20)])
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

The per-case grouping currently eliminates one file per layer, but small cases
still produce small per-case shards. Sweep-wide consolidation remains an open
gate; do not claim that all small cases share a single large dataset yet.

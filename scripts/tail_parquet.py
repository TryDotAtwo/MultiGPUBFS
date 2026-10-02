"""Convert a frozen packed tail to bounded Parquet shards on the GPU host.

Retention remains measured in raw packed bytes, independently of compression.
No CUDA calls. Input manifests and binaries remain immutable.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath

try:
    from .bfs_tail_archive import atomic_json
except ImportError:
    from bfs_tail_archive import atomic_json


def publication_root(root, storage_format):
    """Cache conversion of an immutable completed-case manifest by content."""
    root = Path(root)
    if storage_format == 'packed':
        return root
    if storage_format != 'parquet':
        raise ValueError('unsupported publication format')
    fingerprint = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
    destination = root.parent / ('parquet-' + fingerprint)
    if not (destination / 'manifest.json').exists():
        if destination.exists():
            # An interrupted conversion never has a committed manifest.
            import shutil
            shutil.rmtree(destination)
        convert(root, destination)
    return destination


def convert(root, destination, *, shard_bytes=512_000_000, row_group_rows=262144):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if shard_bytes <= 0 or row_group_rows <= 0:
        raise ValueError('positive Parquet bounds required')
    root = Path(root).resolve()
    destination = Path(destination)
    manifest = json.loads((root / 'manifest.json').read_text())
    width = manifest['packing']['bytes_per_state']
    if width not in (8, 16):
        raise ValueError('unsupported packed width')
    destination.mkdir(parents=True, exist_ok=False)
    schema = pa.schema([
        ('n', pa.uint8()), ('r', pa.uint8()), ('depth', pa.uint32()),
        ('ordinal', pa.uint64()), ('state', pa.binary(width)),
    ], metadata={b'packing': json.dumps(manifest['packing']).encode(),
                 b'graph': json.dumps(manifest['graph']).encode()})
    # Size target counts uncompressed columns. Row groups never straddle layers.
    rows_per_shard = max(1, shard_bytes // (width + 14))
    output, spans = [], []
    writer = None
    path = None
    rows_in_shard = 0

    def close():
        nonlocal writer, rows_in_shard, spans
        if writer is None:
            return
        writer.close()
        with path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        output.append(dict(path=path.name, bytes=path.stat().st_size,
                           sha256=checksum, states=rows_in_shard, layers=spans))
        writer = None
        rows_in_shard = 0
        spans = []

    try:
        for entry in manifest['files']:
            relative = PurePosixPath(entry['path'])
            source = (root / entry['path']).resolve()
            if relative.is_absolute() or '\\' in entry['path'] or not source.is_relative_to(root):
                raise ValueError('unsafe packed input path')
            digest = hashlib.sha256()
            read_rows = 0
            with source.open('rb') as stream:
                while read_rows < entry['states']:
                    if writer is None:
                        path = destination / f'states-{len(output):05d}.parquet'
                        writer = pq.ParquetWriter(path, schema, compression='zstd',
                            use_dictionary=['n', 'r', 'depth'], write_statistics=True)
                    count = min(row_group_rows, rows_per_shard - rows_in_shard,
                                entry['states'] - read_rows)
                    data = stream.read(count * width)
                    if len(data) != count * width:
                        raise ValueError('truncated packed layer')
                    digest.update(data)
                    ordinal = entry.get('first_state_ordinal', 0) + read_rows
                    state = pa.Array.from_buffers(pa.binary(width), count,
                                                  [None, pa.py_buffer(data)])
                    table = pa.Table.from_arrays([
                        pa.array([manifest['graph']['n']] * count, type=pa.uint8()),
                        pa.array([manifest['graph']['r']] * count, type=pa.uint8()),
                        pa.array([entry['depth']] * count, type=pa.uint32()),
                        pa.array(range(ordinal, ordinal + count), type=pa.uint64()),
                        state,
                    ], schema=schema)
                    writer.write_table(table, row_group_size=count)
                    spans.append(dict(depth=entry['depth'], first_state_ordinal=ordinal,
                                      states=count, full_layer=entry['full_layer']))
                    read_rows += count
                    rows_in_shard += count
                    if rows_in_shard == rows_per_shard:
                        close()
                if stream.read(1) or source.stat().st_size != entry['bytes']:
                    raise ValueError('packed layer size mismatch')
            if digest.hexdigest() != entry['sha256']:
                raise ValueError('packed layer checksum mismatch')
        close()
        result = dict(manifest, schema=2, storage_format='parquet',
                      retained_packed_bytes=sum(x['bytes'] for x in manifest['files']),
                      retained_layers=[{k: v for k, v in x.items()
                                        if k not in ('path', 'sha256')} for x in manifest['files']],
                      files=output, parquet=dict(compression='zstd',
                          target_uncompressed_shard_bytes=shard_bytes,
                          row_group_rows=row_group_rows, columns=schema.names))
        atomic_json(destination / 'manifest.json', result)
        return result
    finally:
        if writer is not None:
            writer.close()

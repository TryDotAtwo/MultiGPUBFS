"""Manifest-driven, bounded-memory analysis of local or remote tail Parquet.

Remote reads use HF filesystem range requests, without a local payload cache.
Keep the filesystem pinned to an immutable dataset revision for reproducibility.
"""
import json
from pathlib import PurePosixPath


def load_manifest(filesystem, path):
    with filesystem.open(path, 'rb') as stream:
        return json.load(stream)


def layer_statistics(manifest):
    """All completed layers, including those outside the retained state tail."""
    return manifest['layers']


def iter_case_batches(manifest, filesystem, payload_prefix='', *, depths=None,
                      batch_rows=65536):
    """Yield only this case's retained states; never concatenate entire shards.

    The caller owns the filesystem and authentication. For local data use
    fsspec.filesystem('file'); for HF use huggingface_hub.HfFileSystem().
    Filtering each bounded batch also handles shards shared by several cases.
    Depths missing from the retained tail produce no rows, not fabricated data.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    if type(batch_rows) is not int or batch_rows <= 0:
        raise ValueError('positive batch_rows required')
    if manifest.get('storage_format') != 'parquet':
        raise ValueError('Parquet manifest required')
    depths = None if depths is None else set(depths)
    n, r = manifest['graph']['n'], manifest['graph']['r']
    width = manifest['packing']['bytes_per_state']
    seen = set()
    for entry in manifest['files']:
        if depths is not None and entry.get('layers') is not None:
            if not any(layer['depth'] in depths for layer in entry['layers']):
                continue
        relative = entry.get('repo_path', entry['path'])
        path = PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts or '\\' in relative:
            raise ValueError('unsafe payload path')
        if relative in seen:
            continue
        seen.add(relative)
        with filesystem.open(payload_prefix.rstrip('/')+'/'+relative if payload_prefix else relative,
                             'rb') as stream:
            parquet = pq.ParquetFile(stream)
            if parquet.schema_arrow.field('state').type != pa.binary(width):
                raise ValueError('state packing width differs from manifest')
            # Reject only ranges that provably cannot contain the selection.
            # Missing statistics always fall back to reading and filtering.
            columns = {name: index for index, name in enumerate(parquet.schema_arrow.names)}
            row_groups = []
            for index in range(parquet.metadata.num_row_groups):
                group = parquet.metadata.row_group(index)
                possible = True
                for name, values in [('n', [n]), ('r', [r]), ('depth', depths)]:
                    if values is None:
                        continue
                    stats = group.column(columns[name]).statistics
                    if stats is not None and stats.has_min_max:
                        if not any(stats.min <= value <= stats.max for value in values):
                            possible = False
                            break
                if possible:
                    row_groups.append(index)
            for batch in parquet.iter_batches(batch_size=batch_rows,
                    row_groups=row_groups,
                    columns=['n', 'r', 'depth', 'ordinal', 'state']):
                table = pa.Table.from_batches([batch])
                keep = pc.and_(pc.equal(table['n'], n), pc.equal(table['r'], r))
                if depths is not None:
                    keep = pc.and_(keep, pc.is_in(table['depth'],
                        value_set=pa.array(sorted(depths), type=pa.uint32())))
                selected = table.filter(keep)
                if selected.num_rows:
                    yield selected


def unpack_state(state, n, bits_per_symbol=4):
    """Low nibble is the earlier symbol; trailing padding is excluded."""
    if not 2 <= n <= 128 or bits_per_symbol not in (4,8,16,32,64) or len(state) != ((n*bits_per_symbol+63)//64)*8:
        raise ValueError('packed state shape')
    value=int.from_bytes(state,'little')
    mask=(1<<bits_per_symbol)-1
    return tuple((value>>(bits_per_symbol*index)) & mask for index in range(n))

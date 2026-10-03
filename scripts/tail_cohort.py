"""Immutable bounded groups of cases sharing Parquet payloads on the GPU host."""
import hashlib
import json
from pathlib import Path

try:
    from .bfs_tail_archive import atomic_json
except ImportError:
    from bfs_tail_archive import atomic_json


def publication_cases(root, ledger, *, group_size=20, shard_bytes=512_000_000):
    root = Path(root)
    if group_size <= 0 or shard_bytes <= 0:
        raise ValueError('positive cohort bounds required')
    run_id = ledger['configuration']['base']['run_id']
    if not isinstance(run_id, str) or run_id in ('', '.', '..') or '/' in run_id or '\\' in run_id:
        raise ValueError('unsafe sweep run ID')
    for key in ledger['cases']:
        if key in ('', '.', '..') or '/' in key or '\\' in key:
            raise ValueError('unsafe case key')
    cases = [(key, root/key/'saved') for key, record in ledger['cases'].items()
             if record.get('attempted', True)]
    result = {}
    for begin in range(0, len(cases), group_size):
        # A fixed group of completed cases stays immutable as the sweep grows.
        groups = {}
        for key, saved in cases[begin:begin+group_size]:
            manifest = json.loads((saved/'manifest.json').read_text())
            groups.setdefault(manifest['packing']['bytes_per_state'], []).append((key, saved))
        for width, members in groups.items():
            fingerprint = hashlib.sha256(json.dumps(dict(
                members=[(key, hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest())
                         for key, p in members], shard_bytes=shard_bytes,
                schema=3, run_id=run_id), sort_keys=True).encode()).hexdigest()
            cohort_root = (root/'cohorts').resolve()
            destination = (cohort_root/fingerprint).resolve()
            if not destination.is_relative_to(cohort_root):
                raise ValueError('cohort cache escapes root')
            marker = destination/'cohort.json'
            if not marker.exists():
                if destination.exists():
                    import shutil
                    shutil.rmtree(destination)
                merge(members, destination,
                      f'tail-shards/{run_id}/{fingerprint}/', shard_bytes=shard_bytes)
            summary = json.loads(marker.read_text())
            if summary['cases'] != [key for key, _ in members]:
                raise ValueError('cohort cache case identity mismatch')
            for key in summary['cases']:
                result[key] = (destination, destination/f'{key}.json')
    return result


def merge(members, destination, prefix, *, shard_bytes=512_000_000):
    import pyarrow as pa
    import pyarrow.parquet as pq
    for key, _ in members:
        if key in ('', '.', '..') or '/' in key or '\\' in key:
            raise ValueError('unsafe case key')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    sources = [(key, Path(saved).resolve()) for key, saved in members]
    manifests = {key: json.loads((p/'manifest.json').read_text()) for key, p in sources}
    graphs = {(m['graph']['n'], m['graph']['r']) for m in manifests.values()}
    if len(manifests) != len(members) or len(graphs) != len(members):
        raise ValueError('duplicate cohort case/graph')
    widths = {m['packing']['bytes_per_state'] for m in manifests.values()}
    if len(widths) != 1 or next(iter(widths)) not in (8, 16) or shard_bytes <= 0:
        raise ValueError('cohort packing/bounds')
    limit = max(1, shard_bytes//(next(iter(widths))+14))
    writer, path, count, spans = None, None, 0, []
    outputs = {key: [] for key in manifests}
    files = []

    def close():
        nonlocal writer, count, spans
        if writer is None:
            return
        writer.close()
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        entry = dict(path=path.name, repo_path=prefix+path.name,
                     bytes=path.stat().st_size, sha256=digest, states=count)
        files.append(entry)
        for key in {s['case'] for s in spans}:
            selected = [{k: v for k, v in s.items() if k != 'case'}
                        for s in spans if s['case'] == key]
            outputs[key].append(dict(entry, case_states=sum(s['states'] for s in selected),
                                     layers=selected))
        writer, count, spans = None, 0, []

    try:
        for key, source in sources:
            manifest = manifests[key]
            width = manifest['packing']['bytes_per_state']
            schema = pa.schema([('n', pa.uint8()), ('r', pa.uint8()),
                                ('depth', pa.uint32()), ('ordinal', pa.uint64()),
                                ('state', pa.binary(width))])
            for entry in manifest['files']:
                from pathlib import PurePosixPath
                relative = PurePosixPath(entry['path'])
                payload = (source/entry['path']).resolve()
                if relative.is_absolute() or '\\' in entry['path'] or not payload.is_relative_to(source):
                    raise ValueError('unsafe packed cohort input path')
                digest = hashlib.sha256()
                read_rows = 0
                with payload.open('rb') as stream:
                    while read_rows < entry['states']:
                        if writer is None:
                            path = destination/f'states-{len(files):05d}.parquet'
                            writer = pq.ParquetWriter(path, schema, compression='zstd',
                                use_dictionary=['n', 'r', 'depth'], write_statistics=True)
                        rows = min(limit-count, 262144, entry['states']-read_rows)
                        data = stream.read(rows*width)
                        if len(data) != rows*width:
                            raise ValueError('truncated packed cohort layer')
                        digest.update(data)
                        ordinal = entry.get('first_state_ordinal', 0)+read_rows
                        depth = entry['depth']
                        chunk = pa.Table.from_arrays([
                            pa.array([manifest['graph']['n']]*rows, type=pa.uint8()),
                            pa.array([manifest['graph']['r']]*rows, type=pa.uint8()),
                            pa.array([depth]*rows, type=pa.uint32()),
                            pa.array(range(ordinal, ordinal+rows), type=pa.uint64()),
                            pa.Array.from_buffers(pa.binary(width), rows, [None, pa.py_buffer(data)]),
                        ], schema=schema)
                        writer.write_table(chunk, row_group_size=rows)
                        spans.append(dict(case=key, depth=depth, states=rows,
                            full_layer=entry['full_layer'], first_state_ordinal=ordinal,
                            file_row_offset=count))
                        count += rows
                        read_rows += rows
                        if count == limit:
                            close()
                    if stream.read(1) or payload.stat().st_size != entry['bytes']:
                        raise ValueError('packed cohort layer size mismatch')
                if digest.hexdigest() != entry['sha256']:
                    raise ValueError('cohort input checksum mismatch')
        close()
        for key, manifest in manifests.items():
            atomic_json(destination/f'{key}.json', dict(manifest, schema=3,
                storage_format='parquet', shared_payloads=True, files=outputs[key],
                retained_packed_bytes=sum(x['bytes'] for x in manifest['files']),
                retained_layers=[{k: v for k, v in x.items() if k not in ('path', 'sha256')}
                                 for x in manifest['files']],
                parquet=dict(compression='zstd', target_uncompressed_shard_bytes=shard_bytes,
                             row_group_rows=262144, columns=schema.names)))
        # Only this final marker makes an interrupted conversion reusable.
        atomic_json(destination/'cohort.json', dict(schema=3, cases=list(manifests), files=files))
    finally:
        if writer is not None:
            writer.close()

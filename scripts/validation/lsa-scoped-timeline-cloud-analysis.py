"""LSA scoped evidence adapter; reuse the frozen existing SQLite analysis core."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import urllib.request

root = Path('/tmp/mgbfs-lsa-scoped-input')
root.mkdir(exist_ok=False)
out = Path('/kaggle/working/lsa-scoped-analysis')
out.mkdir(exist_ok=False)
manifest = []
for name, url, expected in globals().pop('_MGBFS_TRACE_DOWNLOADS'):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or len(expected) != 64:
        raise RuntimeError('TRACE_INPUT_IDENTITY')
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(url, timeout=120) as response, target.open('xb') as writer:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > 1024 ** 3:
                    raise RuntimeError('TRACE_INPUT_CAPACITY')
                digest.update(chunk)
                writer.write(chunk)
    except Exception:
        raise RuntimeError('TRACE_INPUT_DOWNLOAD_FAILED:' + name) from None
    if digest.hexdigest() != expected:
        raise RuntimeError('TRACE_INPUT_SHA256:' + name)
    manifest.append(dict(name=name, bytes=size, sha256=expected))
    print('TRACE_INPUT_VERIFIED', name, size, flush=True)
(out / 'input-manifest.json').write_text(json.dumps(manifest, indent=2))
root = root / 'lsa-bfs-gate'
summary = json.loads((root / 'summary.json').read_text())
contracts = {
    '7a9f748ff7f649d3b062e9ee04a26428cbbc65c8': (6, {(profile, owner) for profile in ('DENSE', 'HASH_FIRST') for owner in ('CUB_SORT_MERGE', 'CUCO_RANK', 'BMMA_BUCKET')}),
    'd1c1c405c147a90f1c28c0417b542da887648b6c': (10, {('DENSE', 'CUCO_RANK')}),
    '78805a3f0a21b2988329d3dc701a83c4ea2bed41': (60, {(profile, owner) for profile in ('DENSE', 'HASH_FIRST') for owner in ('CUB_SORT_MERGE', 'CUCO_RANK', 'BMMA_BUCKET')}),
}
if summary.get('status') != 'LSA_SCOPED_TIMELINE_PASS' or summary.get('source') not in contracts:
    raise RuntimeError('LSA_TRACE_SOURCE_OR_STATUS')
expected_panels, expected_pairs = contracts[summary['source']]
all_rows = summary['validation_runs']
rows = [row for row in all_rows if row['label'].startswith('scoped-S8-')]
exports = summary['timeline_exports']
expected_traces = 2 * len(expected_pairs)
if len(all_rows) != expected_panels or len(rows) != len(expected_pairs) or len(exports) != expected_traces or not all(row['passed'] for row in all_rows):
    raise RuntimeError('LSA_TRACE_PANEL_INVENTORY')
if expected_panels in (10, 60):
    capture_rows = [row for row in all_rows if row['label'].startswith('batch-capture-')]
    sanitizer_rows = [row for row in all_rows if row['label'].startswith('sanitizer-')]
    fault_rows = [row for row in all_rows if row['label'].startswith('full-fault-')]
    if expected_panels == 10:
        expected_capture = {'batch-capture-' + pre + '-' + rank_map for pre in ('0', '1') for rank_map in ('01', '10')}
    else:
        expected_capture = {'batch-capture-' + profile + '-' + owner + '-' + pre + '-' + rank_map
            for profile, owner in expected_pairs for pre in ('0', '1') for rank_map in ('01', '10')}
    if len(capture_rows) != len(expected_capture) or {row['label'] for row in capture_rows} != expected_capture:
        raise RuntimeError('LSA_TRACE_CAPTURE_INVENTORY')
    expected_sanitizers = {'sanitizer-' + profile + '-' + owner + '-' + tool
        for profile, owner in expected_pairs for tool in ('memcheck', 'racecheck', 'initcheck', 'synccheck')}
    if len(sanitizer_rows) != len(expected_sanitizers) or {row['label'] for row in sanitizer_rows} != expected_sanitizers:
        raise RuntimeError('LSA_TRACE_SANITIZER_INVENTORY')
    for row in sanitizer_rows:
        if row['detail'].get('process_instrumentation') != row['label'].rsplit('-', 1)[1]:
            raise RuntimeError('LSA_TRACE_SANITIZER_PROVENANCE')
    expected_faults = {'full-fault-' + profile + '-' + owner: 19 if profile == 'DENSE' else 23
        for profile, owner in expected_pairs}
    if len(fault_rows) != len(expected_faults) or {row['label'] for row in fault_rows} != set(expected_faults):
        raise RuntimeError('LSA_TRACE_FAULT_INVENTORY')
    for row in fault_rows:
        if len(row['detail']['cases']) != expected_faults[row['label']]:
            raise RuntimeError('LSA_TRACE_FAULT_CASE_COUNT')
    for row in all_rows:
        if row['detail'].get('base_commit') != summary['source'] or any(not case.get('pass') or case.get('forced_cleanup') for case in row['detail']['cases']):
            raise RuntimeError('LSA_TRACE_PACKAGE_PROVENANCE_OR_FAILURE')
    for row in capture_rows + sanitizer_rows:
        evidence = row.get('batch_capture_evidence', [])
        if len(evidence) != 2 or {item['rank'] for item in evidence} != {0, 1} or not all(item['passed'] and len(item['markers']) == 2 for item in evidence):
            raise RuntimeError('LSA_TRACE_ADJACENT_CAPTURE_MISSING')
seen = set()
profiles = []
for row in rows:
    config_path = root / (row['label'] + '-config.json')
    data = config_path.read_bytes()
    cfg = json.loads(data)
    if hashlib.sha256(data).hexdigest() != row['config_sha256'] or cfg['transport_backend'] != 'NCCL_LSA':
        raise RuntimeError('LSA_TRACE_CONFIG')
    if cfg['graph']['rows'] != 8 or cfg['graph']['expected_max_unique_states'] != 40320 or cfg['parent_batch'] != 128:
        raise RuntimeError('LSA_TRACE_WORKLOAD')
    seen.add((cfg['frontier_profile'], cfg['owner_backend']))
    if row['detail'].get('profile_capture_scope') != 'search_after_full_warmup' or row['detail'].get('base_commit') != summary['source']:
        raise RuntimeError('LSA_TRACE_CAPTURE_SCOPE')
    if any(not case.get('pass') or case.get('forced_cleanup') for case in row['detail']['cases']):
        raise RuntimeError('LSA_TRACE_CASE_FAILURE')
    traces = [trace for trace in exports if trace['label'] == row['label']]
    if len(traces) != 2 or len({trace['rank_trace'] for trace in traces}) != 2:
        raise RuntimeError('LSA_TRACE_RANK_PAIR')
    for trace in traces:
        if trace['source'] != summary['source'] or trace['config_sha256'] != row['config_sha256'] or trace['capture_scope'] != 'search_after_full_warmup':
            raise RuntimeError('LSA_TRACE_EXPORT_PROVENANCE')
        if not any(item['name'] == 'lsa-bfs-gate/' + trace['sqlite'] and item['sha256'] == trace['sqlite_sha256'] for item in manifest):
            raise RuntimeError('LSA_TRACE_EXPORT_DIGEST')
    profiles.append(dict(configuration=row['label'], status='COMPLETE', profiled=True, scope='CUDA profiler start/stop bounds archived measured BFS only; full process warmup excluded', sqlite=[trace['sqlite'] for trace in traces]))
if seen != expected_pairs:
    raise RuntimeError('LSA_TRACE_PROFILE_OWNER_COVERAGE')
summary['profile_rows'] = profiles
summary['batch_size'] = 128
core_ref = '868050ed8786484ff24f8240c469686c453cc528'
core_path = 'scripts/validation/paired-payload-timeline-cloud-analysis.py'
with urllib.request.urlopen('https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/' + core_ref + '/' + core_path, timeout=30) as response:
    code = response.read(1024 * 1024 + 1)
if hashlib.sha256(code).hexdigest() != '1a9e28a3d340cef9c4a48e78aef5aaf395bb9883cab53a98039aabfd7e3a553d':
    raise RuntimeError('FROZEN_ANALYZER_SHA')
tree = ast.parse(code)
containers = [node for node in tree.body if isinstance(node, ast.Try)]
body = next(node.body for node in containers if any(isinstance(item, ast.For) and ast.unparse(item.iter) == "summary['profile_rows']" for item in node.body))
start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'results' for target in node.targets))
loop = next(i for i in range(start, len(body)) if isinstance(body[i], ast.For) and ast.unparse(body[i].iter) == "summary['profile_rows']")
exec(compile(ast.Module(body=body[start:loop + 1], type_ignores=[]), core_path, 'exec'), globals())
if len(results) != expected_traces or len(bounded) != expected_traces or any(record['healthy_window_aggregate']['windows'] == 0 for record in bounded):
    raise RuntimeError('LSA_TRACE_HEALTHY_WINDOWS_MISSING')
(out / 'bounded-aggregate.json').write_text(json.dumps(dict(source=summary['source'], status='EVIDENCE_EXTRACTED', analyzer_ref=core_ref, results=bounded), indent=2))
(out / 'summary.json').write_text(json.dumps(dict(status='EVIDENCE_EXTRACTED', source=summary['source'], analyzer_ref=core_ref, results=results,
    claim_boundary='Measured profile intervals and correlation only; no automatic CPU-free, critical-path, occupancy or A/B speed claim.'), indent=2))
print('LSA_SCOPED_EVIDENCE_EXTRACTED', len(results), flush=True)

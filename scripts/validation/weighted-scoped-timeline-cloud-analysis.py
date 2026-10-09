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
if summary.get('source') != '6f35c433ebd3269da45c8569fba1424f4e20a1d6' or summary.get('status') != 'WEIGHTED_SCOPED_TIMELINE_COLLECTED':
    raise RuntimeError('WEIGHTED_TRACE_SOURCE_STATUS')
rows = summary['weighted_multi_rank_runs']
exports = summary['timeline_exports']
expected_panels = 3 if summary.get('weighted_lsa_status') == 'UNSUPPORTED_P2P_HOST_NOT_RETRIED' else 6
if len(rows) != expected_panels or len(exports) != 2 * expected_panels or not all(r['passed'] for r in rows):
    raise RuntimeError('WEIGHTED_TRACE_INVENTORY')
profiles = []
for row in rows:
    cfg_data = (root / (row['label'] + '-config.json')).read_bytes()
    cfg = json.loads(cfg_data)
    if hashlib.sha256(cfg_data).hexdigest() != row['config_sha256']:
        raise RuntimeError('WEIGHTED_TRACE_CONFIG_HASH')
    if row['detail'].get('base_commit') != summary['source'] or row['detail'].get('profile_capture_scope') != 'search_after_full_warmup':
        raise RuntimeError('WEIGHTED_TRACE_PROVENANCE')
    if any(not c.get('pass') or c.get('forced_cleanup') for c in row['detail']['cases']):
        raise RuntimeError('WEIGHTED_TRACE_ORACLE_FAILURE')
    traces = [x for x in exports if x['label'] == row['label']]
    if len(traces) != 2 or len({x['sqlite'] for x in traces}) != 2:
        raise RuntimeError('WEIGHTED_TRACE_RANK_PAIR')
    for trace in traces:
        if trace['source'] != summary['source'] or trace['capture_scope'] != 'search_after_full_warmup':
            raise RuntimeError('WEIGHTED_TRACE_EXPORT_SCOPE')
        if not any(x['name'] == 'lsa-bfs-gate/' + trace['sqlite'] and x['sha256'] == trace['sqlite_sha256'] for x in manifest):
            raise RuntimeError('WEIGHTED_TRACE_DIGEST')
    profiles.append(dict(configuration=row['label'], status='COMPLETE', profiled=True,
        scope='Actual weighted full BFS measured search after full warmup; archive enabled; not A/B',
        sqlite=[x['sqlite'] for x in traces]))
summary['profile_rows'] = profiles
expected_traces = 2 * expected_panels
core_ref = '868050ed8786484ff24f8240c469686c453cc528'
core_path = 'scripts/validation/paired-payload-timeline-cloud-analysis.py'
with urllib.request.urlopen('https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/' + core_ref + '/' + core_path, timeout=30) as response:
    code = response.read(1024 * 1024 + 1)
if hashlib.sha256(code).hexdigest() != '1a9e28a3d340cef9c4a48e78aef5aaf395bb9883cab53a98039aabfd7e3a553d':
    raise RuntimeError('FROZEN_ANALYZER_SHA')
# Weighted runtime uses its own existing NVTX batch label. Normalize only the
# query's label; keep timestamps, threads, boundaries and API correlation intact.
code = code.decode().replace(
    "SELECT start,end,globalTid,text FROM NVTX_EVENTS",
    "SELECT start,end,globalTid,CASE WHEN text='mgbfs.weighted_batch' THEN 'mgbfs.batch' ELSE text END AS text FROM NVTX_EVENTS")
code = code.replace(
    "('mgbfs.batch','mgbfs.FinalizeDepth','mgbfs.archive_d2h')",
    "('mgbfs.batch','mgbfs.weighted_batch','mgbfs.FinalizeDepth','mgbfs.archive_d2h')")
tree = ast.parse(code)
containers = [node for node in tree.body if isinstance(node, ast.Try)]
body = next(node.body for node in containers if any(isinstance(item, ast.For) and ast.unparse(item.iter) == "summary['profile_rows']" for item in node.body))
start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'results' for target in node.targets))
loop = next(i for i in range(start, len(body)) if isinstance(body[i], ast.For) and ast.unparse(body[i].iter) == "summary['profile_rows']")
exec(compile(ast.Module(body=body[start:loop + 1], type_ignores=[]), core_path, 'exec'), globals())
if len(results) != expected_traces or len(bounded) != expected_traces:
    raise RuntimeError('LSA_TRACE_HEALTHY_WINDOWS_MISSING')
# Reuse full-window API/callchain/overlap analyzer, including cases with no batch NVTX.
with urllib.request.urlopen('https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/8ceaf6309db16e94e90325eb81b3032a7751207d/scripts/nsys_sync_callsites.py', timeout=60) as response:
    callsite_code = response.read(1024 * 1024)
namespace = {'__name__': 'weighted_callsite_analysis'}
exec(compile(callsite_code, 'nsys_sync_callsites.py', 'exec'), namespace)
full_window = []
for profile in profiles:
    for rel in profile['sqlite']:
        record = namespace['summarize'](root / rel)
        full_window.append(dict(configuration=profile['configuration'], database=rel, evidence=record))
(out / 'full-window-callsite-aggregate.json').write_text(json.dumps(full_window, indent=2))
(out / 'bounded-aggregate.json').write_text(json.dumps(dict(source=summary['source'], status='EVIDENCE_EXTRACTED', analyzer_ref=core_ref, results=bounded), indent=2))
(out / 'summary.json').write_text(json.dumps(dict(status='EVIDENCE_EXTRACTED', source=summary['source'], analyzer_ref=core_ref, results=results,
    claim_boundary='Measured profile intervals and correlation only; no automatic CPU-free, critical-path, occupancy or A/B speed claim.'), indent=2))
print('WEIGHTED_SCOPED_EVIDENCE_EXTRACTED', len(results), flush=True)

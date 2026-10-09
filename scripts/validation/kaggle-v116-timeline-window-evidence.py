"""Bounded remote-only inspection of retained v116 Nsight SQLite exports.
This emits evidence, not an automatic dependency or performance admission.
"""
import collections
import json
import sqlite3
from pathlib import Path

out = Path('/kaggle/working/timeline-analysis')
out.mkdir(exist_ok=True)
roots = list(Path('/kaggle/input').rglob('lsa-bfs-gate/summary.json'))
assert len(roots) == 1, roots
root = roots[0].parent
summary = json.loads(roots[0].read_text())
assert summary['status'] == 'TYPED_FOLLOWUP_PASS'
assert summary['source'] == '2fda8d25e84a39ebc3c1420a35e9eb2dc7be4e1f'
results = []
for case in summary['typed_runs']:
    assert case['pass']
    for rel in case['rank_sqlite']:
        path = root / rel
        db = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        schema = {t: [r[1] for r in db.execute('PRAGMA table_info(' + t + ')')]
                  for t in tables if t.startswith(('NVTX','CUPTI_ACTIVITY'))}
        ranges = [dict(r) for r in db.execute(
            "SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE text IN "
            "('mgbfs.batch','mgbfs.FinalizeDepth','mgbfs.archive_d2h') AND end IS NOT NULL ORDER BY start")]
        apis = [dict(r) for r in db.execute(
            "SELECT a.*,s.value AS api FROM CUPTI_ACTIVITY_KIND_RUNTIME a "
            "JOIN StringIds s ON s.id=a.nameId ORDER BY a.start")]
        for table in ('CUPTI_ACTIVITY_KIND_DRIVER',):
            if table in tables:
                apis.extend(dict(r) for r in db.execute(
                    'SELECT a.*,s.value AS api FROM ' + table + ' a JOIN StringIds s ON s.id=a.nameId ORDER BY a.start'))
        kernels = [dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL')] if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables else []
        copies = [dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start')] if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables else []
        copy_by_id = collections.defaultdict(list)
        for copy in copies:
            copy_by_id[copy.get('correlationId')].append(copy)
        batches = collections.defaultdict(list)
        for rng in ranges:
            if rng['text'] == 'mgbfs.batch':
                batches[rng['globalTid']].append(rng)
        def overlap(start,end):
            events = []
            for k in kernels + copies:
                a,b=max(start,k['start']),min(end,k['end'])
                if a<b:
                    stream=(k.get('deviceId'),k.get('streamId'))
                    events.extend(((a,1,stream),(b,-1,stream)))
            events.sort(key=lambda e:e[0])
            active=collections.Counter();last=start;busy=multi=0
            for t,delta,stream in events:
                n=sum(v>0 for v in active.values())
                if n: busy+=t-last
                if n>1: multi+=t-last
                active[stream]+=delta;last=t
            return dict(busy_ns=busy,multiple_streams_ns=multi,
                scope='GPU kernel/copy interval union; multiple active streams, not stage critical-path or occupancy proof')
        windows = []
        for tid, seq in batches.items():
            for left,right in zip(seq,seq[1:]):
                start,end = left['start'],right['start']
                boundary = any(r['text']=='mgbfs.FinalizeDepth' and r['globalTid']==tid
                    and r['start']<end and r['end']>start for r in ranges)
                suspect = []
                counts = collections.Counter()
                for api in apis:
                    if api['globalTid'] != tid or api['start']>=end or api['end']<=start:
                        continue
                    counts[api['api']] += 1
                    if 'Synchronize' not in api['api'] and 'Memcpy' not in api['api']:
                        continue
                    archive = any(r['text']=='mgbfs.archive_d2h' and r['globalTid']==tid
                        and r['start']<=api['start'] and r['end']>=api['end'] for r in ranges)
                    suspect.append(dict(api=api['api'],start=api['start'],end=api['end'],
                        archive_range=archive,correlationId=api.get('correlationId'),
                        device_copies=copy_by_id.get(api.get('correlationId'),[])))
                windows.append(dict(start=start,end=end,batch_end=left['end'],
                    globalTid=tid,finalize_boundary=boundary,apis=dict(counts),sync_copy_events=suspect,gpu_overlap=overlap(start,end)))
        result = dict(case=case['label'],database=rel,scope='Same-thread batch-start to next batch-start, including gap; raw correlation evidence; no automatic dependency admission',
            schema=schema,nvtx_ranges=ranges,windows=windows,
            memcpy_kind_counts=dict(collections.Counter(str(r.get('copyKind')) for r in copies)),
            kernel_inventory=[])
        if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
            ks = schema['CUPTI_ACTIVITY_KIND_KERNEL']
            result['kernel_inventory'] = [dict(r) for r in db.execute(
                'SELECT deviceId,streamId,COUNT(*) AS calls,MIN(start) AS first_start,MAX(end) AS last_end '
                'FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId,streamId')] if {'deviceId','streamId'} <= set(ks) else []
        dest = out / (case['label'] + '-' + Path(rel).stem + '.json')
        dest.write_text(json.dumps(result,indent=2))
        results.append(dict(case=case['label'],database=rel,analysis=dest.name,
            healthy_windows=sum(not w['finalize_boundary'] for w in windows),
            boundary_windows=sum(w['finalize_boundary'] for w in windows)))
        db.close()
(out/'summary.json').write_text(json.dumps(dict(status='EVIDENCE_EXTRACTED',source=summary['source'],results=results),indent=2))
print(json.dumps(results))

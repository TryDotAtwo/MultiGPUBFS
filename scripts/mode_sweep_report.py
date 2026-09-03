"""Recompute five-run statistics; emit CSV only after auditing raw result logs."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics

def summarize(rows):
    result=dict(status='INCOMPLETE',search_s=None,mad_s=None,durable_s=None,peak_mib=None,
                requested_bytes=None,pinned_bytes=None)
    if len(rows)!=5 or {r['repetition'] for r in rows}!=set(range(5)): return result
    if any(r['status']!='COMPLETE' for r in rows):
        result['status']='FAILED'; return result
    times=[r['search_seconds'] for r in rows]
    median=statistics.median(times)
    result.update(status='COMPLETE',search_s=median,
        mad_s=statistics.median(abs(t-median) for t in times),
        durable_s=statistics.median(r['durable_run_commit_seconds'] for r in rows)
            if all('durable_run_commit_seconds' in r for r in rows) else None,
        peak_mib=max(r['smi_process_peak_mib'] for r in rows),
        requested_bytes=rows[0].get('requested_device_bytes'),pinned_bytes=rows[0].get('pinned_bytes'))
    return result

def verify(path):
    report=json.loads((path/'summary.json').read_text())
    if report['status'] not in ('COMPLETE','COMPLETE_WITH_FAILURES'): raise ValueError('INCOMPLETE_SWEEP')
    if report['source_commit']!='5482f3bb9d20db5780bf6b5c915c4d93c8cd321c': raise ValueError('SOURCE_MISMATCH')
    if report['baseline_commit']!='f0f2b8e5ee61173039ab9742f3a7756c9b6365e6': raise ValueError('BASELINE_MISMATCH')
    if len(report['configs'])!=18: raise ValueError('CONFIG_MATRIX')
    pairs={(c['generation'],c['prededup']) for c in report['configs'] if c['batch']==262144 and c['shards']==16}
    if pairs!={(g,p) for g in range(5) for p in (0,1)}: raise ValueError('GENERATION_PREDEDUP_MATRIX')
    reference=None; seen=set(); layers={}; archives={}
    for row in report['rows']:
        if row['label'] in seen: raise ValueError('DUPLICATE_RUN')
        seen.add(row['label'])
        rawpath=path/(row['label']+'.log')
        text=rawpath.read_text(errors='replace')
        if row['status']!='COMPLETE': continue
        raw=[json.loads(line) for line in text.splitlines() if line.startswith('{')]
        if len(raw)!=1 or any(row.get(k)!=v for k,v in raw[0].items()): raise ValueError('RAW_RESULT_MISMATCH')
        if not math.isfinite(row['search_seconds']) or row['search_seconds']<=0: raise ValueError('INVALID_TIME')
        if sum(row['layer_sizes'])!=row['modulus']**6: raise ValueError('CARDINALITY')
        shape=tuple(row['layer_sizes']); m=row['modulus']
        if m in layers and layers[m]!=shape: raise ValueError('LAYER_COUNTS')
        layers[m]=shape
        if not row['smi_samples'] or row['smi_process_peak_mib'] is None: raise ValueError('NO_MEMORY_SAMPLE')
        if row['phase']=='verify':
            digest=row['layer_sha256']
            if not digest: raise ValueError('NO_FULL_STATE_DIGEST')
            if reference is not None and reference!=digest: raise ValueError('FULL_STATE_DIGEST')
            reference=digest
        cfg=row['config']
        if row['batch']!=cfg['batch']: raise ValueError('BATCH')
        if 'generation' in cfg:
            if row['generation_variant']!=cfg['generation'] or int(row['prededup'])!=cfg['prededup']: raise ValueError('NATIVE_MODE')
            if row['durable_run_commit_seconds']<row['search_seconds']: raise ValueError('DURABILITY')
            obj=row['archive_object']; key=(m,cfg['id'],row['phase'])
            if not re.fullmatch('[0-9a-f]{64}',obj['sha256']): raise ValueError('ARCHIVE_DIGEST')
            if key in archives and archives[key]!=obj: raise ValueError('ARCHIVE_REPEAT_IDENTITY')
            archives[key]=obj
    result=[]
    for entry in report['comparisons']:
        cfg=entry['config']; m=entry['modulus']
        rows=[r for r in report['rows'] if r['config_id']==cfg['id'] and r['modulus']==m and r['phase']=='measure']
        stats=summarize(rows)
        if stats['status']!=entry['status']: raise ValueError('SUMMARY_STATUS')
        if stats['status']=='COMPLETE':
            for key, old in [('search_s','median_seconds'),('mad_s','mad_seconds'),('durable_s','durable_median'),('peak_mib','peak_mib')]:
                if stats[key]!=entry[old]: raise ValueError('SUMMARY_STATISTICS')
        result.append(dict(modulus=m,config=cfg['id'],generation=cfg.get('generation',''),
            prededup=cfg.get('prededup',''),batch=cfg['batch'],shards=cfg.get('shards',''),
            job_buckets=cfg.get('job_buckets',''),**stats))
    if len([r for r in result if r['modulus']==16])!=21: raise ValueError('MISSING_MODE_COMPARISONS')
    if len([r for r in result if r['modulus']==24])!=2: raise ValueError('MISSING_LARGE_COMPARISONS')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('path',type=Path); parser.add_argument('--csv',type=Path)
    args=parser.parse_args(); rows=verify(args.path)
    if args.csv:
        with args.csv.open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows,indent=2))
    print('VERIFIED_MODE_SWEEP_RAW_RESULTS_AND_STATISTICS')

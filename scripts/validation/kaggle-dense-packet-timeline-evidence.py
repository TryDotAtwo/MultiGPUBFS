import json, traceback
from pathlib import Path
probe=Path('/kaggle/working/timeline-launch-observation.json')
probe.write_text(json.dumps(dict(status='PYTHON_STARTED')))
try:
    """Bounded remote-only inspection of integrated paired payload before/candidate owners v14 Nsight SQLite exports.
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
    assert summary['status'] == 'PROFILE_PASS'
    assert summary['source'] == '5e2b1a6d9d79d83a7208c5c3ba26da8f2625c992'
    assert summary['before_source'] == 'c38235b8140e56201ec153128b97efae4f70a73a'
    assert summary['old_source'] == '4ef9ce1d16c8cef62fc610cd6d36e32e673e623b'
    assert summary['old_launcher_sha256'] == '3c4067d95a3c7d8afccc89da8f12f4981a56526dcfbf46b14ab51ea779c17754'
    expected_cases = {'old', 'new-cub', 'before-cub', 'new-cuco', 'before-cuco',
                      'new-hash-first-cub', 'before-hash-first-cub'}
    assert len(summary['profile_rows']) == 7
    assert {r['configuration'] for r in summary['profile_rows']} == expected_cases
    assert len({p for r in summary['profile_rows'] for p in r['sqlite']}) == 14
    assert summary['batch_size'] == 32768
    # These are diagnostic captures, NOT the independent v11 five-repeat panel.
    assert len(summary['rows']) == 7
    assert {r['configuration'] for r in summary['rows']} == expected_cases
    expected_layers = summary['rows'][0]['layer_sizes']
    assert len(expected_layers) == 46 and sum(expected_layers) == 3628800
    for row in summary['rows'] + summary['profile_rows']:
        assert row['status'] == 'COMPLETE'
        if row.get('profiled'):
            assert row['scope'].startswith('CUDA profiler start/stop bounds measured BFS only;')
        else:
            assert row['archive_disabled'] is True
        assert row['layer_sizes'] == expected_layers
        expected_source = (summary['old_source'] if row['configuration'] == 'old'
                           else summary['before_source'] if row['configuration'].startswith('before-')
                           else summary['source'])
        assert row['source'] == expected_source
        assert len(row['rank_results']) == 2
        if row['configuration'] != 'old':
            assert all(r.get('archive_enabled') is False for r in row['rank_results'])
    assert all(record['mode'] == 'required' and record['hit'] is True
               for record in summary['compiled_cache'].values())
    assert summary['paired_native_artifacts']['independent_directories'] is True
    print('SCOPED_BEFORE_CANDIDATE_WINDOW_ANALYSIS_STARTED', flush=True)
    results = []
    bounded = []
    for row in summary['profile_rows']:
        assert row['status'] == 'COMPLETE' and row['profiled']
        assert len(row['sqlite']) == 2
        case = dict(label=row['configuration'], rank_sqlite=row['sqlite'])
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
            def overlap(start,end,device=None):
                events = []
                for k in kernels + copies:
                    if device is not None and k.get('deviceId') != device:
                        continue
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
            osrt = []
            if 'OSRT_API' in tables:
                cols = {r[1] for r in db.execute('PRAGMA table_info(OSRT_API)')}
                if {'start','end','globalTid','nameId'} <= cols:
                    osrt = [dict(r) for r in db.execute('SELECT a.*,s.value AS api FROM OSRT_API a JOIN StringIds s ON s.id=a.nameId ORDER BY a.start')]
            devices = sorted({k.get('deviceId') for k in kernels+copies if k.get('deviceId') is not None})
            windows = []
            for tid, seq in batches.items():
                for left,right in zip(seq,seq[1:]):
                    start,end = left['start'],right['start']
                    boundary = any(r['text']=='mgbfs.FinalizeDepth' and r['globalTid']==tid
                        and r['start']<end and r['end']>start for r in ranges)
                    suspect = []
                    counts = collections.Counter()
                    durations = collections.Counter()
                    for api in apis:
                        if api['globalTid'] != tid or api['start']>=end or api['end']<=start:
                            continue
                        counts[api['api']] += 1
                        durations[api['api']] += min(end,api['end'])-max(start,api['start'])
                        if 'Synchronize' not in api['api'] and 'Memcpy' not in api['api']:
                            continue
                        archive = any(r['text']=='mgbfs.archive_d2h' and r['globalTid']==tid
                            and r['start']<=api['start'] and r['end']>=api['end'] for r in ranges)
                        suspect.append(dict(api=api['api'],start=api['start'],end=api['end'],
                            archive_range=archive,correlationId=api.get('correlationId'),
                            device_copies=copy_by_id.get(api.get('correlationId'),[])))
                    host_sleeps = [dict(api=a['api'],start=a['start'],end=a['end'],overlap_ns=min(end,a['end'])-max(start,a['start']))
                        for a in osrt if a['globalTid']==tid and a['start']<end and a['end']>start
                        and any(name in a['api'].lower() for name in ('sleep','yield','poll','wait'))]
                    windows.append(dict(start=start,end=end,batch_end=left['end'],
                        globalTid=tid,finalize_boundary=boundary,apis=dict(counts),api_durations_ns=dict(durations),sync_copy_events=suspect,host_wait_events=host_sleeps,gpu_overlap=overlap(start,end),gpu_overlap_by_device={str(d):overlap(start,end,d) for d in devices}))
            result = dict(case=case['label'],database=rel,scope='Same-thread batch-start to next batch-start, including gap; raw correlation evidence; no automatic dependency admission',
                schema=schema,nvtx_ranges=ranges,windows=windows,
                memcpy_kind_counts=dict(collections.Counter(str(r.get('copyKind')) for r in copies)),
                kernel_inventory=[])
            if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
                ks = schema['CUPTI_ACTIVITY_KIND_KERNEL']
                result['kernel_inventory'] = [dict(r) for r in db.execute(
                    'SELECT deviceId,streamId,COUNT(*) AS calls,MIN(start) AS first_start,MAX(end) AS last_end '
                    'FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId,streamId')] if {'deviceId','streamId'} <= set(ks) else []
            # Measured-search-only capture; initialization and untimed warmup excluded.
            # No profiler duration is substituted for the independent paired A/B.
            result['capture_scope'] = row['scope']
            result['osrt_inventory'] = [dict(api=name,calls=sum(a['api']==name for a in osrt),summed_duration_ns=sum(a['end']-a['start'] for a in osrt if a['api']==name)) for name in sorted({a['api'] for a in osrt})]
            result['nvtx_event_inventory'] = [dict(r) for r in db.execute("SELECT eventType,text,COUNT(*) AS calls,SUM(end IS NULL) AS null_ends,MIN(start) AS first_start,MAX(end) AS last_end FROM NVTX_EVENTS GROUP BY eventType,text")]
            result['nvtx_samples'] = [dict(r) for r in db.execute("SELECT * FROM NVTX_EVENTS WHERE text IN ('mgbfs.batch','mgbfs.FinalizeDepth') ORDER BY start LIMIT 12")]
            result['nvtx_inventory'] = [dict(r) for r in db.execute('SELECT text,COUNT(*) AS calls FROM NVTX_EVENTS GROUP BY text')] if 'NVTX_EVENTS' in tables else []
            result['host_api_inventory'] = [dict(r) for r in db.execute(
                'SELECT s.value AS api,COUNT(*) AS calls,SUM(a.end-a.start) AS api_duration_ns '
                'FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id=a.nameId '
                'GROUP BY s.value ORDER BY api_duration_ns DESC')]
            if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
                cols = set(schema['CUPTI_ACTIVITY_KIND_KERNEL'])
                name_col = next((c for c in ('demangledName','shortName','nameId') if c in cols), None)
                assert name_col, 'KERNEL_NAME_COLUMN_UNAVAILABLE'
                result['named_kernels'] = [dict(r) for r in db.execute(
                    'SELECT s.value AS kernel,k.deviceId,k.streamId,COUNT(*) AS calls,'
                    'SUM(k.end-k.start) AS summed_gpu_duration_ns '
                    'FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.'+name_col+
                    ' GROUP BY s.value,k.deviceId,k.streamId ORDER BY summed_gpu_duration_ns DESC')]
            result['copy_inventory'] = [dict(r) for r in db.execute(
                'SELECT copyKind,bytes,COUNT(*) AS calls,SUM(end-start) AS summed_gpu_duration_ns '
                'FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind,bytes ORDER BY calls DESC')] if copies else []
            result['full_capture_overlap'] = overlap(
                min((k['start'] for k in kernels+copies), default=0),
                max((k['end'] for k in kernels+copies), default=0))
            healthy = [w for w in windows if not w['finalize_boundary']]
            healthy_apis = collections.Counter()
            for w in healthy:
                healthy_apis.update(w['api_durations_ns'])
            result['healthy_window_aggregate'] = dict(
                windows=len(healthy), wall_ns=sum(w['end']-w['start'] for w in healthy),
                host_wait_ns=sum(e['overlap_ns'] for w in healthy for e in w['host_wait_events']),
                api_durations_ns=dict(healthy_apis),
                gpu_by_device={str(d):dict(
                    busy_ns=sum(w['gpu_overlap_by_device'][str(d)]['busy_ns'] for w in healthy),
                    multiple_streams_ns=sum(w['gpu_overlap_by_device'][str(d)]['multiple_streams_ns'] for w in healthy))
                    for d in devices},
                scope='same-thread intra-depth batch windows only; profiler overhead included; not unprofiled A/B or causal speedup')
            if summary.get('batch_size') == 32768 and case['label'] != 'old':
                assert healthy, 'NO_INTRA_DEPTH_BATCH_WINDOWS'
            healthy_counts = collections.Counter()
            healthy_small_copies = collections.Counter()
            for window in healthy:
                healthy_counts.update(window['apis'])
                for event in window['sync_copy_events']:
                    for device_copy in event['device_copies']:
                        if device_copy.get('bytes', 1 << 30) <= 256:
                            identity = (str(device_copy.get('copyKind')), device_copy.get('bytes'))
                            healthy_small_copies[identity] += 1
            bounded.append(dict(case=case['label'], database=rel,
                capture_scope=result['capture_scope'],
                healthy_window_aggregate=result['healthy_window_aggregate'],
                healthy_api_counts=dict(healthy_counts),
                healthy_correlated_small_copy_counts=[dict(copy_kind=k[0],bytes=k[1],calls=v)
                    for k,v in sorted(healthy_small_copies.items())],
                full_capture_host_api_inventory=result['host_api_inventory'],
                full_capture_overlap=result['full_capture_overlap'],
                full_capture_copy_inventory=result['copy_inventory'],
                top_named_kernels=result['named_kernels'][:32],
                boundary_windows=sum(w['finalize_boundary'] for w in windows),
                old_batch_ranges_absent=case['label']=='old' and not windows,
                claim_boundary='API duration is not recoverable critical-path time. Missing old NVTX windows do not mean zero old waits. Small correlated copies are raw byte/kind evidence, not automatically counts.'))
            dest = out / (case['label'] + '-' + Path(rel).stem + '.json')
            dest.write_text(json.dumps(result,indent=2))
            results.append(dict(case=case['label'],database=rel,analysis=dest.name,
                healthy_windows=sum(not w['finalize_boundary'] for w in windows),
                boundary_windows=sum(w['finalize_boundary'] for w in windows)))
            db.close()
    # Compare both-rank sums, not PID ordering or unrelated process durations.
    # Old has no batch annotations; never invent healthy-window old zeros.
    paired = []
    for before, candidate in [('before-cub', 'new-cub'),
                              ('before-cuco', 'new-cuco'),
                              ('before-hash-first-cub', 'new-hash-first-cub')]:
        panel = {}
        for label in (before, candidate):
            records = [r for r in bounded if r['case'] == label]
            assert len(records) == 2, 'PAIRED_TRACE_RANK_INVENTORY'
            small = collections.Counter()
            for record in records:
                for copy in record['healthy_correlated_small_copy_counts']:
                    small[(copy['copy_kind'], copy['bytes'])] += copy['calls']
            panel[label] = dict(
                kernel_launches=sum(k['calls'] for record in records
                    for k in record['full_capture_host_api_inventory']
                    if k['api'] in ('cudaLaunchKernel', 'cudaLaunchKernelExC')
                    or any(k['api'].startswith(base + '_v')
                           and k['api'][len(base) + 2:].isdigit()
                           for base in ('cudaLaunchKernel', 'cudaLaunchKernelExC'))),
                healthy_windows=sum(r['healthy_window_aggregate']['windows'] for r in records),
                gpu_busy_ns=sum(r['full_capture_overlap']['busy_ns'] for r in records),
                multiple_streams_ns=sum(r['full_capture_overlap']['multiple_streams_ns'] for r in records),
                correlated_small_copies=[dict(copy_kind=kind, bytes=size, calls=count)
                    for (kind, size), count in sorted(small.items())])
        paired.append(dict(before=before, candidate=candidate, panel=panel))
    (out/'paired-delta.json').write_text(json.dumps(dict(
        status='EVIDENCE_EXTRACTED', comparisons=paired,
        claim_boundary='Profiler-count/interval comparison only, not critical-path attribution, occupancy or unprofiled speedup. Host-sized count dependencies remain explicit.'), indent=2))
    (out/'bounded-aggregate.json').write_text(json.dumps(dict(source=summary['source'],status='EVIDENCE_EXTRACTED',results=bounded),indent=2))
    (out/'summary.json').write_text(json.dumps(dict(status='EVIDENCE_EXTRACTED',source=summary['source'],results=results),indent=2))
    print(json.dumps(results))
except Exception:
    probe.write_text(json.dumps(dict(status='ANALYZER_ERROR',traceback=traceback.format_exc()),indent=2))
    print(traceback.format_exc(),flush=True)
    raise
else:
    probe.write_text(json.dumps(dict(status='ANALYZER_FINISHED')))

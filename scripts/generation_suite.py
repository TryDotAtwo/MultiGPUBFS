"""Fixed CUTLASS variants: isolated stage measurements and full BFS A/B."""
import json
from pathlib import Path
import statistics

from single_gpu_bench import worker, timing_stats


def suite(source, out, env):
    report = dict(schema=1, status='INCOMPLETE', rows=[], comparisons=[],
        variants={0:'legacy_64x32x64',1:'transpose_64x32x32',2:'transpose_128x32x32',
                  3:'transpose_64x32x64',4:'transpose_64x32x32_u4_vector_output'},
        scope='one physical T4; unchanged owner/hash/sort; experimental unarchived BFS')
    def save():
        (out/'summary.json').write_text(json.dumps(report,indent=2))
    try:
        for count in [4096,65536,262144,524280,1048576]:
            for variant in (range(5) if count in (4096,262144,1048576) else reversed(range(5))):
                if variant==0 and count>524280:
                    report['rows'].append(dict(phase='micro',variant=variant,parents=count,status='UNSUPPORTED_GRID'))
                    continue
                label=f'generate-v{variant}-n{count}'
                row=worker([str(source/'target/release/examples/generation_bench'),str(variant),str(count)],out,label,env)
                row['phase']='micro'
                report['rows'].append(row); save()
                if row['status']!='COMPLETE': raise ValueError('generation worker failed')
        reference=None
        for variant in range(5):
            label=f'full-state-v{variant}'
            cfg=dict(env,MGBFS_BENCH_GENERATION=str(variant),MGBFS_BENCH_CAPACITY='15625')
            row=worker([str(source/'target/release/examples/single_gpu_bench'),'5','4096','1','verify'],out,label,cfg)
            row.update(phase='verify',variant=variant); report['rows'].append(row); save()
            if row['status']!='COMPLETE': raise ValueError('verification worker failed')
            if reference is None: reference=row['layer_sha256']
            if reference!=row['layer_sha256']: raise ValueError('variant full-state digest mismatch')
        # Same large graph, capacity and batch for every variant: only generation changes.
        groups={v:[] for v in range(5)}
        counts=None
        for rep in range(5):
            for variant in (range(5) if rep%2==0 else reversed(range(5))):
                label=f'bfs-m16-v{variant}-r{rep}'
                cfg=dict(env,MGBFS_BENCH_GENERATION=str(variant),MGBFS_BENCH_CAPACITY=str(16**6),MGBFS_BENCH_RESERVE_GIB='1')
                row=worker([str(source/'target/release/examples/single_gpu_bench'),'16','262144','1','time'],out,label,cfg,timeout=1800)
                row.update(phase='bfs',variant=variant,repetition=rep); report['rows'].append(row); save()
                if row['status']!='COMPLETE': raise ValueError('full BFS failed')
                if counts is None: counts=row['layer_sizes']
                if counts!=row['layer_sizes']: raise ValueError('BFS layer counts mismatch')
                groups[variant].append(row)
        for variant,rows in groups.items():
            report['comparisons'].append(dict(variant=variant,**timing_stats(rows),
                device_allocation_delta_bytes=max(r['cuda_fixed_allocation_delta_bytes'] for r in rows),
                smi_process_peak_mib=max(r['smi_process_peak_mib'] for r in rows)))
        report['status']='COMPLETE'
        return report
    finally:
        save()

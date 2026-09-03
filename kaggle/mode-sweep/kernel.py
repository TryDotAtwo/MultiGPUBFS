"""Fixed-source, mandatory-archive mode sweep; every timed worker is fresh."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import urllib.request

SOURCE = '5482f3bb9d20db5780bf6b5c915c4d93c8cd321c'
BASELINE = 'f0f2b8e5ee61173039ab9742f3a7756c9b6365e6'
CUTLASS = 'ffa119a1255d78998536107466cc7097ecefa393'

def configurations():
    result = []
    def add(g, pre, batch=262144, shards=16):
        result.append(dict(id=f'g{g}-p{pre}-b{batch}-h{shards}', generation=g,
                           prededup=pre, batch=batch, shards=shards,
                           job_buckets=min(16, 256//shards)))
    for g in range(5):
        for pre in (0, 1): add(g, pre)
    for shards in (1, 4, 64): add(0, 1, shards=shards)
    for g, batch in ((0,65536),(0,524280),(4,65536),(4,524280),(4,1048576)):
        add(g, 1, batch)
    return result

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def main():
    root=Path(tempfile.mkdtemp(prefix='mgbfs-modes-',dir='/tmp'))
    logs=Path('/kaggle/working/mode-sweep'); logs.mkdir()
    helper=root/'bootstrap.py'
    urllib.request.urlretrieve('https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/77bc9a1f8d8d8bd096912f4b2df2e34e5652fba5/kaggle/native-primitives/kernel.py',helper)
    gate=load(helper,'gate'); env=dict(os.environ)
    env['PATH']='/usr/local/cuda/bin:'+env.get('PATH','')
    def run(cmd,name,cwd=root,timeout=900):
        return gate.run(cmd,cwd=cwd,env=env,logs=logs,name=name,timeout=timeout)
    report=dict(status='INCOMPLETE',source_commit=SOURCE,baseline_commit=BASELINE,
        configs=configurations(),rows=[],comparisons=[],
        unavailable=['HASH_FIRST','BMMA_BUCKET','native multi-rank NCCL'],
        scope='single T4; DENSE/CUB; generation and pre-dedup factorial plus separate batch/shard panels',
        prior_gate='native-runtime v4: same source, 10/10 plain and sanitizer gates',
        archive_policy='Every native worker durably writes an archive; identical repeats share a SHA256-verified gzip object. Compression and archival identity checks are outside BFS/durable timers.')
    def save(): (logs/'summary.json').write_text(json.dumps(report,indent=2))
    retained={}
    def retain(path,key):
        # Only files freshly created inside this task-owned temporary directory.
        if path.resolve().parent != root.resolve(): raise RuntimeError('ARCHIVE_PATH_SCOPE')
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4<<20),b''): digest.update(block)
        sha=digest.hexdigest()
        if key in retained:
            if retained[key]['sha256'] != sha: raise RuntimeError('NONDETERMINISTIC_ARCHIVE_'+key)
        else:
            target=logs/(key+'.archive.gz')
            with path.open('rb') as src, gzip.open(target,'wb',compresslevel=1) as dst:
                for block in iter(lambda:src.read(4<<20),b''): dst.write(block)
            check=hashlib.sha256()
            with gzip.open(target,'rb') as f:
                for block in iter(lambda:f.read(4<<20),b''): check.update(block)
            if check.hexdigest()!=sha: raise RuntimeError('ARCHIVE_COMPRESSION_MISMATCH')
            retained[key]=dict(sha256=sha,path=target.name,raw_bytes=path.stat().st_size,
                               compressed_bytes=target.stat().st_size)
        # Recoverable: exact bytes are retained in the verified gzip object.
        path.unlink()
        return retained[key]
    try:
        report['gpus']=gate.validate_gpus(run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],'inventory'))
        run(['nvidia-smi','-q'],'environment')
        source,baseline,cutlass=root/'source',root/'baseline',root/'cutlass'
        for repo,commit,path,label in [('TryDotAtwo/MultiGPUBFS',SOURCE,source,'source'),('TryDotAtwo/cayleypy',BASELINE,baseline,'baseline'),('NVIDIA/cutlass',CUTLASS,cutlass,'cutlass')]:
            gate.checkout('https://github.com/'+repo+'.git',commit,path,env,logs,label)
        env['CARGO_HOME'],env['RUSTUP_HOME']=str(root/'cargo'),str(root/'rustup')
        installer=root/'rustup.sh'; urllib.request.urlretrieve('https://sh.rustup.rs',installer)
        run(['sh',str(installer),'-y','--no-modify-path','--profile','minimal','--default-toolchain','1.75.0'],'rust-install')
        env['PATH']=str(root/'cargo/bin')+':'+env['PATH']
        build=source/'build/native-cuda'
        env['MGBFS_CUDA_LIB_DIR']=str(build)
        env['LD_LIBRARY_PATH']=str(build)+':/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
        run(['nvcc','--version'],'cuda-version')
        run(['cmake','-S','cuda','-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=75','-DCUTLASS_ROOT='+str(cutlass)],'cmake',source)
        run(['cmake','--build',str(build),'--target','mgbfs_cuda','--parallel','2'],'cuda-build',source)
        run(['cargo','build','--locked','--release','-p','mgbfs-runtime','--features','cuda','--example','native_bench'],'rust-build',source)
        bench=load(source/'scripts/single_gpu_bench.py','bench')
        device=dict(env,CUDA_VISIBLE_DEVICES=report['gpus'][0]['uuid'],PYTHONPATH=str(baseline))
        for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'): device.pop(key,None)
        def sample(cfg,m,phase,rep):
            native='generation' in cfg
            key=f'm{m}-{cfg["id"]}-{phase}'
            label=f'{key}-r{rep}'; path=root/(label+'.archive')
            worker_env=dict(device,MGBFS_BENCH_CAPACITY=str(min(m**6,32_000_000)),
                MGBFS_ARCHIVE_SLOTS=str((m**6+cfg['batch']-1)//cfg['batch']+128))
            if native:
                worker_env.update(MGBFS_BENCH_GENERATION=str(cfg['generation']),
                    MGBFS_SHARDS=str(cfg['shards']),MGBFS_JOB_BUCKETS=str(cfg['job_buckets']))
            operation='verify' if phase=='verify' else 'time'
            command=([str(source/'target/release/examples/native_bench'),str(m),str(cfg['batch']),str(cfg['prededup']),operation,str(path)]
                if native else [sys.executable,str(source/'scripts/single_gpu_bench.py'),'baseline',str(m),str(cfg['batch']),operation])
            row=bench.worker(command,logs,label,worker_env,timeout=1800)
            row.update(config_id=cfg['id'],phase=phase,repetition=rep,modulus=m,config=cfg)
            report['rows'].append(row); save()
            if row['status']=='COMPLETE' and native:
                row['archive_object']=retain(path,key); save()
            return row
        baseline_configs=[dict(id=f'cayleypy-b{b}',batch=b) for b in (65536,262144,1048576)]
        reference=sample(baseline_configs[0],8,'verify',0)
        valid=[]
        for cfg in report['configs']:
            row=sample(cfg,8,'verify',0)
            if row['status']=='COMPLETE': bench.verify_pair(row,reference); valid.append(cfg)
        configs=valid+baseline_configs
        rng=random.Random(20260903)
        for rep in range(5):
            order=list(configs); rng.shuffle(order)
            for cfg in order: sample(cfg,16,'measure',rep)
        def summarize(m,cfgs):
            result=[]
            good_layers=set()
            for cfg in cfgs:
                rows=[r for r in report['rows'] if r['modulus']==m and r['phase']=='measure' and r['config_id']==cfg['id']]
                entry=dict(modulus=m,config=cfg,status='FAILED',successful=sum(r['status']=='COMPLETE' for r in rows))
                if len(rows)==5 and all(r['status']=='COMPLETE' for r in rows):
                    good_layers.update(tuple(r['layer_sizes']) for r in rows)
                    entry.update(status='COMPLETE',**bench.timing_stats(rows),
                        peak_mib=max(r['smi_process_peak_mib'] for r in rows),
                        requested_bytes=rows[0].get('requested_device_bytes'),pinned_bytes=rows[0].get('pinned_bytes'),
                        durable_median=statistics.median(r['durable_run_commit_seconds'] for r in rows) if 'generation' in cfg else None)
                result.append(entry)
            if len(good_layers)>1: raise RuntimeError('MEASURED_LAYER_MISMATCH')
            report['comparisons'].extend(result); save(); return result
        small=summarize(16,configs)
        winners=[]
        for native in (True,False):
            good=[r for r in small if r['status']=='COMPLETE' and ('generation' in r['config'])==native]
            if not good: raise RuntimeError('NO_SUCCESSFUL_BACKEND')
            winners.append(min(good,key=lambda r:r['median_seconds'])['config'])
        report['large_selected']=winners; save()
        for rep in range(5):
            for cfg in (winners if rep%2==0 else winners[::-1]): sample(cfg,24,'measure',rep)
        summarize(24,winners)
        report['status']='COMPLETE_WITH_FAILURES' if any(r['status']!='COMPLETE' for r in report['rows']) else 'COMPLETE'
    except Exception as e:
        report['error']=str(e); raise
    finally:
        save(); print(json.dumps(report),flush=True)

if __name__=='__main__': main()

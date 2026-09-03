"""Archived single-rank native runtime gate and A/B, not a multi-rank claim."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from pathlib import Path
import sys
import tempfile
import urllib.request

SOURCE_COMMIT = '4fea92d2228c19be83ca2d16464daf5623aa21ba'
BASELINE_COMMIT = 'f0f2b8e5ee61173039ab9742f3a7756c9b6365e6'
CUTLASS_COMMIT = 'ffa119a1255d78998536107466cc7097ecefa393'

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    root = Path(tempfile.mkdtemp(prefix='mgbfs-native-', dir='/tmp'))
    logs = Path('/kaggle/working/native-runtime')
    logs.mkdir()
    helper = root/'bootstrap.py'
    urllib.request.urlretrieve('https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/77bc9a1f8d8d8bd096912f4b2df2e34e5652fba5/kaggle/native-primitives/kernel.py', helper)
    gate = load(helper, 'gate')
    env = dict(os.environ)
    env['PATH'] = '/usr/local/cuda/bin:' + env.get('PATH', '')
    def run(command, name, cwd=root, timeout=900, device_env=None):
        return gate.run(command, cwd=cwd, env=device_env or env, logs=logs, name=name, timeout=timeout)
    report = dict(status='INCOMPLETE', source_commit=SOURCE_COMMIT, baseline_commit=BASELINE_COMMIT,
                  scope='single-rank archived DENSE reference; independent tests on two T4; no NCCL', tests=[], rows=[])
    def save(): (logs/'summary.json').write_text(json.dumps(report, indent=2))
    try:
        report['gpus'] = gate.validate_gpus(run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.free','--format=csv,noheader,nounits'], 'inventory'))
        run(['nvidia-smi','-q'], 'environment-before')
        source, baseline, cutlass = root/'source', root/'baseline', root/'cutlass'
        for repo, commit, path, label in [('TryDotAtwo/MultiGPUBFS',SOURCE_COMMIT,source,'source'),
            ('TryDotAtwo/cayleypy',BASELINE_COMMIT,baseline,'baseline'), ('NVIDIA/cutlass',CUTLASS_COMMIT,cutlass,'cutlass')]:
            gate.checkout('https://github.com/'+repo+'.git', commit, path, env, logs, label)
        env['CARGO_HOME'], env['RUSTUP_HOME'] = str(root/'cargo'), str(root/'rustup')
        installer = root/'rustup.sh'
        urllib.request.urlretrieve('https://sh.rustup.rs', installer)
        run(['sh',str(installer),'-y','--no-modify-path','--profile','minimal','--default-toolchain','1.75.0'], 'rust-install')
        env['PATH'] = str(root/'cargo/bin')+':'+env['PATH']
        build = source/'build/native-cuda'
        env['MGBFS_CUDA_LIB_DIR'] = str(build)
        env['LD_LIBRARY_PATH'] = str(build)+':/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
        run(['nvcc','--version'], 'cuda-version')
        run(['cmake','-S','cuda','-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=75','-DCUTLASS_ROOT='+str(cutlass)],'cmake',source)
        run(['cmake','--build',str(build),'--target','mgbfs_cuda','--parallel','2'],'cuda-build',source)
        run(['cargo','test','--locked'],'cpu-tests',source)
        artifacts = run(['cargo','test','--locked','-p','mgbfs-runtime','--features','cuda','--test','native','--no-run','--message-format=json'],'native-test-build',source)
        executables = [json.loads(line)['executable'] for line in artifacts.splitlines() if line.startswith('{')
            and json.loads(line).get('reason')=='compiler-artifact' and json.loads(line).get('executable')]
        if len(executables)!=1: raise RuntimeError('NATIVE_TEST_INVENTORY')
        executable = executables[0]
        def test_gpu(gpu):
            device = dict(env, CUDA_VISIBLE_DEVICES=gpu['uuid'])
            results=[]
            for tool in ['plain', 'memcheck', 'racecheck', 'initcheck', 'synccheck']:
                command=[executable,'--test-threads=1','--nocapture','--skip','native_timing_probe']
                if tool=='plain': command += ['--include-ignored']
                else: command=['compute-sanitizer','--error-exitcode','99','--tool',tool]+command
                output=run(command, f"gpu{gpu['index']}-{tool}", source, 1800, device)
                if '0 failed' not in output: raise RuntimeError('NATIVE_TEST_FAILURE')
                if tool=='racecheck':
                    totals=re.findall(r'RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)',output)
                    if not totals or any(t!=('0','0','0') for t in totals): raise RuntimeError('RACECHECK_FAILURE')
                elif tool!='plain':
                    totals=re.findall(r'ERROR SUMMARY: (\d+) errors',output)
                    if not totals or any(t!='0' for t in totals): raise RuntimeError('SANITIZER_FAILURE')
                results.append(dict(gpu=gpu['uuid'],tool=tool,status='PASS'))
            return results
        with ThreadPoolExecutor(max_workers=2) as pool:
            for results in pool.map(test_gpu, report['gpus']): report['tests'].extend(results)
        save()
        run(['cargo','build','--locked','--release','-p','mgbfs-runtime','--features','cuda','--example','native_bench'],'native-bench-build',source)
        run([sys.executable,'-m','pip','freeze'],'python-environment')
        device=dict(env,CUDA_VISIBLE_DEVICES=report['gpus'][0]['uuid'],PYTHONPATH=str(baseline))
        for key in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE']: device.pop(key,None)
        sys.path.insert(0,str(source/'scripts'))
        bench=load(source/'scripts/single_gpu_bench.py','bench')
        def sample(backend,m,batch,phase,rep=0):
            label=f'm{m}-{backend}-b{batch}-{phase}-{rep}'
            operation='verify' if phase=='verify' else 'time'
            cmd=([str(source/'target/release/examples/native_bench'),str(m),str(batch),'1',operation,str(logs/(label+'.archive'))]
                 if backend=='native' else [sys.executable,str(source/'scripts/single_gpu_bench.py'),'baseline',str(m),str(batch),operation])
            # V2 measured a real 512 MiB archive-ring exhaustion at m20.
            # This is a NEW fixed pre-run configuration, never in-run resizing.
            # Provision a whole-run payload backlog plus 128 partial-run slots.
            archive_slots=max(64,(m**6+batch-1)//batch+128)
            worker_env=dict(device,MGBFS_BENCH_CAPACITY=str(min(m**6,32_000_000)),
                            MGBFS_ARCHIVE_SLOTS=str(archive_slots))
            row=bench.worker(cmd,logs,label,worker_env,timeout=1800)
            row.update(modulus=m,batch=batch,phase=phase,backend_requested=backend,repetition=rep)
            report['rows'].append(row); save(); return row
        native=sample('native',8,4096,'verify')
        baseline_row=sample('cayleypy',8,65536,'verify')
        bench.verify_pair(native,baseline_row)
        configs={}
        for backend in ['native','cayleypy']:
            batches=[65536,262144,524280] if backend=='native' else [65536,262144,1048576]
            trials=[sample(backend,16,b,'calibrate') for b in batches]
            good=[r for r in trials if r['status']=='COMPLETE']
            if not good: raise RuntimeError('NO_VALID_CONFIGURATION_'+backend)
            configs[backend]=min(good,key=lambda r:r['search_seconds'])['batch']
        report['configs']=configs
        measured={b:[] for b in configs}
        for rep in range(5):
            for backend in (['native','cayleypy'] if rep%2==0 else ['cayleypy','native']):
                row=sample(backend,16,configs[backend],'time',rep)
                if row['status']!='COMPLETE': raise RuntimeError('MEASUREMENT_FAILED')
                measured[backend].append(row)
        if len({tuple(r['layer_sizes']) for rows in measured.values() for r in rows})!=1: raise RuntimeError('LAYER_COUNTS_DIFFER')
        report['timings']={b:bench.timing_stats(rows) for b,rows in measured.items()}
        report['output_contracts']='native: mandatory durable state/hash archive; baseline: no archive'
        for m in [20,24]:
            pair=[sample(b,m,configs[b],'capacity') for b in ['native','cayleypy']]
            if any(r['status']!='COMPLETE' for r in pair): break
            if pair[0]['layer_sizes']!=pair[1]['layer_sizes']: raise RuntimeError('CAPACITY_COUNTS_DIFFER')
            if min(r['search_seconds'] for r in pair)>=60: break
        report['status']='COMPLETE_SINGLE_RANK_AB'
    except Exception as e:
        report['error']=str(e); raise
    finally:
        save()
        print(json.dumps(report),flush=True)
if __name__=='__main__': main()

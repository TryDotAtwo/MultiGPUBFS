"""m28 capacity probe of the exact source already gated in native-runtime v3."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

SOURCE='4fea92d2228c19be83ca2d16464daf5623aa21ba'
BASELINE='f0f2b8e5ee61173039ab9742f3a7756c9b6365e6'
CUTLASS='ffa119a1255d78998536107466cc7097ecefa393'
MODULUS=28
CAPACITY=48_000_000
BATCH=524280

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    root=Path(tempfile.mkdtemp(prefix='mgbfs-capacity-',dir='/tmp'))
    logs=Path('/kaggle/working/native-capacity'); logs.mkdir()
    helper=root/'bootstrap.py'
    urllib.request.urlretrieve('https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/77bc9a1f8d8d8bd096912f4b2df2e34e5652fba5/kaggle/native-primitives/kernel.py',helper)
    gate=load(helper,'gate')
    env=dict(os.environ); env['PATH']='/usr/local/cuda/bin:'+env.get('PATH','')
    def run(command,name,cwd=root,timeout=900):
        return gate.run(command,cwd=cwd,env=env,logs=logs,name=name,timeout=timeout)
    report=dict(status='INCOMPLETE',source_commit=SOURCE,baseline_commit=BASELINE,
        modulus=MODULUS,native_layer_capacity=CAPACITY,rows=[],
        scope='single-GPU capacity probe, one native trial; not a timing median or multi-rank run',
        prior_gate='mgbfs-native-runtime-t4/v3, exact same native source, 10/10 plus m16-m24 A/B')
    def save(): (logs/'summary.json').write_text(json.dumps(report,indent=2))
    try:
        report['gpus']=gate.validate_gpus(run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],'inventory'))
        run(['nvidia-smi','-q'],'environment')
        source,baseline,cutlass=root/'source',root/'baseline',root/'cutlass'
        for repo,commit,path,label in [('TryDotAtwo/MultiGPUBFS',SOURCE,source,'source'),
                ('TryDotAtwo/cayleypy',BASELINE,baseline,'baseline'),('NVIDIA/cutlass',CUTLASS,cutlass,'cutlass')]:
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
        sys.path.insert(0,str(source/'scripts')); bench=load(source/'scripts/single_gpu_bench.py','bench')
        device=dict(env,CUDA_VISIBLE_DEVICES=report['gpus'][0]['uuid'],PYTHONPATH=str(baseline),
                    MGBFS_BENCH_CAPACITY=str(CAPACITY),MGBFS_ARCHIVE_SLOTS=str((MODULUS**6+BATCH-1)//BATCH+128))
        for key in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE']: device.pop(key,None)
        for batch in [1048576,262144,65536]:
            label=f'm28-cayleypy-b{batch}'
            row=bench.worker([sys.executable,str(source/'scripts/single_gpu_bench.py'),'baseline',str(MODULUS),str(batch),'time'],logs,label,device,timeout=1800)
            row.update(backend_requested='cayleypy',batch=batch); report['rows'].append(row); save()
        label=f'm28-native-b{BATCH}'
        row=bench.worker([str(source/'target/release/examples/native_bench'),str(MODULUS),str(BATCH),'1','time',str(logs/(label+'.archive'))],logs,label,device,timeout=1800)
        row.update(backend_requested='native',batch=BATCH); report['rows'].append(row)
        report['status']='COMPLETE_CAPACITY_PROBE' if row['status']=='COMPLETE' else 'NATIVE_CAPACITY_FAILURE'
    except Exception as e:
        report['error']=str(e); raise
    finally:
        save(); print(json.dumps(report),flush=True)
if __name__=='__main__': main()

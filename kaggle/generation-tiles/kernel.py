"""Fixed generation layouts/tiles: two-T4 correctness, one-T4 timing."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

SOURCE_COMMIT = 'a8c18b57c2a480c8bc5ac99359fe543082f39db9'
CUTLASS_COMMIT = 'ffa119a1255d78998536107466cc7097ecefa393'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root=Path(tempfile.mkdtemp(prefix='mgbfs-tiles-',dir='/tmp'))
    logs=Path('/kaggle/working/generation-tiles'); logs.mkdir()
    helper=root/'bootstrap.py'
    urllib.request.urlretrieve('https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/c6b501c5e245ff15d92bfbc018c6bc25b0e68c98/kaggle/native-primitives/kernel.py',helper)
    gate=load(helper,'gate')
    env=os.environ.copy(); env['PATH']='/usr/local/cuda/bin:'+env.get('PATH','')
    def run(command,name,cwd=root,timeout=900,device_env=None):
        return gate.run(command,cwd=cwd,env=device_env or env,logs=logs,name=name,timeout=timeout)
    gpus=gate.validate_gpus(run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],'inventory'))
    run(['nvidia-smi','-q'],'gpu-environment')
    source,cutlass=root/'source',root/'cutlass'
    gate.checkout('https://github.com/TryDotAtwo/MultiGPUBFS.git',SOURCE_COMMIT,source,env,logs,'source')
    gate.checkout('https://github.com/NVIDIA/cutlass.git',CUTLASS_COMMIT,cutlass,env,logs,'cutlass')
    env['CARGO_HOME']=str(root/'cargo'); env['RUSTUP_HOME']=str(root/'rustup')
    installer=root/'rustup.sh'; urllib.request.urlretrieve('https://sh.rustup.rs',installer)
    run(['sh',str(installer),'-y','--no-modify-path','--profile','minimal','--default-toolchain','1.75.0'],'rust-install')
    env['PATH']=str(root/'cargo/bin')+':'+env['PATH']
    build=source/'build/native-cuda'
    env['MGBFS_CUDA_LIB_DIR']=str(build)
    env['LD_LIBRARY_PATH']=str(build)+':/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
    run(['nvcc','--version'],'cuda-version')
    run(['cmake','-S','cuda','-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=75','-DCUTLASS_ROOT='+str(cutlass)],'cmake',source)
    run(['cmake','--build',str(build),'--parallel','2'],'cuda-build',source)
    run(['cargo','test','--locked'],'cpu-contracts',source)
    for package,example in [('mgbfs-cuda','generation_bench'),('mgbfs-runtime','single_gpu_bench')]:
        run(['cargo','build','--locked','--release','-p',package,'--features','cuda','--example',example],example+'-build',source)
    exes={}
    for package,target in [('mgbfs-cuda','generate'),('mgbfs-runtime','ping_pong')]:
        output=run(['cargo','test','--locked','-p',package,'--features','cuda','--test',target,'--no-run','--message-format=json'],target+'-build',source)
        for line in output.splitlines():
            if line.startswith('{'):
                item=json.loads(line)
                if item.get('reason')=='compiler-artifact' and item.get('executable') and item['target']['name']==target:
                    exes[target]=item['executable']
    if set(exes)!={'generate','ping_pong'}: raise ValueError('test executable inventory')
    checks=[]
    for gpu in gpus:
        cfg=dict(env,CUDA_VISIBLE_DEVICES=str(gpu['index']))
        for tool in ['plain','memcheck','racecheck','initcheck','synccheck']:
            for target in ['generate','ping_pong']:
                command=[exes[target],'--test-threads=1','--nocapture']
                if target=='generate' and tool!='plain': command+=['--skip','large_batch_crosses_old_grid_y_boundary']
                if target=='ping_pong': command+=['generation_variants_preserve_full_layers' if tool=='plain' else 'generation_variants_small_feedback','--exact']
                if tool!='plain': command=['compute-sanitizer','--error-exitcode','99','--tool',tool]+command
                label=f"gpu{gpu['index']}-{target}-{tool}"
                run(command,label,source,timeout=1800,device_env=cfg)
                checks.append(dict(gpu=gpu['uuid'],target=target,tool=tool,status='PASS'))
                (logs/'checks.json').write_text(json.dumps(checks,indent=2))
    sys.path.insert(0,str(source/'scripts'))
    bench=load(source/'scripts/generation_suite.py','bench')
    cfg=dict(env,CUDA_VISIBLE_DEVICES='0')
    (logs/'environment.json').write_text(json.dumps(dict(source=SOURCE_COMMIT,cutlass=CUTLASS_COMMIT,gpus=gpus,rust='1.75.0',timing_gpu=0),indent=2))
    bench.suite(source,logs,cfg)
    # Hardware-counter access is optional evidence, never an algorithm fallback.
    ncu=shutil.which('ncu',path=env['PATH'])
    profile=dict(available=bool(ncu),results=[])
    if ncu:
        for variant in [0,4]:
            try:
                run([ncu,'--set','full','--clock-control','none','--kernel-name','regex:.*materialize.*','--launch-count','1','--csv',
                     str(source/'target/release/examples/generation_bench'),str(variant),'262144'],f'ncu-v{variant}',source,device_env=cfg,timeout=600)
                profile['results'].append(dict(variant=variant,status='PASS'))
            except Exception as error:
                profile['results'].append(dict(variant=variant,status='UNAVAILABLE_OR_FAILED',error=str(error)))
    (logs/'profiling.json').write_text(json.dumps(profile,indent=2))


if __name__=='__main__':
    main()

"""Two-T4 correctness and sanitizer gate for bounded macro settlement."""
import importlib.util,json,os,tempfile,urllib.request
from pathlib import Path

SOURCE='6c439424178698978ece9d8b2dc0104062c5cac8'
CUTLASS='ffa119a1255d78998536107466cc7097ecefa393'
def load(path):
    spec=importlib.util.spec_from_file_location('gate',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def main():
    root=Path(tempfile.mkdtemp(prefix='mgbfs-macro-settle-',dir='/tmp'));logs=Path('/kaggle/working/macro-settle');logs.mkdir()
    helper=root/'gate.py';urllib.request.urlretrieve('https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/c6b501c5e245ff15d92bfbc018c6bc25b0e68c98/kaggle/native-primitives/kernel.py',helper);gate=load(helper)
    env=os.environ.copy();env['PATH']='/usr/local/cuda/bin:'+env.get('PATH','')
    def run(command,name,cwd=root,timeout=1200,device_env=None):return gate.run(command,cwd=cwd,env=device_env or env,logs=logs,name=name,timeout=timeout)
    gpus=gate.validate_gpus(run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],'inventory'))
    source,cutlass=root/'source',root/'cutlass';gate.checkout('https://github.com/TryDotAtwo/MultiGPUBFS.git',SOURCE,source,env,logs,'source');gate.checkout('https://github.com/NVIDIA/cutlass.git',CUTLASS,cutlass,env,logs,'cutlass')
    env['CARGO_HOME']=str(root/'cargo');env['RUSTUP_HOME']=str(root/'rustup');installer=root/'rustup.sh';urllib.request.urlretrieve('https://sh.rustup.rs',installer);run(['sh',str(installer),'-y','--no-modify-path','--profile','minimal','--default-toolchain','1.75.0'],'rust-install');env['PATH']=str(root/'cargo/bin')+':'+env['PATH']
    build=source/'build/macro-settle';env['MGBFS_CUDA_LIB_DIR']=str(build);env['LD_LIBRARY_PATH']=str(build)+':/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
    run(['cmake','-S','cuda','-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=75','-DCUTLASS_ROOT='+str(cutlass)],'cmake',source);run(['cmake','--build',str(build),'--parallel','2'],'build',source)
    output=run(['cargo','test','--locked','-p','mgbfs-cuda','--features','cuda','--test','macro_settle','--no-run','--message-format=json'],'rust-test-build',source)
    executable=None
    for line in output.splitlines():
        if line.startswith('{'):
            item=json.loads(line)
            if item.get('reason')=='compiler-artifact' and item.get('executable') and item['target']['name']=='macro_settle':executable=item['executable']
    if not executable:raise ValueError('missing test executable')
    checks=[]
    for gpu in gpus:
        cfg=dict(env,CUDA_VISIBLE_DEVICES=str(gpu['index']))
        for tool in ['plain','memcheck','racecheck','initcheck','synccheck']:
            command=[executable,'--test-threads=1','--nocapture']
            if tool!='plain':command=['compute-sanitizer','--error-exitcode','99','--tool',tool]+command
            run(command,f"gpu{gpu['index']}-{tool}",source,device_env=cfg)
            checks.append(dict(gpu=gpu['uuid'],tool=tool,status='PASS'));(logs/'checks.json').write_text(json.dumps(checks,indent=2))
    (logs/'environment.json').write_text(json.dumps(dict(source=SOURCE,cutlass=CUTLASS,gpus=gpus),indent=2))
if __name__=='__main__':main()

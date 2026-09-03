"""Physical 2xT4 exact-state gate for the native NCCL BFS runtime."""
import importlib.util,json,os,tempfile,urllib.request
from pathlib import Path
SOURCE='d19dc4adbb12a4f467812c249a281d0ae814c68e';CUTLASS='ffa119a1255d78998536107466cc7097ecefa393'
def load(path):
    spec=importlib.util.spec_from_file_location('gate',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def oracle():
    import numpy as np
    n=4;m=2;start=np.eye(n,dtype=np.uint8);generators=[]
    for delta in [1,m-1]:
        for i in range(n-1):
            x=np.eye(n,dtype=np.uint8);x[i,i+1]=delta;generators.append(x)
    seen={start.tobytes()};layer={start.tobytes()};layers=[]
    while layer:
        layers.append(layer);future=set()
        for raw in layer:
            state=np.frombuffer(raw,dtype=np.uint8).reshape(n,n)
            for g in generators:
                child=((g.astype(np.uint16)@state.astype(np.uint16))%m).astype(np.uint8).tobytes()
                if child not in seen:future.add(child)
        seen|=future;layer=future
    return [[x.hex() for x in sorted(layer)] for layer in layers]
def main():
    root=Path(tempfile.mkdtemp(prefix='mgbfs-dist-',dir='/tmp'));logs=Path('/kaggle/working/distributed-native-smoke');logs.mkdir()
    helper=root/'gate.py';urllib.request.urlretrieve('https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/c6b501c5e245ff15d92bfbc018c6bc25b0e68c98/kaggle/native-primitives/kernel.py',helper);gate=load(helper)
    env=os.environ.copy();env['PATH']='/usr/local/cuda/bin:'+env.get('PATH','')
    def run(command,name,cwd=root,timeout=1200,extra=None):return gate.run(command,cwd=cwd,env=extra or env,logs=logs,name=name,timeout=timeout)
    gpus=gate.validate_gpus(run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],'inventory'));source,cutlass=root/'source',root/'cutlass';gate.checkout('https://github.com/TryDotAtwo/MultiGPUBFS.git',SOURCE,source,env,logs,'source');gate.checkout('https://github.com/NVIDIA/cutlass.git',CUTLASS,cutlass,env,logs,'cutlass')
    env['CARGO_HOME']=str(root/'cargo');env['RUSTUP_HOME']=str(root/'rustup');installer=root/'rustup.sh';urllib.request.urlretrieve('https://sh.rustup.rs',installer);run(['sh',str(installer),'-y','--no-modify-path','--profile','minimal','--default-toolchain','1.75.0'],'rust-install');env['PATH']=str(root/'cargo/bin')+':'+env['PATH']
    build=source/'build/distributed';env['MGBFS_CUDA_LIB_DIR']=str(build);env['LD_LIBRARY_PATH']=str(build)+':/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
    run(['cmake','-S','cuda','-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=75','-DCUTLASS_ROOT='+str(cutlass)],'cmake',source);run(['cmake','--build',str(build),'--parallel','2'],'cuda-build',source);run(['cargo','build','--locked','--release','-p','mgbfs-runtime','--features','cuda','--example','distributed_smoke'],'rust-build',source)
    expected=oracle();checks=[]
    for rank_map in ['0,1','1,0']:
        bootstrap=root/f'bootstrap-{rank_map.replace(",","")}.bin';cfg=dict(env,MGBFS_RANK_MAP=rank_map,NCCL_DEBUG='INFO')
        output=run(['torchrun','--standalone','--nproc-per-node=2','--no-python',str(source/'target/release/examples/distributed_smoke'),str(bootstrap)],f'run-{rank_map.replace(",","")}',source,300,cfg)
        rows=[json.loads(line) for line in output.splitlines() if line.startswith('{"status"')];rows.sort(key=lambda x:x['rank'])
        if len(rows)!=2:raise ValueError('rank output inventory')
        actual=[]
        for depth in range(len(expected)):
            values=sorted(rows[0]['states'][depth]+rows[1]['states'][depth])
            if len(values)!=len(set(values)) or values!=expected[depth]:raise ValueError(f'exact mismatch {rank_map} depth {depth}')
            actual.append(len(values))
        checks.append(dict(rank_map=rank_map,status='PASS',layer_sizes=actual))
    (logs/'checks.json').write_text(json.dumps(checks,indent=2));(logs/'environment.json').write_text(json.dumps(dict(source=SOURCE,cutlass=CUTLASS,gpus=gpus),indent=2))
if __name__=='__main__':main()

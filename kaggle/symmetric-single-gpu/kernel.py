"""Physical T4 S_n matrix BFS A/B gate."""
import importlib.util,json,os,sys,tempfile,urllib.request
from pathlib import Path
SOURCE='50b346650379be9da5c7101abe5b012974b8c7bb'
BASELINE='f0f2b8e5ee61173039ab9742f3a7756c9b6365e6'
CUTLASS='ffa119a1255d78998536107466cc7097ecefa393'
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def main():
    root=Path(tempfile.mkdtemp(prefix='mgbfs-symmetric-ab-',dir='/tmp'));logs=Path('/kaggle/working/symmetric-single-gpu');logs.mkdir()
    helper=root/'gate.py';urllib.request.urlretrieve('https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBFS/c6b501c5e245ff15d92bfbc018c6bc25b0e68c98/kaggle/native-primitives/kernel.py',helper);gate=load(helper,'gate')
    env=os.environ.copy();env['PATH']='/usr/local/cuda/bin:'+env.get('PATH','')
    def run(command,name,cwd=root,timeout=1800):return gate.run(command,cwd=cwd,env=env,logs=logs,name=name,timeout=timeout)
    gpus=gate.validate_gpus(run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],'inventory'));env['CUDA_VISIBLE_DEVICES']='0'
    source,baseline,cutlass=root/'source',root/'baseline',root/'cutlass';gate.checkout('https://github.com/TryDotAtwo/MultiGPUBFS.git',SOURCE,source,env,logs,'native');gate.checkout('https://github.com/TryDotAtwo/cayleypy.git',BASELINE,baseline,env,logs,'baseline');gate.checkout('https://github.com/NVIDIA/cutlass.git',CUTLASS,cutlass,env,logs,'cutlass')
    env['CARGO_HOME']=str(root/'cargo');env['RUSTUP_HOME']=str(root/'rustup');installer=root/'rustup.sh';urllib.request.urlretrieve('https://sh.rustup.rs',installer);run(['sh',str(installer),'-y','--no-modify-path','--profile','minimal','--default-toolchain','1.75.0'],'rust-install');env['PATH']=str(root/'cargo/bin')+':'+env['PATH']
    build=source/'build/symmetric';env['MGBFS_CUDA_LIB_DIR']=str(build);env['LD_LIBRARY_PATH']=str(build)+':/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','');env['PYTHONPATH']=str(baseline)
    run(['cmake','-S','cuda','-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=75','-DCUTLASS_ROOT='+str(cutlass)],'cmake',source);run(['cmake','--build',str(build),'--parallel','2'],'cuda-build',source);run(['cargo','build','--locked','--release','-p','mgbfs-runtime','--features','cuda','--example','macro_bench'],'rust-build',source)
    (logs/'environment.json').write_text(json.dumps(dict(source=SOURCE,baseline=BASELINE,cutlass=CUTLASS,measured_gpu=gpus[0],unused_gpu=gpus[1],archive='native mandatory; baseline none'),indent=2))
    bench=load(source/'scripts/symmetric_gpu_bench.py','bench');bench.suite(source/'target/release/examples/macro_bench',logs,env)
if __name__=='__main__':main()

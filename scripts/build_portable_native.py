"""Rebuild native artifacts on the runner's Linux ABI, without a GPU."""
import ast, hashlib, json, os, shutil, subprocess, sys, tarfile, urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1];work=Path('/tmp/mgbfs-portable');work.mkdir()
def run(args,env=None):subprocess.run(list(map(str,args)),check=True,env=env)
def constant(path,name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):return ast.literal_eval(node.value)
    raise RuntimeError('Missing pin '+name)
def extract(archive, destination):
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target=(destination/member.name).resolve()
            if not target.is_relative_to(destination.resolve()):raise RuntimeError('Archive path escape')
        tar.extractall(destination)
sdk=work/'cuda';sdk.mkdir()
components=constant(root/'kaggle/library-owner/kernel.py','CUDA_COMPONENTS')
for component,version,sha in components:
    name=f'{component}-linux-x86_64-{version}-archive';archive=work/(name+'.tar.xz')
    url=f'https://developer.download.nvidia.com/compute/cuda/redist/{component}/linux-x86_64/{archive.name}'
    urllib.request.urlretrieve(url,archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=sha:raise RuntimeError('Toolchain SHA mismatch '+component)
    extract(archive,work);shutil.copytree(work/name,sdk,dirs_exist_ok=True)
(sdk/'lib64').symlink_to('lib',target_is_directory=True)
cutlass_commit=constant(root/'kaggle/native-primitives/kernel.py','CUTLASS_COMMIT')
archive=work/'cutlass.tar.gz';urllib.request.urlretrieve('https://codeload.github.com/NVIDIA/cutlass/tar.gz/'+cutlass_commit,archive)
extract(archive,work);cutlass=work/('cutlass-'+cutlass_commit)
deps=work/'deps'
run([sys.executable,'-m','pip','install','--no-deps','--target',deps,'nvidia-nccl-cu12==2.30.7'])
nccl=deps/'nvidia/nccl';native=work/'native-build'
env=dict(os.environ,CUDACXX=str(sdk/'bin/nvcc'))
env['PATH']=str(sdk/'bin')+':'+env['PATH']
env['LD_LIBRARY_PATH']=str(nccl/'lib')+':'+str(sdk/'lib')
run(['cmake','-S',root/'cuda','-B',native,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=75;86','-DCMAKE_CUDA_COMPILER='+str(sdk/'bin/nvcc'),'-DBUILD_TESTING=OFF','-DMGBFS_NCCL_LSA=OFF','-DMGBFS_NCCL_ROOT='+str(nccl),'-DCUTLASS_ROOT='+str(cutlass)],env)
run(['cmake','--build',native,'--target','mgbfs_cuda','--parallel','2'],env)
env.update(MGBFS_CUDA_LIB_DIR=str(native),MGBFS_CUDART_LIB_DIR=str(sdk/'lib'),CARGO_TARGET_DIR=str(work/'target'))
run(['cargo','+1.75.0','build','--manifest-path',root/'Cargo.toml','--locked','--release','-p','mgbfs-cli','--features','cuda'],env)
print('PORTABLE_NATIVE_COMPILE_FINISHED_NOT_GPU_ACCEPTANCE',flush=True)

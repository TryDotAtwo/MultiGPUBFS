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
variant=os.environ.get('MGBFS_DIAGNOSTIC_VARIANT')
choices={'shared-legacy':('75;86','Shared'),'static-sm120':('75;86;120','Static'),'shared-sm120':('75;86;120','Shared')}
if variant is not None and variant not in choices:raise RuntimeError('UNKNOWN_DIAGNOSTIC_VARIANT')
architectures,runtime=choices.get(variant,('75;86',None))
flags=[] if runtime is None else ['-DCMAKE_CUDA_RUNTIME_LIBRARY='+runtime]
run(['cmake','-S',root/'cuda','-B',native,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES='+architectures,'-DCMAKE_CUDA_COMPILER='+str(sdk/'bin/nvcc'),'-DBUILD_TESTING=OFF','-DMGBFS_NCCL_LSA=OFF','-DMGBFS_NCCL_ROOT='+str(nccl),'-DCUTLASS_ROOT='+str(cutlass),*flags],env)
run(['cmake','--build',native,'--target','mgbfs_cuda','--parallel','2'],env)
if variant:
    library=native/'libmgbfs_cuda.so';out=work/'diagnostic-variant'/variant;out.mkdir(parents=True)
    shutil.copy2(library,out/'libmgbfs_cuda.so')
    dynamic=subprocess.check_output(['readelf','-d',str(library)],text=True)
    symbols=subprocess.check_output(['nm','--defined-only',str(library)],text=True,stderr=subprocess.DEVNULL)
    sass=subprocess.check_output([sdk/'bin/cuobjdump','--list-elf',library],text=True)
    ptx=subprocess.check_output([sdk/'bin/cuobjdump','--list-ptx',library],text=True)
    assert all('sm_'+a in sass for a in architectures.split(';'))
    assert ('libcudart.so' in dynamic)==(runtime=='Shared')
    defined=[s for s in symbols.splitlines() if s.endswith((' cudaSetDevice',' cudaGetDeviceProperties_v2'))]
    assert bool(defined)==(runtime=='Static')
    receipt={'schema':1,'id':variant,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'architectures':list(map(int,architectures.split(';'))),'runtime':runtime,'cuda_toolkit':'12.9.86','sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'elf_dynamic':dynamic,'runtime_defined_symbols':defined,'sass_inventory':sass,'ptx_inventory':ptx,'scope':'compiled/ELF-verified diagnostic variant; not GPU acceptance'}
    (out/'build-receipt.json').write_text(json.dumps(receipt,indent=2))
    print('DIAGNOSTIC_VARIANT_COMPILE_AND_ELF_VERIFIED',variant,flush=True)
    raise SystemExit(0)

env.update(MGBFS_CUDA_LIB_DIR=str(native),MGBFS_CUDART_LIB_DIR=str(sdk/'lib'),CARGO_TARGET_DIR=str(work/'target'))
run(['cargo','+1.75.0','build','--manifest-path',root/'Cargo.toml','--locked','--release','-p','mgbfs-cli','--features','cuda'],env)
print('PORTABLE_NATIVE_COMPILE_FINISHED_NOT_GPU_ACCEPTANCE',flush=True)

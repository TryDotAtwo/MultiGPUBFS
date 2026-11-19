"""Build the Rust CLI against a pinned CUDA component and emit provenance."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
from scripts.package_diagnostic_variants import cuda_source_identity

ROOT=Path(__file__).resolve().parents[1]
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(4<<20),b''):h.update(part)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--cuda-dir',type=Path,required=True)
 p.add_argument('--common-dir',type=Path,required=True)
 p.add_argument('--receipt',type=Path,required=True)
 a=p.parse_args()
 status=subprocess.check_output(['git','status','--porcelain','--untracked-files=all','--','Cargo.toml','Cargo.lock','crates','cuda'],cwd=ROOT,text=True)
 if status.strip():raise RuntimeError('CLI_DIRTY_NATIVE_SOURCE')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 proof=json.loads((a.cuda_dir/'build-receipt.json').read_text());library=a.cuda_dir/'libmgbfs_cuda.so'
 if proof['architectures']!=[86] or proof['runtime']!='Shared' or digest(library)!=proof['sha256']:
  raise RuntimeError('CLI_CUDA_COMPONENT_INPUT')
 if cuda_source_identity(ROOT,commit)['cuda_kernel_source_sha256']!=cuda_source_identity(ROOT,proof['source_commit'])['cuda_kernel_source_sha256']:
  raise RuntimeError('CLI_CUDA_SOURCE_MISMATCH')
 common=a.common_dir/'lib'
 if not (common/'libcudart.so.12').is_file() or not (common/'libnccl.so.2').is_file():raise RuntimeError('CLI_RUNTIME_MISSING')
 link=common/'libcudart.so'
 if not link.exists():link.symlink_to('libcudart.so.12')
 env=dict(os.environ,MGBFS_CUDA_LIB_DIR=str(a.cuda_dir.resolve()),MGBFS_CUDART_LIB_DIR=str(common.resolve()),LD_LIBRARY_PATH=str(a.cuda_dir.resolve())+':'+str(common.resolve()))
 subprocess.run(['cargo','+1.75.0','build','--locked','--release','-p','mgbfs-cli','--features','cuda'],cwd=ROOT,env=env,check=True)
 binary=ROOT/'target/release/mgbfs'
 info=json.loads(subprocess.check_output([str(binary),'key-info'],env=env,text=True,timeout=30))
 if info.get('segmented24_v1') is not True or info.get('permutation_bitmap_v1') is not True:
  raise RuntimeError('CLI_MEMORY_BACKEND_CAPABILITIES')
 receipt={'schema':1,'source_commit':commit,'binary_sha256':digest(binary),'cuda_input_sha256':proof['sha256'],
          'cuda_build_source':proof['source_commit'],'rustc':subprocess.check_output(['rustc','+1.75.0','--version'],text=True).strip(),
          'features':['cuda'],'capabilities':info,'scope':'Compiled CLI; GPU execution requires separate acceptance'}
 a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

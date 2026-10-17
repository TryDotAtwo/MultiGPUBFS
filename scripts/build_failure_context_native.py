"""Rebuild only the Rust CLI; preserve all immutable CUDA architecture libraries."""
import gzip,hashlib,io,json,os,shutil,subprocess,tarfile,tempfile
from pathlib import Path
from multigpubfs.native_assets import AssetManager
from scripts.package_diagnostic_variants import cuda_source_identity
from multigpubfs.artifact_metadata import elf_runtime_requirements

ROOT=Path(__file__).resolve().parents[1]
TAG='native-components-20261017-v5'
BASE='https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/'+TAG+'/'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(*args,**kw):return subprocess.run(list(map(str,args)),check=True,cwd=ROOT,**kw)
def main():
 old_bytes=(ROOT/'multigpubfs/native_assets.json').read_bytes()
 assert hashlib.sha256(old_bytes).hexdigest()=='32564ff0a869dc1cf24d8255bc5d71acfdf6a3145de7497f757f4d28efec6ba5'
 old=json.loads(old_bytes);identity=cuda_source_identity(ROOT,'HEAD')
 assert all(c['build']['cuda_kernel_source_sha256']==identity['cuda_kernel_source_sha256'] for c in old['components'].values() if 'build' in c),'CUDA_KERNEL_CHANGED'
 work=Path('/tmp/mgbfs-fault-native');work.mkdir()
 manager=AssetManager(old,work/'cache');manager.prefetch([75])
 common=work/'cache/components'/old['components'][old['common']]['sha256']
 library=work/'cache/components'/old['components'][old['production']['75']]['sha256']/'libmgbfs_cuda.so'
 vendor=work/'link';vendor.mkdir()
 for name,path in [('libmgbfs_cuda.so',library),('libcudart.so',common/'lib/libcudart.so.12'),('libcudart.so.12',common/'lib/libcudart.so.12'),('libnccl.so.2',common/'lib/libnccl.so.2')]:
  (vendor/name).symlink_to(path)
 env=dict(os.environ,MGBFS_CUDA_LIB_DIR=str(vendor),MGBFS_CUDART_LIB_DIR=str(vendor),CARGO_TARGET_DIR=str(work/'target'),LD_LIBRARY_PATH=str(vendor))
 run('cargo','+1.75.0','test','--locked','--no-fail-fast',env=env)
 run('cargo','+1.75.0','build','--locked','--release','-p','mgbfs-cli','--features','cuda',env=env)
 binary=work/'target/release/mgbfs';run('patchelf','--set-rpath','$ORIGIN/../lib',binary)
 key=subprocess.run([str(binary),'key-info'],env=env,check=True,text=True,capture_output=True,timeout=30)
 assert json.loads(key.stdout)['lossless_bitpack128_feistel_v1'] is True
 assert json.loads(key.stdout)['resident_plan_rejection_v1'] is True
 output=ROOT/'dist-fault-native';output.mkdir();archive=output/'common.tar.gz';files={}
 with archive.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=1) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
  for relative,info in sorted(old['components'][old['common']]['files'].items()):
   source=binary if relative=='bin/mgbfs' else common/relative
   item=tarfile.TarInfo(relative);item.size=source.stat().st_size;item.mode=info['mode'];item.mtime=0
   with source.open('rb') as stream:tar.addfile(item,stream)
   files[relative]={'sha256':sha(source),'size':source.stat().st_size,'mode':info['mode']}
   if relative!='bin/mgbfs':assert files[relative]==info
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 index=json.loads(old_bytes);index['id']=TAG;index['components'][index['common']]={'url':BASE+'common.tar.gz','size':archive.stat().st_size,'sha256':sha(archive),'files':files}
 index['native_metadata'].update(native_binary_source_commit=head,native_failure_context_schema=1,native_primary_error_preserved=True,native_abort_failed_generic_comm=True,native_expected_plan_rejections_nonfatal=True)
 index['native_metadata']['elf_runtime_requirements']=elf_runtime_requirements([binary,common/'lib/libcudart.so.12',common/'lib/libnccl.so.2',library])
 AssetManager(index,work/'new-schema')
 data=(json.dumps(index,sort_keys=True,separators=(',',':'))+'\n').encode();(output/'native-assets.json').write_bytes(data)
 receipt={'schema':1,'source_commit':head,'native_binary_sha256':sha(binary),'base_index_sha256':hashlib.sha256(old_bytes).hexdigest(),'index_sha256':hashlib.sha256(data).hexdigest(),'common_sha256':sha(archive),'common_size':archive.stat().st_size,'cuda_library_components_unchanged':True,'native_expected_plan_rejections_nonfatal':True,'kernel_source_sha256':identity['cuda_kernel_source_sha256'],'status':'COMPILED_CPU_TESTED_NEEDS_GPU_ACCEPTANCE','scope':'Only Rust resident plan rejection handling rebuilt; no changes to CUDA kernels, SM libraries, CUDA runtime, NCCL or BFS math'}
 (output/'component-release.json').write_text(json.dumps(receipt,indent=2))
 (output/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(output.iterdir()) if p.is_file()))
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

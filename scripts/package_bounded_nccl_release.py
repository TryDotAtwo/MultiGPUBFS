"""Replace production CUDA components with one verified bounded-transport build.

The v29 executable/runtime/NCCL and explicitly old diagnostic controls are reused.
Every production SM, including universal, must have the new transport source.
"""
import argparse,copy,gzip,hashlib,io,json,os,shutil,subprocess,tarfile
from pathlib import Path
from multigpubfs.architecture_support import NATIVE_ARCHITECTURES,PTX_ARCHITECTURES
from multigpubfs.artifact_metadata import elf_runtime_requirements
from multigpubfs.native_assets import AssetManager
from scripts.fatbinary_inventory import verify_inventory
from scripts.package_diagnostic_variants import cuda_source_identity

TAG='native-pip-20261019-v30'
BUILD_SOURCE='fcc00f7c7aca517297eb546eea5fc3c8fb74e7b1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(proof,library,target):
 arches=list(NATIVE_ARCHITECTURES) if target=='universal' else [int(target)]
 ptx=list(PTX_ARCHITECTURES) if target=='universal' else []
 if proof['source_commit']!=BUILD_SOURCE:raise ValueError('BUILD_SOURCE')
 if proof['architectures']!=arches or proof.get('ptx_architectures')!=ptx or proof['runtime']!='Shared' or proof['cuda_toolkit']!='12.9.86':raise ValueError('BUILD_TARGET')
 if sha(library)!=proof['sha256']:raise ValueError('BUILD_CHECKSUM')
 verify_inventory(proof['sass_inventory'],proof['ptx_inventory'],arches,ptx)
 return arches,ptx
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--builds',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 root=Path(__file__).resolve().parents[1];index=json.loads((root/'multigpubfs/native_assets.json').read_text())
 expected=[str(x) for x in NATIVE_ARCHITECTURES]+['universal']
 assert set(index['production'])==set(expected)
 a.output.mkdir(exist_ok=False);identity=cuda_source_identity(root,BUILD_SOURCE)
 index=copy.deepcopy(index);index['id']='bounded-nccl-20261019-v30'
 index['scope']='Bounded NCCL transport production components; GPU acceptance recorded separately. Legacy diagnostic controls are intentionally unchanged.'
 for target in expected:
  folder=a.builds/('native-component-sm'+target);proof=json.loads((folder/'build-receipt.json').read_text());original=folder/'libmgbfs_cuda.so';arches,ptx=validate(proof,original,target)
  library=a.output/('staged-'+target+'.so');shutil.copy2(original,library)
  subprocess.run(['patchelf','--set-rpath','$ORIGIN',str(library)],check=True)
  dynamic=subprocess.check_output(['readelf','-d',str(library)],text=True);assert 'libcudart.so.12' in dynamic
  name=index['production'][target];filename=name+'.tar.gz';dest=a.output/filename;data=library.read_bytes()
  with dest.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
   item=tarfile.TarInfo('libmgbfs_cuda.so');item.size=len(data);item.mode=0o644;item.mtime=0;tar.addfile(item,io.BytesIO(data))
  build=dict(proof,architectures=arches,ptx_architectures=ptx,cuda_source_tree=identity['cuda_source_tree'],cuda_kernel_source_sha256=identity['cuda_kernel_source_sha256'],packaged_sha256=sha(library),elf_dynamic=dynamic,elf_runtime_requirements=elf_runtime_requirements([library]),scope='Compiled and inventory-verified; this target requires its own GPU acceptance')
  index['components'][name]={'url':'https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/'+TAG+'/'+filename,'sha256':sha(dest),'size':dest.stat().st_size,'files':{'libmgbfs_cuda.so':{'sha256':sha(library),'size':len(data),'mode':0o644}},'build':build}
  library.unlink()
 AssetManager(index,a.output/'schema-check')
 raw=(json.dumps(index,sort_keys=True,separators=(',',':'))+'\n').encode();(a.output/'native-assets.json').write_bytes(raw);(root/'multigpubfs/native_assets.json').write_bytes(raw)
 receipt={'schema':1,'build_source':BUILD_SOURCE,'index_sha256':hashlib.sha256(raw).hexdigest(),'common_unchanged':index['common'],'production':{k:index['components'][v]['sha256'] for k,v in index['production'].items()},'scope':'All production CUDA components rebuilt; common binary and old diagnostic contrasts retained'}
 (a.output/'component-release.json').write_text(json.dumps(receipt,indent=2)+'\n')
 (a.output/'SHA256SUMS').write_text(''.join(sha(f)+'  '+f.name+'\n' for f in sorted(a.output.iterdir()) if f.is_file()))
 print(json.dumps(receipt))
if __name__=='__main__':main()

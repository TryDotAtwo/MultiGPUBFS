"""Assemble same-kernel components from CI and the accepted v18 payload."""
import argparse,gzip,hashlib,io,json,os,shutil,subprocess,tarfile,tempfile,zipfile
from pathlib import Path
from multigpubfs.architecture_support import NATIVE_ARCHITECTURES,PTX_ARCHITECTURES
from scripts.fatbinary_inventory import verify_inventory
from scripts.package_diagnostic_variants import cuda_source_identity
from multigpubfs.artifact_metadata import elf_runtime_requirements

BASE_SHA='1eda27dcd590528dc972dedffcf978c26854832e70703cb8e3d3233910769fa9'
TAG='native-components-20261015-v1'
URL='https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/'+TAG+'/'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def command(args):return subprocess.check_output(list(map(str,args)),text=True,stderr=subprocess.STDOUT,timeout=120)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base-wheel',type=Path,required=True);p.add_argument('--builds',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cuobjdump',type=Path,required=True);a=p.parse_args()
 assert sha(a.base_wheel)==BASE_SHA,'ACCEPTED_BASE_CHECKSUM';a.output.mkdir(exist_ok=False)
 root=Path(__file__).resolve().parents[1];identity=cuda_source_identity(root,'HEAD')
 with zipfile.ZipFile(a.base_wheel) as z,tempfile.TemporaryDirectory() as temp:
  work=Path(temp)
  def member(name):
   matches=[n for n in z.namelist() if n.endswith('/multigpubfs/_native/'+name) or n=='multigpubfs/_native/'+name]
   assert len(matches)==1,('NATIVE_MEMBER',name,matches);return z.read(matches[0])
  old=json.loads(member('manifest.json'));assert old['cuda_kernel_source_sha256']==identity['cuda_kernel_source_sha256'],'KERNEL_INPUTS_CHANGED'
  variants=json.loads(member('diagnostic_variants/variants.json'))
  index={'schema':1,'id':TAG,'platform':'linux_x86_64','driver_minimum':12000,'common':'common','components':{},'production':{},'diagnostics':{},'native_metadata':{k:old[k] for k in ('cuda_toolkit','native_binary_source_commit','bundled_runtime','elf_runtime_requirements') if k in old},'scope':'Pinned precompiled components. Execution acceptance is separate; no target GPU is inferred from compilation.'}
  def emit(name,files,build=None):
   filename=name+'.tar.gz';path=a.output/filename;info={}
   with path.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
    for relative,(source,mode) in sorted(files.items()):
     data=Path(source).read_bytes();item=tarfile.TarInfo(relative);item.size=len(data);item.mode=mode;item.mtime=0;tar.addfile(item,io.BytesIO(data));info[relative]={'sha256':hashlib.sha256(data).hexdigest(),'size':len(data),'mode':mode}
   row={'url':URL+filename,'sha256':sha(path),'size':path.stat().st_size,'files':info}
   if build is not None:row['build']=build
   index['components'][name]=row
  common={}
  for path in old['sha256']:
   if path not in ('bin/mgbfs','lib/libcudart.so.12','lib/libnccl.so.2') and not path.startswith('licenses/'):continue
   data=member(path);assert hashlib.sha256(data).hexdigest()==old['sha256'][path]
   destination=work/'common'/path;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data);common[path]=(destination,0o755 if path=='bin/mgbfs' else 0o644)
  assert {'bin/mgbfs','lib/libcudart.so.12','lib/libnccl.so.2'}<=set(common) and any(k.startswith('licenses/') for k in common)
  emit('common',common)
  def verify(library,proof,arches,ptx,runtime):
   subprocess.run(['patchelf','--set-rpath','$ORIGIN' if runtime=='Shared' and len(arches)==1 else '$ORIGIN',str(library)],check=True)
   dynamic=command(['readelf','-d',library]);assert ('libcudart.so' in dynamic)==(runtime=='Shared')
   sass=command([a.cuobjdump,'--list-elf',library]);portable=command([a.cuobjdump,'--list-ptx',library]);inventory=verify_inventory(sass,portable,arches,ptx)
   assert set(inventory['native_module_counts'])=={str(x) for x in arches}
   assert cuda_source_identity(root,proof['source_commit'])['cuda_kernel_source_sha256']==identity['cuda_kernel_source_sha256']
   return dict(proof,runtime=runtime,architectures=arches,ptx_architectures=ptx,cuda_source_tree=cuda_source_identity(root,proof['source_commit'])['cuda_source_tree'],cuda_kernel_source_sha256=identity['cuda_kernel_source_sha256'],packaged_sha256=sha(library),fatbinary_inventory=inventory,elf_dynamic=dynamic,sass_inventory=sass,ptx_inventory=portable,elf_runtime_requirements=elf_runtime_requirements([library]),scope='Compiled and inventory-verified; actual GPU correctness requires execution')
  for sm in NATIVE_ARCHITECTURES:
   folder=a.builds/('sm'+str(sm));proof=json.loads((folder/'build-receipt.json').read_text());source=folder/'libmgbfs_cuda.so'
   assert proof['architectures']==[sm] and proof['runtime']=='Shared' and proof['cuda_toolkit']=='12.9.86' and proof.get('ptx_architectures')==[]
   assert sha(source)==proof['sha256'] and proof['source_commit']==os.environ['NATIVE_BUILD_SOURCE']
   library=work/('sm'+str(sm)+'.so');shutil.copy2(source,library);build=verify(library,proof,[sm],[],'Shared');name='cuda-sm'+str(sm)
   emit(name,{'libmgbfs_cuda.so':(library,0o644)},build);index['production'][str(sm)]=name
  for row in variants['variants']:
   proof_bytes=member(row['receipt']);assert hashlib.sha256(proof_bytes).hexdigest()==row['receipt_sha256'];proof=json.loads(proof_bytes)
   data=member(row['library']);assert hashlib.sha256(data).hexdigest()==row['sha256'];library=work/(row['id']+'.so');library.write_bytes(data)
   # The legacy runtime path remains valid in its eventual diagnostic directory.
   if row['id']=='shared-sm120':
    build=verify(library,proof,list(NATIVE_ARCHITECTURES),list(PTX_ARCHITECTURES),'Shared');name='cuda-universal';index['production']['universal']=name
   else:
    assert proof['cuda_kernel_source_sha256']==identity['cuda_kernel_source_sha256'];build=proof;name='contrast-'+row['id'];index['diagnostics'][row['id']]=name
   emit(name,{'libmgbfs_cuda.so':(library,0o644)},build)
  # Reuse the production cache validator itself before freezing the public index.
  from multigpubfs.native_assets import AssetManager
  AssetManager(index,work/'schema-validation')
  index_bytes=(json.dumps(index,sort_keys=True,separators=(',',':'))+'\n').encode();(a.output/'native-assets.json').write_bytes(index_bytes)
  sums=[]
  for path in sorted(a.output.glob('*')):sums.append(sha(path)+'  '+path.name)
  (a.output/'SHA256SUMS').write_text('\n'.join(sums)+'\n')
  receipt={'schema':1,'source_commit':command(['git','rev-parse','HEAD']).strip(),'native_build_source':os.environ['NATIVE_BUILD_SOURCE'],'base_wheel_sha256':BASE_SHA,'index_sha256':hashlib.sha256(index_bytes).hexdigest(),'index_size':len(index_bytes),'components':{k:{'size':v['size'],'sha256':v['sha256']} for k,v in index['components'].items()},'status':'BUILT_AND_INVENTORY_VERIFIED_NEEDS_GPU_ACCEPTANCE'}
  (a.output/'component-release.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

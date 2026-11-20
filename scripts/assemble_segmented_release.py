"""Assemble segmented BFS release components from pinned, verified CI builds.

The existing CUDA/NCCL runtime archive is reused verbatim.  The executable is
separate so installations that already cached that archive transfer only the new
CLI and requested CUDA target.  GPU acceptance remains an independent step.
"""
import argparse
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from multigpubfs.architecture_support import NATIVE_ARCHITECTURES, PTX_ARCHITECTURES
from multigpubfs.artifact_metadata import elf_runtime_requirements
from multigpubfs.native_assets import AssetManager
from scripts.fatbinary_inventory import verify_inventory
from scripts.package_diagnostic_variants import cuda_source_identity

EXPECTED_BUILD_SOURCE = os.environ.get('MGBFS_COMPONENT_BUILD_SOURCE', '684cdc36f477fa84bf5181cd84d25d56acc78b2e')
COMMON_SHA = '7d44d15a64d3032974801321420fda50027ce3724dba4d6d316cb759ffcb2c99'
ROOT = Path(__file__).resolve().parents[1]

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(4 << 20),b''):h.update(part)
 return h.hexdigest()

def command(*argv):
 return subprocess.check_output(list(map(str,argv)),text=True,stderr=subprocess.STDOUT,timeout=120).strip()

def emit(index,output,tag,name,files,build=None):
 path=output/(name+'.tar.gz');metadata={}
 with path.open('wb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz, tarfile.open(fileobj=gz,mode='w') as archive:
  for relative,(source,mode) in sorted(files.items()):
   source=Path(source);entry=tarfile.TarInfo(relative);entry.size=source.stat().st_size;entry.mode=mode;entry.mtime=0;entry.uid=entry.gid=0;entry.uname=entry.gname=''
   with source.open('rb') as stream:archive.addfile(entry,stream)
   metadata[relative]={'sha256':digest(source),'size':entry.size,'mode':mode}
 row={'url':'https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/'+tag+'/'+path.name,'sha256':digest(path),'size':path.stat().st_size,'files':metadata}
 if build is not None:row['build']=build
 index['components'][name]=row

def verify_build(folder, expected_arches, expected_ptx, current_identity, work):
 folder=Path(folder);proof=json.loads((folder/'build-receipt.json').read_text());source=folder/'libmgbfs_cuda.so'
 if proof.get('source_commit')!=EXPECTED_BUILD_SOURCE or proof.get('runtime')!='Shared' or proof.get('cuda_toolkit')!='12.9.86':
  raise RuntimeError('COMPONENT_BUILD_PROVENANCE')
 if proof.get('architectures')!=expected_arches or proof.get('ptx_architectures',[])!=expected_ptx or digest(source)!=proof.get('sha256'):
  raise RuntimeError('COMPONENT_BUILD_TARGET_OR_SHA')
 identity=cuda_source_identity(ROOT,proof['source_commit'])
 if identity['cuda_kernel_source_sha256']!=current_identity['cuda_kernel_source_sha256']:
  raise RuntimeError('COMPONENT_KERNEL_SOURCE_MISMATCH')
 inventory=verify_inventory(proof.get('sass_inventory',''),proof.get('ptx_inventory',''),expected_arches,expected_ptx)
 if inventory!=proof.get('fatbinary_inventory'):
  raise RuntimeError('COMPONENT_BUILD_INVENTORY')
 name='universal' if len(expected_arches)>1 else str(expected_arches[0])
 library=Path(work)/('lib-'+name+'.so');shutil.copy2(source,library)
 subprocess.run(['patchelf','--set-rpath','$ORIGIN',str(library)],check=True)
 dynamic=command('readelf','-d',library)
 if 'libcudart.so.12' not in dynamic or 'libnccl.so.2' not in dynamic or '$ORIGIN' not in dynamic:
  raise RuntimeError('COMPONENT_LINKAGE')
 build=dict(proof,cuda_source_tree=identity['cuda_source_tree'],cuda_kernel_source_sha256=identity['cuda_kernel_source_sha256'],
            packaged_sha256=digest(library),elf_dynamic=dynamic,fatbinary_inventory=inventory,
            elf_runtime_requirements=elf_runtime_requirements([library]),
            scope='Source and compiled inventory verified; GPU execution requires separate acceptance')
 return library,build

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--builds',type=Path,required=True)
 parser.add_argument('--binary',type=Path,required=True)
 parser.add_argument('--binary-receipt',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--tag',required=True)
 parser.add_argument('--work',type=Path,required=True)
 args=parser.parse_args()
 if not args.binary.is_file():raise RuntimeError('CLI_NOT_FOUND')
 args.output.mkdir(parents=True,exist_ok=False);args.work.mkdir(parents=True,exist_ok=False)
 old=json.loads((ROOT/'multigpubfs/native_assets.json').read_text())
 common=copy.deepcopy(old['components'][old['common']])
 if common['sha256']!=COMMON_SHA:raise RuntimeError('COMMON_RUNTIME_IDENTITY')
 source=command('git','-C',ROOT,'rev-parse','HEAD')
 cli_receipt=json.loads(args.binary_receipt.read_text())
 sm86_receipt=json.loads((args.builds/'sm86/build-receipt.json').read_text())
 if os.environ.get('MGBFS_COMPONENT_BUILD_SOURCE') and (cli_receipt.get('capabilities',{}).get('segmented32_v2') is not True or cli_receipt.get('capabilities',{}).get('segmented_memory_presets_v1') is not True):
  raise RuntimeError('CLI_SEGMENTED32_ABI_MISSING')
 if cli_receipt.get('source_commit')!=source or cli_receipt.get('binary_sha256')!=digest(args.binary) or cli_receipt.get('cuda_input_sha256')!=sm86_receipt.get('sha256') or cli_receipt.get('cuda_build_source')!=EXPECTED_BUILD_SOURCE or cli_receipt.get('capabilities',{}).get('segmented24_v1') is not True or cli_receipt.get('capabilities',{}).get('permutation_bitmap_v1') is not True:raise RuntimeError('CLI_BUILD_IDENTITY')
 identity=cuda_source_identity(ROOT,'HEAD')
 common_metadata=copy.deepcopy(old['native_metadata'])
 common_metadata['native_binary_source_commit']=source
 common_metadata.pop('package_source_commit',None)
 index={'schema':1,'id':args.tag,'platform':'linux_x86_64','driver_minimum':old['driver_minimum'],
        'common':'common','executable':'native-executable','components':{'common':common},
        'production':{},'diagnostics':{},
        'retired_diagnostic_controls':copy.deepcopy(old.get('retired_diagnostic_controls',[])),
        'native_metadata':common_metadata,
        'scope':'Precompiled experimental segmented BFS components. CUDA inventory is not GPU execution acceptance.'}
 emit(index,args.output,args.tag,'native-executable',{'bin/mgbfs':(args.binary,0o755)},
      dict(cli_receipt,scope='Rust CLI, matched to the segmented CUDA source; GPU acceptance separate'))
 all_libraries=[]
 for sm in NATIVE_ARCHITECTURES:
  library,build=verify_build(args.builds/('sm'+str(sm)),[sm],[],identity,args.work)
  name='cuda-sm'+str(sm);emit(index,args.output,args.tag,name,{'libmgbfs_cuda.so':(library,0o644)},build)
  index['production'][str(sm)]=name;all_libraries.append(library)
 library,build=verify_build(args.builds/'universal',list(NATIVE_ARCHITECTURES),list(PTX_ARCHITECTURES),identity,args.work)
 emit(index,args.output,args.tag,'cuda-universal',{'libmgbfs_cuda.so':(library,0o644)},build)
 index['production']['universal']='cuda-universal';all_libraries.append(library)
 index['native_metadata']['elf_runtime_requirements']=elf_runtime_requirements([args.binary,*all_libraries])
 AssetManager(index,args.work/'schema-check')
 path=args.output/'native-assets.json';path.write_text(json.dumps(index,sort_keys=True,separators=(',',':'))+'\n')
 receipt={'schema':1,'status':'COMPONENTS_BUILT_NEEDS_GPU_ACCEPTANCE','source_commit':source,
          'cuda_build_source':EXPECTED_BUILD_SOURCE,'cuda_kernel_source_sha256':identity['cuda_kernel_source_sha256'],
          'native_index_sha256':digest(path),
          'components':{k:{'sha256':v['sha256'],'size':v['size']} for k,v in index['components'].items()},
          'scope':'Compiled target inventory; execute separately on each target hardware'}
 (args.output/'component-release.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()

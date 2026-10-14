"""Attach verified, same-source diagnostic builds; production artifacts stay intact."""
import hashlib,json,shutil,subprocess
from pathlib import Path
from multigpubfs.runtime_variants import BASELINE,SPECS

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def output(command):return subprocess.check_output(list(map(str,command)),text=True,stderr=subprocess.STDOUT,timeout=120)

def attach_variants(stage,inputs,manifest,cuobjdump):
 root=Path(__file__).resolve().parents[1];stage=Path(stage);inputs=Path(inputs)
 baseline_source=manifest['source_commit']
 def tree(commit):return output(['git','rev-parse',commit+':cuda']).strip()
 cuda_tree=tree(baseline_source);entries=[];destination=stage/'diagnostic_variants';destination.mkdir()
 for identifier,(runtime,architectures) in SPECS.items():
  folder=destination/identifier;folder.mkdir()
  if identifier==BASELINE:
   library=stage/'lib/libmgbfs_cuda.so';proof={'source_commit':baseline_source,'cuda_toolkit':manifest.get('cuda_toolkit'),'sha256':sha(library)}
  else:
   candidates=[]
   for path in inputs.rglob('build-receipt.json'):
    row=json.loads(path.read_text())
    if row.get('id')==identifier:candidates.append((path,row))
   if len(candidates)!=1:raise RuntimeError('DIAGNOSTIC_BUILD_ARTIFACT_COUNT_'+identifier)
   original,proof=candidates[0];source=original.parent/'libmgbfs_cuda.so'
   if not source.resolve().is_relative_to(inputs.resolve()) or source.is_symlink() or sha(source)!=proof['sha256']:raise RuntimeError('DIAGNOSTIC_BUILD_ARTIFACT_CHECKSUM_'+identifier)
   if proof.get('runtime')!=runtime or proof.get('architectures')!=architectures:raise RuntimeError('DIAGNOSTIC_BUILD_FACTOR_'+identifier)
   library=folder/'libmgbfs_cuda.so';shutil.copy2(source,library)
   subprocess.run(['patchelf','--set-rpath','$ORIGIN/../../lib',str(library)],check=True)
  if tree(proof['source_commit'])!=cuda_tree:raise RuntimeError('DIAGNOSTIC_BUILD_CUDA_SOURCE_MISMATCH_'+identifier)
  if proof.get('cuda_toolkit')!='12.9.86':raise RuntimeError('DIAGNOSTIC_BUILD_TOOLKIT_MISMATCH_'+identifier)
  dynamic=output(['readelf','-d',library]);symbols=output(['nm','--defined-only',library])
  defined=[s for s in symbols.splitlines() if s.endswith((' cudaSetDevice',' cudaGetDeviceProperties_v2'))]
  sass=output([cuobjdump,'--list-elf',library]);ptx=output([cuobjdump,'--list-ptx',library])
  if ('libcudart.so' in dynamic)!=(runtime=='Shared') or bool(defined)!=(runtime=='Static'):raise RuntimeError('DIAGNOSTIC_BUILD_LINKAGE_MISMATCH_'+identifier)
  if not all('sm_'+str(a) in sass for a in architectures):raise RuntimeError('DIAGNOSTIC_BUILD_CUBIN_MISSING_'+identifier)
  proof=dict(proof,id=identifier,runtime=runtime,architectures=architectures,cuda_source_tree=cuda_tree,packaged_sha256=sha(library),packaged_elf_dynamic=dynamic,runtime_defined_symbols=defined,sass_inventory=sass,ptx_inventory=ptx,scope='Compiled same-source diagnostic library; actual GPU acceptance is recorded separately')
  receipt=folder/'build-receipt.json';receipt.write_text(json.dumps(proof,indent=2))
  entries.append({'id':identifier,'runtime':runtime,'architectures':architectures,'library':str(library.relative_to(stage)),'sha256':sha(library),'receipt':str(receipt.relative_to(stage)),'receipt_sha256':sha(receipt),'source_commit':proof['source_commit'],'cuda_source_tree':cuda_tree})
 metadata={'schema':1,'production_default':BASELINE,'variants':entries,'scope':'Diagnostic-only child selection; same CUDA source, different runtime linkage and architectures; no production fallback'}
 (destination/'variants.json').write_text(json.dumps(metadata,indent=2));return metadata

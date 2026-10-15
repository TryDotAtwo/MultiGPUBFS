"""Verified build contrasts for isolated diagnostic children only.

The production package uses a fixed shared-runtime multi-architecture build. Alternative
builds are selected only in isolated diagnostic children, with no silent fallback.
"""
import hashlib,json,os
from pathlib import Path
from .architecture_support import NATIVE_ARCHITECTURES

BASELINE='baseline-static-legacy'
PRODUCTION='shared-sm120'
UNIVERSAL='shared-universal'
SPECS={BASELINE:('Static',[75,86]),'shared-legacy':('Shared',[75,86]),'static-sm120':('Static',list(NATIVE_ARCHITECTURES)),'shared-sm120':('Shared',list(NATIVE_ARCHITECTURES))}

def digest(path):
 value=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for part in iter(lambda:stream.read(4<<20),b''):value.update(part)
 return value.hexdigest()

def inventory(root=None,*,environ=None):
 if root is None:
  from .native_distribution import native_root
  root=native_root(prepare=True)
 root=Path(root).resolve()
 env=os.environ if environ is None else environ
 override=env.get('MGBFS_EXECUTABLE')
 if override and Path(override).resolve()!=(root/'bin/mgbfs').resolve():
  return {'status':'UNAVAILABLE','reason':'Custom native executable selected; packaged diagnostic builds are not substituted.'}
 metadata=root/'diagnostic_variants/variants.json'
 if not metadata.is_file():return {'status':'UNAVAILABLE','reason':'This installation has no packaged diagnostic runtime/native-code contrasts.'}
 manifest=json.loads((root/'manifest.json').read_text())
 if manifest.get('schema')!=1:raise RuntimeError('DIAGNOSTIC_VARIANT_MANIFEST_SCHEMA')
 hashes=manifest.get('sha256',{});lazy=bool(manifest.get('native_delivery'))
 def checked(name,expected=None):
  path=root/name
  if not path.resolve().is_relative_to(root) or path.is_symlink():raise RuntimeError('DIAGNOSTIC_VARIANT_PATH')
  expected=hashes.get(name) if expected is None else expected
  if expected is None or not path.is_file() or digest(path)!=expected:raise RuntimeError('DIAGNOSTIC_VARIANT_CHECKSUM_'+name)
  return path
 checked('diagnostic_variants/variants.json');checked('bin/mgbfs')
 for name in ('lib/libcudart.so.12','lib/libnccl.so.2'):checked(name)
 data=json.loads(metadata.read_text())
 if data.get('schema')!=1 or data.get('production_default')!=PRODUCTION:raise RuntimeError('DIAGNOSTIC_VARIANT_METADATA_SCHEMA')
 entries=data.get('variants',[])
 expected_specs=dict(SPECS)
 if lazy:expected_specs[UNIVERSAL]=('Shared',list(NATIVE_ARCHITECTURES))
 if len(entries)!=len(expected_specs) or {x.get('id') for x in entries}!=set(expected_specs):raise RuntimeError('DIAGNOSTIC_VARIANT_MATRIX')
 sources=set();verified=[]
 fingerprints=any('cuda_kernel_source_sha256' in row for row in entries)
 for row in entries:
  identifier=row['id'];runtime,architectures=expected_specs[identifier]
  if lazy and identifier==PRODUCTION:architectures=manifest.get('architectures')
  if row.get('runtime')!=runtime or row.get('architectures')!=architectures:raise RuntimeError('DIAGNOSTIC_VARIANT_FACTOR_'+identifier)
  expected='lib/libmgbfs_cuda.so' if identifier==PRODUCTION else 'diagnostic_variants/'+identifier+'/libmgbfs_cuda.so'
  receipt='diagnostic_variants/'+identifier+'/build-receipt.json'
  if row.get('library')!=expected or row.get('receipt')!=receipt:raise RuntimeError('DIAGNOSTIC_VARIANT_PATH_'+identifier)
  path=root/expected;installed=path.is_file()
  if installed or not lazy or identifier==PRODUCTION:path=checked(expected,row.get('sha256'))
  elif path.is_symlink() or not path.resolve().is_relative_to(root):raise RuntimeError('DIAGNOSTIC_VARIANT_PATH_'+identifier)
  proof=json.loads(checked(receipt,row.get('receipt_sha256')).read_text())
  if not row.get('sha256') or not row.get('receipt_sha256'):raise RuntimeError('DIAGNOSTIC_VARIANT_CHECKSUM_METADATA_'+identifier)
  if not row.get('cuda_source_tree') or row.get('cuda_source_tree')!=proof.get('cuda_source_tree') or proof.get('id')!=identifier or proof.get('runtime')!=runtime or proof.get('architectures')!=architectures:raise RuntimeError('DIAGNOSTIC_VARIANT_SOURCE_'+identifier)
  fingerprint=row.get('cuda_kernel_source_sha256')
  if fingerprints and (not isinstance(fingerprint,str) or len(fingerprint)!=64 or set(fingerprint)-set('0123456789abcdef') or proof.get('cuda_kernel_source_sha256')!=fingerprint):raise RuntimeError('DIAGNOSTIC_VARIANT_SOURCE_'+identifier)
  sources.add(fingerprint if fingerprints else row['cuda_source_tree']);verified.append(dict(row,library_path=str(path.resolve()),build_receipt=proof,installed=installed,verification='CHECKSUM_VERIFIED' if installed else 'PINNED_DECLARATION_NOT_INSTALLED'))
 if len(sources)!=1:raise RuntimeError('DIAGNOSTIC_VARIANT_SOURCE_MISMATCH')
 return {'schema':1,'status':'AVAILABLE','root':str(root),'binary':str(root/'bin/mgbfs'),'binary_sha256':hashes['bin/mgbfs'],'lazy_delivery':lazy,'variants':verified,'scope':'checksummed same-source diagnostic libraries; isolated children only; production default unchanged'}

def environment(identifier,verified,parent=None,*,deadline=None):
 if (identifier not in SPECS and identifier!=UNIVERSAL) or verified.get('status')!='AVAILABLE':raise RuntimeError('DIAGNOSTIC_VARIANT_SELECTION')
 selected=next(x for x in verified['variants'] if x['id']==identifier)
 root=Path(verified['root']);env=dict(os.environ if parent is None else parent)
 if verified.get('lazy_delivery'):
  from .native_assets import ensure_variant
  ensure_variant(root,identifier,deadline=deadline)
  if digest(selected['library_path'])!=selected['sha256']:raise RuntimeError('DIAGNOSTIC_VARIANT_CHECKSUM_'+identifier)
 directories=[str(Path(selected['library_path']).parent),str(root/'lib')]
 driver=Path('/usr/lib/x86_64-linux-gnu')
 if driver.is_dir():directories.append(str(driver))
 if env.get('LD_LIBRARY_PATH'):directories.append(env['LD_LIBRARY_PATH'])
 env.update(MGBFS_EXECUTABLE=verified['binary'],LD_LIBRARY_PATH=':'.join(dict.fromkeys(directories)))
 return env

def confirm_resolution(identifier,verified,libraries):
 expected=next(x for x in verified['variants'] if x['id']==identifier)
 observed=libraries.get('libmgbfs_cuda.so')
 if not observed or Path(observed).resolve()!=Path(expected['library_path']).resolve() or digest(observed)!=expected['sha256']:raise RuntimeError('DIAGNOSTIC_VARIANT_RESOLUTION_'+identifier)
 for name in ('libcudart.so.12','libnccl.so.2'):
  if name not in libraries or Path(libraries[name]).resolve()!=(Path(verified['root'])/'lib'/name).resolve():raise RuntimeError('DIAGNOSTIC_VARIANT_VENDOR_RESOLUTION_'+name)
 return {'id':identifier,'runtime':expected['runtime'],'architectures':expected['architectures'],'sha256':expected['sha256'],'resolved_libraries':libraries,'same_cuda_source_tree':expected['cuda_source_tree']}

def cases(devices,*,production_only=False,contrasts_only=False,include_universal=False):
 if production_only and contrasts_only:raise ValueError('Select one diagnostic partition')
 result=[]
 identifiers=(PRODUCTION,) if production_only else ('static-sm120',BASELINE,'shared-legacy') if contrasts_only else (PRODUCTION,'static-sm120',BASELINE,'shared-legacy')
 # Complete production API, resident and BFS checks before legacy JIT contrasts.
 if include_universal and not production_only:identifiers+=(UNIVERSAL,)
 for identifier in identifiers:
  for path in ('api','persistent','bfs'):
   for device in devices:
    if path=='api':config={'kind':'api','device':device,'module':'EAGER','data_loading':'EAGER','order':'engine-first','control_thread':True}
    elif path=='persistent':config={'kind':'component','stage':'persistent','device':device,'seconds':40}
    else:config={'kind':'bfs','fixture':'permutation','devices':[device],'history':'hash','generator':'cuda','shards':1}
    config['variant']=identifier
    result.append({'name':f'deep-build-{identifier}-gpu{device}-{path}','config':config,'seconds':40 if path!='bfs' else 45})
 return result

def choose_coverage(steps,devices):
 observed={}
 for step in steps:
  config=step.get('diagnostic_case',{})
  if config.get('coverage_build') or config.get('variant') not in (*SPECS,UNIVERSAL):continue
  path='persistent' if config.get('stage')=='persistent' else config.get('kind')
  if path not in ('api','persistent','bfs'):continue
  device=config.get('device',(config.get('devices') or [None])[0])
  observed[(config['variant'],device,path)]=step
 for identifier in (PRODUCTION,'static-sm120',BASELINE,'shared-legacy',UNIVERSAL):
  evidence=[observed.get((identifier,device,path)) for device in devices for path in ('api','persistent','bfs')]
  if evidence and all(x is not None and x['status']=='PASS' for x in evidence):
   return {'status':'VERIFIED_FOR_SYNTHETIC_COVERAGE','id':identifier,'devices':devices,'evidence':[x['name'] for x in evidence],'scope':'Subsequent diagnostic probes only; baseline failures remain in the report; no production repair or default change'}
 return {'status':'UNAVAILABLE','id':None,'reason':'No common build passed API/kernel, native resident inventory and exact S3 BFS on every selected GPU; no unverified alternative is used.'}

def call_boundary(phase):
 if not phase:return None
 if 'hardware-cudaSetDevice-enter' in phase:return 'C++ wrapper entered; cudaSetDevice return was not observed'
 if 'hardware-cudaGetDeviceProperties-enter' in phase:return 'C++ cudaSetDevice returned; cudaGetDeviceProperties return was not observed'
 if 'inventory-hardware-info' in phase or 'graph-info-hardware-info' in phase:return 'Executable reached hardware wrapper call; no later wrapper marker was recorded'
 return 'Last flushed native marker: '+phase

def compare(steps):
 universal=any(x.get('diagnostic_case',{}).get('variant')==UNIVERSAL for x in steps)
 comparison_specs=(*SPECS,UNIVERSAL) if universal else tuple(SPECS)
 shared_control=UNIVERSAL if universal else PRODUCTION
 groups={}
 for step in steps:
  config=step.get('diagnostic_case',{});identifier=config.get('variant')
  if identifier not in comparison_specs or config.get('coverage_build'):continue
  path='persistent' if config.get('stage')=='persistent' else config.get('kind')
  device=config.get('device',(config.get('devices') or [None])[0]);key=(device,path)
  groups.setdefault(key,{})[identifier]={'name':step['name'],'status':step['status'],'seconds':step.get('seconds'),'reason':step.get('reason'),'last_native_phase':step.get('last_native_phase'),'call_boundary':call_boundary(step.get('last_native_phase')) if step['status'] in ('FAIL','TIMEOUT') else None,'last_api':step.get('partial_evidence',{}).get('probe.json',{}).get('last_operation')}
 comparisons=[]
 for (device,path),values in groups.items():
  statuses=[values.get(identifier,{}).get('status','NOT_RUN') for identifier in comparison_specs]
  if any(x in ('NOT_RUN','SKIPPED','RUNNING') for x in statuses):outcome='INCOMPLETE'
  elif all(x=='PASS' for x in statuses):outcome='ALL_PASS'
  elif any(x=='PASS' for x in statuses):outcome='VARIANT_DEPENDENT'
  else:outcome='ALL_FAILED'
  controlled=[]
  for a,b,factor in [(BASELINE,'shared-legacy','runtime'),('static-sm120',shared_control,'runtime'),(BASELINE,'static-sm120','native-fatbinary'),('shared-legacy',shared_control,'native-fatbinary')]:
   left=values.get(a,{});right=values.get(b,{})
   controlled.append({'factor':factor,'left':a,'right':b,'left_status':left.get('status','NOT_RUN'),'right_status':right.get('status','NOT_RUN'),'changed_pass_outcome':None if left.get('status') in (None,'SKIPPED') or right.get('status') in (None,'SKIPPED') else (left['status']=='PASS')!=(right['status']=='PASS')})
  comparisons.append({'device':device,'path':path,'outcome':outcome,'variants':values,'controlled_pairs':controlled})
 return {'root_cause_confirmed':False,'comparisons':comparisons,'scope':'Same CUDA kernel inputs, controlled build configurations, fresh processes. Selective delivery adds a deferred universal shared control so the Shared/Static comparison keeps identical architecture sets. Status/time correlations are not proof of a unique cause; inherited JIT cache is not cleared and shared runtime does not remove every vendor runtime instance.'}

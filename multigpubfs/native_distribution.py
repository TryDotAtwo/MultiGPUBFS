"""Resolve an installed native bundle, with no dependency on the source checkout."""
import hashlib,json,os,shutil,subprocess,sys,sysconfig
from pathlib import Path

def _stream_digest(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(4<<20),b''):digest.update(chunk)
 return digest.hexdigest()

def _file_digest(path):
 from .generic_session import _active
 from .cluster import current_cluster
 owner=_active if _active is not None else current_cluster()
 if owner is None:return _stream_digest(path)
 from .autotune import _stat_identity
 key=(str(Path(path).resolve()),_stat_identity(path))
 cache=getattr(owner,'file_digests',None)
 if cache is None:cache={};owner.file_digests=cache
 if key not in cache:
  if len(cache)>=64:cache.clear()
  cache[key]=_stream_digest(path)
 return cache[key]

def _external_ranks():
 # The cluster launcher exports WORLD_SIZE for both torchrun and SLURM.
 world=os.environ.get('WORLD_SIZE','1')
 return world.isdecimal() and int(world)>1

def native_root(*,prepare=False):
 """Resolve a full installed bundle or this package's verified user cache."""
 root=Path(__file__).resolve().parent/'_native'
 if (root/'bin/mgbfs').is_file():return root
 index=Path(__file__).resolve().parent/'native_assets.json'
 if not index.is_file():return root
 from .native_assets import ensure_runtime,current_root
 universal=_external_ranks()
 return ensure_runtime(index,universal=universal) if prepare else current_root(index,universal=universal) or root

def prepare_for_activation():
 """Prefetch outside the first BFS budget, when a GPU node is present."""
 package=Path(__file__).resolve().parent;index=package/"native_assets.json"
 if os.environ.get("MGBFS_EXECUTABLE") or (package/"_native/bin/mgbfs").is_file() or not index.is_file() or not os.path.exists("/dev/nvidiactl"):return
 from .native_assets import current_root
 if current_root(index,universal=_external_ranks()) is None:native_runtime()

def native_runtime(explicit=None):
 env=dict(os.environ)
 override=explicit or env.get('MGBFS_EXECUTABLE')
 if override:return str(override),env
 root=native_root(prepare=True);binary=root/'bin/mgbfs'
 if binary.is_file():
  manifest=json.loads((root/'manifest.json').read_text())
  if manifest.get('schema')!=1:raise RuntimeError('NATIVE_BUNDLE_MANIFEST_SCHEMA')
  for name in manifest['sha256']:
   if _file_digest(root/name)!=manifest['sha256'][name]:raise RuntimeError('NATIVE_BUNDLE_CHECKSUM_'+name)
  libraries=[str(root/'lib')];pure=Path(sysconfig.get_paths()['purelib'])
  toolkit=None if manifest.get('bundled_runtime') else manifest.get('cuda_toolkit')
  if toolkit:
   major=int(toolkit.split('.')[0]);location=pure/'nvidia'/('cu'+str(major))/'lib'
   if location.is_dir():libraries.append(str(location))
  for name in (() if manifest.get('bundled_runtime') else ('cuda_runtime','nccl')):
   location=pure/'nvidia'/name/'lib'
   if location.is_dir():libraries.append(str(location))
  # Packaged NCCL/runtime must precede unrelated system versions.
  # Keep the real host driver directory before inherited toolkit stubs.
  driver=Path('/usr/lib/x86_64-linux-gnu')
  if driver.is_dir():libraries.append(str(driver))
  inherited=env.get('LD_LIBRARY_PATH','')
  if inherited:libraries.append(inherited)
  env['LD_LIBRARY_PATH']=':'.join(libraries)
  return str(binary),env
 binary=shutil.which('mgbfs')
 if binary:return binary,env
 raise RuntimeError('MGBFS_EXECUTABLE_NOT_FOUND: install a Linux CUDA native wheel or build scripts/build_native_wheel.py and install its wheel')

_nccl_profile_digests={}

def _with_profile_nccl(identity,dependency,env):
 # An external config or loader interposition cannot be attested by a library digest.
 if dependency is None or 'NCCL_CONF_FILE' in env or env.get('LD_PRELOAD') or env.get('LD_AUDIT'):
  return identity
 paths={path for library,path,*_ in dependency['resolved_dependencies']
        if library=='libnccl.so' or library.startswith('libnccl.so.')}
 if len(paths)!=1:return identity
 path=paths.pop()
 from .autotune import _stat_identity
 from .process_control import profile_timeout
 stamp=_stat_identity(path)
 if stamp is None or not 0<stamp[2]<=1<<30 or not Path(path).is_file():return identity
 key=(path,stamp)
 digest=_nccl_profile_digests.get(key)
 if digest is None:
  code="""import hashlib, sys
digest = hashlib.sha256()
with open(sys.argv[1], 'rb') as library:
    for block in iter(lambda: library.read(4 << 20), b''):
        digest.update(block)
print(digest.hexdigest())
"""
  try:
   result=subprocess.run([sys.executable,'-I','-c',code,path],capture_output=True,text=True,
                         timeout=profile_timeout(1))
  except (OSError,subprocess.TimeoutExpired):return identity
  digest=result.stdout.strip()
  if (result.returncode or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)
      or _stat_identity(path)!=stamp):return identity
  if len(_nccl_profile_digests)>=8:_nccl_profile_digests.clear()
  _nccl_profile_digests[key]=digest
 return dict(identity,nccl_library_sha256=digest)

def _runtime_identity_uncached(native,env,*,profile_nccl=False):
 """Cold provenance for the artifacts actually resolved by the launcher."""
 binary=Path(native);digest=hashlib.sha256(binary.read_bytes()).hexdigest()
 python_root=Path(__file__).resolve().parent
 python_digest=hashlib.sha256()
 for path in sorted(python_root.glob('*.py')):
  python_digest.update(path.name.encode());python_digest.update(b'\0');python_digest.update(path.read_bytes());python_digest.update(b'\0')
 identity={'native_sha256':digest,'python_runtime_sha256':python_digest.hexdigest(),'source_commit':None,'cuda_library_sha256':None,'cuda_toolkit':None,'architectures':None}
 from .autotune import _dependency_identity
 dependency=_dependency_identity(str(binary),env)
 if dependency is not None:identity['cuda_library_sha256']=dependency['cuda_library_sha256']
 root=binary.parent.parent;manifest_path=root/'manifest.json'
 if manifest_path.is_file():
  manifest=json.loads(manifest_path.read_text());library=root/'lib/libmgbfs_cuda.so'
  if manifest.get('schema')==1 and manifest.get('sha256',{}).get('bin/mgbfs')==digest and library.is_file():
   library_digest=hashlib.sha256(library.read_bytes()).hexdigest()
   if library_digest!=manifest['sha256'].get('lib/libmgbfs_cuda.so'):raise RuntimeError('NATIVE_IDENTITY_LIBRARY_CHECKSUM')
   # Package provenance applies only if the actual loader resolves its bytes.
   if dependency is not None and dependency['cuda_library_sha256']==library_digest:
    identity.update(source_commit=manifest.get('native_binary_source_commit',manifest.get('source_commit')),cuda_library_source_commit=manifest.get('cuda_library_source_commit',manifest.get('source_commit')),ptx_architectures=manifest.get('ptx_architectures'),cuda_toolkit=None if manifest.get('bundled_runtime') else manifest.get('cuda_toolkit'),architectures=manifest.get('architectures'))
 return _with_profile_nccl(identity,dependency,env) if profile_nccl else identity

_runtime_identity_cache={}
def runtime_identity(native,env,*,profile_nccl=False):
 from .generic_session import _active
 if _active is None:return _runtime_identity_uncached(native,env,profile_nccl=profile_nccl)
 from .autotune import _dependency_identity,_stat_identity
 dependency=_dependency_identity(native,env)
 if dependency is None:return _runtime_identity_uncached(native,env)
 root=Path(__file__).resolve().parent
 files=[Path(native),*sorted(root.glob('*.py')),Path(native).parent.parent/'manifest.json']
 key=(tuple((str(p),_stat_identity(p)) for p in files),dependency['cuda_library_sha256'])
 if key not in _runtime_identity_cache:
  _runtime_identity_cache.clear();_runtime_identity_cache[key]=_runtime_identity_uncached(native,env)
 identity=dict(_runtime_identity_cache[key])
 return _with_profile_nccl(identity,dependency,env) if profile_nccl else identity

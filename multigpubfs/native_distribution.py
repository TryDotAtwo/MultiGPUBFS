"""Resolve an installed native bundle, with no dependency on the source checkout."""
import hashlib,json,os,shutil,sysconfig
from pathlib import Path

def native_runtime(explicit=None):
 env=dict(os.environ)
 override=explicit or env.get('MGBFS_EXECUTABLE')
 if override:return str(override),env
 root=Path(__file__).resolve().parent/'_native';binary=root/'bin/mgbfs'
 if binary.is_file():
  manifest=json.loads((root/'manifest.json').read_text())
  if manifest.get('schema')!=1:raise RuntimeError('NATIVE_BUNDLE_MANIFEST_SCHEMA')
  for name in ('bin/mgbfs','lib/libmgbfs_cuda.so'):
   if hashlib.sha256((root/name).read_bytes()).hexdigest()!=manifest['sha256'][name]:raise RuntimeError('NATIVE_BUNDLE_CHECKSUM_'+name)
  libraries=[str(root/'lib')];pure=Path(sysconfig.get_paths()['purelib'])
  toolkit=manifest.get('cuda_toolkit')
  if toolkit:
   major=int(toolkit.split('.')[0]);location=pure/'nvidia'/('cu'+str(major))/'lib'
   if location.is_dir():libraries.append(str(location))
  for name in ('cuda_runtime','nccl'):
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

def runtime_identity(native,env):
 """Cold provenance for the artifacts actually resolved by the launcher."""
 binary=Path(native);digest=hashlib.sha256(binary.read_bytes()).hexdigest()
 python_root=Path(__file__).resolve().parent
 python_digest=hashlib.sha256()
 for path in sorted(python_root.glob('*.py')):
  python_digest.update(path.name.encode());python_digest.update(b'\0');python_digest.update(path.read_bytes());python_digest.update(b'\0')
 identity={'native_sha256':digest,'python_runtime_sha256':python_digest.hexdigest(),'source_commit':None,'cuda_library_sha256':None,'cuda_toolkit':None,'architectures':None}
 root=binary.parent.parent;manifest_path=root/'manifest.json'
 if manifest_path.is_file():
  manifest=json.loads(manifest_path.read_text());library=root/'lib/libmgbfs_cuda.so'
  if manifest.get('schema')==1 and manifest.get('sha256',{}).get('bin/mgbfs')==digest and library.is_file():
   library_digest=hashlib.sha256(library.read_bytes()).hexdigest()
   if library_digest!=manifest['sha256'].get('lib/libmgbfs_cuda.so'):raise RuntimeError('NATIVE_IDENTITY_LIBRARY_CHECKSUM')
   identity.update(source_commit=manifest.get('source_commit'),cuda_library_sha256=library_digest,cuda_toolkit=manifest.get('cuda_toolkit'),architectures=manifest.get('architectures'))
 else:
  from .autotune import _dependency_identity
  dependency=_dependency_identity(str(binary),env)
  if dependency is not None:identity['cuda_library_sha256']=dependency['cuda_library_sha256']
 return identity

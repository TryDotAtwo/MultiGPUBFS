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

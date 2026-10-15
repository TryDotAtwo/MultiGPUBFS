"""Build trace-only CLI changes against immutable compatible CUDA ABI, on CI."""
import hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(root/'scripts/reuse_portable_native.py'),'native-pip-20261014-v14'],check=True)
work=Path('/tmp/mgbfs-portable');receipt=json.loads((work/'native-manifest.json').read_text())
assert hashlib.sha256((work/'accepted.whl').read_bytes()).hexdigest()=='5fa2adb9d048e2025b873f3b616cb1530c01ad4a5cb8877348e7da993181fa69'
vendor=work/'cli-link-vendor';vendor.mkdir()
with zipfile.ZipFile(work/'accepted.whl') as z:
 for name in z.namelist():
  marker='multigpubfs/_native/lib/'
  if marker in name and name.endswith(('.so.12','.so.2')):
   basename=name.split(marker,1)[1]
   if '/' not in basename:(vendor/basename).write_bytes(z.read(name))
for name in ('libcudart.so.12','libnccl.so.2'):
 assert hashlib.sha256((vendor/name).read_bytes()).hexdigest()==receipt['sha256']['lib/'+name]
(vendor/'libcudart.so').symlink_to('libcudart.so.12')
env=dict(os.environ,MGBFS_CUDA_LIB_DIR=str(work/'native-build'),MGBFS_CUDART_LIB_DIR=str(vendor),CARGO_TARGET_DIR=str(work/'cli-target'),LD_LIBRARY_PATH=str(work/'native-build')+':'+str(vendor))
subprocess.run(['cargo','+1.75.0','build','--locked','--release','-p','mgbfs-cli','--features','cuda'],cwd=root,env=env,check=True)
binary=work/'cli-target/release/mgbfs';out=work/'cli-artifact';out.mkdir();import shutil;shutil.copy2(binary,out/'mgbfs')
trace=work/'count-startup.log';probe_env=dict(env,MGBFS_DIAGNOSTIC_TRACE_FILE=str(trace))
probe=subprocess.run([str(binary),'graph-count'],env=probe_env,text=True,capture_output=True,timeout=30)
text=trace.read_text()
assert 'cli-main-enter' in text and 'cli-policy-ready' in text
assert text.index('graph-count-cudaGetDeviceCount-enter')<text.index('graph-count-cudaGetDeviceCount-returned')
value={'schema':1,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'cuda_source_tree':subprocess.check_output(['git','rev-parse','HEAD:cuda'],cwd=root,text=True).strip(),'trace_probe_exit_code':probe.returncode,'trace':text,'scope':'Compiled CLI and exact count-call logging on CPU runner; no GPU acceptance'}
(out/'cli-receipt.json').write_text(json.dumps(value,indent=2));print('COMPILED_NATIVE_COUNT_TRACE_VERIFIED_NO_GPU',value['sha256'],flush=True)

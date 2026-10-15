"""Package Python-only updates around an immutable accepted native payload."""
import hashlib,json,shutil,subprocess,sys,tempfile,urllib.request,zipfile
from pathlib import Path

def extract_native(archive,root):
 import re
 root=Path(root).resolve();stage=root/'multigpubfs/_native';copied=0
 for name in archive.namelist():
  match=re.fullmatch(r'(?:[^/]+[.]data/platlib/)?(multigpubfs/_native/.+)',name)
  if not match or name.endswith('/'):continue
  path=(root/match[1]).resolve()
  if not path.is_relative_to(stage):raise RuntimeError('NATIVE_PAYLOAD_PATH')
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(name));copied+=1
 if not copied:raise RuntimeError('NATIVE_PAYLOAD_NOT_FOUND')
 return copied

def main():
 root=Path(__file__).resolve().parents[1];stage=root/'multigpubfs/_native'
 if stage.exists():raise RuntimeError('Refusing to replace native staging')
 url='https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261015-v16/multigpubfs-0.2.0.dev16-py3-none-linux_x86_64.whl'
 with tempfile.TemporaryDirectory() as d:
  wheel=Path(d)/'accepted.whl';urllib.request.urlretrieve(url,wheel)
  if hashlib.sha256(wheel.read_bytes()).hexdigest()!='65941655242e7a48fa3f27989992fe5f8d5d898c716d481ce0c92cdd9d4a5a6b':raise RuntimeError('ACCEPTED_NATIVE_WHEEL_CHECKSUM')
  with zipfile.ZipFile(wheel) as z:extract_native(z,root)
 (stage/'bin/mgbfs').chmod(0o755)
 sys.path.insert(0,str(root))
 from multigpubfs.runtime_variants import inventory
 before=inventory()
 if before['status']!='AVAILABLE':raise RuntimeError('NATIVE_INVENTORY_UNAVAILABLE')
 manifest=json.loads((stage/'manifest.json').read_text())
 for name,value in manifest['sha256'].items():
  if hashlib.sha256((stage/name).read_bytes()).hexdigest()!=value:raise RuntimeError('NATIVE_BASE_CHECKSUM_'+name)
 manifest['reused_native_release']='native-pip-20261015-v16'
 manifest['package_source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
 (stage/'manifest.json').write_text(json.dumps(manifest,indent=2))
 output=root/'dist-native';output.mkdir(exist_ok=False)
 subprocess.run([sys.executable,'-m','build','--wheel','--outdir',str(output)],cwd=root,check=True)
 wheel=next(output.glob('*.whl'));sha=hashlib.sha256(wheel.read_bytes()).hexdigest()
 receipt={'schema':1,'wheel':wheel.name,'sha256':sha,'size':wheel.stat().st_size,'manifest':manifest,'status':'BUILT_NEEDS_GPU_ACCEPTANCE','native_payload_unchanged':True}
 (output/'manifest.json').write_text(json.dumps(receipt,indent=2));(output/'SHA256SUMS').write_text(sha+'  '+wheel.name+'\n')
 print(json.dumps({k:receipt[k] for k in ('wheel','sha256','size','native_payload_unchanged')}))

if __name__=='__main__':main()

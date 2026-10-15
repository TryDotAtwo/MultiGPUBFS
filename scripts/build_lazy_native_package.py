"""Build a small Python-only wheel with a pinned native component index."""
import hashlib,json,subprocess,sys,zipfile
from pathlib import Path
def main():
 root=Path(__file__).resolve().parents[1];index=root/'multigpubfs/native_assets.json'
 if not index.is_file() or (root/'multigpubfs/_native').exists():raise RuntimeError('LAZY_WHEEL_REQUIRES_INDEX_WITHOUT_NATIVE_STAGING')
 from multigpubfs.native_assets import AssetManager
 value=json.loads(index.read_text());AssetManager(value,Path('/tmp/native-schema-only'))
 output=root/'dist-native';output.mkdir(exist_ok=False)
 subprocess.run([sys.executable,'-m','build','--wheel','--outdir',str(output)],cwd=root,check=True)
 wheel=next(output.glob('*.whl'));raw=wheel.read_bytes();assert wheel.name.endswith('-py3-none-any.whl') and len(raw)<2_000_000
 with zipfile.ZipFile(wheel) as z:
  assert not any('/_native/' in n or n.endswith(('.so','.so.12','.so.2')) for n in z.namelist())
  assert json.loads(z.read('multigpubfs/native_assets.json'))==value
  info=json.loads(z.read('multigpubfs/_package_info.json'));assert info['source_commit']==subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
 receipt={'schema':1,'wheel':wheel.name,'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw),'source_commit':info['source_commit'],'native_index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),'native_components':value['id'],'status':'BUILT_NEEDS_INSTALLED_GPU_ACCEPTANCE'}
 (output/'manifest.json').write_text(json.dumps(receipt,indent=2));(output/'SHA256SUMS').write_text(receipt['sha256']+'  '+wheel.name+'\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

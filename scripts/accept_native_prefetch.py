"""Acceptance of the public prefetch package on a CPU-only Linux host."""
import argparse,contextlib,ctypes,hashlib,json,os,shutil,subprocess,sys,tempfile,time,urllib.request
from pathlib import Path
from unittest.mock import patch

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--wheel-url',required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 assert args.wheel_url.startswith('https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/')
 m=json.loads(urllib.request.urlopen(args.wheel_url.rsplit('/',1)[0]+'/manifest.json',timeout=60).read())
 assert m['sha256']==args.sha256 and m['size']<2_000_000
 import importlib.metadata,multigpubfs
 from multigpubfs import native_assets as assets
 assert importlib.metadata.version('multigpubfs')=='0.2.0.dev20'
 assert not Path('/dev/nvidiactl').exists() and not list(Path('/dev').glob('nvidia[0-9]*')),'Acceptance requires a CPU-only host'
 package=Path(multigpubfs.__file__).parent;before={str(f.relative_to(package)):digest(f) for f in package.glob('*.py')};before['native_assets.json']=digest(assets.index_path())
 index=json.loads(assets.index_path().read_text());cache=Path(tempfile.mkdtemp(prefix='mgbfs-prefetch-cpu-'))/'shared-cache'
 receipt={'schema':1,'status':'RUNNING','package':'0.2.0.dev20','package_source_commit':m['source_commit'],'wheel_sha256':m['sha256'],'wheel_bytes':m['size'],'native_index_sha256':m['native_index_sha256'],'platform':'Linux x86_64 GitHub CPU-only runner','gpu_device_nodes':[],'scope':'Actual GitHub component prefetch and CPU cache checks; no GPU execution is claimed.','checks':[]}
 started=time.monotonic()
 def record(name,**data):
  receipt['checks'].append(dict(name=name,status='PASS',**data));args.output.write_text(json.dumps(receipt,indent=2));print('PREFETCH_CPU_CHECK',json.dumps(receipt['checks'][-1]),flush=True)
 def forbidden(*a,**k):raise AssertionError('CPU prefetch attempted subprocess/driver/runtime library access')
 @contextlib.contextmanager
 def guard():
  with patch.object(subprocess,'Popen',side_effect=forbidden),patch.object(ctypes,'CDLL',side_effect=forbidden),patch.object(ctypes,'PyDLL',side_effect=forbidden),patch.object(assets,'run_hardware_probe',side_effect=forbidden):yield
 with guard():plan=multigpubfs.prefetch_native(all_architectures=True,cache=cache,dry_run=True)
 expected={index['common'],*index['production'].values()};total=sum(index['components'][n]['size'] for n in expected)
 assert plan['status']=='PLAN_ONLY' and plan['maximum_additional_download_bytes']==total and plan['downloaded_bytes']==0 and not cache.exists()
 record('all-dry-run-no-gpu-no-writes',maximum_bytes=total,component_count=len(expected))
 with guard():selected=multigpubfs.prefetch_native(sm=[100,120],cache=cache)
 names={index['common'],index['production']['100'],index['production']['120']};selected_bytes=sum(index['components'][n]['size'] for n in names)
 assert selected['status']=='PREFETCHED' and selected['downloaded_bytes']==selected_bytes and {r['component'] for r in selected['transfers']}==names
 record('selected-real-github-prefetch-on-cpu',targets=selected['targets'],downloaded_bytes=selected['downloaded_bytes'],gpu_calls_forbidden=True)
 with guard():all_builds=multigpubfs.prefetch_native(all_architectures=True,cache=cache)
 assert all_builds['status']=='PREFETCHED' and all_builds['downloaded_bytes']==total-selected_bytes and all_builds['maximum_additional_download_bytes']==total-selected_bytes
 assert not any(r['component']==index['common'] for r in all_builds['transfers'])
 assert set(all_builds['targets'])==set(index['production'])
 record('all-real-github-prefetch-reuses-common',targets=all_builds['targets'],additional_downloaded_bytes=all_builds['downloaded_bytes'],total_downloaded_bytes=selected_bytes+all_builds['downloaded_bytes'],diagnostic_controls_downloaded=False)
 verified=[]
 for name in sorted(expected):
  c=index['components'][name];folder=cache/'components'/c['sha256']
  for rel,f in c['files'].items():
   actual=folder/rel;assert actual.stat().st_size==f['size'] and digest(actual)==f['sha256']
  verified.append({'component':name,'archive_sha256':c['sha256'],'file_count':len(c['files'])})
 record('independent-cache-file-hashes',components=verified)
 with guard(),patch.object(urllib.request,'urlopen',side_effect=forbidden),patch.dict(os.environ,{'MGBFS_NATIVE_OFFLINE':'1'}):warm=multigpubfs.prefetch_native(all_architectures=True,cache=cache)
 assert warm['downloaded_bytes']==0 and warm['maximum_additional_download_bytes']==0
 record('all-warm-prefetch-network-forbidden',downloaded_bytes=0,gpu_calls_forbidden=True)
 env=dict(os.environ,MGBFS_NATIVE_CACHE=str(cache),MGBFS_NATIVE_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
 entry=Path(sys.executable).with_name('multigpubfs');assert entry.is_file()
 fresh=subprocess.run([str(entry),'prefetch','--all','--json'],env=env,capture_output=True,text=True,check=True,timeout=180)
 result=json.loads(fresh.stdout);assert result['status']=='PREFETCHED' and result['downloaded_bytes']==0 and result['maximum_additional_download_bytes']==0
 record('fresh-process-installed-cli-all-offline',downloaded_bytes=0,gpu_accessed=result['gpu_accessed'])
 after={str(f.relative_to(package)):digest(f) for f in package.glob('*.py')};after['native_assets.json']=digest(assets.index_path())
 assert before==after and assets.status()['status']=='NOT_PREPARED';assert hashlib.sha256(assets.index_path().read_bytes()).hexdigest()==m['native_index_sha256']
 record('installed-package-and-native-index-unchanged',no_runtime_or_gpu_initialized=True)
 receipt.update(status='PASS',wall_seconds=time.monotonic()-started);args.output.write_text(json.dumps(receipt,indent=2));print('PREFETCH_CPU_ACCEPTANCE_PASS',json.dumps(receipt),flush=True)
 shutil.rmtree(cache.parent)
 return 0

if __name__=='__main__':raise SystemExit(main())

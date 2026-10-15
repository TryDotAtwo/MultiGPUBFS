import contextlib,ctypes,importlib,io,json,multiprocessing,os,subprocess,unittest,urllib.request
from pathlib import Path
from unittest.mock import patch
import test_native_assets_diagnostics as fixture

class Prefetch(unittest.TestCase):
 setUp=fixture.Assets.setUp
 manager=fixture.Assets.manager
 def require_prefetch(self):
  self.assertTrue(hasattr(self.module.AssetManager,'prefetch'),'GPU-free selected/all native prefetch is missing')
 @contextlib.contextmanager
 def no_gpu(self):
  with patch.object(self.module,'run_hardware_probe',side_effect=AssertionError('Prefetch attempted GPU discovery')),patch.object(subprocess,'Popen',side_effect=AssertionError('Prefetch started a process')),patch.object(ctypes,'CDLL',side_effect=AssertionError('Prefetch loaded a CUDA library')):yield
 @contextlib.contextmanager
 def public_fixture(self):
  path=Path(self.tmp.name)/'native_assets.json';path.write_text(json.dumps(self.index));original=urllib.request.urlopen;port=self.server.server_address[1]
  def redirect(request,timeout):
   return original(urllib.request.Request('http://127.0.0.1:'+str(port)+'/'+request.full_url.rsplit('/',1)[1],headers=dict(request.header_items())),timeout=timeout)
  with patch.object(self.module,'index_path',return_value=path),patch.object(urllib.request,'urlopen',side_effect=redirect),self.no_gpu():yield
 def test_selected_prefetch_uses_no_gpu_and_only_requested_components(self):
  self.require_prefetch();m=self.manager()
  with self.no_gpu(),patch.object(m,'select',side_effect=AssertionError('Prefetch invented hardware')):r=m.prefetch([100,120])
  self.assertEqual(r['status'],'PREFETCHED');self.assertIs(r['gpu_accessed'],False)
  self.assertEqual(r['targets'],['100','120']);self.assertEqual({n for n,_ in self.calls},{'common.tar.gz','cuda-sm100.tar.gz','cuda-sm120.tar.gz'})
  self.assertEqual(r['downloaded_bytes'],sum(self.index['components'][n]['size'] for n in ('common','cuda-sm100','cuda-sm120')))
 def test_all_covers_every_production_target_including_universal_but_no_contrasts(self):
  self.require_prefetch()
  with self.no_gpu():r=self.manager().prefetch(all_architectures=True)
  expected={self.index['common'],*self.index['production'].values()}
  self.assertEqual(set(r['targets']),set(self.index['production']));self.assertEqual({n for n,_ in self.calls},{n+'.tar.gz' for n in expected});self.assertEqual(len(self.calls),len(expected))
 def test_dry_run_reports_conservative_bytes_without_network_gpu_or_cache_writes(self):
  self.require_prefetch()
  with self.no_gpu():r=self.manager().prefetch([100,120],dry_run=True)
  self.assertEqual(r['status'],'PLAN_ONLY');self.assertEqual(r['downloaded_bytes'],0);self.assertEqual(self.calls,[]);self.assertFalse(self.cache.exists())
  self.assertEqual(r['maximum_additional_download_bytes'],sum(self.index['components'][n]['size'] for n in ('common','cuda-sm100','cuda-sm120')))
 def test_dry_run_recognizes_shared_runtime_and_other_preloaded_gpu(self):
  self.require_prefetch();self.manager().prefetch([100]);before=len(self.calls)
  r=self.manager().prefetch([100,120],dry_run=True)
  self.assertEqual(r['maximum_additional_download_bytes'],self.index['components']['cuda-sm120']['size']);self.assertEqual(len(self.calls),before)
 def test_distinct_gpu_bundles_work_offline_after_one_no_gpu_prefetch(self):
  self.require_prefetch()
  with self.no_gpu():self.manager().prefetch([100,120])
  def forbidden(*a,**k):raise AssertionError('A preloaded GPU caused a new transfer')
  with patch.dict(os.environ,{'MGBFS_NATIVE_OFFLINE':'1'}):
   roots=[self.manager(opener=forbidden).prepare(fixture.hardware(sm)) for sm in (100,120,100)]
  self.assertEqual(roots[0],roots[2]);self.assertNotEqual(roots[0],roots[1]);self.assertEqual((roots[0]/'lib/libmgbfs_cuda.so').read_bytes(),b'cuda 100');self.assertEqual((roots[1]/'lib/libmgbfs_cuda.so').read_bytes(),b'cuda 120')
  self.assertEqual(len(self.calls),3)
 def test_all_cache_supports_a_mixed_gpu_set_offline(self):
  self.require_prefetch();self.manager().prefetch(all_architectures=True)
  def forbidden(*a,**k):raise AssertionError('Mixed GPUs did not reuse universal component')
  with patch.dict(os.environ,{'MGBFS_NATIVE_OFFLINE':'1'}):root=self.manager(opener=forbidden).prepare(fixture.hardware(75,100))
  self.assertEqual((root/'lib/libmgbfs_cuda.so').read_bytes(),b'universal')
 def test_warm_prefetch_does_not_fetch_again(self):
  self.require_prefetch();self.manager().prefetch([100])
  def forbidden(*a,**k):raise AssertionError('Warm prefetch accessed network')
  with self.no_gpu():r=self.manager(opener=forbidden).prefetch([100])
  self.assertEqual(r['downloaded_bytes'],0);self.assertEqual(r['maximum_additional_download_bytes'],0)
 def test_invalid_target_is_rejected_before_any_download_or_write(self):
  self.require_prefetch()
  for targets in ([100,999],[],['static-sm120']):
   with self.assertRaises(self.module.AssetError):self.manager().prefetch(targets)
  self.assertEqual(self.calls,[]);self.assertFalse(self.cache.exists())
 def test_all_and_selection_cannot_be_combined(self):
  self.require_prefetch()
  with self.assertRaises(self.module.AssetError):self.manager().prefetch([100],all_architectures=True)
  self.assertEqual(self.calls,[])
 def test_repeated_targets_share_one_download(self):
  self.require_prefetch();r=self.manager().prefetch(['100,120','100'])
  self.assertEqual(r['targets'],['100','120']);self.assertEqual(len(self.calls),3)
 def test_corrupt_preloaded_component_is_reported_without_network(self):
  self.require_prefetch();self.manager().prefetch([100]);c=self.index['components']['cuda-sm100'];(self.cache/'components'/c['sha256']/'libmgbfs_cuda.so').write_bytes(b'corrupt')
  before=len(self.calls)
  with self.assertRaisesRegex(self.module.AssetError,'SIZE|CHECKSUM'):self.manager().prefetch([100])
  self.assertEqual(len(self.calls),before)
 def test_wrong_download_cannot_be_published_as_prefetched(self):
  self.require_prefetch();self.payload['cuda-sm100.tar.gz']=self.payload['cuda-sm120.tar.gz']
  with self.assertRaisesRegex(self.module.AssetError,'SIZE|CHECKSUM'):self.manager().prefetch([100])
  self.assertFalse((self.cache/'components'/self.index['components']['cuda-sm100']['sha256']).exists())
 def test_prefetch_reuses_partial_download_with_hash_check(self):
  self.require_prefetch();c=self.index['components']['cuda-sm100'];d=self.cache/'downloads';d.mkdir(parents=True);(d/(c['sha256']+'.part')).write_bytes(self.payload['cuda-sm100.tar.gz'][:30])
  r=self.manager().prefetch([100]);self.assertEqual(r['status'],'PREFETCHED');self.assertIn(('cuda-sm100.tar.gz','bytes=30-'),self.calls)
 def test_offline_missing_component_fails_without_network(self):
  self.require_prefetch()
  with patch.dict(os.environ,{'MGBFS_NATIVE_OFFLINE':'1'}):
   with self.assertRaisesRegex(self.module.AssetError,'OFFLINE_MISSING'):self.manager().prefetch([100])
  self.assertEqual(self.calls,[])
 def test_parallel_different_gpu_prefetch_downloads_common_only_once(self):
  self.require_prefetch()
  if os.name!='posix':self.skipTest('POSIX shared cache locks')
  ctx=multiprocessing.get_context('fork');queue=ctx.Queue()
  def worker(targets):
   try:queue.put(self.manager().prefetch(targets)['status'])
   except BaseException as e:queue.put(repr(e))
  jobs=[ctx.Process(target=worker,args=(targets,)) for targets in ([100],[120],[100,120])]
  for job in jobs:job.start()
  for job in jobs:job.join(12);self.assertEqual(job.exitcode,0)
  self.assertEqual([queue.get(timeout=2) for _ in jobs],['PREFETCHED']*3)
  self.assertEqual(sorted(n for n,_ in self.calls),['common.tar.gz','cuda-sm100.tar.gz','cuda-sm120.tar.gz'])
 def test_public_api_uses_shared_cache_without_environment_or_index_mutation(self):
  self.require_prefetch();import multigpubfs
  self.assertTrue(hasattr(multigpubfs,'prefetch_native'),'Public Python-only job prefetch API is missing')
  with self.public_fixture(),patch.dict(os.environ,{'MGBFS_NATIVE_CACHE':str(self.cache)}):
   r=multigpubfs.prefetch_native(sm=[100,120]);self.assertEqual(os.environ['MGBFS_NATIVE_CACHE'],str(self.cache))
  self.assertEqual(r['cache'],str(self.cache.resolve()));self.assertEqual(r['environment']['MGBFS_NATIVE_OFFLINE'],'1');self.assertEqual(r['index_sha256'],self.module.AssetManager(self.index,self.cache).identity)
 def test_cli_json_is_parseable_and_downloads_selected_components_without_gpu(self):
  self.require_prefetch();from multigpubfs import cli
  out=io.StringIO();err=io.StringIO()
  with self.public_fixture(),contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):code=cli.main(['prefetch','--sm','100,120','--cache',str(self.cache),'--json'])
  self.assertEqual(code,0);r=json.loads(out.getvalue());self.assertEqual(r['status'],'PREFETCHED');self.assertEqual(r['targets'],['100','120']);self.assertEqual(len(self.calls),3)
 def test_cli_all_plan_needs_no_gpu_driver_and_no_network(self):
  self.require_prefetch();from multigpubfs import cli
  out=io.StringIO()
  with self.public_fixture(),contextlib.redirect_stdout(out):code=cli.main(['prefetch','--all','--dry-run','--cache',str(self.cache),'--json'])
  self.assertEqual(code,0);self.assertEqual(json.loads(out.getvalue())['status'],'PLAN_ONLY');self.assertFalse(self.cache.exists());self.assertEqual(self.calls,[])
 def test_cli_invalid_selection_reports_failure_without_downloading(self):
  self.require_prefetch();from multigpubfs import cli
  out=io.StringIO();err=io.StringIO()
  with self.public_fixture(),contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):code=cli.main(['prefetch','--sm','100,999','--cache',str(self.cache),'--json'])
  self.assertEqual(code,1);self.assertEqual(json.loads(out.getvalue())['status'],'ERROR');self.assertEqual(self.calls,[])

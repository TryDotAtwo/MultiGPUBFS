import contextlib,copy,hashlib,http.server,importlib,io,json,multiprocessing,os,tarfile,tempfile,threading,unittest,urllib.request
from pathlib import Path
from unittest.mock import patch

BASE='https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-components-fixture/'
ARCHES=[75,80,86,89,90,100,103,120]
def h(b):return hashlib.sha256(b).hexdigest()
def hardware(*sm,driver=13020):return {'driver_api':driver,'devices':[{'logical_id':i,'compute_capability':[x//10,x%10]} for i,x in enumerate(sm)]}
def archive(files,extra=None):
 out=io.BytesIO()
 with tarfile.open(fileobj=out,mode='w:gz') as tar:
  for name,data in files.items():
   item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o755 if name=='bin/mgbfs' else 0o644;tar.addfile(item,io.BytesIO(data))
  if extra:tar.addfile(extra[0],io.BytesIO(extra[1]))
 return out.getvalue()
def fixture():
 index={'schema':1,'id':'native-components-fixture','platform':'linux_x86_64','driver_minimum':12000,'common':'common','components':{},'production':{},'diagnostics':{},'native_metadata':{'cuda_toolkit':'12.9.86','native_binary_source_commit':'a'*40,'bundled_runtime':[{'scope':'fixture'}]}}
 payload={}
 def add(name,files,runtime=None,arches=None,ptx=None):
  data=archive(files);payload[name+'.tar.gz']=data
  c={'url':BASE+name+'.tar.gz','size':len(data),'sha256':h(data),'files':{p:{'size':len(b),'sha256':h(b),'mode':0o755 if p=='bin/mgbfs' else 0o644} for p,b in files.items()}}
  if runtime:c['build']={'source_commit':'b'*40,'cuda_source_tree':'c'*40,'cuda_kernel_source_sha256':'d'*64,'runtime':runtime,'architectures':arches,'ptx_architectures':ptx or []}
  index['components'][name]=c
 add('common',{'bin/mgbfs':b'fixture executable','lib/libcudart.so.12':b'private runtime','lib/libnccl.so.2':b'private nccl','licenses/notice.txt':b'fixture'})
 for sm in ARCHES:
  key='cuda-sm'+str(sm);add(key,{'libmgbfs_cuda.so':('cuda '+str(sm)).encode()},'Shared',[sm]);index['production'][str(sm)]=key
 add('cuda-universal',{'libmgbfs_cuda.so':b'universal'},'Shared',ARCHES,[80]);index['production']['universal']='cuda-universal'
 for identifier,runtime,arches in [('static-sm120','Static',ARCHES),('shared-legacy','Shared',[75,86]),('baseline-static-legacy','Static',[75,86])]:
  key='contrast-'+identifier;add(key,{'libmgbfs_cuda.so':identifier.encode()},runtime,arches);index['diagnostics'][identifier]=key
 return index,payload

class Assets(unittest.TestCase):
 def setUp(self):
  try:self.module=importlib.import_module('multigpubfs.native_assets')
  except ModuleNotFoundError:self.fail('Selective native asset delivery is not implemented')
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.cache=Path(self.tmp.name)/'cache'
  self.index,self.payload=fixture();self.calls=[];outer=self
  class Handler(http.server.BaseHTTPRequestHandler):
   def do_GET(self):
    name=self.path.lstrip('/');outer.calls.append((name,self.headers.get('Range')))
    data=outer.payload.get(name)
    if data is None:self.send_error(404);return
    offset=int(self.headers.get('Range','bytes=0-')[6:].split('-')[0])
    self.send_response(206 if offset else 200)
    if offset:self.send_header('Content-Range',f'bytes {offset}-{len(data)-1}/{len(data)}')
    self.send_header('Content-Length',str(len(data)-offset));self.end_headers();self.wfile.write(data[offset:])
   def log_message(self,*args):pass
  self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
  self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown);self.addCleanup(self.thread.join,2)
  port=self.server.server_address[1]
  def opener(request,timeout):
   local=urllib.request.Request('http://127.0.0.1:'+str(port)+'/'+request.full_url.rsplit('/',1)[1],headers=dict(request.header_items()))
   return urllib.request.urlopen(local,timeout=timeout)
  self.opener=opener
 def manager(self,index=None,opener=None):return self.module.AssetManager(index or self.index,self.cache,opener=opener or self.opener,timeout=15)
 def test_exact_and_same_major_compatible_native_selection(self):
  m=self.manager()
  self.assertEqual(m.select(hardware(100)),'cuda-sm100');self.assertEqual(m.select(hardware(102)),'cuda-sm100');self.assertEqual(m.select(hardware(103)),'cuda-sm103');self.assertEqual(m.select(hardware(80,86)),'cuda-sm80')
 def test_mixed_major_and_future_gpu_use_verified_universal(self):
  self.assertEqual(self.manager().select(hardware(75,100)),'cuda-universal');self.assertEqual(self.manager().select(hardware(130)),'cuda-universal')
 def test_unsupported_gpu_old_driver_or_no_visible_gpu_fails_before_download(self):
  for hw in (hardware(70),hardware(75,driver=11000),hardware()):
   with self.assertRaises(self.module.AssetError):self.manager().prepare(hw)
  self.assertEqual(self.calls,[])
 def test_cold_download_has_only_common_and_matching_native(self):
  root=self.manager().prepare(hardware(75,75));manifest=json.loads((root/'manifest.json').read_text())
  self.assertEqual({p[0] for p in self.calls},{'common.tar.gz','cuda-sm75.tar.gz'});self.assertEqual(manifest['architectures'],[75]);self.assertEqual((root/'lib/libmgbfs_cuda.so').read_bytes(),b'cuda 75')
  self.assertTrue(os.access(root/'bin/mgbfs',os.X_OK));self.assertEqual(manifest['native_delivery']['selected_component'],'cuda-sm75')
 def test_warm_cache_works_with_all_network_disabled(self):
  a=self.manager().prepare(hardware(86))
  def offline(*a,**k):raise AssertionError('Warm native cache accessed the network')
  b=self.manager(opener=offline).prepare(hardware(86));self.assertEqual(a,b);self.assertEqual(len(self.calls),2)
 def test_optional_contrast_download_is_deferred_and_manifest_stays_stable(self):
  m=self.manager();root=m.prepare(hardware(75));before=h((root/'manifest.json').read_bytes())
  self.assertFalse((root/'diagnostic_variants/static-sm120/libmgbfs_cuda.so').exists())
  m.ensure_variant(root,'static-sm120');self.assertTrue((root/'diagnostic_variants/static-sm120/libmgbfs_cuda.so').is_file())
  self.assertEqual(before,h((root/'manifest.json').read_bytes()));self.assertEqual({p[0] for p in self.calls},{'common.tar.gz','cuda-sm75.tar.gz','contrast-static-sm120.tar.gz'})
 def test_wrong_download_hash_is_never_published(self):
  self.payload['cuda-sm75.tar.gz']=self.payload['cuda-sm86.tar.gz']
  with self.assertRaisesRegex(self.module.AssetError,'SIZE|CHECKSUM'):self.manager().prepare(hardware(75))
  self.assertFalse(list(self.cache.glob('bundles/*/manifest.json')))
 def test_changed_cached_library_is_rejected(self):
  root=self.manager().prepare(hardware(75));(root/'lib/libmgbfs_cuda.so').write_bytes(b'corrupt')
  with self.assertRaisesRegex(self.module.AssetError,'CHECKSUM|SIZE'):self.manager().prepare(hardware(75))
 def test_untrusted_url_is_rejected_without_fetch(self):
  for url in ('http://github.com/file','https://evil.invalid/payload','file:///tmp/payload'):
   index=copy.deepcopy(self.index);index['components']['common']['url']=url
   with self.assertRaises(self.module.AssetError):self.manager(index).prepare(hardware(75))
  self.assertEqual(self.calls,[])
 def test_archive_cannot_escape_or_add_unlisted_file(self):
  for name in ('../../outside','/absolute','unexpected.txt'):
   index=copy.deepcopy(self.index);item=tarfile.TarInfo(name);item.size=4
   data=archive({'libmgbfs_cuda.so':b'cuda 75'},(item,b'evil'));self.payload['cuda-sm75.tar.gz']=data
   index['components']['cuda-sm75'].update(sha256=h(data),size=len(data))
   with self.assertRaisesRegex(self.module.AssetError,'ARCHIVE'):self.manager(index).prepare(hardware(75))
   self.assertFalse((Path(self.tmp.name)/'outside').exists())
 def test_archive_symlink_is_rejected(self):
  index=copy.deepcopy(self.index);item=tarfile.TarInfo('libmgbfs_cuda.so');item.type=tarfile.SYMTYPE;item.linkname='/tmp/foreign'
  data=archive({},(item,b''));self.payload['cuda-sm75.tar.gz']=data;index['components']['cuda-sm75'].update(sha256=h(data),size=len(data))
  with self.assertRaisesRegex(self.module.AssetError,'ARCHIVE'):self.manager(index).prepare(hardware(75))
 def test_partial_fetch_leaves_no_ready_bundle(self):
  index=copy.deepcopy(self.index);self.payload['cuda-sm75.tar.gz']=self.payload['cuda-sm75.tar.gz'][:30]
  with self.assertRaisesRegex(self.module.AssetError,'SIZE'):self.manager(index).prepare(hardware(75))
  self.assertFalse(list(self.cache.glob('bundles/*/manifest.json')))
 def test_concurrent_processes_publish_one_complete_bundle(self):
  if os.name!='posix':self.skipTest('POSIX CUDA-host cache locking')
  ctx=multiprocessing.get_context('fork');queue=ctx.Queue()
  def worker():
   try:queue.put(('OK',str(self.manager().prepare(hardware(75)))))
   except BaseException as error:queue.put(('FAIL',repr(error)))
  jobs=[ctx.Process(target=worker) for _ in range(3)]
  for job in jobs:job.start()
  for job in jobs:job.join(12);self.assertEqual(job.exitcode,0)
  values=[queue.get(timeout=2) for _ in jobs];self.assertTrue(all(v[0]=='OK' for v in values),values);self.assertEqual(len({v[1] for v in values}),1);self.assertEqual(len(self.calls),2)
 def test_probe_timeout_is_bounded_and_its_own_group_reaped(self):
  import sys,time
  began=time.monotonic()
  with self.assertRaisesRegex(self.module.AssetError,'PROBE_TIMEOUT'):
   self.module.run_hardware_probe([sys.executable,'-c','import time;time.sleep(20)'],timeout=.1)
  self.assertLess(time.monotonic()-began,6)
 def test_probe_preserves_logical_device_evidence(self):
  import sys
  value=hardware(100,100);code='import json;print('+repr(json.dumps(value))+')'
  self.assertEqual(self.module.run_hardware_probe([sys.executable,'-c',code],timeout=5),value)

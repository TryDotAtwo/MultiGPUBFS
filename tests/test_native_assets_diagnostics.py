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
  self.addCleanup(self.thread.join,2);self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)
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

 def test_native_runtime_uses_prepared_index_root_and_preserves_explicit_override(self):
  from multigpubfs import native_distribution as d
  self.assertTrue(hasattr(d,"native_root"),"Native runtime does not resolve selectively prepared roots")
  package=Path(self.tmp.name)/"package";package.mkdir();(package/"native_assets.json").write_text(json.dumps(self.index))
  manager=self.manager();expected=manager.prepare(hardware(75))
  with patch.object(d,"__file__",str(package/"native_distribution.py")),patch.object(self.module,"ensure_runtime",return_value=expected),patch.dict(os.environ,{},clear=True):
   binary,env=d.native_runtime();self.assertEqual(Path(binary),expected/"bin/mgbfs");self.assertEqual(env["LD_LIBRARY_PATH"].split(":")[0],str(expected/"lib"))
   self.assertEqual(d.native_runtime("/explicit/native")[0],"/explicit/native")
 def test_inventory_declares_optional_payloads_without_fetch_and_environment_verifies_on_selection(self):
  from multigpubfs import runtime_variants as variants
  manager=self.manager();root=manager.prepare(hardware(75));verified=variants.inventory(root,environ={})
  self.assertEqual(verified["status"],"AVAILABLE");self.assertEqual(len(self.calls),2)
  selected=next(x for x in verified["variants"] if x["id"]=="static-sm120");self.assertIs(selected["installed"],False)
  with patch.object(self.module,"ensure_variant",side_effect=lambda r,i,**kw:manager.ensure_variant(r,i)):
   env=variants.environment("static-sm120",verified,parent={})
  self.assertEqual(Path(env["LD_LIBRARY_PATH"].split(":")[0]),root/"diagnostic_variants/static-sm120");self.assertTrue((root/"diagnostic_variants/static-sm120/libmgbfs_cuda.so").is_file());self.assertEqual(len(self.calls),3)

 def test_index_cannot_mislabel_native_architecture_or_kernel_source(self):
  for field,value in [("architectures",[100]),("cuda_kernel_source_sha256","f"*64)]:
   index=copy.deepcopy(self.index);index["components"]["cuda-sm75"]["build"][field]=value
   with self.assertRaises(self.module.AssetError):self.manager(index)
  self.assertEqual(self.calls,[])
 def test_partial_download_resumes_and_final_hash_is_verified(self):
  component=self.index["components"]["cuda-sm75"];folder=self.cache/"downloads";folder.mkdir(parents=True);(folder/(component["sha256"]+".part")).write_bytes(self.payload["cuda-sm75.tar.gz"][:30])
  root=self.manager().prepare(hardware(75));self.assertTrue((root/"bin/mgbfs").is_file());self.assertIn(("cuda-sm75.tar.gz","bytes=30-"),self.calls)
 def test_nested_hardware_probe_stays_in_owned_diagnostic_group(self):
  import sys
  if os.name!="posix":self.skipTest("POSIX CUDA-host groups")
  code="import json,os;print(json.dumps({\"group\":os.getpgrp()}))"
  with patch.dict(os.environ,{"MGBFS_DIAGNOSTIC_TRACE_FILE":str(Path(self.tmp.name)/"trace.log")}):
   data=self.module.run_hardware_probe([sys.executable,"-c",code],timeout=5)
  self.assertEqual(data["group"],os.getpgrp(),"Hardware child escaped the owned diagnostic process group")
 def test_inherited_group_probe_timeout_does_not_kill_parent(self):
  import sys,time
  began=time.monotonic()
  with patch.dict(os.environ,{"MGBFS_DIAGNOSTIC_TRACE_FILE":str(Path(self.tmp.name)/"trace.log")}):
   with self.assertRaisesRegex(self.module.AssetError,"PROBE_TIMEOUT"):
    self.module.run_hardware_probe([sys.executable,"-c","import time;time.sleep(20)"],timeout=.1)
  self.assertLess(time.monotonic()-began,6)

 def test_selective_inventory_retains_deferred_universal_shared_control(self):
  from multigpubfs import runtime_variants as v
  m=self.manager();root=m.prepare(hardware(75));value=v.inventory(root,environ={})
  self.assertEqual({r["id"] for r in value["variants"]},set(v.SPECS)|{"shared-universal"})
  self.assertEqual(len(self.calls),2);m.ensure_variant(root,"shared-universal");self.assertEqual(len(self.calls),3)
 def test_deep_matrix_can_compare_same_architectures_for_runtime_factor(self):
  from multigpubfs import runtime_variants as v
  import inspect
  self.assertIn("include_universal",inspect.signature(v.cases).parameters,"Deep contrasts lack matching universal shared/static pair")
  cases=v.cases([0],contrasts_only=True,include_universal=True)
  self.assertIn("shared-universal",{c["config"]["variant"] for c in cases})
  steps=[{"name":i,"status":"PASS","diagnostic_case":{"variant":i,"kind":"api","device":0}} for i in list(v.SPECS)+["shared-universal"]]
  comparison=v.compare(steps)["comparisons"][0]
  pair=next(p for p in comparison["controlled_pairs"] if p["left"]=="static-sm120" and p["factor"]=="runtime")
  self.assertEqual(pair["right"],"shared-universal");self.assertFalse(v.compare(steps)["root_cause_confirmed"])

 def test_server_recipe_search_includes_universal_shared_only_for_selective_install(self):
  from multigpubfs import server_recipe as r
  import inspect
  self.assertIn("include_universal",inspect.signature(r.candidates).parameters,"Selective recipe search omits universal shared fallback")
  self.assertNotIn("shared-universal",{x["variant"] for x in r.candidates([0])})
  c=list(r.candidates([0],include_universal=True));self.assertIn("shared-universal",{x["variant"] for x in c})
  chosen=next(x for x in c if x["variant"]=="shared-universal");r.overrides(chosen)

 def test_delivery_report_keeps_early_component_source_and_hash_identity(self):
  manager=self.manager();hw=hardware(75);root=manager.prepare(hw)
  with patch.dict(self.module._roots,{"fixture":(root,manager,hw)},clear=True):data=self.module.status(root)
  self.assertIn("native_build_declaration",data,"Early native delivery report omits component provenance")
  declared=data["native_build_declaration"];self.assertEqual(declared["native_binary_source_commit"],"a"*40);self.assertEqual(declared["cuda_library_source_commit"],"b"*40);self.assertEqual(declared["architectures"],[75]);self.assertEqual(declared["declared_artifact_sha256"]["bin/mgbfs"],h(b"fixture executable"))

 def test_explicit_activation_prepares_assets_before_replacing_graph_method(self):
  from multigpubfs import native_distribution as d,cayleypy_backend as backend
  self.assertTrue(hasattr(d,"prepare_for_activation"),"Explicit native activation does not prepare missing components")
  try:from cayleypy import CayleyGraph
  except ImportError:self.skipTest("CayleyPy API runs in full CPU and installed GPU acceptance")
  backend.disable_native();original=CayleyGraph.bfs;package=Path(self.tmp.name)/"activation-package";package.mkdir();(package/"native_assets.json").write_text(json.dumps(self.index));manager=self.manager()
  try:
   with patch.object(d,"__file__",str(package/"native_distribution.py")),patch.object(d.os.path,"exists",return_value=True),patch.object(self.module,"ensure_runtime",side_effect=lambda path:manager.prepare(hardware(75))),patch.dict(os.environ,{},clear=True):backend.enable_native()
   self.assertIsNot(CayleyGraph.bfs,original);self.assertEqual({p[0] for p in self.calls},{"common.tar.gz","cuda-sm75.tar.gz"})
  finally:backend.disable_native()
 def test_activation_download_failure_leaves_original_api_intact(self):
  from multigpubfs import native_distribution as d,cayleypy_backend as backend
  self.assertTrue(hasattr(d,"prepare_for_activation"),"Explicit native preparation hook is missing")
  try:from cayleypy import CayleyGraph
  except ImportError:self.skipTest("CayleyPy API runs in full CPU and installed GPU acceptance")
  backend.disable_native();original=CayleyGraph.bfs
  with patch.object(d,"prepare_for_activation",side_effect=self.module.AssetError("download failed")):
   with self.assertRaisesRegex(self.module.AssetError,"download failed"):backend.enable_native()
  self.assertIs(CayleyGraph.bfs,original)
 def test_activation_without_gpu_does_not_start_discovery_or_download(self):
  from multigpubfs import native_distribution as d
  self.assertTrue(hasattr(d,"prepare_for_activation"),"CPU-safe activation deferral is missing")
  with patch.object(d.os.path,"exists",return_value=False),patch.object(d,"native_runtime",side_effect=AssertionError("CPU-only activation initialized CUDA")):d.prepare_for_activation()
  self.assertEqual(self.calls,[])

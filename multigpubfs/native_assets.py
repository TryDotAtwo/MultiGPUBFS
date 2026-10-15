"""Pinned precompiled native components, with an atomic user-local cache.

This module never patches a driver, invokes a compiler or retries a BFS.
Only a native request prepares components; importing the library is offline.
"""
import contextlib,hashlib,json,os,platform,re,shutil,subprocess,sys,tarfile,tempfile,time,urllib.error,urllib.request
from pathlib import Path,PurePosixPath

LAYOUT_SCHEMA=1
class AssetError(RuntimeError):pass
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
def safe_path(name):
 p=PurePosixPath(name)
 if not name or not p.parts or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or p.as_posix()!=name:raise AssetError('NATIVE_ASSET_ARCHIVE_PATH '+name)
 return p
def checksum(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
def checked_file(root,name,info):
 safe_path(name);root=Path(root);path=root/name
 if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():raise AssetError('NATIVE_ASSET_ARCHIVE_SYMLINK '+name)
 if not path.is_file() or path.stat().st_size!=info['size']:raise AssetError('NATIVE_ASSET_SIZE '+name)
 if digest(path)!=info['sha256']:raise AssetError('NATIVE_ASSET_CHECKSUM '+name)
 return path

_probe_blocked=False
def _stop_hardware_probe(process,owned_group):
 if owned_group:
  from .deep_diagnostics import bounded_stop
  return bounded_stop(process)['complete']
 # Nested probes inherit the diagnostic group; never kill that group here.
 for method in ('terminate','kill'):
  if process.poll() is not None:return True
  try:getattr(process,method)()
  except ProcessLookupError:return True
  try:process.wait(timeout=1)
  except subprocess.TimeoutExpired:pass
 return process.poll() is not None

def run_hardware_probe(command=None,timeout=20):
 global _probe_blocked
 if _probe_blocked:raise AssetError('OWNED_DIAGNOSTIC_PROCESS_COULD_NOT_BE_REAPED: no more hardware probes')
 command=command or [sys.executable,'-m','multigpubfs._native_hardware']
 owned_group=os.name=='posix' and not os.environ.get('MGBFS_DIAGNOSTIC_TRACE_FILE')
 p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=owned_group)
 try:
  try:out,err=p.communicate(timeout=timeout)
  except subprocess.TimeoutExpired as error:
   if not _stop_hardware_probe(p,owned_group):
    _probe_blocked=True;raise AssetError('NATIVE_ASSET_PROBE_TIMEOUT OWNED_DIAGNOSTIC_PROCESS_COULD_NOT_BE_REAPED') from error
   raise AssetError('NATIVE_ASSET_PROBE_TIMEOUT: CUDA hardware discovery exceeded its separate deadline') from error
  if p.returncode:raise AssetError('NATIVE_ASSET_HARDWARE_UNAVAILABLE: '+(err or out)[-4000:])
  try:value=json.loads(out)
  except ValueError as error:raise AssetError('NATIVE_ASSET_HARDWARE_RESPONSE') from error
  if not isinstance(value,dict):raise AssetError('NATIVE_ASSET_HARDWARE_RESPONSE')
  return value
 finally:
  if p.poll() is None and not _stop_hardware_probe(p,owned_group):_probe_blocked=True
  for stream in (p.stdout,p.stderr):
   if stream:stream.close()

class AssetManager:
 def __init__(self,index,cache,*,opener=None,timeout=600):
  self.index=index;self.cache=Path(cache).expanduser().resolve();self.opener=opener or urllib.request.urlopen;self.timeout=timeout;self.events=[]
  if index.get('schema')!=1 or index.get('platform')!='linux_x86_64' or not re.fullmatch('[a-z0-9_-]+',index.get('id','')):raise AssetError('NATIVE_ASSET_INDEX_SCHEMA')
  if not isinstance(timeout,(int,float)) or not 0<timeout<=1800:raise AssetError('NATIVE_ASSET_TIMEOUT')
  components=index.get('components',{})
  for name,c in components.items():
   if not re.fullmatch('[a-z0-9_-]+',name):raise AssetError('NATIVE_ASSET_COMPONENT_ID')
   url=c.get('url','')
   if not re.fullmatch(r'https://github\.com/TryDotAtwo/MultiGPUBFS/releases/download/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+\.tar\.gz',url):raise AssetError('NATIVE_ASSET_URL '+name)
   if not checksum(c.get('sha256')) or type(c.get('size')) is not int or not 0<c['size']<=1_500_000_000:raise AssetError('NATIVE_ASSET_INDEX_CHECKSUM '+name)
   if not c.get('files') or len(c['files'])>256:raise AssetError('NATIVE_ASSET_INDEX_FILES '+name)
   for path,f in c['files'].items():
    safe_path(path)
    if not checksum(f.get('sha256')) or type(f.get('size')) is not int or not 0<=f['size']<=2_000_000_000 or f.get('mode') not in (0o644,0o755):raise AssetError('NATIVE_ASSET_INDEX_FILE '+path)
  refs=[index.get('common'),*index.get('production',{}).values(),*index.get('diagnostics',{}).values()]
  if not refs or any(x not in components for x in refs):raise AssetError('NATIVE_ASSET_INDEX_REFERENCE')
  core=components[index['common']]['files']
  if not {'bin/mgbfs','lib/libcudart.so.12','lib/libnccl.so.2'}<=set(core) or not any(p.startswith('licenses/') for p in core):raise AssetError('NATIVE_ASSET_COMMON_LAYOUT')
  kernels=set()
  for target,name in index['production'].items():
   build=components[name].get('build',{})
   if build.get('runtime')!='Shared' or (target!='universal' and build.get('architectures')!=[int(target)]):raise AssetError('NATIVE_ASSET_NATIVE_TARGET_MISMATCH')
  for name in [*index['production'].values(),*index['diagnostics'].values()]:
   build=components[name].get('build',{});fingerprint=build.get('cuda_kernel_source_sha256')
   if not checksum(fingerprint) or 'libmgbfs_cuda.so' not in components[name]['files']:raise AssetError('NATIVE_ASSET_BUILD_PROVENANCE')
   kernels.add(fingerprint)
  if len(kernels)!=1:raise AssetError('NATIVE_ASSET_KERNEL_INPUT_MISMATCH')
  self.identity=hashlib.sha256(canonical(index)).hexdigest()
 def _remaining(self,deadline):
  remaining=deadline-time.monotonic()
  if remaining<=0:raise AssetError('NATIVE_ASSET_DOWNLOAD_TIMEOUT: component preparation exceeded its separate deadline')
  return remaining
 @contextlib.contextmanager
 def _lock(self,name,deadline):
  import fcntl
  folder=self.cache/'locks';folder.mkdir(parents=True,exist_ok=True);path=folder/(name+'.lock')
  if path.is_symlink():raise AssetError('NATIVE_ASSET_CACHE_SYMLINK')
  with path.open('a+b') as f:
   while True:
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:self._remaining(deadline);time.sleep(.05)
   try:yield
   finally:fcntl.flock(f,fcntl.LOCK_UN)
 def select(self,hardware):
  if type(hardware.get('driver_api')) is not int or hardware['driver_api']<self.index.get('driver_minimum',12000):raise AssetError('NATIVE_ASSET_DRIVER_UNSUPPORTED: compatible CUDA driver is required')
  devices=hardware.get('devices',[])
  if not devices or len(devices)>128 or {d.get('logical_id') for d in devices}!=set(range(len(devices))):raise AssetError('NATIVE_ASSET_NO_VISIBLE_GPU: run inside the allocated CUDA GPU job')
  caps=[]
  for d in devices:
   cc=d.get('compute_capability')
   if not isinstance(cc,list) or len(cc)!=2 or any(type(v) is not int or v<0 for v in cc):raise AssetError('NATIVE_ASSET_COMPUTE_CAPABILITY')
   caps.append(10*cc[0]+cc[1])
  candidates=[]
  for key,name in self.index['production'].items():
   if key=='universal':continue
   target=int(key)
   if all(target//10==cc//10 and target%10<=cc%10 for cc in caps):candidates.append((target,name))
  if candidates:return max(candidates)[1]
  name=self.index['production'].get('universal');build=self.index['components'].get(name,{}).get('build',{})
  def supported(cc):return any(a//10==cc//10 and a%10<=cc%10 for a in build.get('architectures',[])) or any(a<=cc for a in build.get('ptx_architectures',[]))
  if name and all(supported(cc) for cc in caps):return name
  raise AssetError('NATIVE_ASSET_ARCHITECTURE_UNSUPPORTED: no compatible published cubin/PTX component')
 def _component(self,name,deadline):
  c=self.index['components'][name];folder=self.cache/'components'/c['sha256']
  with self._lock('component-'+c['sha256'],deadline):
   if folder.is_symlink():raise AssetError('NATIVE_ASSET_CACHE_SYMLINK')
   if folder.exists():
    for p,f in c['files'].items():checked_file(folder,p,f)
    return folder
   if os.environ.get('MGBFS_NATIVE_OFFLINE')=='1':raise AssetError('NATIVE_ASSET_OFFLINE_MISSING '+name)
   downloads=self.cache/'downloads';downloads.mkdir(parents=True,exist_ok=True);part=downloads/(c['sha256']+'.part')
   if part.is_symlink():raise AssetError('NATIVE_ASSET_CACHE_SYMLINK')
   offset=part.stat().st_size if part.exists() else 0
   if offset>c['size']:part.unlink();offset=0
   if offset<c['size']:
    request=urllib.request.Request(c['url'],headers={'User-Agent':'MultiGPUBFS/native-components','Accept-Encoding':'identity',**({'Range':'bytes='+str(offset)+'-'} if offset else {})})
    print('NATIVE ASSET download '+name+' ('+str(c['size'])+' bytes)',flush=True)
    try:
     with self.opener(request,timeout=min(30,self._remaining(deadline))) as response:
      status=getattr(response,'status',200)
      if offset and status==206:
       if not response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-'):raise AssetError('NATIVE_ASSET_RANGE '+name)
      elif status==200:offset=0
      else:raise AssetError('NATIVE_ASSET_HTTP '+str(status)+' '+name)
      transferred=0
      with part.open('ab' if offset else 'wb') as out:
       while True:
        self._remaining(deadline);chunk=response.read(1<<20)
        if not chunk:break
        if out.tell()+len(chunk)>c['size']:raise AssetError('NATIVE_ASSET_SIZE '+name)
        out.write(chunk);transferred+=len(chunk)
       out.flush();os.fsync(out.fileno())
      self.events.append({'component':name,'downloaded_bytes':transferred,'resumed_from_bytes':offset,'sha256':c['sha256']})
    except AssetError:raise
    except (OSError,urllib.error.URLError) as error:raise AssetError('NATIVE_ASSET_DOWNLOAD '+name+': '+str(error)) from error
   if part.stat().st_size!=c['size']:raise AssetError('NATIVE_ASSET_SIZE '+name)
   if digest(part)!=c['sha256']:part.unlink();raise AssetError('NATIVE_ASSET_CHECKSUM '+name)
   folder.parent.mkdir(parents=True,exist_ok=True)
   stage=Path(tempfile.mkdtemp(prefix='extract-',dir=folder.parent))
   try:
    seen=set()
    with tarfile.open(part,'r:gz') as archive:
     for member in archive:
      safe_path(member.name)
      if not member.isfile() or member.name in seen or member.name not in c['files']:raise AssetError('NATIVE_ASSET_ARCHIVE_MEMBER '+member.name)
      seen.add(member.name);f=c['files'][member.name]
      if member.size!=f['size']:raise AssetError('NATIVE_ASSET_ARCHIVE_SIZE '+member.name)
      target=stage/member.name;target.parent.mkdir(parents=True,exist_ok=True)
      with archive.extractfile(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,1<<20)
      checked_file(stage,member.name,f);target.chmod(f['mode'])
    if seen!=set(c['files']):raise AssetError('NATIVE_ASSET_ARCHIVE_MISSING '+name)
    os.replace(stage,folder);part.unlink()
   except (tarfile.TarError,EOFError) as error:raise AssetError('NATIVE_ASSET_ARCHIVE '+name+': '+str(error)) from error
   finally:
    if stage.exists():shutil.rmtree(stage)
  return folder
 def _layout(self,selected):
  components=self.index['components'];production=components[selected];core=components[self.index['common']]
  files=dict(core['files']);files['lib/libmgbfs_cuda.so']=production['files']['libmgbfs_cuda.so'];metadata={};rows=[]
  choices={'shared-sm120':selected,'shared-universal':self.index['production']['universal'],**self.index['diagnostics']}
  for identifier,name in choices.items():
   c=components[name];build=dict(c['build']);library='lib/libmgbfs_cuda.so' if identifier=='shared-sm120' else 'diagnostic_variants/'+identifier+'/libmgbfs_cuda.so'
   proof=dict(build,id=identifier,packaged_sha256=c['files']['libmgbfs_cuda.so']['sha256'])
   receipt='diagnostic_variants/'+identifier+'/build-receipt.json';data=canonical(proof);metadata[receipt]=data
   rows.append({'id':identifier,'runtime':proof['runtime'],'architectures':proof['architectures'],'ptx_architectures':proof.get('ptx_architectures',[]),'library':library,'sha256':proof['packaged_sha256'],'receipt':receipt,'receipt_sha256':hashlib.sha256(data).hexdigest(),'source_commit':proof['source_commit'],'cuda_source_tree':proof['cuda_source_tree'],'cuda_kernel_source_sha256':proof['cuda_kernel_source_sha256'],'component_id':name})
  variants='diagnostic_variants/variants.json';metadata[variants]=canonical({'schema':1,'production_default':'shared-sm120','variants':rows,'scope':'Pinned components; absent diagnostic libraries are declarations until downloaded and verified'})
  build=production['build'];manifest=dict(self.index['native_metadata'])
  manifest.pop('package_source_commit',None)
  manifest.update(schema=1,architectures=build['architectures'],ptx_architectures=build.get('ptx_architectures',[]),production_build='shared-sm120',cuda_runtime_linkage='Shared',cuda_library_source_commit=build['source_commit'],cuda_source_tree=build['cuda_source_tree'],cuda_kernel_source_sha256=build['cuda_kernel_source_sha256'],native_delivery={'schema':1,'index_sha256':self.identity,'layout_schema':LAYOUT_SCHEMA,'selected_component':selected,'index_id':self.index['id']})
  manifest['sha256']={p:f['sha256'] for p,f in files.items()};manifest['sha256'][variants]=hashlib.sha256(metadata[variants]).hexdigest()
  return files,metadata,manifest
 def _verify(self,root,selected):
  files,metadata,manifest=self._layout(selected)
  if (root/'manifest.json').is_symlink() or (root/'manifest.json').read_bytes()!=canonical(manifest):raise AssetError('NATIVE_ASSET_MANIFEST_CHECKSUM')
  for p,f in files.items():checked_file(root,p,f)
  for p,b in metadata.items():
   if (root/p).is_symlink() or not (root/p).is_file() or (root/p).read_bytes()!=b:raise AssetError('NATIVE_ASSET_METADATA_CHECKSUM '+p)
  return root
 @staticmethod
 def _link(source,destination):
  destination.parent.mkdir(parents=True,exist_ok=True)
  try:os.link(source,destination)
  except OSError:shutil.copyfile(source,destination);destination.chmod(source.stat().st_mode&0o777)
 def prepare(self,hardware):
  selected=self.select(hardware);deadline=time.monotonic()+self.timeout;root=self.cache/'bundles'/(self.identity+'-v'+str(LAYOUT_SCHEMA)+'-'+selected)
  with self._lock('bundle-'+root.name,deadline):
   if root.exists():return self._verify(root,selected)
   common=self._component(self.index['common'],deadline);cuda=self._component(selected,deadline)
   root.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='assemble-',dir=root.parent))
   try:
    files,metadata,manifest=self._layout(selected)
    for p in self.index['components'][self.index['common']]['files']:self._link(common/p,stage/p)
    self._link(cuda/'libmgbfs_cuda.so',stage/'lib/libmgbfs_cuda.so')
    for p,b in metadata.items():(stage/p).parent.mkdir(parents=True,exist_ok=True);(stage/p).write_bytes(b)
    (stage/'manifest.json').write_bytes(canonical(manifest));self._verify(stage,selected);os.replace(stage,root)
   finally:
    if stage.exists():shutil.rmtree(stage)
  return root
 def ensure_variant(self,root,identifier,timeout=None):
  if identifier=='shared-sm120':return Path(root)/'lib/libmgbfs_cuda.so'
  controls={**self.index['diagnostics'],'shared-universal':self.index['production']['universal']}
  if identifier not in controls:raise AssetError('NATIVE_ASSET_DIAGNOSTIC_VARIANT')
  root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text());selected=manifest.get('native_delivery',{}).get('selected_component')
  if selected not in self.index['production'].values() or root!=self.cache/'bundles'/(self.identity+'-v'+str(LAYOUT_SCHEMA)+'-'+selected):raise AssetError('NATIVE_ASSET_ROOT_IDENTITY')
  self._verify(root,selected);name=controls[identifier];c=self.index['components'][name];target=root/'diagnostic_variants'/identifier/'libmgbfs_cuda.so';deadline=time.monotonic()+min(self.timeout,self.timeout if timeout is None else max(.001,timeout))
  with self._lock('variant-'+self.identity+'-'+identifier,deadline):
   if target.exists():return checked_file(target.parent,'libmgbfs_cuda.so',c['files']['libmgbfs_cuda.so'])
   folder=self._component(name,deadline)
   fd,temp=tempfile.mkstemp(prefix='native-',dir=target.parent);os.close(fd);temporary=Path(temp);temporary.unlink()
   try:self._link(folder/'libmgbfs_cuda.so',temporary);os.replace(temporary,target)
   finally:temporary.unlink(missing_ok=True)
  return target

_roots={}
def index_path():return Path(__file__).with_name('native_assets.json')
def cache_path():return Path(os.environ.get('MGBFS_NATIVE_CACHE') or Path(os.environ.get('XDG_CACHE_HOME') or Path.home()/'.cache')/'multigpubfs/native')
def _key(path):
 path=Path(path)
 try:st=path.stat();revision=(st.st_size,st.st_mtime_ns)
 except OSError:revision=None
 return (str(path.resolve()),revision,str(cache_path().expanduser().resolve()),os.environ.get('CUDA_VISIBLE_DEVICES'),os.environ.get('NVIDIA_VISIBLE_DEVICES'),os.environ.get('LD_LIBRARY_PATH'),os.environ.get('LD_PRELOAD'))
def current_root(path=None):
 value=_roots.get(_key(path or index_path()));return value[0] if value and (value[0]/'bin/mgbfs').is_file() else None
def ensure_runtime(path=None):
 if _probe_blocked:raise AssetError('OWNED_DIAGNOSTIC_PROCESS_COULD_NOT_BE_REAPED: native launch is blocked')
 path=Path(path or index_path());key=_key(path)
 if key in _roots and (_roots[key][0]/'bin/mgbfs').is_file():return _roots[key][0]
 if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):raise AssetError('NATIVE_ASSET_PLATFORM_UNSUPPORTED: Linux x86_64 NVIDIA execution is required')
 if not path.is_file():raise AssetError('NATIVE_ASSET_INDEX_NOT_INSTALLED')
 index=json.loads(path.read_text());manager=AssetManager(index,cache_path());hardware=run_hardware_probe();root=manager.prepare(hardware);_roots[key]=(root,manager,hardware);return root
def ensure_variant(root,identifier,*,deadline=None):
 root=Path(root).resolve()
 for prepared,manager,hardware in _roots.values():
  if prepared==root:return manager.ensure_variant(root,identifier,timeout=None if deadline is None else deadline-time.monotonic())
 ensure_runtime()
 for prepared,manager,hardware in _roots.values():
  if prepared==root:return manager.ensure_variant(root,identifier,timeout=None if deadline is None else deadline-time.monotonic())
 raise AssetError('NATIVE_ASSET_ROOT_IDENTITY')
def status(root=None):
 for prepared,manager,hardware in _roots.values():
  if root is None or Path(root).resolve()==prepared:
   return {'status':'READY','root':str(prepared),'index_id':manager.index['id'],'index_sha256':manager.identity,'hardware':hardware,'selected_component':json.loads((prepared/'manifest.json').read_text())['native_delivery']['selected_component'],'transfers':list(manager.events),'downloaded_bytes':sum(x['downloaded_bytes'] for x in manager.events),'scope':'Verified files only; transfers are cumulative in this Python process; actual GPU/BFS acceptance is separate'}
 return {'status':'NOT_PREPARED','scope':'Import and activation do not download or initialize CUDA'}

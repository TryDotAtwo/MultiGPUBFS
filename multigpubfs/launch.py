"""Single-host native launch. CPU control never generates or deduplicates states."""
import hashlib,json,os,secrets,shutil,socket,subprocess,tempfile,time,signal
from pathlib import Path
from .process_control import check_graph_admission, query as native_query, run as native_run
from .graph_definition import _normalize_graph
from .memory_admission import bitmap_pack4_bits, bounded_hash_available
from .native_distribution import native_runtime

_LEGACY_ARENA_CAPACITY = 0x7fff_fffe
_LEGACY_ROLLING_CAPACITY = (0xffff_ffff - 3) // 3
_PATH_COMPACT = 'MGBFS_GENERIC_PATH_COMPACT'
_PATH_SEGMENTED_NEAREST = 'MGBFS_SEGMENT_NEAREST_V1'
_PATH_SEGMENTED_WINDOW = 'MGBFS_SEGMENT_WINDOW_V1'
_SEGMENTED_NEAREST_CAPABILITY = 'graph_path_segmented_nearest_v1'
_PACKET_ADDR = 'MGBFS_SEGMENT_PACKET_CONTROL_ADDR'
_PACKET_BIND = 'MGBFS_SEGMENT_PACKET_BIND_ADDR'
_PACKET_RING = 'MGBFS_SEGMENT_PACKET_RING_SLOTS'
_PACKET_RX = 'MGBFS_SEGMENT_PACKET_RX_SLOTS'
_PATH_PACKET_ENV = (_PACKET_ADDR,_PACKET_BIND,_PACKET_RING,_PACKET_RX)

def _rank_launch_config(admission, digest, *, bitmap=False, depth=None, path_targets=None,
                        path_bidirectional=False, path_window=None, path_max_depth=None):
 if admission['graph_digest']!=digest:raise RuntimeError('GRAPH_ADMISSION_IDENTITY')
 config=dict(admission)
 if bitmap:
  # The admission planner returns hash cut points. Bitmap uses the
  # exact rank partition and cannot use those cuts.
  config['owner_cuts']=None
 if depth is not None:config['profile_max_layers']=depth
 if path_targets is not None:config.update(path_targets=path_targets,path_bidirectional=path_bidirectional)
 if path_window is not None:config['path_window']=path_window
 if path_max_depth is not None:config['path_max_depth']=path_max_depth
 return config

def _with_packet_job_token(config,env):
 plan=config['plan']
 if config.get('path_targets') is not None and any(
     env.get(key)=='1' for key in (_PATH_COMPACT,_PATH_SEGMENTED_NEAREST,_PATH_SEGMENTED_WINDOW)):
  if (plan.get('compact_transport') is not True or plan.get('compact_packet_slots')!=3
      or plan.get('compact_receive_slots')!=3 or plan['world']<2 or not env.get(_PACKET_ADDR)):
   raise RuntimeError('NATIVE_PATH_COMPACT_PLAN_MISMATCH')
 uses_packets=plan.get('compact_transport') or env.get('MGBFS_EXPERIMENTAL_SEGMENTED24')=='1'
 if uses_packets and _PACKET_ADDR in env and plan['world']>1:
  # The communicator may be parked and reused by a later launch.
  return dict(config,packet_job_token=secrets.token_hex(16))
 return config

def _normalize_memory_backend(name):
 # Keep the previous spelling as an alias for existing clients and receipts;
 # it is no longer the width of the local index.
 if name in ('segmented','segmented32','segmented24'):
  return 'segmented24'
 if name in ('legacy','permutation_bitmap'):
  return name
 raise ValueError('INVALID_MEMORY_BACKEND')

def _normalize_segment_strategy(name):
 if name not in ('atomic','warp','batch'):raise ValueError('INVALID_SEGMENT_STRATEGY')
 return name

def _apply_memory_preset(env,backend,preset):
 if preset not in ('balanced','max_capacity'):raise ValueError('INVALID_MEMORY_PRESET')
 selected=dict(env,MGBFS_MEMORY_PRESET=preset)
 if preset=='max_capacity':
  if backend not in ('segmented24','permutation_bitmap'):raise ValueError('MEMORY_PRESET_UNSUPPORTED_FOR_LEGACY')
  # Keep the conservative values without treating them as user constraints.
  # The scoped markers survive nested pilot launches, but do not modify
  # os.environ or change the user's explicit geometry.
  # The larger window still uses one bounded producer tile and one TX/RX
  # pair. Pack4 keeps their footprint small even on the heavy S13 layers.
  pack4_packets = (backend=='permutation_bitmap'
                   and env.get('MGBFS_SEGMENT_PACK4')=='1'
                   and 'MGBFS_SEGMENT_PACKET_CONTROL_ADDR' in env)
  queue_default = '32768' if pack4_packets else '2048'
  for key,value,marker in (
   ('MGBFS_SEGMENT_OWNER_LANES','1','_MGBFS_PRESET_IMPLICIT_LANES'),
   ('MGBFS_SEGMENT_QUEUE_CAP',queue_default,'_MGBFS_PRESET_IMPLICIT_QUEUE')):
   if key not in selected:
    selected[key]=value;selected[marker]='1'
  layout_key,layout_value=('MGBFS_SEGMENT_COMPACT_POS','1') if backend=='segmented24' else ('MGBFS_BITMAP_BANKS','2')
  if layout_key not in selected:
   selected[layout_key]=layout_value;selected['_MGBFS_PRESET_IMPLICIT_LAYOUT']='1'
 else:
  # Private settings supplied by the user remain explicit; defaults do not
  # contaminate a later balanced job in a persistent Python process.
  pass
 return selected

# Cache only successful capability queries. The key follows the executable
# identity and loader search path, so replacing a binary or library path
# cannot silently reuse a capability learned from a different installation.
_memory_backend_capabilities={}
def _native_key_info(native,env):
 resolved=shutil.which(native) or native
 st=os.stat(resolved)
 # A wheel may update its CUDA library without changing the executable or
 # loader path. Never reuse capabilities from different dependency bytes.
 # In persistent mode the resolver checks file stamps and caches the digest.
 from .autotune import _dependency_identity
 dependency=_dependency_identity(str(resolved),env)
 dependency_key=None if dependency is None else (
  dependency.get('cuda_library_sha256'),
  tuple(tuple(item) for item in dependency.get('resolved_dependencies',())))
 loader=tuple(env.get(k,'') for k in ('LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT'))
 test_options=tuple(env.get(k,'') for k in ('MGBFS_SEGMENT_TEST_BATCH','MGBFS_SEGMENT_TEST_BATCH_ABI_MISSING'))
 identity=(str(Path(resolved).resolve()),st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,loader,test_options,dependency_key)
 if dependency is not None and identity in _memory_backend_capabilities:return _memory_backend_capabilities[identity]
 result=native_query([str(native),'key-info'],env=env,capture_output=True,text=True)
 try: info=json.loads(result.stdout)
 except (TypeError,ValueError):info={}
 if result.returncode or not isinstance(info,dict) or info.get('schema')!=1:return {}
 if dependency is not None:
  if len(_memory_backend_capabilities)>=64:_memory_backend_capabilities.clear()
  _memory_backend_capabilities[identity]=info
 return info

def _require_memory_backend(native,env,backend):
 key_name={'segmented24':'segmented24_v1','permutation_bitmap':'permutation_bitmap_v1'}[backend]
 info=_native_key_info(native,env)
 if info.get(key_name) is not True or info.get('segmented32_v2') is not True or info.get('segmented_memory_presets_v1') is not True:
  raise RuntimeError('NATIVE_MEMORY_BACKEND_UNAVAILABLE: '+backend+' requires a matching native executable and CUDA library built from the segmented backend source')
 return info

def _compact_path_capability(bidirectional):
 return 'graph_path_compact_bidirectional_v1' if bidirectional else 'graph_path_compact_forward_v1'


def _segmented_nearest_capability(bidirectional):
 return 'graph_path_segmented_nearest_bidirectional_v1' if bidirectional else _SEGMENTED_NEAREST_CAPABILITY


def _segmented_window_capability(bidirectional):
 return 'graph_path_segmented_window_bidirectional_v1' if bidirectional else 'graph_path_segmented_window_v1'


def _require_segmented_path_capabilities(info,bidirectional,*,window=False):
 mode='SEGMENTED_WINDOW' if window else 'SEGMENTED_NEAREST'
 if (info.get(_segmented_nearest_capability(bidirectional)) is not True
     or window and info.get(_segmented_window_capability(bidirectional)) is not True):
  raise RuntimeError(f'NATIVE_PATH_{mode}_UNAVAILABLE: rebuild the native executable and CUDA library')


def _window_shards(env):
 if env.get(_PATH_SEGMENTED_WINDOW) != '1':
  return 1
 text=env.get('MGBFS_SEGMENT_WINDOW_SHARDS','1')
 if not text.isascii() or not text.isdecimal() or len(text)>4 or not 1<=int(text)<=4096:
  raise ValueError('INVALID_NATIVE_SEGMENTED_WINDOW_SHARDS')
 return int(text)


def _compact_path_env(native,env,world,bidirectional,*,external):
 if world<2:raise ValueError('NATIVE_PATH_COMPACT_REQUIRES_MULTIPLE_GPUS')
 address=env.get(_PACKET_ADDR)
 if external and not address:raise ValueError('NATIVE_PATH_COMPACT_REQUIRES_EXTERNAL_PACKET_ADDR')
 if address=='':raise ValueError('NATIVE_PATH_COMPACT_PACKET_ADDR')
 info=_native_key_info(native,env)
 if env.get(_PATH_SEGMENTED_WINDOW)=='1' or env.get(_PATH_SEGMENTED_NEAREST)=='1':
  _require_segmented_path_capabilities(info,bidirectional,window=env.get(_PATH_SEGMENTED_WINDOW)=='1')
 elif info.get(_compact_path_capability(bidirectional)) is not True:
  raise RuntimeError('NATIVE_PATH_COMPACT_UNAVAILABLE: rebuild the native executable and CUDA library')
 if address is None:
  with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
   probe.bind(('127.0.0.1',0))
   address=f'127.0.0.1:{probe.getsockname()[1]}'
 return dict(env,**{_PACKET_ADDR:address,_PACKET_RING:'3',_PACKET_RX:'3'})

def _check_memory_backend_receipt(report,backend):
 expected={'segmented24':'EXPERIMENTAL_SEGMENTED24','permutation_bitmap':'EXPERIMENTAL_PERMUTATION_BITMAP'}.get(backend)
 if expected is not None and report.get('backend')!=expected:
  raise RuntimeError('NATIVE_MEMORY_BACKEND_RECEIPT_MISMATCH: expected '+expected+', got '+str(report.get('backend')))

def _receipt(output,digest):
 report=json.loads((output/'report.json').read_text(encoding='utf-8'))
 if report['graph_digest']!=digest:raise RuntimeError('GRAPH_RECEIPT_IDENTITY_MISMATCH')
 if report['status'] not in ('COMPLETE','INCOMPLETE'):raise RuntimeError('GRAPH_RECEIPT_STATUS')
 if hashlib.sha256((output/'states.json').read_bytes()).hexdigest()!=report['states_sha256']:raise RuntimeError('GRAPH_RECEIPT_CHECKSUM')
 return report

def _job_phase(job,output,rank):
 if hasattr(job,'phase'):return job.phase()
 path=Path(output)/('rank-'+str(rank)+'.phase.json')
 try:
  from .metadata_io import read_json
  value=read_json(path)
 except ValueError:return None
 if not isinstance(value,dict) or value.get('schema')!=1 or value.get('pid')!=job.pid:return None
 if not isinstance(value.get('entered'),dict) or not isinstance(value.get('phase'),str):return None
 return value

def _phase_failures(phases):
 failures=[]
 for rank,phase in enumerate(phases):
  error=phase.get('native_error') if isinstance(phase,dict) else None
  if isinstance(error,dict) and type(error.get('code')) is int and error['code']!=0:failures.append((rank,1))
 return failures

def _native_log_faults(jobs,output,cursors):
 """Watch the separate native log even if writing the phase file fails.

 Each cursor begins at this job's offset in a resident worker, so an old
 fault cannot contaminate a later job. Read bounded increments only.
 """
 failures=[]
 for rank,job in enumerate(jobs):
  worker=getattr(job,'worker',None)
  path=worker.root/'worker.log' if worker is not None else Path(output)/f'rank-{rank}.log'
  offset,tail=cursors.get(rank,(getattr(job,'log_offset',0),b''))
  try:
   with path.open('rb') as stream:
    stream.seek(offset);chunk=stream.read(65536);offset=stream.tell()
  except OSError:
   continue
  pieces=(tail+chunk).split(b'\n');cursors[rank]=(offset,pieces.pop()[-8192:])
  for line in pieces:
   marker=b'MGBFS_NATIVE_ERROR ';pos=line.find(marker)
   if pos<0:continue
   try:fault=json.loads(line[pos+len(marker):])
   except (ValueError,UnicodeDecodeError):fault={'message':line[pos:][-4096:].decode('utf-8','replace')}
   failures.append((rank,fault))
 return failures

def _stop_rank_jobs(jobs,grace=10.0,kill_grace=5.0):
 """Terminate a failed group concurrently under shared teardown deadlines."""
 for job in jobs:
  if job.poll() is None:
   try:job.terminate()
   except ProcessLookupError:pass
 deadline=time.monotonic()+grace
 for job in jobs:
  if job.poll() is None:
   try:job.wait(timeout=max(0.0,deadline-time.monotonic()))
   except subprocess.TimeoutExpired:pass
 for job in jobs:
  if job.poll() is None:
   try:job.kill()
   except ProcessLookupError:pass
 deadline=time.monotonic()+kill_grace
 for job in jobs:
  if job.poll() is None:
   try:job.wait(timeout=max(0.0,deadline-time.monotonic()))
   except subprocess.TimeoutExpired:pass
 return [getattr(job,'pid',None) for job in jobs if job.poll() is None]

from .work_limits import normalize, wire, remaining as remaining_work, phase_expired

def run_graph(graph,output,*,device=None,devices=None,capacity=None,max_seconds=None,executable=None,shards=None,autotune=True,backend='auto',memory_backend=None,segment_rows=None,memory_preset='balanced',segment_strategy=None,peer_transport='auto',transport='auto',candidate_order='auto',bitmap_owner_page=None,bitmap_pack4=None,resident_alphabet8=None,queue_capacity=None,owner_lanes=None,_batch=None,_profile_layers=None,_native_env=None,_profile_options=None,_profile_deadline=None,_path_targets=None,_path_bidirectional=False,_path_window=None,_path_max_depth=None):
 """Launch on all visible GPUs by default, or an explicit device/device list.

 Memory is admitted from actual free VRAM. With shards omitted, bounded
 GPU prefix measurements select a cached profile for substantial workloads.
 With WORLD_SIZE/RANK/LOCAL_RANK and MASTER_ADDR/MASTER_PORT it uses
 the external-rank network launcher; one process owns one local GPU.
 With storage unspecified, a requested capacity beyond the legacy index
 range selects segmented storage for generic traversal.
 bitmap_pack4 opts into the four-bit frontier for atomic permutation bitmap
 with a 64-bit word per state. Wider small-alphabet graphs use 1–3 bits
 per coordinate. None honors MGBFS_SEGMENT_PACK4.
 resident_alphabet8 selects exact byte labels for wide segmented permutations.
 """
 graph=_normalize_graph(graph)
 if device is not None and devices is not None:raise ValueError('DEVICE_SELECTION_CONFLICT')
 if device is not None:
  if type(device) is not int or device<0:raise ValueError('INVALID_DEVICE')
  devices=[device]
 if devices is not None and (not isinstance(devices,(list,tuple)) or not devices or any(type(v) is not int or v<0 for v in devices) or len(set(devices))!=len(devices)):raise ValueError('INVALID_DEVICES')
 max_seconds=normalize(max_seconds)
 if shards is not None and (type(shards) is not int or not 1<=shards<=4096):raise ValueError('INVALID_SHARDS')
 if type(autotune) is not bool:raise ValueError('INVALID_AUTOTUNE')
 if bitmap_pack4 is not None and type(bitmap_pack4) is not bool:raise ValueError('BITMAP_PACK4_INVALID')
 if segment_rows is not None and (type(segment_rows) is not int or not 1<=segment_rows<=1<<32):raise ValueError('INVALID_SEGMENT_ROWS')
 if bitmap_owner_page is not None and (type(bitmap_owner_page) is not int or bitmap_owner_page<0 or bitmap_owner_page>1<<20 or (bitmap_owner_page and bitmap_owner_page&(bitmap_owner_page-1))):raise ValueError('INVALID_BITMAP_OWNER_PAGE')
 if backend not in ('auto','generic','shard_ab_hash','shard_ab_sort_merge'):raise ValueError('INVALID_BACKEND')
 external=os.environ.get('WORLD_SIZE','1')!='1'
 strategy_source=_native_env if _native_env is not None else os.environ
 native_selection=None
 segmented_nearest=segmented_window=False
 requested_window=_path_window is not None and strategy_source.get(_PATH_SEGMENTED_WINDOW)=='1'
 requested_nearest=_path_window is None and strategy_source.get(_PATH_SEGMENTED_NEAREST)=='1'
 if _path_targets is not None and (requested_nearest or requested_window):
  if external:
   world=int(os.environ['WORLD_SIZE'])
  else:
   if devices is None:
    native_selection=native_runtime(executable)
    native,env=native_selection
    discovery_env=dict(_native_env if _native_env is not None else env)
    discovery_env.pop(_PATH_SEGMENTED_NEAREST,None)
    discovery_env.pop(_PATH_SEGMENTED_WINDOW,None)
    count=native_query([native,'graph-count'],env=discovery_env,capture_output=True,text=True)
    if count.returncode:raise RuntimeError('NATIVE_GPU_DISCOVERY_FAILED: '+count.stderr[-4000:])
    devices=list(range(int(count.stdout.strip())))
   world=len(devices)
  segmented_nearest=requested_nearest and world>1
  if requested_window and world<2:raise ValueError('NATIVE_PATH_SEGMENTED_WINDOW_REQUIRES_MULTIPLE_GPUS')
  segmented_window=requested_window
 segmented_path=segmented_nearest or segmented_window
 if segmented_path:
  if (memory_backend is not None and _normalize_memory_backend(memory_backend)!='segmented24'
      or segment_strategy not in (None,'batch') or bitmap_pack4 or resident_alphabet8
      or graph.generator_count>=0x7fff_ffff):
   raise ValueError('INVALID_NATIVE_SEGMENTED_PATH_LAUNCH')
  if strategy_source.get('MGBFS_NATIVE_DISTRIBUTED_NEAREST')=='0':
   raise ValueError('NATIVE_PATH_SEGMENTED_NEAREST_DISTRIBUTED_DISABLED')
  memory_backend,segment_strategy,bitmap_pack4,resident_alphabet8='segmented24','batch',False,False
 if _path_max_depth is not None and (
     not segmented_nearest or type(_path_max_depth) is not int or not 0<=_path_max_depth<=0xffff_ffff):
  raise ValueError('INVALID_NATIVE_PATH_MAX_DEPTH')
 if memory_backend is None:
  memory_backend='legacy'
  if (_path_targets is None and backend in ('auto','generic') and type(capacity) is int
      and capacity>_LEGACY_ROLLING_CAPACITY
      and (capacity>_LEGACY_ARENA_CAPACITY or graph.inverse_closed())):
   memory_backend='segmented24'
 memory_backend=_normalize_memory_backend(memory_backend)
 if resident_alphabet8 is not None and type(resident_alphabet8) is not bool:
  raise ValueError('INVALID_RESIDENT_ALPHABET8')
 if resident_alphabet8 and (memory_backend!='segmented24' or graph.action['kind']!='permutation'
                           or graph.state_elements<=24 or len(set(graph.start))>256):
  raise ValueError('RESIDENT_ALPHABET8_REQUIRES_WIDE_SEGMENTED_PERMUTATION')
 dictionary_requested = resident_alphabet8 and any(not 0<=value<=255 for value in graph.start)
 if queue_capacity is not None and (type(queue_capacity) is not int or not 1<=queue_capacity<=0xffffffff):
  raise ValueError('INVALID_SEGMENTED_QUEUE_CAPACITY')
 if owner_lanes is not None and (type(owner_lanes) is not int or not 1<=owner_lanes<=64):
  raise ValueError('INVALID_SEGMENTED_OWNER_LANES')
 if memory_backend=='legacy' and (queue_capacity is not None or owner_lanes is not None):
  raise ValueError('QUEUE_AND_OWNER_GEOMETRY_REQUIRES_SEGMENTED_BACKEND')
 if bitmap_owner_page and memory_backend!='permutation_bitmap':raise ValueError('BITMAP_OWNER_PAGE_REQUIRES_BITMAP')
 implicit_strategy=segment_strategy is None and 'MGBFS_SEGMENT_STRATEGY' not in strategy_source
 if segment_strategy is None:segment_strategy=strategy_source.get('MGBFS_SEGMENT_STRATEGY','atomic')
 segment_strategy=_normalize_segment_strategy(segment_strategy)
 packed=strategy_source.get('MGBFS_SEGMENT_PACK4','0')
 if bitmap_pack4 is not None:packed='1' if bitmap_pack4 else '0'
 if packed not in ('0','1'):raise ValueError('BITMAP_PACK4_INVALID')
 if packed=='1':
  if memory_backend!='permutation_bitmap':raise ValueError('BITMAP_PACK4_REQUIRES_BITMAP')
  if segment_strategy!='atomic':raise ValueError('BITMAP_PACK4_STRATEGY_UNSUPPORTED')
  if bitmap_pack4_bits(graph) is None:raise ValueError('BITMAP_PACK4_UNSUPPORTED_DEGREE')
 if memory_backend=='legacy' and segment_strategy!='atomic':raise ValueError('SEGMENT_STRATEGY_REQUIRES_SEGMENTED_BACKEND')
 if memory_preset=='maxmemory':memory_preset='max_capacity'
 if memory_preset not in ('balanced','max_capacity'):raise ValueError('INVALID_MEMORY_PRESET')
 if memory_backend=='permutation_bitmap':
  if graph.action['kind']!='permutation':raise ValueError('BITMAP_REQUIRES_PERMUTATION_ACTION')
  if not 1<=len(graph.start)<=64:raise ValueError('BITMAP_UNSUPPORTED_DEGREE')
 if peer_transport not in ('auto','host','lsa'):raise ValueError('INVALID_PEER_TRANSPORT')
 if peer_transport=='lsa' and not backend.startswith('shard_ab_'):raise ValueError('LSA_REQUIRES_SPECIALIZED_BACKEND')
 if transport not in ('auto','full','parent'):raise ValueError('INVALID_TRANSPORT')
 if candidate_order not in ('auto','none','radix'):raise ValueError('INVALID_CANDIDATE_ORDER')
 if capacity is not None and (type(capacity) is not int or not 1<=capacity<=(0xffffffffffffffff if memory_backend in ('segmented24','permutation_bitmap') else _LEGACY_ARENA_CAPACITY)):raise ValueError('INVALID_CAPACITY')
 from .host_memory import admit
 host_memory=admit(graph,output=output) if not external else None
 native,native_env=native_selection or native_runtime(executable)
 if _native_env is not None:native_env=dict(_native_env)
 if _PATH_SEGMENTED_NEAREST in native_env or _PATH_SEGMENTED_WINDOW in native_env:
  native_env=dict(native_env)
  native_env.pop(_PATH_SEGMENTED_NEAREST,None)
  native_env.pop(_PATH_SEGMENTED_WINDOW,None)
 if segmented_path:
  key=_PATH_SEGMENTED_WINDOW if segmented_window else _PATH_SEGMENTED_NEAREST
  native_env=dict(native_env,**{key:'1'},
                  MGBFS_SEGMENT_HASH_SERIES='1',MGBFS_SEGMENT_EXACT_EXCHANGE='0')
  native_env.pop(_PATH_COMPACT,None)
 for value,key,marker in (
  (queue_capacity,'MGBFS_SEGMENT_QUEUE_CAP','_MGBFS_PRESET_IMPLICIT_QUEUE'),
  (owner_lanes,'MGBFS_SEGMENT_OWNER_LANES','_MGBFS_PRESET_IMPLICIT_LANES'),
 ):
  if value is not None:
   native_env[key]=str(value)
   native_env.pop(marker,None)
 native_env.pop('MGBFS_SEGMENT_EXTERNAL_LINK',None)
 if _path_window is not None and _path_targets is None:raise ValueError('INVALID_NATIVE_WINDOW_LAUNCH')
 if _path_bidirectional and _path_targets is None:raise ValueError('INVALID_NATIVE_PATH_LAUNCH')
 if _path_targets is not None:
  native_env['MGBFS_GENERIC_PATH_MAXMEMORY']='1' if memory_preset=='max_capacity' else '0'
  native_env['MGBFS_GENERIC_PATH_WINDOW']='1' if _path_window is not None else '0'
  native_env['MGBFS_GENERIC_PATH_BIDIRECTIONAL']='1' if _path_bidirectional else '0'
  native_env['MGBFS_GENERIC_PATH_BYTE_TARGETS']='1' if (
   _path_bidirectional and all(0 <= value <= 255 for goal in _path_targets for value in goal)
  ) else '0'
  if (memory_backend!=('segmented24' if segmented_path else 'legacy') or backend!='generic'
      or autotune or _profile_layers is not None):raise ValueError('INVALID_NATIVE_PATH_LAUNCH')
  native_env.update(MGBFS_GENERIC_PATH_APPEND_ONLY='1',MGBFS_GENERIC_HISTORY='hash',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_TRANSPORT='full',MGBFS_GENERIC_SORT='none')
 else:native_env.pop(_PATH_COMPACT,None)
 if bitmap_pack4 is not None:native_env['MGBFS_SEGMENT_PACK4']=packed
 native_env=_apply_memory_preset(native_env,memory_backend,'balanced' if _path_targets is not None else memory_preset)
 if resident_alphabet8 is not None:native_env['MGBFS_SEGMENT_RESIDENT_ALPHABET8']='1' if resident_alphabet8 else '0'
 if bitmap_owner_page is not None:native_env['MGBFS_SEGMENT_BITMAP_OWNER_PAGE']=str(bitmap_owner_page)
 segmented=(memory_backend in ('segmented24','permutation_bitmap'))
 if not segmented:
  # An inherited private experiment flag must not override an explicit API choice.
  native_env.pop('MGBFS_EXPERIMENTAL_SEGMENTED24',None);native_env.pop('MGBFS_EXPERIMENTAL_BITMAP',None);native_env.pop('MGBFS_SEGMENT_MAX_RANK_STATES',None)
 if segmented:
  if external:
   from .cluster import current_cluster
   if current_cluster() is None:raise ValueError('SEGMENTED_EXTERNAL_RANK_NOT_VERIFIED')
  if segment_rows is not None:native_env['MGBFS_SEGMENT_MAX_ROWS']=str(segment_rows)
  else:native_env.setdefault('MGBFS_SEGMENT_MAX_ROWS',str(1<<32))
  if backend not in ('auto','generic') or peer_transport=='lsa' or transport not in ('auto','full') or candidate_order not in ('auto','none'):raise ValueError('SEGMENTED_BACKEND_OPTION_CONFLICT')
  capabilities=_require_memory_backend(native,native_env,memory_backend)
  if implicit_strategy and memory_backend=='segmented24' and bounded_hash_available(capabilities,native_env):
   segment_strategy='batch'
  if dictionary_requested and capabilities.get('segmented_resident_alphabet8_v1') is not True:
   raise RuntimeError('SEGMENTED_RESIDENT_ALPHABET8_NATIVE_UNAVAILABLE')
  if 'MGBFS_SEGMENT_PACKET_CONTROL_ADDR' in native_env and capabilities.get('segmented_packet_transport_v1') is not True:
   raise RuntimeError('SEGMENT_PACKET_CUDA_ABI_UNAVAILABLE: rebuild the native executable and CUDA library')
  if packed=='1' and (capabilities.get('bitmap_pack4_experimental_v1') is not True or
      (graph.state_elements>16 and capabilities.get('bitmap_pack4_adaptive_v2') is not True)):
   raise RuntimeError('BITMAP_PACK4_NATIVE_ABI_MISSING: experimental executable and CUDA library must match')
  if external:
   if capabilities.get('segmented_network_agreement_v1') is not True:
    raise ValueError('SEGMENTED_EXTERNAL_RANK_NOT_VERIFIED')
   native_env['MGBFS_SEGMENT_EXTERNAL_LINK']='1'
  if memory_backend=='segmented24' and segment_strategy=='warp' and capabilities.get('segmented_warp_hash_safe_v1') is not True:raise RuntimeError('SEGMENT_WARP_SAFE_NATIVE_ABI_MISSING')
  if memory_backend=='permutation_bitmap' and native_env.get('MGBFS_SEGMENT_BITMAP_OWNER_PAGE','0') not in ('0','') and capabilities.get('bitmap_owner_immutable_v1') is not True:raise RuntimeError('BITMAP_OWNER_PAGE_NATIVE_ABI_MISSING')
  if segment_strategy!='atomic':
   supported=capabilities.get('segmented_strategies_v1')
   if not isinstance(supported,dict) or supported.get(segment_strategy) is not True:
    raise RuntimeError('NATIVE_SEGMENT_STRATEGY_UNAVAILABLE: '+segment_strategy+' requires a matching verified native executable and CUDA library')
  if memory_backend=='segmented24' and segment_strategy=='batch' and native_env.get('MGBFS_SEGMENT_HASH_SERIES')!='0':
   if capabilities.get('segmented_hash_series_v1') is not True:
    raise RuntimeError('SEGMENT_HASH_SERIES_CUDA_ABI_UNAVAILABLE: rebuild the native executable and CUDA library')
  native_env['MGBFS_SEGMENT_STRATEGY']=segment_strategy
  exchange_capable=capabilities.get('segmented_exact_exchange_v1') is True
  if native_env.get('MGBFS_SEGMENT_EXACT_EXCHANGE')=='1' and not exchange_capable:raise RuntimeError('NATIVE_SEGMENTED_EXCHANGE_UNAVAILABLE: update the native runtime before requesting exact exchange')
  native_env['_MGBFS_SEGMENT_EXCHANGE_CAPABLE']='1' if exchange_capable else '0'
  native_env=dict(native_env,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_GENERIC_HISTORY='hash',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_TRANSPORT='full',MGBFS_GENERIC_SORT='none')
  if memory_backend=='permutation_bitmap':
   native_env['MGBFS_EXPERIMENTAL_BITMAP']='1'
   native_env['_MGBFS_BITMAP_POSITION_ORBITS_CAPABLE']='1' if capabilities.get('permutation_bitmap_position_orbits_v1') is True else '0'
  else:native_env.pop('MGBFS_EXPERIMENTAL_BITMAP',None)
  if capacity is not None:
   prior=int(native_env.get('MGBFS_SEGMENT_MAX_RANK_STATES',str(capacity)))
   native_env['MGBFS_SEGMENT_MAX_RANK_STATES']=str(min(prior,capacity))
   # Legacy graph-info capacity is only used to size bounded transport; it
   # must not become an implicit rank-wide segment capacity.
   capacity=None
 communication=None
 if _profile_layers is None and not external and not backend.startswith('shard_ab_'):
  if Path(output).exists():raise FileExistsError(output)
  selected=devices
  if selected is None:
   count=native_query([native,'graph-count'],env=native_env,capture_output=True,text=True)
   if count.returncode:raise RuntimeError('NATIVE_GPU_DISCOVERY_FAILED: '+count.stderr[-4000:])
   selected=list(range(int(count.stdout.strip())))
  path_compact=_path_targets is not None and (segmented_path or native_env.get(_PATH_COMPACT)=='1')
  if path_compact:
   native_env=_compact_path_env(native,native_env,len(selected),_path_bidirectional,external=False)
   devices=selected
  from .nccl_preflight import prepare
  # The communication prefix uses its own admission. Request-local limits
  # belong to the actual search, where a resource stop preserves its prefix.
  preflight_env=dict(native_env)
  search_limits={}
  for key in (
   'MGBFS_SEGMENT_MAX_RANK_STATES',
   'MGBFS_SEGMENT_HASH_INDEX_VRAM_BYTES',
   'MGBFS_SEGMENT_HASH_HOST_BYTES',
   'MGBFS_SEGMENT_TEST_MAX_LOOKUP_SERIES',
  ):
   if key in preflight_env:
    search_limits[key]=preflight_env.pop(key)
  if _path_targets is not None:
   for key in tuple(preflight_env):
    if (key.startswith('MGBFS_GENERIC_PATH_') or key in (_PATH_SEGMENTED_NEAREST,_PATH_SEGMENTED_WINDOW)
        or (path_compact and key in _PATH_PACKET_ENV)):
     search_limits[key]=preflight_env.pop(key)
  native_env,communication=prepare(native,selected,preflight_env,output,graph)
  native_env.update(search_limits)
 geometry_bound=None;shard_geometry=None
 if _path_targets is None and capacity is None and _profile_layers is None and not segmented and not backend.startswith('shard_ab_'):
  from .workload_geometry import certified_frontier_bound
  geometry_bound=certified_frontier_bound(graph)
  if geometry_bound is not None and not external:
   from .autotune import _admit
   with tempfile.TemporaryDirectory(prefix='mgbfs-bound-') as bound_dir:
    bound_admission=_admit(graph,devices,None,shards or 1,native,native_env,bound_dir)
   capacity=min(geometry_bound['maximum_frontier'],bound_admission['plan']['capacity'])

 native_env=dict(native_env,MGBFS_PEER_TRANSPORT=peer_transport)
 def finish(report):
  _check_memory_backend_receipt(report,memory_backend)
  if dictionary_requested:
   ranks=report.get('segment_memory_by_rank')
   if not isinstance(ranks,list) or not ranks or any(not isinstance(rank,dict) or rank.get('resident_codec')!='alphabet8-v1' for rank in ranks):
    raise RuntimeError('RESIDENT_ALPHABET8_RECEIPT_MISMATCH')
  if packed=='1':
   ranks=report.get('segment_memory_by_rank')
   if not isinstance(ranks,list) or not ranks or any(not isinstance(rank,dict) or rank.get('resident_codec')!='pack4-v1' for rank in ranks):
    raise RuntimeError('BITMAP_PACK4_RECEIPT_MISMATCH')
  if communication is not None:
   report['communication_preflight']=communication
   if communication.get('status')=='VERIFIED':
    from .generic_session import _active
    from .nccl_preflight import bind_communication_group
    bind_communication_group(_active)
  if geometry_bound is not None:report['workload_geometry_bound']=geometry_bound
  if shard_geometry is not None:report['shard_geometry']=shard_geometry
  from .size_profiles import save_online_profiles
  online_cache_status=save_online_profiles(report)
  if online_cache_status is not None:report['online_profile_cache_write_status']=online_cache_status
  from .native_distribution import runtime_identity
  report['runtime_identity']=runtime_identity(native,native_env)
  if host_memory is not None:report['host_memory']=[host_memory]
  path=Path(output)/'report.json';temporary=path.with_name('report.json.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(path)
  return report
 if transport!='auto':native_env=dict(native_env,MGBFS_GENERIC_TRANSPORT=transport)
 if candidate_order!='auto':native_env=dict(native_env,MGBFS_GENERIC_SORT=candidate_order)
 if external:
  if backend not in ('auto','generic') or peer_transport=='lsa':raise ValueError('SPECIALIZED_EXTERNAL_RANK_UNSUPPORTED')
  from .distributed_launch import run_external
  if _path_targets is not None:
   return run_external(graph,output,devices=devices,capacity=capacity,max_seconds=max_seconds,native=native,native_env=native_env,shards=shards,autotune=autotune,batch=_batch,path_targets=_path_targets,path_bidirectional=_path_bidirectional,path_window=_path_window,path_max_depth=_path_max_depth)
  return finish(run_external(graph,output,devices=devices,capacity=capacity,max_seconds=max_seconds,native=native,native_env=native_env,shards=shards,autotune=autotune,batch=_batch))
 output=Path(output).absolute()
 if output.exists():raise FileExistsError(output)
 profile=None
 if backend.startswith('shard_ab_'):
  if shards is not None or transport!='auto' or candidate_order!='auto':raise ValueError('SPECIALIZED_ADAPTIVE_GEOMETRY_OR_GENERIC_OPTION_CONFLICT')
  return finish(_run_specialized_admitted(graph,output,native,native_env,devices,capacity,max_seconds,backend,_batch,_profile_layers,None))
 fixed_segmented_tuning=segmented and autotune and shards is not None and _batch is None and _profile_layers is None
 if shards is None or fixed_segmented_tuning:
  if autotune:
   from .autotune import choose_size_profile,choose_profile
   # Reserve most of the caller's deadline for BFS, not cold tuning.
   tuning_budget=60 if max_seconds is None else min(60,max(1,max_seconds//3))
   profile=(choose_profile(graph,devices,capacity,tuning_budget,native,native_env,allow_specialized=False,fixed_shards=shards if fixed_segmented_tuning else None,**(_profile_options or {})) if segmented else choose_size_profile(graph,devices,capacity,tuning_budget,native,native_env,allow_specialized=backend=='auto'))
   shards=profile['shards']
   if segmented:shard_geometry=profile.get('shard_geometry')
   if profile.get('backend','generic')!='generic':
    remaining=remaining_work(max_seconds,profile['seconds'])
    return finish(_run_specialized_admitted(graph,output,native,native_env,devices,capacity,remaining,profile['backend'],_batch,_profile_layers,profile))
   from .autotune import _profile_environment
   native_env=_profile_environment(native_env,profile)
   if (profile.get('measured') or profile.get('apply_profile_batch') is True or profile.get('status')=='REUSED_COMPATIBLE_GPU_PROFILE') and profile.get('backend','generic')=='generic':_batch=profile['batch']
   max_seconds=remaining_work(max_seconds,profile['seconds'])
  elif segmented:
   from .autotune import _segmented_shard_plan
   with tempfile.TemporaryDirectory(prefix='mgbfs-segment-shards-') as shard_dir:
    shard_geometry,_=_segmented_shard_plan(graph,devices,capacity,native,native_env,shard_dir)
   shards=shard_geometry['admitted_shards']
  else:shards=1
 output=Path(output).absolute()
 if output.exists():raise FileExistsError(output)
 output.parent.mkdir(parents=True,exist_ok=True);digest=graph.digest()
 with tempfile.TemporaryDirectory(prefix='mgbfs-definition-',dir=output.parent) as temporary:
  definition=Path(temporary)/'graph.json';definition.write_text(graph.to_json(),encoding='utf-8')
  selection='auto' if devices is None else ','.join(map(str,devices));command=[str(native),'graph-info',str(definition),selection,str(shards)]
  # A bounded pilot may reuse the exclusive startup's inventory. Real runs
  # always query live VRAM again; no shared inventory on production admission.
  if _profile_layers is not None and native_env.get('_MGBFS_PROFILE_INVENTORY'):
   cached_inventory=Path(temporary)/'inventory.json';cached_inventory.write_text(native_env['_MGBFS_PROFILE_INVENTORY'])
   command=[str(native),'graph-plan',str(definition),str(cached_inventory),str(shards)]
  if capacity is not None:command.append(str(capacity))
  if _batch is None and profile and profile.get('batch_fraction',1.0)!=1.0:
   initial=native_query(command,capture_output=True,text=True,env=native_env)
   check_graph_admission(initial)
   _batch=max(1,int(json.loads(initial.stdout)['plan']['batch']*profile['batch_fraction']))
  if _batch is not None:
   if capacity is None:command.append('auto')
   command.append(str(_batch))
  from .generic_session import take_launch_admission
  accepted=take_launch_admission(graph,native,native_env,devices,capacity,shards,_batch) if profile is not None and _profile_layers is None else None
  probe=subprocess.CompletedProcess(command,0,json.dumps(accepted),'') if accepted is not None else native_query(command,capture_output=True,text=True,env=native_env)
  if profile is not None:profile['launch_admission_reused']=accepted is not None
  check_graph_admission(probe)
  admission=_rank_launch_config(
   json.loads(probe.stdout),digest,bitmap=memory_backend=='permutation_bitmap',depth=_profile_layers,
   path_targets=_path_targets,path_bidirectional=_path_bidirectional,path_window=_path_window,
   path_max_depth=_path_max_depth)
  devices=admission['devices']
  if segmented:
   # Scale a known maximum frontier per bank to the admitted GPU and shard
   # count. Other graph families are sized natively from their orbit bound.
   from .workload_geometry import coxeter_frontier_bound
   frontier=coxeter_frontier_bound(graph) if admission['plan'].get('history_layers',3)==3 else None
   if frontier:
    divisor=4*len(devices)*shards
    expected=(frontier['maximum_frontier']*5+divisor-1)//divisor
    native_env['MGBFS_SEGMENT_EXPECTED_ROWS']=str(max(32,expected))
  if admission.get('owned_resource_admission'):native_env=dict(native_env,MGBFS_REQUIRE_OWNED_RAW='1')
  if profile and profile.get('size_profiles'):admission.update(size_profiles=profile['size_profiles'],online_size_tuning=profile.get('online_size_tuning',False))
  if _path_targets is None and not segmented and max_seconds is not None and admission['plan'].get('generator_backend','cuda')!='gemm' and admission['plan'].get('history_algorithm','HASH')=='HASH' and len(devices)==1 and shards==1 and _profile_layers is None and not (profile and profile.get('size_profiles')) and admission['plan'].get('history_layers',1)==1 and not admission['plan'].get('parent_transport',False) and not admission['plan'].get('sort_candidates',False):
   command=[str(native),'graph',str(definition),str(output),'--device',str(devices[0]),'--seconds',str(wire(max_seconds))]
   if capacity is not None:command+=['--capacity',str(capacity)]
   process=native_run(command,capture_output=True,text=True,env=native_env,timeout=None if max_seconds is None else max_seconds+180)
   if process.returncode:raise RuntimeError('NATIVE_GRAPH_FAILED: '+process.stderr[-4000:])
   result=_receipt(output,digest)
   if profile:
    result['autotune']=profile;(output/'report.json').write_text(json.dumps(result,indent=2))
   return finish(result)
  launch_config=_with_packet_job_token(admission,native_env)
  output.mkdir();configuration=output/'launch.json';configuration.write_text(json.dumps(launch_config));bootstrap=Path(temporary)/'nccl-id';jobs=[];logs=[];failure=None;started=time.monotonic()
  try:
   if _profile_deadline is not None and time.monotonic()>=_profile_deadline:raise RuntimeError('AUTOTUNE_PILOT_WALL_DEADLINE')
   from .generic_session import launch_group_selection
   selection=launch_group_selection(str(native),devices,native_env)
   for rank in range(len(devices)):
    log=(output/f'rank-{rank}.log').open('w');logs.append(log)
    env=dict(native_env,RANK=str(rank),WORLD_SIZE=str(len(devices)),LOCAL_RANK=str(devices[rank]),MGBFS_JOB_PHASE_PATH=str(output/f'rank-{rank}.phase.json'))
    from .generic_session import launch as launch_rank
    jobs.append(launch_rank([str(native),'graph-rank',str(definition),str(configuration),str(rank),str(bootstrap),str(output/f'rank-{rank}'),str(wire(max_seconds))],env,log,devices,_selection=selection))
   next_phase_check=0.;log_cursors={}
   while any(p.poll() is None for p in jobs):
    failed=[(rank,p.returncode) for rank,p in enumerate(jobs) if p.poll() not in (None,0)]
    if failed:raise RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED '+repr(failed))
    now=time.monotonic()
    if _profile_deadline is not None and now>=_profile_deadline:raise RuntimeError('AUTOTUNE_PILOT_WALL_DEADLINE')
    if now>=next_phase_check:
     log_faults=_native_log_faults(jobs,output,log_cursors)
     if log_faults:raise RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED '+repr([(rank,1) for rank,_ in log_faults])+'; native_fault='+repr(log_faults))
     phases=[_job_phase(p,output,rank) for rank,p in enumerate(jobs)]
     failed=_phase_failures(phases)
     if failed:raise RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED '+repr(failed))
     timed_out=any(
      phase_expired(phase,time.time(),max_seconds)
      for phase in phases if phase is not None)
     if timed_out or (not all(phases) and now-started>(max_seconds or 0)+1200):raise RuntimeError('DISTRIBUTED_COMPLETION_TIMEOUT: worker state remains in rank logs; no restart')
     next_phase_check=now+.05
    time.sleep(.002)
   if any(p.returncode for p in jobs):raise RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED')
   failed=_phase_failures([_job_phase(p,output,rank) for rank,p in enumerate(jobs)])
   if failed:raise RuntimeError('NATIVE_DISTRIBUTED_RANK_FAILED '+repr(failed))

  except BaseException as error:
   failure=error
   from .failure_evidence import preserve_native_failure
   preserve_native_failure(jobs,output,error,when='before_cleanup')
   if isinstance(error,RuntimeError) and str(error)!='AUTOTUNE_PILOT_WALL_DEADLINE':error.args=(str(error)+'; evidence='+str(output/'failure-evidence'),)
   raise
  finally:
   survivors=_stop_rank_jobs(jobs);cleanup_error=None
   if survivors:
    note='DISTRIBUTED_TEARDOWN_INCOMPLETE pids='+repr(survivors)
    if failure is not None:failure.args=(str(failure)+'; '+note,)
    else:cleanup_error=RuntimeError(note)
   if failure is not None:
    from .failure_evidence import preserve_native_failure
    preserve_native_failure(jobs,output,failure,when='after_cleanup')
   for log in logs:log.close()
   if cleanup_error is not None:raise cleanup_error
  if _path_targets is not None:
   try:
    replies=[json.loads((output/f'rank-{rank}'/'path.json').read_text()) for rank in range(len(devices))]
   except (OSError, ValueError) as error:
    raise RuntimeError('NATIVE_DISTRIBUTED_PATH_RECEIPT_INVALID') from error
   if _path_window is not None:
    from .path_search import _agree_window_receipts
    result=_agree_window_receipts(replies)
   else:
    if not replies or any(reply!=replies[0] for reply in replies[1:]):
     raise RuntimeError('NATIVE_DISTRIBUTED_PATH_CONSENSUS_MISMATCH')
    result=replies[0]
   if communication is not None and communication.get('status')=='VERIFIED':
    from .generic_session import _active
    from .nccl_preflight import bind_communication_group
    bind_communication_group(_active)
   from .generic_session import path_workspace
   return dict(result, native_workspace={'ranks':[path_workspace(job) for job in jobs]})
  try:
   parts=[_receipt(output/f'rank-{rank}',digest) for rank in range(len(devices))]
   from .traversal_receipt import merge_rank_traversal
   snapshots=[json.loads((output/f'rank-{rank}'/'states.json').read_text()) for rank in range(len(devices))]
   receipt_plan=dict(devices=devices,rank_plans=admission.get('rank_plans',[admission['plan']]*len(devices)),
                     owner_cuts=admission.get('owner_cuts'))
   report,states=merge_rank_traversal(parts,snapshots,receipt_plan,error_prefix='DISTRIBUTED',
      segmented=segmented,bitmap=native_env.get('MGBFS_EXPERIMENTAL_BITMAP')=='1',sample_limit_field=True)
   (output/'states.json').write_bytes(json.dumps(states).encode())
   report.update(rank_receipts=[f'rank-{r}/report.json' for r in range(len(devices))],
                 launch_wall_seconds=time.monotonic()-started,rank_worker_pids=[p.pid for p in jobs],
                 scope='single host general exact retained-history path; bounded profile evidence in autotune receipt when enabled; larger hardware not verified')
   if profile:report['autotune']=profile;report['profile_status']=profile['status']
   (output/'report.json.tmp').write_text(json.dumps(report,indent=2));(output/'report.json.tmp').replace(output/'report.json')
   return finish(_receipt(output,digest))
  except BaseException:
   from .generic_session import invalidate_launch_group
   invalidate_launch_group(selection)
   raise


def _run_specialized_admitted(graph,output,native,env,devices,capacity,max_seconds,backend,batch,profile_layers,profile):
 from .specialized import admit_specialized,match_lrx,run_specialized,exact_specialized_supported
 if not exact_specialized_supported(graph):raise RuntimeError('SPECIALIZED_LOSSLESS_KEY_DOMAIN_UNSUPPORTED: no proved injective key for this graph')
 if match_lrx(graph) is None:raise ValueError('SPECIALIZED_GRAPH_UNSUPPORTED')
 output=Path(output).absolute();output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise FileExistsError(output)
 started=time.monotonic()
 with tempfile.TemporaryDirectory(prefix='mgbfs-specialized-admit-',dir=output.parent) as temporary:
  definition=Path(temporary)/'definition.json';definition.write_text(graph.to_json());selection='auto' if devices is None else ','.join(map(str,devices))
  probe=native_query([native,'graph-info',str(definition),selection,'1'],capture_output=True,text=True,env=env)
  check_graph_admission(probe)
  inventory=json.loads(probe.stdout)
  if inventory['graph_digest']!=graph.digest():raise RuntimeError('GRAPH_ADMISSION_IDENTITY')
  selected=inventory['devices'];mode='HASH' if backend=='shard_ab_hash' else 'SORT_MERGE'
  from .transport import select_peer_transport
  request=('lsa' if profile['specialized_transport']=='NCCL_LSA' else 'host') if profile and profile.get('specialized_transport') else env.get('MGBFS_PEER_TRANSPORT','auto')
  decision=select_peer_transport(native,env,selected,Path(temporary)/'transport',request=request,seconds=60 if max_seconds is None else max_seconds)
  env=dict(env,MGBFS_SPECIALIZED_TRANSPORT=decision['selected'])
  if decision['selected']=='NCCL_LSA':env['NCCL_CUMEM_ENABLE']='1'
  plan=admit_specialized(graph,native,env,selected,Path(temporary)/'queries',capacity=capacity,mode=mode,seconds=60 if max_seconds is None else max(1,max_seconds-(time.monotonic()-started)))
  actual_batch=plan['batch'] if batch is None else batch
  if actual_batch!=plan['batch']:raise ValueError('SPECIALIZED_BATCH_REQUIRES_MATCHED_ADMISSION')
  report=run_specialized(graph,output,native=native,env=env,devices=selected,capacity=plan['capacity'],batch=actual_batch,max_seconds=remaining_work(max_seconds,time.monotonic()-started),mode=mode,profile_layers=profile_layers)
  report['plan'].update(plan);report['startup_admission_seconds']=time.monotonic()-started-report['launch_wall_seconds']
  report['transport_selection']=decision
  if profile:report['autotune']=profile
  (output/'report.json').write_text(json.dumps(report,indent=2));return _receipt(output,graph.digest())

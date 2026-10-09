"""Lossless startup bridge for the existing compact LRX SHARD_AB owner.

No CPU successor generation/dedup. Only bounded committed snapshots are read.
"""
import hashlib,json,math,os,re,select,struct,subprocess,threading,time,uuid
from pathlib import Path

def exact_specialized_supported(graph=None):
 if graph is None:return False
 match=match_lrx(graph)
 if match is None:return False
 bits=max(1,(len(match['labels'])-1).bit_length())
 return match['n']*bits<=128

def require_lossless_native(native,env):
 p=subprocess.run([native,'key-info'],env=env,capture_output=True,text=True,timeout=10)
 try:capability=json.loads(p.stdout)
 except ValueError:capability={}
 if p.returncode or capability.get('schema')!=1 or capability.get('lossless_bitpack128_feistel_v1') is not True:raise RuntimeError('SPECIALIZED_NATIVE_LOSSLESS_KEYS_UNAVAILABLE')
 return capability

def match_lrx(graph):
 if graph.action['kind']!='permutation' or not 2<=graph.state_elements<=128:return None
 n=graph.state_elements;expected={tuple(list(range(1,n))+[0]),tuple([n-1]+list(range(n-1))),tuple([1,0]+list(range(2,n)))}
 if {tuple(g) for g in graph.action['generators']}!=expected:return None
 labels=[];index={};normalized=[]
 for value in graph.start:
  if value not in index:index[value]=len(labels);labels.append(value)
  normalized.append(index[value])
 r=n-len(labels)+1
 if normalized!=list(range(n-r+1))+[n-r]*(r-1):return None
 return dict(n=n,r=r,labels=labels,order=math.factorial(n)//math.factorial(r))

class _Pipe:
 def __init__(self,path,stopped):self.fd=os.open(path,os.O_RDWR|os.O_NONBLOCK);self.stopped=stopped
 def read(self,n):
  while True:
   try:
    data=os.read(self.fd,n)
    if data:return data
   except BlockingIOError:pass
   if self.stopped.is_set():return b''
   select.select([self.fd],[],[],.02)
 def close(self):os.close(self.fd)

def _exact(stream,n):
 data=bytearray()
 while len(data)<n:
  part=stream.read(n-len(data))
  if not part:raise EOFError('SPECIALIZED_ARCHIVE_TRUNCATED')
  data.extend(part)
 return bytes(data)

def read_prefix(stream,n):
 header=_exact(stream,48)
 if header[:8]!=b'MGBFSAS2' or struct.unpack_from('<Q',header,8)[0]!=n:raise RuntimeError('SPECIALIZED_ARCHIVE_HEADER')
 chain=hashlib.sha256(header).digest();seq=depth=total=rows=previous_rows=0;buffer=bytearray();layers={};replacement=False;replaced=False
 while True:
  frame=_exact(stream,80);kind,at,count,size,sequence=struct.unpack_from('<QQQQQ',frame,8)
  expected=depth-1 if kind==4 else depth
  if frame[:8]!=b'MGBFSFR1' or at!=expected or sequence!=seq or frame[48:]!=chain or size>1000*n:raise RuntimeError('SPECIALIZED_ARCHIVE_ORDER_BOUND')
  payload=_exact(stream,size);chain=hashlib.sha256(frame+payload).digest()
  if _exact(stream,32)!=chain:raise RuntimeError('SPECIALIZED_ARCHIVE_CHECKSUM')
  if kind==1:
   if not count or size!=count*n or rows+count>1000:raise RuntimeError('SPECIALIZED_ARCHIVE_RECORD_BOUND')
   buffer.extend(payload);rows+=count
  elif kind==2:
   if size or count!=rows:raise RuntimeError('SPECIALIZED_ARCHIVE_LAYER')
   layers[depth]=bytes(buffer)
   for old in sorted(layers)[:-2]:del layers[old]
   total+=rows;previous_rows=rows;rows=0;depth+=1;buffer.clear();replacement=False
  elif kind==4:
   if size or count or rows or not depth or replaced:raise RuntimeError('SPECIALIZED_ARCHIVE_REPLACEMENT')
   replaced=replacement=True;depth-=1;total-=previous_rows
  elif kind==3:
   if size or rows or not depth or count!=total:raise RuntimeError('SPECIALIZED_ARCHIVE_COMMIT')
   return dict(depths=depth,layers=layers,config_digest=header[16:].hex(),terminal_replaced=replaced)
  else:raise RuntimeError('SPECIALIZED_ARCHIVE_FRAME')
  seq+=1

def _environment(env,match,devices,capacity,batch,mode,query=False,profile_layers=None,shard_target=1<<20):
 if mode not in ('HASH','SORT_MERGE'):raise ValueError('SPECIALIZED_OWNER_MODE')
 world=len(devices)
 if world not in (1,2,4,8):raise ValueError('SPECIALIZED_WORLD_UNSUPPORTED')
 e=dict(env);e.update(MGBFS_EXACT_PACKED_KEYS='1',MGBFS_PROFILE='DENSE',MGBFS_OWNER_BACKEND='SHARD_AB',MGBFS_PRE_DEDUP='OFF',MGBFS_CAPACITY_MODE='max_per_rank',MGBFS_BENCH_CAPACITY=str(capacity),MGBFS_FUTURE_CAPACITY=str(capacity*3),MGBFS_BUCKET_CAPACITY=str(capacity),MGBFS_BUCKETS='16',MGBFS_SHARDS='8',MGBFS_JOB_BUCKETS='2',MGBFS_STATE_CODEC='permutation_u8',MGBFS_ARCHIVE_CODEC='permutation_u8',MGBFS_ARCHIVE_SELECTION='last_complete_prefix_1000',MGBFS_ARCHIVE_ROWS='1000',MGBFS_ARCHIVE_SLOTS='16',MGBFS_ARCHIVE_INITIAL_SLOTS='16',MGBFS_ARCHIVE_CREDIT_MODE='wait',MGBFS_SELECTED_WHOLE_ONLY='0',MGBFS_TRACE_DEPTHS='1',MGBFS_BENCH_WARMUP='0',MGBFS_BENCH_WORLD_SIZE=str(world),MGBFS_RANK_MAP=','.join(map(str,range(world))),MGBFS_MACRO_DEPTH='1',MGBFS_CUDA_GRAPH_BATCHES='0',MGBFS_TRANSPORT_BACKEND=env.get('MGBFS_SPECIALIZED_TRANSPORT','HOST_SIZED_NCCL'),MGBFS_SHARD_AB_DEDUP=mode,MGBFS_SHARD_AB_CAPACITY=str(shard_target),MGBFS_VRAM_RESERVE_BYTES=str(256<<20),MGBFS_BENCH_SKIP_ARCHIVE='1' if query else '0',MGBFS_ARCHIVE_STREAM='0' if query else '1')
 for key in ('MGBFS_LIBRARY_POOL_BYTES','MGBFS_LIBRARY_POOL_AUTOSIZE','MGBFS_CALIBRATION_LAYERS','MGBFS_SHARD_AB_SHARDS','MGBFS_MEMORY_QUERY','MGBFS_SEARCH_ONLY'):e.pop(key,None)
 for key in ('MGBFS_SHARD_AB_KEY_FIRST','MGBFS_COMPACT_DIRECT_HASH','MGBFS_SHARD_AB_PEER_METADATA','MGBFS_SHARD_AB_ASYNC_MATERIALIZE','MGBFS_SHARD_AB_DIRECT_INPUT','MGBFS_SORT_HISTORY_LOOKUP'):e[key]='1'
 for key in ('MGBFS_SHARD_AB_COMBINED_STATUS','MGBFS_SHARD_AB_FUSED_RESPONSE_META','MGBFS_SHARD_AB_REUSE_HISTORY'):e[key]='0'
 e['MGBFS_SHARD_AB_REUSE_PREOWNER_STATUS']='1' if mode=='HASH' else '0'
 if query:e['MGBFS_MEMORY_QUERY']='1'
 if profile_layers is not None:e['MGBFS_CALIBRATION_LAYERS']=str(profile_layers)
 return e

def _workers(native,root,match,devices,batch,env,seconds,query=False):
 (root/'native').mkdir();jobs=[];logs=[];started=time.monotonic();stop_at=None
 try:
  for rank,device in enumerate(devices):
   log=(root/f'rank-{rank}.log').open('w');logs.append(log);e=dict(env,RANK=str(rank),WORLD_SIZE=str(len(devices)),LOCAL_RANK=str(device),TORCHELASTIC_RUN_ID=root.name)
   command=[native,'bench','--reference',f"lrx{match['n']}r{match['r']}",str(batch),str(root/'bootstrap'),str(root/'archive'),str(root/'native')]
   if query:command.append('--search-only')
   jobs.append(subprocess.Popen(command,env=e,stdout=log,stderr=subprocess.STDOUT))
  while any(p.poll() is None for p in jobs):
   if time.monotonic()-started>seconds and stop_at is None:
    stop_at=time.monotonic()
    for p in jobs:
     if p.poll() is None:p.terminate()
   if stop_at is not None and time.monotonic()-stop_at>10:
    raise RuntimeError('SPECIALIZED_COMPLETION_TIMEOUT_NO_RESTART')
   if any(p.poll() not in (None,0,1) for p in jobs) and stop_at is None:
    stop_at=time.monotonic()
    for p in jobs:
     if p.poll() is None:p.terminate()
   time.sleep(.02)
  return [p.returncode for p in jobs],time.monotonic()-started,stop_at is not None
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   if p.poll() is None:
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for log in logs:log.close()

def query(native,env,graph,root,devices,capacity,batch,mode='HASH',seconds=60,shard_target=1<<20):
 match=match_lrx(graph)
 if match is None:raise ValueError('SPECIALIZED_GRAPH_UNSUPPORTED')
 if not exact_specialized_supported(graph):raise ValueError('SPECIALIZED_LOSSLESS_KEY_DOMAIN_UNSUPPORTED')
 require_lossless_native(native,env)
 root=Path(root);root.mkdir(parents=True);e=_environment(env,match,devices,capacity,batch,mode,query=True,shard_target=shard_target)
 codes,wall,stopped=_workers(native,root,match,devices,batch,e,seconds,query=True);records=[]
 for rank in range(len(devices)):
  text=(root/f'rank-{rank}.log').read_text();rows=[];errors=[]
  for line in text.splitlines():
   if line.startswith('MGBFS_MEMORY_QUERY '):rows.append(json.loads(line.split(' ',1)[1]))
   try:value=json.loads(line)
   except ValueError:continue
   if isinstance(value,dict) and value.get('status')=='ERROR':errors.append(value.get('error'))
  if stopped or codes[rank]!=1 or len(rows)!=1 or errors!=['MEMORY_QUERY_DONE'] or rows[0]['rank']!=rank:raise RuntimeError('SPECIALIZED_QUERY_FAILED: '+str(root))
  record=rows[0];record['query_to_run_margin_bytes']=64<<20;records.append(record)
 (root/'admission.json').write_text(json.dumps(records));return records

def run_specialized(graph,output,*,native,env,devices,capacity,batch,max_seconds,mode='HASH',profile_layers=None,shard_target=1<<20):
 match=match_lrx(graph)
 if match is None:raise ValueError('SPECIALIZED_GRAPH_UNSUPPORTED')
 if not exact_specialized_supported(graph):raise ValueError('SPECIALIZED_LOSSLESS_KEY_DOMAIN_UNSUPPORTED')
 require_lossless_native(native,env)
 output=Path(output);output.mkdir(parents=True);stopped=threading.Event();pipes=[];threads=[];receipts={};errors={};n=match['n']
 try:
  for rank in range(len(devices)):
   path=Path(str(output/'archive')+f'-rank-{rank}.mgbfsar1');os.mkfifo(path);pipe=_Pipe(path,stopped);pipes.append(pipe)
   def read(rank=rank,pipe=pipe):
    try:receipts[rank]=read_prefix(pipe,n)
    except BaseException as error:errors[rank]=str(error)
   t=threading.Thread(target=read,daemon=True);threads.append(t);t.start()
  e=_environment(env,match,devices,capacity,batch,mode,profile_layers=profile_layers,shard_target=shard_target)
  codes,wall,deadline=_workers(native,output,match,devices,batch,e,max_seconds)
 finally:
  stopped.set()
  for t in threads:t.join(timeout=10)
  for pipe in pipes:pipe.close()
 if errors or len(receipts)!=len(devices) or any(t.is_alive() for t in threads):raise RuntimeError('SPECIALIZED_COMMITTED_SNAPSHOT_MISSING '+repr(errors))
 if len({v['config_digest'] for v in receipts.values()})!=1:raise RuntimeError('SPECIALIZED_ARCHIVE_IDENTITY_MISMATCH')
 native_parts=[];traces=[]
 for rank in range(len(devices)):
  text=(output/f'rank-{rank}.log').read_text();counts={};times={}
  for line in text.splitlines():
   if line.startswith(('MGBFS_DEPTH_BEGIN ','MGBFS_DEPTH_END ')):
    fields=dict(token.split('=',1) for token in line.split()[1:]);depth=int(fields['depth'])
    if int(fields['rank'])!=rank:raise RuntimeError('SPECIALIZED_TRACE_RANK')
    if line.startswith('MGBFS_DEPTH_BEGIN '):counts[depth]=int(fields['count'])
    else:times[depth]=float(fields['seconds'])
  path=output/'native'/f'rank-{rank}.json';native_parts.append(json.loads(path.read_text()) if path.exists() else None);traces.append((counts,times,text))
 depth=min(receipts[rank]['depths'] for rank in receipts)
 if depth<1 or any(v['depths']!=depth for v in receipts.values()) or any(set(c)!=set(range(depth)) for c,_,_ in traces):raise RuntimeError('SPECIALIZED_LAYER_CONSENSUS')
 sizes=[sum(c[i] for c,_,_ in traces) for i in range(depth)]
 complete=all(code==0 for code in codes) and all(p and p['status']=='COMPLETE' for p in native_parts)
 calibration=all(code==0 for code in codes) and all(p and p['status']=='INCOMPLETE' and p.get('stop_reason')=='calibration layer limit' for p in native_parts)
 if not complete and not calibration:
  allowed=('GROUP_STATE_RING_RETIRE_FATAL_16','GROUP_STATE_RING_RETIRE_FATAL_112','SHARD_AB_PREPARE_FATAL_16','SHARD_AB_PREPARE_FATAL_112','GROUP_STATE_RING_RETIRE_FATAL_11','GROUP_STATE_RING_RETIRE_FATAL_12','SHARD_AB_PREPARE_FATAL_11','SHARD_AB_PREPARE_FATAL_12','REMOTE_SEARCH_CANCELLED')
  if not all(any(code in text for code in allowed) for _,_,text in traces):raise RuntimeError('SPECIALIZED_FATAL_NOT_RESOURCE_OR_CANCEL')
 if complete or calibration:
  marker=output/'native'/('group-complete.json' if complete else 'group-calibration.json');m=json.loads(marker.read_text())
  if m['world_size']!=len(devices):raise RuntimeError('SPECIALIZED_GROUP_GEOMETRY')
  for rank in range(len(devices)):
   if list(hashlib.sha256((output/'native'/f'rank-{rank}.json').read_bytes()).digest())!=m['rank_sha256'][rank]:raise RuntimeError('SPECIALIZED_GROUP_CHECKSUM')
 def decode(at):
  raw=b''.join(receipts[rank]['layers'].get(at,b'') for rank in range(len(devices)))[:1000*n]
  if len(raw)%n:raise RuntimeError('SPECIALIZED_STATE_SHAPE')
  if any(v>=len(match['labels']) for v in raw):raise RuntimeError('SPECIALIZED_STATE_ALPHABET')
  return [[match['labels'][v] for v in raw[i:i+n]] for i in range(0,len(raw),n)]
 current=decode(depth-1);previous=decode(depth-2) if depth>1 and sizes[-2]<1000 else []
 if (complete or calibration) and len(current)!=min(sizes[-1],1000):raise RuntimeError('SPECIALIZED_TERMINAL_PREFIX_SHAPE')
 if len(previous)!=(sizes[-2] if depth>1 and sizes[-2]<1000 else 0):raise RuntimeError('SPECIALIZED_PREVIOUS_SHAPE')
 state={'schema':2,'state_encoding':'signed_int64_vectors','current':current,'previous_small':previous,'current_sample_limit':1000};raw=json.dumps(state).encode();(output/'states.json').write_bytes(raw)
 layer_seconds=[max(t[i] for _,t,_ in traces) for i in range(depth-1)]
 report={'status':'COMPLETE' if complete else 'INCOMPLETE','reason':'GRAPH_EXHAUSTED' if complete else ('PROFILE_LAYER_LIMIT' if calibration else ('DEADLINE' if deadline else 'RESOURCE_STOP')),'backend':'SHARD_AB_'+mode,'graph_digest':graph.digest(),'devices':list(devices),'world':len(devices),'layer_sizes':sizes,'layer_seconds':layer_seconds,'bfs_seconds':sum(layer_seconds),'launch_wall_seconds':wall,'states_sha256':hashlib.sha256(raw).hexdigest(),'plan':{'capacity':capacity,'batch':batch,'history_layers':3,'state_bytes':1,'shard_target':shard_target,'peer_transport':e['MGBFS_TRANSPORT_BACKEND']},'native_rank_receipts':[str(p.relative_to(output)) for p in (output/'native').glob('rank-*.json')],'terminal_wire_committed':True,'sample_of_unprocessed_current':not(complete or calibration),'scope':'lossless LRX definition bridge; existing GPU key-first SHARD_AB with admitted peer transport; bounded RAM-only prefix snapshots; physical ranks limited to1/2/4/8 by reference owner'}
 (output/'report.json').write_text(json.dumps(report,indent=2));return report


def select_capacity(query, upper, *, minimum=32768):
    """Find largest admitted capacity. Exact queries correct affine prediction."""
    minimum = min(minimum, upper)
    cache = {}

    def probe(rows):
        if rows not in cache:
            records = query(rows)
            if not records:
                raise ValueError('missing native memory query records')
            for record in records:
                for key in ('required_bytes', 'reserve_bytes', 'free_after_nccl_warmup_bytes'):
                    if type(record.get(key)) is not int or record[key] < 0:
                        raise ValueError('invalid native memory query')
                margin = record.get('query_to_run_margin_bytes', 0)
                if type(margin) is not int or margin < 0:
                    raise ValueError('invalid query-to-run margin')
            cache[rows] = records
        return cache[rows]

    def fits(rows):
        return all(x['required_bytes'] + x['reserve_bytes'] +
                   x.get('query_to_run_margin_bytes', 0) <=
                   x['free_after_nccl_warmup_bytes'] for x in probe(rows))

    if not fits(minimum):
        raise ValueError('minimum fast batch does not fit warmed VRAM')
    if upper == minimum:
        return minimum, cache
    second = min(upper, minimum * 2)
    first_records, second_records = probe(minimum), probe(second)
    if len(first_records) != len(second_records):
        raise ValueError('memory query rank count changed')
    prediction = upper
    for first, other in zip(first_records, second_records):
        slope = (other['required_bytes'] - first['required_bytes']) / (second - minimum)
        if slope <= 0:
            raise ValueError('non-increasing native allocation plan')
        free = min(first['free_after_nccl_warmup_bytes'], other['free_after_nccl_warmup_bytes'])
        reserve = max(x['reserve_bytes'] + x.get('query_to_run_margin_bytes', 0)
                      for x in (first, other))
        room = free - reserve - first['required_bytes']
        prediction = min(prediction, minimum + int(room / slope))
    candidate = max(minimum, min(upper, prediction))
    low, high = minimum, upper + 1
    if fits(candidate):
        low = candidate
        if candidate == upper:
            return low, cache
        # Plans are almost affine: test the immediately adjacent row first.
        if not fits(candidate + 1):
            return low, cache
        low = candidate + 1
    else:
        high = candidate
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low, cache


def admit_specialized(graph,native,env,devices,root,*,capacity=None,mode='HASH',shard_target=1<<20,seconds=60):
 match=match_lrx(graph)
 if match is None:raise ValueError('SPECIALIZED_GRAPH_UNSUPPORTED')
 if not exact_specialized_supported(graph):raise ValueError('SPECIALIZED_LOSSLESS_KEY_DOMAIN_UNSUPPORTED')
 require_lossless_native(native,env)
 root=Path(root);root.mkdir(parents=True);upper=min(match['order'],(2**31-1)//3)
 def probe(rows):
  batch=min(rows,65536)
  return query(native,env,graph,root/('query-'+str(rows)),devices,rows,batch,mode,seconds,shard_target)
 if capacity is None:rows,records=select_capacity(probe,max(1,upper))
 else:
  rows=capacity;records={rows:probe(rows)}
  if not all(r['required_bytes']+r['reserve_bytes']+r['query_to_run_margin_bytes']<=r['free_after_nccl_warmup_bytes'] for r in records[rows]):raise ValueError('SPECIALIZED_REQUESTED_CAPACITY_EXCEEDS_ADMISSION')
 plan={'capacity':rows,'batch':min(rows,65536),'mode':mode,'shard_target':shard_target,'devices':list(devices),'history_layers':3,'state_bytes':1,'policy':'actual_native_nccl_warmed_allocation_query','rank_queries':records[rows],'maximum_hardware_capacity_proven':False}
 (root/'plan.json').write_text(json.dumps(plan,indent=2));return plan

"""GPU three-bank primitive stress; synthetic queues, not end-to-end BFS acceptance."""
import ctypes as C,json,sys,os,hashlib
from ctypes.util import find_library
from pathlib import Path
u=C.c_uint32;h=C.c_uint64;p=C.c_void_p
class Route(C.Structure):_fields_=[('hash',h),('parent',h),('source',u),('generator',u),('shard',u),('reserved',u)]
lib=C.CDLL(sys.argv[1]);cuda=C.CDLL(os.environ.get('MGBFS_CUDART_LIBRARY') or find_library('cudart') or 'libcudart.so.13')
cuda.cudaMalloc.argtypes=[C.POINTER(p),C.c_size_t];cuda.cudaMemcpy.argtypes=[p,p,C.c_size_t,C.c_int];cuda.cudaFree.argtypes=[p];cuda.cudaStreamCreateWithFlags.argtypes=[C.POINTER(p),u];cuda.cudaStreamDestroy.argtypes=[p]
def ck(v):
 if v:raise RuntimeError('CUDA_STATUS_'+str(v))
checks=[]
for device in (0,1):
 ck(cuda.cudaSetDevice(device))
 for codec,width,packed,bits in [(c,w,p,b) for c,w,p in [(1,4,True),(1,17,False),(8,4,False)] for b in (0,64)]:
  cap=64;stride=cap*3;sc=64;world=2;shards=2;q=64;boxes=world*shards;alloc=[];streams=[]
  def allocbuf(count,ty):
   ptr=p();ck(cuda.cudaMalloc(C.byref(ptr),max(1,count*C.sizeof(ty))));alloc.append(ptr);return ptr
  def upload(ptr,values,ty):
   host=(ty*len(values))(*values);ck(cuda.cudaMemcpy(ptr,host,C.sizeof(host),1))
  def read(ptr,count,ty):
   host=(ty*count)();ck(cuda.cudaMemcpy(host,ptr,C.sizeof(host),2));return list(host)
  ty=C.c_uint8 if codec==1 else C.c_int64
  arena=allocbuf(stride*width,ty);slots=allocbuf(sc,h);positions=allocbuf(stride,u);accepted=allocbuf(1,u);future=allocbuf(cap,u);error=allocbuf(1,u)
  local=allocbuf(boxes*q*width,ty);remote=allocbuf(boxes*q*width,ty);lm=allocbuf(boxes*q,Route);rm=allocbuf(boxes*q,Route);lc=allocbuf(boxes,u);rc=allocbuf(boxes,u)
  upload(arena,[0]*(stride*width),ty);upload(slots,[(1<<64)-1]*sc,h);upload(positions,[sc]*stride,u);upload(error,[0],u)
  for _ in range(shards):v=p();ck(cuda.cudaStreamCreateWithFlags(C.byref(v),1));streams.append(v)
  fn=getattr(lib,'mgbfs_generic_accept_rolling_'+('packed_u8' if packed else 'u8' if codec==1 else 'i64'));fn.argtypes=[u,p,p,p,p,p,p,u,u,u,u,u,p,u,p,u,u,u,p,p,p,p,p];fn.restype=C.c_int
  retire=lib.mgbfs_generic_retire_rows;retire.argtypes=[p,u,p,u,u,u,p,p];retire.restype=C.c_int
  seed=getattr(lib,'mgbfs_generic_reseed_rows_'+('u8' if codec==1 else 'i64'));seed.argtypes=[u,p,u,u,u,p,u,p,h,u,p,p];seed.restype=C.c_int
  banks={};history=[]
  def mix(x):
   x&=(1<<64)-1;x^=x>>30;x=x*0xbf58476d1ce4e5b9&((1<<64)-1);x^=x>>27;x=x*0x94d049bb133111eb&((1<<64)-1);return x^(x>>31)
  def key(x):
   h=0
   for e,v in enumerate(x):h=mix(h^(v&((1<<64)-1))^e)
   return mix(h) if bits else 0
  def state(i):return tuple(((i+e*7)%251) if codec==1 else (i-(1<<40)+e*7) for e in range(width))
  try:
   for turn in range(120):
    bank=turn%3;base=bank*cap
    if bank in banks:
     previous=banks.pop(bank);ck(retire(slots,sc,positions,stride,base,len(previous),error,None));ck(cuda.cudaDeviceSynchronize());assert read(error,1,u)==[0]
    retained=set().union(*map(set,banks.values())) if banks else set()
    # Duplicates occur after retired holes; two owner streams receive identical keys.
    fresh=[state((turn*10+i)%230) for i in range(10)];candidates=list(retained)[:20]+fresh+fresh
    expected=set(candidates)-retained
    ls=[0]*(boxes*q*width);rs=[0]*len(ls);ml=[Route() for _ in range(boxes*q)];mr=[Route() for _ in ml];cl=[0]*boxes;cr=[0]*boxes
    for shard in range(shards):
     queue=shard;cl[queue]=len(candidates)
     for row,x in enumerate(candidates):
      ml[queue*q+row]=Route(key(x),sum(int(v)<<(8*e) for e,v in enumerate(x[:8])),sum(int(v)<<(8*(e-8)) for e,v in enumerate(x[8:12],8)),sum(int(v)<<(8*(e-12)) for e,v in enumerate(x[12:16],12)),shard,0) if packed else Route(key(x),0,0,0,0,0)
      for e,v in enumerate(x):ls[queue*q*width+e*q+row]=v
    for ptr,vals,t in [(local,ls,ty),(remote,rs,ty),(lm,ml,Route),(rm,mr,Route),(lc,cl,u),(rc,cr,u),(accepted,[0],u)]:upload(ptr,vals,t)
    for shard in range(shards):ck(fn(width,local,remote,lm,rm,lc,rc,0,world,shard,shards,q,slots,sc,arena,stride,base,cap,accepted,future,positions,error,streams[shard]))
    ck(cuda.cudaDeviceSynchronize());assert read(error,1,u)==[0];n=read(accepted,1,u)[0];assert n==len(expected),(turn,n,len(expected))
    raw=read(arena,stride*width,ty);actual={tuple(raw[e*stride+base+i] for e in range(width)) for i in range(n)};assert actual==expected
    banks[bank]=list(actual);history.append(n)
    # Bounded periodic tombstone maintenance, after all owners retire.
    if bits==0 and turn%8==7:
     upload(slots,[(1<<64)-1]*sc,h)
     for b,values in banks.items():ck(seed(width,arena,stride,b*cap,len(values),slots,sc,positions,0,bits,error,None))
     ck(cuda.cudaDeviceSynchronize());assert read(error,1,u)==[0]
   # A stale position must reject a second retirement, never delete another key.
   b=119%3;ck(retire(slots,sc,positions,stride,b*cap,len(banks[b]),error,None));ck(cuda.cudaDeviceSynchronize());ck(retire(slots,sc,positions,stride,b*cap,len(banks[b]),error,None));ck(cuda.cudaDeviceSynchronize());assert read(error,1,u)[0]&64
   checks.append({'device':device,'codec':codec,'width':width,'rotations':120,'concurrent_owner_streams':2,'hash_bits':bits,'stale_retirement_rejected':True,'accepted_counts':history})
  finally:
   ck(cuda.cudaDeviceSynchronize())
   for v in streams:ck(cuda.cudaStreamDestroy(v))
   for v in alloc:ck(cuda.cudaFree(v))
receipt={'status':'VERIFIED_THREE_BANK_GPU_PRIMITIVES','library_sha256':hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),'checks':checks,'scope':'synthetic queue primitives on two actual RTX3060; runtime integration and complete graph oracle still required'}
Path(sys.argv[2]).write_text(json.dumps(receipt));print(json.dumps(receipt))

"""Exact origin regeneration under full hash collisions and malformed leases."""
import ctypes as C,json,os,sys
from pathlib import Path
from ctypes.util import find_library
u=C.c_uint32;h=C.c_uint64;p=C.c_void_p
class Route(C.Structure):_fields_=[('hash',h),('parent',h),('source',u),('generator',u),('shard',u),('reserved',u)]
lib=C.CDLL(sys.argv[1]);cuda=C.CDLL(find_library('cudart') or 'libcudart.so.13');cuda.cudaMalloc.argtypes=[C.POINTER(p),C.c_size_t];cuda.cudaMemcpy.argtypes=[p,p,C.c_size_t,C.c_int];cuda.cudaFree.argtypes=[p]
fn=lib.mgbfs_generic_accept_parent_origin_wide;fn.argtypes=[u]*6+[p]*4+[u,u,h]+[p]*6+[u]*5+[p,h,p,u,u,u]+[p]*5+[u,p,p];fn.restype=C.c_int
sort=lib.mgbfs_generic_sort_origins;sort.argtypes=[p]*4+[u]*5+[p]*5+[h,p,p];sort.restype=C.c_int
checks=[]
def ck(x):
 if x:raise RuntimeError('CUDA_STATUS_'+str(x))
for device in (0,1):
 ck(cuda.cudaSetDevice(device))
 for world in (1,2,3,8,128):
  for shards in (1,3,16):
   for width,bytes,ty in [(25,1,C.c_uint8),(257,8,C.c_int64)]:
    rank=world//2;shard=shards-1;q=2;capacity=256;alloc=[]
    def upload(values,ctype):
     host=(ctype*len(values))(*values);v=p();ck(cuda.cudaMalloc(C.byref(v),max(1,C.sizeof(host))));alloc.append(v);ck(cuda.cudaMemcpy(v,host,C.sizeof(host),1));return v
    def read(v,n,ctype):
     host=(ctype*n)();ck(cuda.cudaMemcpy(host,v,C.sizeof(host),2));return list(host)
    try:
     boxes=world*shards;lm=[Route() for _ in range(boxes*q)];rm=[Route() for _ in lm];lc=[0]*boxes;rc=[0]*boxes;cache=[0]*(world*q*width);expected=set();cursors=[r*100 for r in range(world)];frontiers=[1+(r%3) for r in range(world)];begin=1
     for source in range(world):
      queue=source*shards+shard;count=source%3;records=lm if source==rank else rm;(lc if source==rank else rc)[queue]=count
      for row in range(count):
       state=[0]*(width-2)+[(source+row)%7,(source*3+row)%11 if bytes==1 else -(1<<40)+(source*3+row)%11]
       expected.add(tuple(state));records[queue*q+row]=Route(0,cursors[source]+begin+row,source,0,shard,0)
       for e,v in enumerate(state):cache[source*q*width+e*q+row]=v
     # A foreign-shard pending reference reads the same parent cache.
     if shards>1:
      records=lm if rank==0 else rm;(lc if rank==0 else rc)[0]=1;records[0]=Route(0,cursors[0]+begin,0,0,0,0)
      foreign=[123]*width
      for e,v in enumerate(foreign):cache[e*q]=v
     meta=[upload(lm,Route),upload(rm,Route),upload(lc,u),upload(rc,u)];parents=upload(cache,ty);cursor=upload(cursors,h);frontier=upload(frontiers,u);perms=upload(list(range(width)),u);dummy=upload([0],u)
     tokens=[(1<<64)-1]*512
     if shards>1:tokens[0]=0x80000000
     slots=upload(tokens,h);arena=upload([0]*(capacity*width),ty);vc=upload([0],u);future=upload([0]*capacity,u);accepted=upload([0],u);error=upload([0],u)
     args=[bytes,0,width,width,1,1,perms,None,None,parents,q,q,begin,cursor,frontier,*meta,rank,world,shard,shards,q,slots,512,arena,capacity,0,capacity,vc,future,accepted,None,error,0,None,None]
     keys=upload([0]*(world*q),h);sorted_keys=upload([0]*(world*q),h);origins=upload([0]*(world*q),u);ordered=upload([0]*(world*q),u);scratch=upload([0]*(world*q*64+65536),C.c_uint8)
     ck(sort(*meta,rank,world,shard,shards,q,keys,sorted_keys,origins,ordered,scratch,world*q*64+65536,error,None));ck(cuda.cudaDeviceSynchronize())
     valid=sum((lc if source==rank else rc)[source*shards+shard] for source in range(world));assert read(sorted_keys,world*q,h)==[0]*valid+[(1<<64)-1]*(world*q-valid)
     args[-2]=ordered
     ck(fn(*args));ck(cuda.cudaDeviceSynchronize());assert read(error,1,u)==[0],(device,world,shards,width)
     n=read(vc,1,u)[0];raw=read(arena,capacity*width,ty);actual={tuple(raw[e*capacity+i] for e in range(width)) for i in range(n)};assert actual==expected,(device,world,shards,width,len(actual),len(expected));assert read(accepted,1,u)==[n]
     # Stale/foreign origin is a fatal lease error, never a partial successful layer.
     bad=[Route() for _ in lm];bad[rank*shards*q+shard*q]=Route(0,cursors[rank],rank,0,shard,0);badcounts=[0]*boxes;badcounts[rank*shards+shard]=1
     args[15]=upload(bad,Route);args[17]=upload(badcounts,u);ck(fn(*args));ck(cuda.cudaDeviceSynchronize());assert read(error,1,u)[0]&128
     checks.append({'device':device,'logical_sources':world,'shards':shards,'elements':width,'state_bytes':bytes,'forced_hash_collision_exact':True,'stale_origin_rejected':True,'cross_shard_pending_exact':shards>1})
    finally:
     for v in alloc:ck(cuda.cudaFree(v))
v={'status':'VERIFIED_SORTED_PARENT_FORCED_COLLISION_LEASES','checks':checks,'scope':'actual kernels on two GPUs with synthetic source layouts; no physical3/8/128rank claim'};Path(sys.argv[2]).write_text(json.dumps(v));print(json.dumps(v))

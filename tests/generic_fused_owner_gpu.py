import os
"""Exact fused owner with logical 1/2/3/8/128-source layouts on actual GPUs."""
import ctypes as C,json,sys
from pathlib import Path
u=C.c_uint32;h=C.c_uint64;p=C.c_void_p
class Route(C.Structure):_fields_=[('hash',h),('parent',h),('source',u),('generator',u),('shard',u),('reserved',u)]
lib=C.CDLL(sys.argv[1]);cuda=C.CDLL(os.environ.get('MGBFS_CUDART_LIBRARY','libcudart.so.13'))
cuda.cudaMalloc.argtypes=[C.POINTER(p),C.c_size_t];cuda.cudaMemcpy.argtypes=[p,p,C.c_size_t,C.c_int];cuda.cudaFree.argtypes=[p]
def ck(code):
 if code:raise RuntimeError('CUDA_STATUS_'+str(code))
checks=[]
for device in (0,1):
 ck(cuda.cudaSetDevice(device))
 for world in (1,2,3,8,128):
  for shards in (1,3,16):
   for codec,ty in ((1,C.c_uint8),(8,C.c_int64)):
    rank=world//2;shard=shards-1;q=2;width=4;boxes=world*shards;capacity=256;alloc=[]
    def upload(values,ctype):
     host=(ctype*len(values))(*values);ptr=p();ck(cuda.cudaMalloc(C.byref(ptr),max(1,C.sizeof(host))));alloc.append(ptr);ck(cuda.cudaMemcpy(ptr,host,C.sizeof(host),1));return ptr
    def read(ptr,count,ctype):
     host=(ctype*count)();ck(cuda.cudaMemcpy(host,ptr,C.sizeof(host),2));return list(host)
    try:
     local=[0]*(boxes*q*width);remote=[0]*len(local);lm=[Route() for _ in range(boxes*q)];rm=[Route() for _ in lm];lc=[0]*boxes;rc=[0]*boxes;expected=set()
     for source in range(world):
      queue=source*shards+shard;count=source%3
      values=local if source==rank else remote;counts=lc if source==rank else rc;records=lm if source==rank else rm;counts[queue]=count
      for row in range(count):
       state=[(source+row)%7,(source*3+row)%11,255,0] if codec==1 else [-((source+row)%7),(1<<40)+(source*3+row)%11,255,0]
       expected.add(tuple(state));records[queue*q+row]=Route(0,source,source,0,shard,0)
       for e,value in enumerate(state):values[queue*q*width+e*q+row]=value
     pointers=[upload(local,ty),upload(remote,ty),upload(lm,Route),upload(rm,Route),upload(lc,u),upload(rc,u)]
     slots=upload([(1<<64)-1]*512,h);visited=upload([0]*(capacity*width),ty);vc=upload([0],u);future=upload([0]*capacity,u);fc=upload([0],u);error=upload([0],u)
     fn=getattr(lib,'mgbfs_generic_accept_all_'+('u8' if codec==1 else 'i64'));fn.argtypes=[u,p,p,p,p,p,p,u,u,u,u,u,p,u,p,u,p,p,u,p,p,p];fn.restype=C.c_int
     ck(fn(width,*pointers,rank,world,shard,shards,q,slots,512,visited,capacity,vc,future,capacity,fc,error,None));ck(cuda.cudaDeviceSynchronize())
     assert read(error,1,u)==[0];count=read(vc,1,u)[0];assert count==len(expected) and read(fc,1,u)==[count]
     raw=read(visited,capacity*width,ty);actual={tuple(raw[e*capacity+i] for e in range(width)) for i in range(count)};assert actual==expected,(device,world,shards,codec)
     # Malformed count cannot authorize future completion, even with other empty sources.
     bad=[0]*boxes;bad[rank*shards+shard]=q+1;ck(cuda.cudaMemcpy(pointers[4],(u*boxes)(*bad),boxes*4,1));ck(cuda.cudaMemcpy(pointers[5],(u*boxes)(*([0]*boxes)),boxes*4,1))
     ck(fn(width,*pointers,rank,world,shard,shards,q,slots,512,visited,capacity,vc,future,capacity,fc,error,None));ck(cuda.cudaDeviceSynchronize());assert read(error,1,u)[0]&32
     checks.append({'device':device,'logical_sources':world,'shards':shards,'state_bytes':codec,'unique_states':count,'forced_hash_collision_exact':True,'bad_count_rejected':True})
    finally:
     for ptr in alloc:ck(cuda.cudaFree(ptr))
receipt={'status':'VERIFIED_FUSED_OWNER_LOGICAL_SOURCE_LAYOUTS','checks':checks,'scope':'actual kernels on two RTX3060 with synthetic logical source layouts; no 3/8/128-rank NCCL or hardware scaling claim'}
Path('/root/universal/grouped-owner-logical-verification.json').write_text(json.dumps(receipt));print(json.dumps(receipt))

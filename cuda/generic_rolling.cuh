// Three-bank exact history primitives. Included after IncomingAllAction.
static constexpr uint64_t RETIRED=EMPTY-1;
template<class Candidate,class State> __global__ void accept_rolling(Candidate action,uint64_t* slots,uint32_t slot_capacity,
 State* arena,uint32_t arena_stride,uint32_t base,uint32_t capacity,uint32_t* accepted,uint32_t* future,
 uint32_t* positions,uint32_t* error){
 for(uint32_t iteration=blockIdx.x*blockDim.x+threadIdx.x;iteration<action.count;iteration+=blockDim.x*gridDim.x){
  const uint32_t child=action.child(iteration);if(atomicAdd(error,0u))return;if(!action.valid(child,error))continue;
  const uint64_t hash=action.hash(child,0,64),prefix=hash&0xffffffff00000000ULL,pending=prefix|0x80000000ULL|child;
  bool done=false;
  while(!done){
   uint32_t first=slot_capacity,start=action.bucket(hash,slot_capacity);uint64_t available=EMPTY;bool duplicate=false;
   for(uint32_t probe=0;probe<slot_capacity;probe++){
    uint32_t slot=(start+probe)&(slot_capacity-1);cuda::atomic_ref<uint64_t,cuda::thread_scope_device> cell(slots[slot]);uint64_t observed=cell.load(cuda::memory_order_acquire);
    if(observed==RETIRED){if(first==slot_capacity){first=slot;available=RETIRED;}continue;}
    if(observed==EMPTY){if(first==slot_capacity){first=slot;available=EMPTY;}break;}
    if((observed&0xffffffff00000000ULL)!=prefix)continue;
    uint32_t ref=uint32_t(observed),other=ref&0x7fffffffU;bool transient=ref&0x80000000U;
    if((transient&&other>=action.origin_limit())||(!transient&&other>=arena_stride)){atomicOr(error,8u);done=true;break;}
    bool equal=true;for(uint32_t e=0;e<action.elements;e++){int64_t old=transient?action.value(other,e):arena[uint64_t(e)*arena_stride+other];if(old!=action.value(child,e)){equal=false;break;}}
    if(equal){duplicate=true;break;}
   }
   if(done||duplicate)break;
   if(first==slot_capacity){atomicOr(error,1u);break;}
   cuda::atomic_ref<uint64_t,cuda::thread_scope_device> cell(slots[first]);uint64_t observed=available;
   if(!cell.compare_exchange_strong(observed,pending,cuda::memory_order_acq_rel,cuda::memory_order_acquire))continue;
   // A failed claim restarts the complete duplicate search; no alternate tombstone shortcut.
   uint32_t position=atomicAdd(accepted,1u);if(position>=capacity){atomicOr(error,2u);break;}
   uint32_t row=base+position;for(uint32_t e=0;e<action.elements;e++)arena[uint64_t(e)*arena_stride+row]=action.value(child,e);
   positions[row]=first;future[position]=row;cell.store(prefix|row,cuda::memory_order_release);done=true;
  }
 }
}
__global__ void retire_rows(uint64_t* slots,uint32_t slot_capacity,const uint32_t* positions,uint32_t base,uint32_t count,uint32_t* error){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x){
  uint32_t row=base+i,slot=positions[row];if(slot>=slot_capacity){atomicOr(error,64u);continue;}
  cuda::atomic_ref<uint64_t,cuda::thread_scope_device> cell(slots[slot]);uint64_t observed=cell.load(cuda::memory_order_acquire);
  if(uint32_t(observed)!=row||observed==EMPTY||observed==RETIRED||!cell.compare_exchange_strong(observed,RETIRED,cuda::memory_order_acq_rel,cuda::memory_order_acquire))atomicOr(error,64u);
 }
}
template<class State> __global__ void reseed_rows(uint32_t elements,const State* arena,uint32_t stride,uint32_t base,uint32_t count,uint64_t* slots,uint32_t slot_capacity,uint32_t* positions,uint64_t seed,uint32_t bits,uint32_t* error){
 for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x){
  uint32_t row=base+i;uint64_t hash=seed;for(uint32_t e=0;e<elements;e++)hash=mix64(hash^uint64_t(arena[uint64_t(e)*stride+row])^uint64_t(e));hash=finish_hash(hash,bits);
  uint32_t slot=uint32_t((uint64_t(uint32_t(hash))*slot_capacity)>>32);bool done=false;
  for(uint32_t probe=0;probe<slot_capacity;probe++,slot=(slot+1)&(slot_capacity-1)){if(atomicCAS(reinterpret_cast<unsigned long long*>(slots+slot),EMPTY,(hash&0xffffffff00000000ULL)|row)==EMPTY){positions[row]=slot;done=true;break;}}
  if(!done)atomicOr(error,1u);
 }
}
extern "C" int mgbfs_generic_retire_rows(uint64_t* slots,uint32_t slot_capacity,const uint32_t* positions,uint32_t stride,uint32_t base,uint32_t count,uint32_t* error,void* stream){
 if(!slots||!positions||!error||!slot_capacity||(slot_capacity&(slot_capacity-1))||base>stride||count>stride-base)return int(cudaErrorInvalidValue);
 if(count)retire_rows<<<grid(count),256,0,static_cast<cudaStream_t>(stream)>>>(slots,slot_capacity,positions,base,count,error);return int(cudaGetLastError());
}
template<class State,bool Packed=false> int rolling_accept(uint32_t elements,const State* local,const State* remote,const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,
 uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint64_t* slots,uint32_t slot_capacity,State* arena,uint32_t stride,uint32_t base,uint32_t cap,uint32_t* accepted,uint32_t* future,uint32_t* positions,uint32_t* error,void* stream){
 if(!elements||(Packed&&elements>16)||!world||world>128||rank>=world||!shards||shards>4096||shard>=shards||!q||uint64_t(world)*shards*q>=0x7fffffffULL||!slot_capacity||(slot_capacity&(slot_capacity-1))||!stride||stride>=0x80000000U||base>stride||!cap||cap>stride-base||!local||!remote||!lm||!rm||!lc||!rc||!slots||!arena||!accepted||!future||!positions||!error)return int(cudaErrorInvalidValue);
 IncomingAllAction<State,Packed,true> action{elements,world*q,q,1,rank,world,shard,shards,local,remote,lm,rm,lc,rc};
 accept_rolling<<<grid(uint64_t(world)*q),256,0,static_cast<cudaStream_t>(stream)>>>(action,slots,slot_capacity,arena,stride,base,cap,accepted,future,positions,error);return int(cudaGetLastError());
}
#define ROLLING_EXPORT(NAME,STATE,PACKED) extern "C" int NAME(uint32_t e,const STATE* l,const STATE* r,const GenericRouteRecord* lm,const GenericRouteRecord* rm,const uint32_t* lc,const uint32_t* rc,uint32_t rank,uint32_t world,uint32_t shard,uint32_t shards,uint32_t q,uint64_t* slots,uint32_t sc,STATE* arena,uint32_t stride,uint32_t base,uint32_t cap,uint32_t* accepted,uint32_t* future,uint32_t* positions,uint32_t* error,void* stream){return rolling_accept<STATE,PACKED>(e,l,r,lm,rm,lc,rc,rank,world,shard,shards,q,slots,sc,arena,stride,base,cap,accepted,future,positions,error,stream);}
ROLLING_EXPORT(mgbfs_generic_accept_rolling_i64,int64_t,false)
ROLLING_EXPORT(mgbfs_generic_accept_rolling_u8,uint8_t,false)
ROLLING_EXPORT(mgbfs_generic_accept_rolling_packed_u8,uint8_t,true)
#define RESEED_EXPORT(NAME,STATE) extern "C" int NAME(uint32_t e,const STATE* arena,uint32_t stride,uint32_t base,uint32_t count,uint64_t* slots,uint32_t sc,uint32_t* positions,uint64_t seed,uint32_t bits,uint32_t* error,void* stream){if(!e||!arena||!slots||!positions||!error||!sc||(sc&(sc-1))||!stride||stride>=0x80000000U||base>stride||count>stride-base||bits>64)return int(cudaErrorInvalidValue);if(count)reseed_rows<<<grid(count),256,0,static_cast<cudaStream_t>(stream)>>>(e,arena,stride,base,count,slots,sc,positions,seed,bits,error);return int(cudaGetLastError());}
RESEED_EXPORT(mgbfs_generic_reseed_rows_u8,uint8_t)
RESEED_EXPORT(mgbfs_generic_reseed_rows_i64,int64_t)

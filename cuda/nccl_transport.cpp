#include "mgbfs_cuda.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <memory>
#include <climits>
#ifdef MGBFS_NCCL_LSA
#include <nccl_device.h>
#if NCCL_VERSION_CODE < 22900
#error "MGBFS_NCCL_LSA requires NCCL 2.29 or newer"
#endif
#endif
struct Comm {
  ncclComm_t value{};
  uint32_t rank{}, world{};
#ifdef MGBFS_NCCL_LSA
  ncclDevComm device{};
  ncclWindow_t window{};
  unsigned char* symmetric{};
  uint32_t candidate_capacity{}, state_stride{};
  size_t states_offset{};
  bool window_ready{},device_ready{};
#endif
  ~Comm(){
#ifdef MGBFS_NCCL_LSA
    // Normal teardown requires the caller to drain all LSA stream consumers.
    // After ncclCommAbort the process is terminal; avoid using an invalid comm.
    if(value && device_ready)ncclDevCommDestroy(value,&device);
    if(value && window_ready)ncclCommWindowDeregister(value,window);
    if(symmetric)ncclMemFree(symmetric);
#endif
    if(value)ncclCommDestroy(value);
  }
};
extern "C" int mgbfs_nccl_unique_id(void* out){if(!out)return 1;static_assert(sizeof(ncclUniqueId)==128);return ncclGetUniqueId(static_cast<ncclUniqueId*>(out))==ncclSuccess?0:2;}
extern "C" int mgbfs_nccl_create(uint32_t rank,uint32_t world,uint32_t device,const void* raw_id,void** out,char* error,size_t error_capacity){
  if(!out||!raw_id||!world||rank>=world)return 1;*out=nullptr;auto p=std::make_unique<Comm>();p->rank=rank;p->world=world;cudaError_t ce=cudaSetDevice(int(device));if(ce!=cudaSuccess){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",cudaGetErrorString(ce));return 2;}ncclUniqueId id;std::memcpy(&id,raw_id,sizeof(id));ncclResult_t e=ncclCommInitRank(&p->value,int(world),id,int(rank));if(e!=ncclSuccess){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",ncclGetErrorString(e));return 3;}*out=p.release();return 0;
}
extern "C" int mgbfs_nccl_send_recv(void* raw,const void* send,uint64_t send_bytes,uint32_t peer,void* recv,uint64_t recv_bytes,void* raw_stream){
  auto* p = static_cast<Comm*>(raw);
  if(!p || !p->value || (!send && send_bytes) || (!recv && recv_bytes)) return 1;
  auto s = static_cast<cudaStream_t>(raw_stream);
  if(ncclGroupStart() != ncclSuccess) return 2;
  int status = 0;
  if(ncclSend(send,size_t(send_bytes),ncclUint8,int(peer),p->value,s) != ncclSuccess) status = 3;
  else if(ncclRecv(recv,size_t(recv_bytes),ncclUint8,int(peer),p->value,s) != ncclSuccess) status = 4;
  // Close every successfully opened group, including the immediate-error path.
  // Preserve the first operation error if group cleanup also reports failure.
  const auto end = ncclGroupEnd();
  return status ? status : (end == ncclSuccess ? 0 : 5);
}
extern "C" int mgbfs_nccl_all_gather_u32(void* raw,const uint32_t* send,uint32_t* recv,void* raw_stream){auto*p=static_cast<Comm*>(raw);if(!p||!p->value||!send||!recv)return 1;return ncclAllGather(send,recv,1,ncclUint32,p->value,static_cast<cudaStream_t>(raw_stream))==ncclSuccess?0:2;}
extern "C" int mgbfs_nccl_all_reduce_max_u32(void* raw,const uint32_t* send,uint32_t* recv,void* raw_stream){auto*p=static_cast<Comm*>(raw);if(!p||!p->value||!send||!recv)return 1;return ncclAllReduce(send,recv,1,ncclUint32,ncclMax,p->value,static_cast<cudaStream_t>(raw_stream))==ncclSuccess?0:2;}
extern "C" void mgbfs_nccl_destroy(void* raw){delete static_cast<Comm*>(raw);}
extern "C" int mgbfs_nccl_abort(void* raw){
  auto* p = static_cast<Comm*>(raw);
  if(!p) return 1;
  if(!p->value) return 0;
  const auto value = p->value;
  p->value = nullptr; // Terminal even if NCCL reports an abort error.
  return ncclCommAbort(value) == ncclSuccess ? 0 : 2;
}
extern "C" int mgbfs_nccl_poll(void* raw){
  auto* p = static_cast<Comm*>(raw);
  if(!p || !p->value) return 1;
  ncclResult_t state = ncclSuccess;
  if(ncclCommGetAsyncError(p->value, &state) != ncclSuccess) return 2;
  return state == ncclSuccess ? 0 : 3;
}
extern "C" int mgbfs_nccl_scatter(void* raw,uint32_t source,const void* send,uint64_t send_capacity,const uint64_t* sizes,void* recv,uint64_t recv_bytes,uint64_t recv_capacity,void* stream) {
  auto* p = static_cast<Comm*>(raw);
  if(!p || !p->value || source >= p->world) return 1;
  if(p->rank == source) {
    if(!sizes) return 1;
    uint64_t total = 0;
    for(uint32_t rank=0; rank<p->world; ++rank) {
      if(sizes[rank] > send_capacity-total) return 1;
      total += sizes[rank];
    }
    if(total && !send) return 1;
  } else if(recv_bytes > recv_capacity || (recv_bytes && !recv)) return 1;
  if(ncclGroupStart() != ncclSuccess) return 2;
  int status = 0;
  auto s = static_cast<cudaStream_t>(stream);
  if(p->rank == source) {
    uint64_t offset = 0;
    for(uint32_t rank=0; rank<p->world; ++rank) {
      const void* ptr = offset ? static_cast<const char*>(send)+offset : send;
      if(rank != source && ncclSend(ptr,size_t(sizes[rank]),ncclUint8,int(rank),p->value,s) != ncclSuccess) {
        status = 3;
        break;
      }
      offset += sizes[rank];
    }
  } else if(ncclRecv(recv,size_t(recv_bytes),ncclUint8,int(source),p->value,s) != ncclSuccess) status = 4;
  const auto end = ncclGroupEnd();
  return status ? status : (end == ncclSuccess ? 0 : 5);
}

#ifdef MGBFS_NCCL_LSA
namespace {
constexpr size_t lsa_control_bytes=256;
constexpr unsigned lsa_copy_ctas=16,lsa_copy_threads=256;
// The symmetric slot has one writer for each field in each exchange round:
// [recv_count, local_fatal, global_fatal] followed by dense hash/state planes.
__global__ void lsa_publish_count(ncclDevComm dev,ncclWindow_t win,
    const uint32_t* counts,uint32_t logical_owner,uint32_t peer,uint32_t cap){
  ncclLsaBarrierSession<ncclCoopCta> barrier{
    ncclCoopCta(),dev,ncclTeamTagLsa(),0};
  barrier.sync(ncclCoopCta(),cuda::memory_order_acquire);
  if(threadIdx.x==0){
    auto* local=static_cast<uint32_t*>(ncclGetLsaPointer(win,0,dev.lsaRank));
    auto* remote=static_cast<uint32_t*>(ncclGetLsaPointer(win,0,peer));
    uint64_t total=0;
    for(unsigned owner=0;owner<unsigned(dev.nRanks);++owner)total+=counts[owner];
    // This one capacity covers both the sorted source slot and the peer's
    // symmetric receive slot. Do not read past the source when only the
    // outgoing partition itself fits.
    uint32_t bad=local[1]||local[2]||total>cap;
    local[1]=bad;
    remote[0]=bad?0:counts[logical_owner];
  }
  barrier.sync(ncclCoopCta(),cuda::memory_order_release);
}
__global__ void lsa_copy_exact(ncclDevComm dev,ncclWindow_t win,
    const uint4* hashes,const uint4* states,const uint32_t* counts,
    uint32_t logical_owner,uint32_t peer,uint32_t cap,uint32_t stride,
    size_t states_offset){
  ncclLsaBarrierSession<ncclCoopCta> barrier{
    ncclCoopCta(),dev,ncclTeamTagLsa(),blockIdx.x};
  barrier.sync(ncclCoopCta(),cuda::memory_order_acquire);
  auto* local=static_cast<uint32_t*>(ncclGetLsaPointer(win,0,dev.lsaRank));
  bool bad=false;
  for(unsigned rank=0;rank<unsigned(dev.nRanks);++rank){
    auto* control=static_cast<const uint32_t*>(ncclGetLsaPointer(win,0,rank));
    bad|=control[1]!=0;
  }
  if(blockIdx.x==0&&threadIdx.x==0&&bad)local[2]=1;
  if(!bad){
    uint64_t begin=0;
    for(unsigned owner=0;owner<logical_owner;++owner)begin+=counts[owner];
    const uint32_t rows=counts[logical_owner];
    auto* remote=static_cast<unsigned char*>(ncclGetLsaPointer(win,0,peer));
    auto* dest_hashes=reinterpret_cast<uint4*>(remote+lsa_control_bytes);
    auto* dest_states=reinterpret_cast<uint4*>(remote+states_offset);
    const uint64_t t=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const uint64_t step=uint64_t(gridDim.x)*blockDim.x;
    for(uint64_t i=t;i<rows;i+=step)dest_hashes[i]=hashes[begin+i];
    const uint64_t words=uint64_t(rows)*(stride/16);
    for(uint64_t i=t;i<words;i+=step)
      dest_states[i]=states[begin*(stride/16)+i];
  }
  barrier.sync(ncclCoopCta(),cuda::memory_order_release);
}
void lsa_error(char* error,size_t capacity,const char* where,ncclResult_t code){
  if(error&&capacity)std::snprintf(error,capacity,"%s: %s",where,ncclGetErrorString(code));
}
}
extern "C" int mgbfs_nccl_lsa_prepare(void* raw,uint32_t cap,uint32_t stride,
    char* error,size_t error_capacity){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->value||p->symmetric||p->window_ready||p->device_ready||!cap||cap>INT_MAX||
     !stride||(stride&15u)||!p->world||p->world>8)return 1;
  int version=0;
  if(ncclGetVersion(&version)!=ncclSuccess||version<22900)return 2;
  ncclCommProperties_t props=NCCL_COMM_PROPERTIES_INITIALIZER;
  auto result=ncclCommQueryProperties(p->value,&props);
  if(result!=ncclSuccess){lsa_error(error,error_capacity,"query_properties",result);return 3;}
  if(!props.deviceApiSupport||props.nLsaTeams!=1)return 4;
  const uint64_t hash_bytes=uint64_t(cap)*16;
  const uint64_t state_bytes=uint64_t(cap)*stride;
  const uint64_t state_offset=lsa_control_bytes+hash_bytes;
  if(state_offset>SIZE_MAX||state_bytes>SIZE_MAX-state_offset)return 1;
  p->states_offset=size_t(state_offset);
  p->candidate_capacity=cap;
  p->state_stride=stride;
  result=ncclMemAlloc(reinterpret_cast<void**>(&p->symmetric),
                      size_t(state_offset+state_bytes));
  if(result!=ncclSuccess){lsa_error(error,error_capacity,"mem_alloc",result);return 5;}
  if(cudaMemset(p->symmetric,0,lsa_control_bytes)!=cudaSuccess)return 6;
  return 0;
}
extern "C" int mgbfs_nccl_lsa_activate(void* raw,char* error,size_t error_capacity){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->value||!p->symmetric||p->window_ready||p->device_ready)return 1;
  auto result=ncclSuccess;
  const size_t slot_bytes=p->states_offset+
      size_t(p->candidate_capacity)*p->state_stride;
  result=ncclCommWindowRegister(p->value,p->symmetric,
      slot_bytes,&p->window,NCCL_WIN_COLL_SYMMETRIC);
  if(result!=ncclSuccess){lsa_error(error,error_capacity,"window_register",result);return 7;}
  p->window_ready=true;
  ncclDevCommRequirements reqs=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.lsaBarrierCount=lsa_copy_ctas;
  result=ncclDevCommCreate(p->value,&reqs,&p->device);
  if(result!=ncclSuccess){lsa_error(error,error_capacity,"device_comm_create",result);return 8;}
  p->device_ready=true;
  if(p->device.lsaSize!=int(p->world))return 9;
  return 0;
}
extern "C" int mgbfs_nccl_lsa_exchange(void* raw,const void* sorted_hashes,
    const void* packed_states,const uint32_t* owner_counts,
    uint32_t logical_owner,uint32_t peer,void* raw_stream){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->device_ready||!sorted_hashes||!packed_states||!owner_counts||
     logical_owner>=p->world||peer>=p->world||peer==p->rank)return 1;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  lsa_publish_count<<<1,32,0,stream>>>(p->device,p->window,owner_counts,
      logical_owner,peer,p->candidate_capacity);
  if(cudaGetLastError()!=cudaSuccess)return 2;
  lsa_copy_exact<<<lsa_copy_ctas,lsa_copy_threads,0,stream>>>(
      p->device,p->window,static_cast<const uint4*>(sorted_hashes),
      static_cast<const uint4*>(packed_states),owner_counts,
      logical_owner,peer,p->candidate_capacity,p->state_stride,p->states_offset);
  return cudaGetLastError()==cudaSuccess?0:3;
}
extern "C" int mgbfs_nccl_lsa_view(void* raw,const uint32_t** count,
    const uint32_t** fatal,const void** hashes,const void** states){
  auto* p=static_cast<Comm*>(raw);
  if(!p||!p->device_ready||!count||!fatal||!hashes||!states)return 1;
  *count=reinterpret_cast<const uint32_t*>(p->symmetric);
  *fatal=reinterpret_cast<const uint32_t*>(p->symmetric+8);
  *hashes=p->symmetric+lsa_control_bytes;
  *states=p->symmetric+p->states_offset;
  return 0;
}
#else
extern "C" int mgbfs_nccl_lsa_prepare(void*,uint32_t,uint32_t,char*,size_t){return 7;}
extern "C" int mgbfs_nccl_lsa_activate(void*,char*,size_t){return 7;}
extern "C" int mgbfs_nccl_lsa_exchange(void*,const void*,const void*,const uint32_t*,
    uint32_t,uint32_t,void*){return 7;}
extern "C" int mgbfs_nccl_lsa_view(void*,const uint32_t**,const uint32_t**,
    const void**,const void**){return 7;}
#endif

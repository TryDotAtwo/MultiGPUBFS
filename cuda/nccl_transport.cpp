#include "mgbfs_cuda.h"
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <memory>
struct Comm { ncclComm_t value{}; ~Comm(){if(value)ncclCommDestroy(value);} };
extern "C" int mgbfs_nccl_unique_id(void* out){if(!out)return 1;static_assert(sizeof(ncclUniqueId)==128);return ncclGetUniqueId(static_cast<ncclUniqueId*>(out))==ncclSuccess?0:2;}
extern "C" int mgbfs_nccl_create(uint32_t rank,uint32_t world,uint32_t device,const void* raw_id,void** out,char* error,size_t error_capacity){
  if(!out||!raw_id||!world||rank>=world)return 1;*out=nullptr;auto p=std::make_unique<Comm>();cudaError_t ce=cudaSetDevice(int(device));if(ce!=cudaSuccess){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",cudaGetErrorString(ce));return 2;}ncclUniqueId id;std::memcpy(&id,raw_id,sizeof(id));ncclResult_t e=ncclCommInitRank(&p->value,int(world),id,int(rank));if(e!=ncclSuccess){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",ncclGetErrorString(e));return 3;}*out=p.release();return 0;
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

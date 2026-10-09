#include "generic_sorted_native.h"
#include "generic_sorted_disjoint_carry_graph.cuh"
#include <new>
struct NativeCarryGraph {cudaGraph_t graph{};cudaGraphExec_t exec{};cudaStream_t stream{};int device{};};
extern "C" int mgbfs_generic_sorted_native_carry_shape(uint32_t page,uint32_t classes,uint64_t* control,uint64_t* handles){
 if(!control||!handles)return cudaErrorInvalidValue;*control=*handles=0;
 GenericSortedDisjointCarryShape shape{};auto e=generic_sorted_disjoint_carry_shape(page,classes,&shape);
 if(e==cudaSuccess){*control=generic_sorted_merge_align256(shape.control_bytes);*handles=generic_sorted_merge_align256(shape.handle_bytes);}return e;
}
extern "C" int mgbfs_generic_sorted_native_carry_create(const GenericSortedNativeCarry* p,void* stream,void** out){
 if(!out)return cudaErrorInvalidValue;*out=nullptr;
 if(!p||!p->pool||!p->control||!p->handles||(p->state_bytes!=1&&p->state_bytes!=8))return cudaErrorInvalidValue;
 auto* g=new(std::nothrow) NativeCarryGraph;if(!g)return cudaErrorMemoryAllocation;
 auto e=cudaGetDevice(&g->device);g->stream=(cudaStream_t)stream;
 auto* raw=(char*)p->control;GenericSortedCarryWorkspace w{};
 w.left=(GenericSortedHistoryRun*)raw;w.right=w.left+1;w.unique=(uint32_t*)(w.right+1);w.device_handles=(cudaGraphConditionalHandle*)p->handles;
 auto pool=*(const GenericSortedRunPool*)p->pool;
 if(e==cudaSuccess){
  if(p->state_bytes==1)e=generic_sorted_disjoint_carry_graph_create(&g->graph,pool,(GenericSortedRunTiers*)p->tiers,(GenericSortedRunCarry*)p->carry,(uint64_t*)p->hashes,(uint32_t*)p->rows,(const uint8_t*)p->arena,p->stride,p->width,(uint32_t*)p->error,p->classes,w);
  else e=generic_sorted_disjoint_carry_graph_create(&g->graph,pool,(GenericSortedRunTiers*)p->tiers,(GenericSortedRunCarry*)p->carry,(uint64_t*)p->hashes,(uint32_t*)p->rows,(const int64_t*)p->arena,p->stride,p->width,(uint32_t*)p->error,p->classes,w);
 }
 if(e==cudaSuccess)e=cudaGraphInstantiate(&g->exec,g->graph,nullptr,nullptr,0);
 if(e!=cudaSuccess){if(g->graph)cudaGraphDestroy(g->graph);delete g;return e;}*out=g;return cudaSuccess;
}
extern "C" int mgbfs_generic_sorted_native_carry_launch(void* handle,void* stream){
 if(!handle)return cudaErrorInvalidValue;auto* g=(NativeCarryGraph*)handle;
 // Fixed owner stream: scratch sharing cannot silently cross stream boundaries.
 if(g->stream!=(cudaStream_t)stream)return cudaErrorInvalidValue;
 return cudaGraphLaunch(g->exec,g->stream);
}
extern "C" int mgbfs_generic_sorted_native_carry_destroy(void* handle){
 if(!handle)return cudaSuccess;auto* g=(NativeCarryGraph*)handle;int prior;auto e=cudaGetDevice(&prior);if(e!=cudaSuccess)return e;
 e=cudaSetDevice(g->device);if(e!=cudaSuccess)return e;
 // Cold teardown joins all readers/writers before caller can free graph buffers.
 e=cudaStreamSynchronize(g->stream);
 auto a=cudaGraphExecDestroy(g->exec),b=cudaGraphDestroy(g->graph);delete g;
 auto restore=cudaSetDevice(prior);return e!=cudaSuccess?e:a!=cudaSuccess?a:b!=cudaSuccess?b:restore;
}

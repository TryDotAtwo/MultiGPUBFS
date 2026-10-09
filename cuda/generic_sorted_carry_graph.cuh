#pragma once
#include "generic_sorted_run_tiers.cuh"
#include "generic_sorted_run_merge.cuh"
inline cudaError_t generic_sorted_graph_add_node(cudaGraphNode_t* node,cudaGraph_t graph,
 const cudaGraphNode_t* dependencies,size_t count,cudaGraphNodeParams* parameters){
#if CUDART_VERSION >= 13000
 return cudaGraphAddNode(node,graph,dependencies,nullptr,count,parameters);
#else
 return cudaGraphAddNode(node,graph,dependencies,count,parameters);
#endif
}
// A reusable owner-stream graph dispatches one admitted size class per carry.
// Conditions, views, output pointers and commit counts stay on device.
__global__ void generic_sorted_carry_select(GenericSortedRunPool pool,GenericSortedRunTiers* tiers,
 GenericSortedRunCarry* carry,uint64_t* hashes,uint32_t* rows,uint32_t* error,
 cudaGraphConditionalHandle loop,const cudaGraphConditionalHandle* handles,uint32_t classes,
 GenericSortedHistoryRun* a,GenericSortedHistoryRun* b){
 if(blockIdx.x||threadIdx.x)return;
 for(uint32_t c=0;c<classes;++c)cudaGraphSetConditional(handles[c],0);
 generic_sorted_tier_prepare(pool,tiers,carry,hashes,rows,error);
 if(carry->stage!=SORTED_CARRY_MERGE){cudaGraphSetConditional(loop,0);return;}
 uint32_t cls=carry->ticket.size_class;
 if(cls>=classes){generic_sorted_tier_abort(pool,carry,error);atomicOr(error,32u);carry->stage=SORTED_CARRY_FAILED;cudaGraphSetConditional(loop,0);return;}
 *a=carry->ticket.left_view;*b=carry->ticket.right_view;cudaGraphSetConditional(handles[cls],1);
}
__global__ void generic_sorted_carry_scatter(GenericSortedRunCarry* carry,const uint64_t* hashes,
 const uint32_t* rows,const uint32_t* flags,const uint32_t* prefix,uint32_t capacity,uint32_t* count){
 if(!blockIdx.x&&!threadIdx.x)*count=prefix[capacity-1]+flags[capacity-1];
 auto ticket=carry->ticket;
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<capacity;i+=uint64_t(blockDim.x)*gridDim.x)
  if(flags[i]){ticket.destination_hashes[prefix[i]]=hashes[i];ticket.destination_rows[prefix[i]]=rows[i];}
}
__global__ void generic_sorted_carry_finish(GenericSortedRunPool pool,GenericSortedRunTiers* tiers,
 GenericSortedRunCarry* carry,const uint32_t* count,uint32_t* error,cudaGraphConditionalHandle loop){
 if(blockIdx.x||threadIdx.x)return;generic_sorted_tier_commit(pool,tiers,carry,*count,error);
 cudaGraphSetConditional(loop,carry->stage==SORTED_CARRY_NEXT&&!(*error));
}
struct GenericSortedCarryWorkspace {
 GenericSortedHistoryRun *left,*right;uint64_t* hashes;uint32_t *rows,*flags,*prefix,*count,*unique;
 void* scan_temporary;size_t scan_temporary_bytes;cudaGraphConditionalHandle* device_handles;
};
struct GenericSortedCarryGraphShape {
 uint32_t capacity;uint64_t hash_bytes,row_bytes,flag_bytes,prefix_bytes,control_bytes,handle_bytes;
 size_t scan_temporary_bytes;uint64_t aligned_workspace_bytes;
};
inline cudaError_t generic_sorted_carry_graph_shape(uint32_t page_entries,uint32_t classes,GenericSortedCarryGraphShape* shape){
 if(!shape)return cudaErrorInvalidValue;*shape={};
 if(!classes||classes>31||!page_entries||(uint64_t(page_entries)<<classes)>0x7fffffff)return cudaErrorInvalidValue;
 shape->capacity=page_entries<<classes;shape->hash_bytes=uint64_t(shape->capacity)*8;
 shape->row_bytes=shape->flag_bytes=shape->prefix_bytes=uint64_t(shape->capacity)*4;
 shape->control_bytes=2*sizeof(GenericSortedHistoryRun)+2*sizeof(uint32_t);
 shape->handle_bytes=uint64_t(classes)*sizeof(cudaGraphConditionalHandle);
 for(uint32_t c=0;c<classes;++c){size_t bytes=0;auto status=cub::DeviceScan::ExclusiveSum(nullptr,bytes,static_cast<const uint32_t*>(nullptr),static_cast<uint32_t*>(nullptr),page_entries<<(c+1));if(status!=cudaSuccess)return status;if(bytes>shape->scan_temporary_bytes)shape->scan_temporary_bytes=bytes;}
 shape->aligned_workspace_bytes=generic_sorted_merge_align256(shape->hash_bytes)+generic_sorted_merge_align256(shape->row_bytes)+generic_sorted_merge_align256(shape->flag_bytes)+generic_sorted_merge_align256(shape->prefix_bytes)+generic_sorted_merge_align256(shape->control_bytes)+generic_sorted_merge_align256(shape->handle_bytes)+generic_sorted_merge_align256(shape->scan_temporary_bytes);
 return cudaSuccess;
}
// Caller admits every workspace plane and the pool/state arena together, then
// retains all allocations until graph destruction. Construction is cold-path.
template<class State> cudaError_t generic_sorted_carry_graph_create(cudaGraph_t* out,
 GenericSortedRunPool pool,GenericSortedRunTiers* tiers,GenericSortedRunCarry* carry,
 uint64_t* hashes,uint32_t* rows,const State* arena,uint32_t stride,uint32_t width,
 uint32_t* error,uint32_t classes,GenericSortedCarryWorkspace w){
 GenericSortedCarryGraphShape shape{};auto admission=generic_sorted_carry_graph_shape(pool.page_entries,classes,&shape);if(admission!=cudaSuccess)return admission;
 if(!out||!tiers||!carry||!hashes||!rows||!arena||!stride||!width||!error||!pool.occupied||!pool.descriptors||!pool.regions||!w.left||!w.right||!w.hashes||!w.rows||!w.flags||!w.prefix||!w.count||!w.unique||!w.device_handles||!w.scan_temporary||w.scan_temporary_bytes<shape.scan_temporary_bytes)return cudaErrorInvalidValue;
 #define SORTED_GRAPH_TRY(x) do{auto status_=(x);if(status_!=cudaSuccess){cudaGraphDestroy(graph);return status_;}}while(0)
 cudaGraph_t graph;auto status=cudaGraphCreate(&graph,0);if(status!=cudaSuccess)return status;
 cudaGraphConditionalHandle loop;SORTED_GRAPH_TRY(cudaGraphConditionalHandleCreate(&loop,graph,1,cudaGraphCondAssignDefault));
 cudaGraphNodeParams lp{};lp.type=cudaGraphNodeTypeConditional;lp.conditional.handle=loop;lp.conditional.type=cudaGraphCondTypeWhile;lp.conditional.size=1;cudaGraphNode_t loopnode;SORTED_GRAPH_TRY(generic_sorted_graph_add_node(&loopnode,graph,nullptr,0,&lp));auto body=lp.conditional.phGraph_out[0];
 cudaGraphConditionalHandle handles[31];for(uint32_t c=0;c<classes;++c)SORTED_GRAPH_TRY(cudaGraphConditionalHandleCreate(&handles[c],graph,0,cudaGraphCondAssignDefault));
 SORTED_GRAPH_TRY(cudaMemcpy(w.device_handles,handles,classes*sizeof(handles[0]),cudaMemcpyHostToDevice));
 void* args[]={&pool,&tiers,&carry,&hashes,&rows,&error,&loop,&w.device_handles,&classes,&w.left,&w.right};
 cudaGraphNodeParams selector{};selector.type=cudaGraphNodeTypeKernel;selector.kernel.func=(void*)generic_sorted_carry_select;selector.kernel.gridDim=dim3(1);selector.kernel.blockDim=dim3(1);selector.kernel.kernelParams=args;
 cudaGraphNode_t previous;SORTED_GRAPH_TRY(generic_sorted_graph_add_node(&previous,body,nullptr,0,&selector));
 cudaStream_t stream;SORTED_GRAPH_TRY(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 for(uint32_t c=0;c<classes;++c){cudaGraphNodeParams ip{};ip.type=cudaGraphNodeTypeConditional;ip.conditional.handle=handles[c];ip.conditional.type=cudaGraphCondTypeIf;ip.conditional.size=1;cudaGraphNode_t node;
  status=generic_sorted_graph_add_node(&node,body,&previous,1,&ip);if(status!=cudaSuccess){cudaStreamDestroy(stream);cudaGraphDestroy(graph);return status;}previous=node;auto child=ip.conditional.phGraph_out[0];uint32_t capacity=pool.page_entries<<(c+1);
  status=cudaStreamBeginCaptureToGraph(stream,child,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal);if(status!=cudaSuccess){cudaStreamDestroy(stream);cudaGraphDestroy(graph);return status;}
  generic_sorted_run_merge<<<(capacity+255)/256,256,0,stream>>>(w.left,w.right,arena,stride,width,w.hashes,w.rows,capacity/2,capacity/2,capacity,w.count,error);
  generic_sorted_run_unique_flags<<<(capacity+255)/256,256,0,stream>>>(w.hashes,w.rows,w.count,capacity,arena,stride,width,w.flags,error);
  size_t bytes=w.scan_temporary_bytes;status=cub::DeviceScan::ExclusiveSum(w.scan_temporary,bytes,w.flags,w.prefix,capacity,stream);
  if(status==cudaSuccess){generic_sorted_carry_scatter<<<(capacity+255)/256,256,0,stream>>>(carry,w.hashes,w.rows,w.flags,w.prefix,capacity,w.unique);generic_sorted_carry_finish<<<1,1,0,stream>>>(pool,tiers,carry,w.unique,error,loop);}
  cudaGraph_t ended;auto endstatus=cudaStreamEndCapture(stream,&ended);if(status!=cudaSuccess||endstatus!=cudaSuccess){cudaStreamDestroy(stream);cudaGraphDestroy(graph);return status!=cudaSuccess?status:endstatus;}
 }
 cudaStreamDestroy(stream);*out=graph;return cudaSuccess;
 #undef SORTED_GRAPH_TRY
}

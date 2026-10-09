#pragma once
#include "generic_sorted_carry_graph.cuh"
#include "generic_sorted_disjoint_merge.cuh"
static __global__ void generic_sorted_disjoint_carry_select(GenericSortedRunPool pool,GenericSortedRunTiers* tiers,
 GenericSortedRunCarry* carry,uint64_t* hashes,uint32_t* rows,uint32_t* error,
 cudaGraphConditionalHandle loop,const cudaGraphConditionalHandle* handles,uint32_t classes,
 GenericSortedHistoryRun* a,GenericSortedHistoryRun* b){
 if(blockIdx.x||threadIdx.x)return;
 for(uint32_t c=0;c<classes;++c)cudaGraphSetConditional(handles[c],0);
 generic_sorted_tier_prepare(pool,tiers,carry,hashes,rows,error);
 if(carry->stage!=SORTED_CARRY_MERGE){
  // Same bounded second chance as the old two host graph launches, but only
  // on actual pressure. Input ownership is unchanged by failed allocation.
  bool retry=carry->stage==SORTED_CARRY_PRESSURE&&carry->valid&&!(*error)&&carry->retries_remaining;
  if(retry)--carry->retries_remaining;
  cudaGraphSetConditional(loop,retry);return;
 }
 uint32_t cls=carry->ticket.size_class;
 if(cls>=classes){generic_sorted_tier_abort(pool,carry,error);atomicOr(error,32u);carry->stage=SORTED_CARRY_FAILED;cudaGraphSetConditional(loop,0);return;}
 *a=carry->ticket.left_view;*b=carry->ticket.right_view;cudaGraphSetConditional(handles[cls],1);
}
struct GenericSortedDisjointCarryShape {uint32_t capacity;uint64_t control_bytes,handle_bytes,aligned_workspace_bytes;};
inline cudaError_t generic_sorted_disjoint_carry_shape(uint32_t page_entries,uint32_t classes,GenericSortedDisjointCarryShape* out){
 if(!out)return cudaErrorInvalidValue;*out={};if(!page_entries||!classes||classes>31||(uint64_t(page_entries)<<classes)>0x7fffffff)return cudaErrorInvalidValue;
 out->capacity=page_entries<<classes;out->control_bytes=2*sizeof(GenericSortedHistoryRun)+sizeof(uint32_t);out->handle_bytes=uint64_t(classes)*sizeof(cudaGraphConditionalHandle);
 out->aligned_workspace_bytes=generic_sorted_merge_align256(out->control_bytes)+generic_sorted_merge_align256(out->handle_bytes);return cudaSuccess;
}
// The owner must have completed exact origin dedup and checked all future tiers
// before publication. Duplicated roots fail transactionally, never silently.
template<class State> cudaError_t generic_sorted_disjoint_carry_graph_create(cudaGraph_t* out,
 GenericSortedRunPool pool,GenericSortedRunTiers* tiers,GenericSortedRunCarry* carry,
 uint64_t* hashes,uint32_t* rows,const State* arena,uint32_t stride,uint32_t width,
 uint32_t* error,uint32_t classes,GenericSortedCarryWorkspace w){
 GenericSortedDisjointCarryShape shape{};auto admission=generic_sorted_disjoint_carry_shape(pool.page_entries,classes,&shape);if(admission!=cudaSuccess)return admission;
 if(!out||!tiers||!carry||!hashes||!rows||!arena||!stride||!width||!error||!pool.occupied||!pool.descriptors||!pool.regions||!w.left||!w.right||!w.unique||!w.device_handles)return cudaErrorInvalidValue;
 #define SORTED_GRAPH_TRY(x) do{auto status_=(x);if(status_!=cudaSuccess){cudaGraphDestroy(graph);return status_;}}while(0)
 cudaGraph_t graph;auto status=cudaGraphCreate(&graph,0);if(status!=cudaSuccess)return status;
 cudaGraphConditionalHandle loop;SORTED_GRAPH_TRY(cudaGraphConditionalHandleCreate(&loop,graph,1,cudaGraphCondAssignDefault));
 cudaGraphNodeParams lp{};lp.type=cudaGraphNodeTypeConditional;lp.conditional.handle=loop;lp.conditional.type=cudaGraphCondTypeWhile;lp.conditional.size=1;cudaGraphNode_t loopnode;SORTED_GRAPH_TRY(generic_sorted_graph_add_node(&loopnode,graph,nullptr,0,&lp));auto body=lp.conditional.phGraph_out[0];
 cudaGraphConditionalHandle handles[31];for(uint32_t c=0;c<classes;++c)SORTED_GRAPH_TRY(cudaGraphConditionalHandleCreate(&handles[c],graph,0,cudaGraphCondAssignDefault));
 SORTED_GRAPH_TRY(cudaMemcpy(w.device_handles,handles,classes*sizeof(handles[0]),cudaMemcpyHostToDevice));
 void* args[]={&pool,&tiers,&carry,&hashes,&rows,&error,&loop,&w.device_handles,&classes,&w.left,&w.right};
 cudaGraphNodeParams selector{};selector.type=cudaGraphNodeTypeKernel;selector.kernel.func=(void*)generic_sorted_disjoint_carry_select;selector.kernel.gridDim=dim3(1);selector.kernel.blockDim=dim3(1);selector.kernel.kernelParams=args;
 cudaGraphNode_t previous;SORTED_GRAPH_TRY(generic_sorted_graph_add_node(&previous,body,nullptr,0,&selector));
 cudaStream_t stream;SORTED_GRAPH_TRY(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 for(uint32_t c=0;c<classes;++c){cudaGraphNodeParams ip{};ip.type=cudaGraphNodeTypeConditional;ip.conditional.handle=handles[c];ip.conditional.type=cudaGraphCondTypeIf;ip.conditional.size=1;cudaGraphNode_t node;
  status=generic_sorted_graph_add_node(&node,body,&previous,1,&ip);if(status!=cudaSuccess){cudaStreamDestroy(stream);cudaGraphDestroy(graph);return status;}previous=node;auto child=ip.conditional.phGraph_out[0];uint32_t capacity=pool.page_entries<<(c+1);
  status=cudaStreamBeginCaptureToGraph(stream,child,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal);if(status!=cudaSuccess){cudaStreamDestroy(stream);cudaGraphDestroy(graph);return status;}
  generic_sorted_disjoint_merge<<<(capacity+255)/256,256,0,stream>>>(carry,w.left,w.right,arena,stride,width,capacity,w.unique,error);
  generic_sorted_carry_finish<<<1,1,0,stream>>>(pool,tiers,carry,w.unique,error,loop);
  cudaGraph_t ended;auto endstatus=cudaStreamEndCapture(stream,&ended);if(status!=cudaSuccess||endstatus!=cudaSuccess){cudaStreamDestroy(stream);cudaGraphDestroy(graph);return status!=cudaSuccess?status:endstatus;}
 }
 cudaStreamDestroy(stream);*out=graph;return cudaSuccess;
 #undef SORTED_GRAPH_TRY
}

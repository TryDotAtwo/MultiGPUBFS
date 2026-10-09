#include "generic_sorted_run_pool.cuh"
__device__ uint32_t failures_left,allocation_attempts;
__device__ inline GenericSortedRunAcquire inject_allocate(GenericSortedRunPool pool,uint32_t cls,uint32_t begin,GenericSortedRunToken* token,uint32_t* error){
 atomicAdd(&allocation_attempts,1u);
 if(failures_left){--failures_left;return SORTED_RUN_PRESSURE;}
 return generic_sorted_run_allocate(pool,cls,begin,token,error);
}
#define generic_sorted_run_allocate inject_allocate
#include "generic_sorted_disjoint_carry_graph.cuh"
#undef generic_sorted_run_allocate
#include <cstdio>
#include <cstdlib>
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>struct Dev{T* p;Dev(size_t n){CK(cudaMalloc(&p,n*sizeof(T)));}~Dev(){cudaFree(p);}T get(){T x;CK(cudaMemcpy(&x,p,sizeof(T),cudaMemcpyDeviceToHost));return x;}};
__global__ void seed(GenericSortedRunPool pool,GenericSortedRunCarry* c,uint64_t* hashes,uint32_t* rows,uint32_t row,uint32_t* error){
 if(threadIdx.x||blockIdx.x)return;GenericSortedRunToken t{};if(generic_sorted_run_allocate(pool,0,0,&t,error)!=SORTED_RUN_ACQUIRED){atomicOr(error,2u);return;}
 hashes[t.slot]=0;rows[t.slot]=row;if(!generic_sorted_run_publish(pool,t,1,error))return;c->token=t;c->valid=1;c->stage=SORTED_CARRY_NEXT;c->retries_remaining=1;
}
__global__ void cleanup(GenericSortedRunPool p,GenericSortedRunTiers* t,GenericSortedRunCarry* c,uint32_t* error){if(threadIdx.x||blockIdx.x)return;if(c->valid){generic_sorted_run_release(p,c->token,true,error);c->valid=0;}for(uint32_t k=0;k<32;++k)if(t->present&(1u<<k))generic_sorted_run_release(p,t->tokens[k],true,error);t->present=0;}
void fixture(int device,uint32_t injected){
 Dev<unsigned long long> bits(1);Dev<GenericSortedRunDescriptor> descriptors(64);Dev<uint64_t> hashes(64);Dev<uint32_t> rows(64),error(1),unique(1);Dev<GenericSortedRunTiers> tiers(1);Dev<GenericSortedRunCarry> carry(1);Dev<GenericSortedHistoryRun> left(1),right(1);Dev<uint8_t> arena(4);Dev<cudaGraphConditionalHandle> handles(2);
 CK(cudaMemset(bits.p,0,8));CK(cudaMemset(descriptors.p,0,64*sizeof(GenericSortedRunDescriptor)));CK(cudaMemset(tiers.p,0,sizeof(GenericSortedRunTiers)));CK(cudaMemset(carry.p,0,sizeof(GenericSortedRunCarry)));CK(cudaMemset(error.p,0,4));uint8_t states[]={1,2,3,4};CK(cudaMemcpy(arena.p,states,4,cudaMemcpyHostToDevice));
 uint32_t zero=0;CK(cudaMemcpyToSymbol(failures_left,&zero,4));CK(cudaMemcpyToSymbol(allocation_attempts,&zero,4));GenericSortedRunPool pool{bits.p,descriptors.p,1,1};GenericSortedCarryWorkspace w{left.p,right.p,nullptr,nullptr,nullptr,nullptr,nullptr,unique.p,nullptr,0,handles.p};cudaGraph_t graph;CK(generic_sorted_disjoint_carry_graph_create(&graph,pool,tiers.p,carry.p,hashes.p,rows.p,arena.p,2,2,error.p,2,w));cudaGraphExec_t exec;CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 seed<<<1,1>>>(pool,carry.p,hashes.p,rows.p,0,error.p);CK(cudaGraphLaunch(exec,0));CK(cudaDeviceSynchronize());auto initial=tiers.get();if(initial.present!=1||carry.get().valid)exit(3);
 seed<<<1,1>>>(pool,carry.p,hashes.p,rows.p,1,error.p);CK(cudaDeviceSynchronize());auto original_bits=bits.get();auto original_carry=carry.get();CK(cudaMemcpyToSymbol(failures_left,&injected,4));CK(cudaMemcpyToSymbol(allocation_attempts,&zero,4));CK(cudaGraphLaunch(exec,0));CK(cudaDeviceSynchronize());uint32_t attempts;CK(cudaMemcpyFromSymbol(&attempts,allocation_attempts,4));auto result=carry.get();auto roots=tiers.get();if(error.get())exit(4);
 if(injected<2){if(result.valid||roots.present!=2||attempts!=1+injected||result.retries_remaining!=1-injected)exit(5);auto token=roots.tokens[1];GenericSortedRunDescriptor d;CK(cudaMemcpy(&d,descriptors.p+token.slot,sizeof(d),cudaMemcpyDeviceToHost));uint32_t rr[2];CK(cudaMemcpy(rr,rows.p+token.slot,8,cudaMemcpyDeviceToHost));if(d.count!=2||rr[0]!=0||rr[1]!=1)exit(6);}
 else{if(!result.valid||result.stage!=SORTED_CARRY_PRESSURE||result.retries_remaining||attempts!=2||bits.get()!=original_bits||roots.present!=initial.present||roots.tokens[0].slot!=initial.tokens[0].slot||result.token.slot!=original_carry.token.slot)exit(7);}
 cleanup<<<1,1>>>(pool,tiers.p,carry.p,error.p);CK(cudaDeviceSynchronize());if(bits.get()||error.get())exit(8);CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));printf("BOUNDED_PRESSURE_RETRY_PASS device=%d injected=%u attempts=%u\n",device,injected,attempts);
}
int main(){int n;CK(cudaGetDeviceCount(&n));if(n<2)return 9;for(int d=0;d<2;++d){CK(cudaSetDevice(d));for(uint32_t f=0;f<3;++f)fixture(d,f);}}

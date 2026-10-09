#include "generic_sorted_run_tiers.cuh"
#include "generic_sorted_carry_graph.cuh"
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#define CK(x) do{auto status_=(x);if(status_!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(status_));exit(2);}}while(0)
template<class T> struct Dev{T* p;Dev(size_t n){CK(cudaMalloc(&p,(n?n:1)*sizeof(T)));}~Dev(){cudaFree(p);}void put(const std::vector<T>& v){if(v.size())CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T> get(size_t n){std::vector<T> v(n);if(n)CK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}};
__global__ void root(GenericSortedRunPool pool,GenericSortedRunCarry* carry,uint64_t* hashes,uint32_t* rows,const uint64_t* in_hash,const uint32_t* in_row,uint32_t cls,uint32_t n,uint32_t* error){
 if(blockIdx.x||threadIdx.x)return;GenericSortedRunToken token{};
 if(generic_sorted_run_allocate(pool,cls,0,&token,error)!=SORTED_RUN_ACQUIRED){atomicOr(error,1024u);return;}
 auto offset=uint64_t(token.slot)*pool.page_entries;for(uint32_t i=0;i<n;++i){hashes[offset+i]=in_hash[i];rows[offset+i]=in_row[i];}
 if(!generic_sorted_run_publish(pool,token,n,error))return;carry->token=token;carry->valid=1;carry->stage=SORTED_CARRY_NEXT;
}
__global__ void prepare(GenericSortedRunPool pool,GenericSortedRunTiers* tiers,GenericSortedRunCarry* carry,uint64_t* hashes,uint32_t* rows,uint32_t* error){if(!blockIdx.x&&!threadIdx.x)generic_sorted_tier_prepare(pool,tiers,carry,hashes,rows,error);}
__global__ void views(GenericSortedRunCarry* carry,GenericSortedHistoryRun* a,GenericSortedHistoryRun* b){if(!blockIdx.x&&!threadIdx.x){*a=carry->ticket.left_view;*b=carry->ticket.right_view;}}
__global__ void commit(GenericSortedRunPool pool,GenericSortedRunTiers* tiers,GenericSortedRunCarry* carry,const uint32_t* count,uint32_t* error){if(!blockIdx.x&&!threadIdx.x)generic_sorted_tier_commit(pool,tiers,carry,*count,error);}
__global__ void cleanup(GenericSortedRunPool pool,GenericSortedRunTiers* tiers,uint32_t* error){if(!blockIdx.x&&!threadIdx.x){for(uint32_t c=0;c<32;++c)if(tiers->present&(1u<<c))generic_sorted_run_release(pool,tiers->tokens[c],true,error);tiers->present=0;}}
__global__ void trim_contract(GenericSortedRunPool pool,uint32_t* error,uint32_t* failed){
 if(blockIdx.x||threadIdx.x)return;GenericSortedRunToken large{},reused{};
 if(generic_sorted_run_allocate(pool,9,0,&large,error)!=SORTED_RUN_ACQUIRED){++*failed;return;}
 if(!generic_sorted_run_trim(pool,large,3,error)||pool.descriptors[large.slot].size_class!=0)++*failed;
 if(!generic_sorted_run_publish(pool,large,3,error))++*failed;
 if(generic_sorted_run_allocate(pool,8,0,&reused,error)!=SORTED_RUN_ACQUIRED)++*failed;
 if(reused.slot==large.slot)++*failed;
 generic_sorted_run_release(pool,large,true,error);
 if(!generic_sorted_run_publish(pool,reused,1,error)||!generic_sorted_run_read_acquire(pool,reused,error))++*failed;
 generic_sorted_run_release(pool,reused,true,error);generic_sorted_run_release(pool,reused,false,error);
 for(uint32_t i=0;i<pool.regions;++i)if(pool.occupied[i])++*failed;
 // Published writers cannot trim, and the failed attempt preserves their allocation.
 if(generic_sorted_run_allocate(pool,7,0,&large,error)!=SORTED_RUN_ACQUIRED||!generic_sorted_run_publish(pool,large,513,error))++*failed;
 if(generic_sorted_run_trim(pool,large,1,error)||!*error||pool.descriptors[large.slot].size_class!=7)++*failed;*error=0;
 generic_sorted_run_release(pool,large,true,error);
 for(uint32_t i=0;i<pool.regions;++i)if(pool.occupied[i])++*failed;
}
template<class State> void fixture(int device,uint32_t width,uint32_t mode){
 constexpr uint32_t stride=4096,regions=32,page=8;GenericSortedRunPoolShape shape{};if(!generic_sorted_run_pool_shape(regions,page,&shape))exit(6);
 Dev<unsigned long long> bitmap(regions);Dev<GenericSortedRunDescriptor> desc(regions*64);Dev<uint64_t> hashes(shape.entries);Dev<uint32_t> rows(shape.entries),error(1),failed(1);Dev<GenericSortedRunTiers> tiers(1);Dev<GenericSortedRunCarry> carry(1);
 bitmap.put(std::vector<unsigned long long>(regions));desc.put(std::vector<GenericSortedRunDescriptor>(regions*64));tiers.put({{}});carry.put({{}});error.put({0});failed.put({0});GenericSortedRunPool pool{bitmap.p,desc.p,regions,page};
 trim_contract<<<1,1>>>(pool,error.p,failed.p);CK(cudaDeviceSynchronize());if(error.get(1)[0]||failed.get(1)[0])exit(7);
 std::vector<State> arena(uint64_t(stride)*width);for(uint32_t row=0;row<stride;++row)for(uint32_t c=0;c<width;++c)arena[uint64_t(c)*stride+row]=State((row%997)*(c+1)+c*17);Dev<State> states(arena.size());states.put(arena);
 auto hash=[&](uint32_t row){uint64_t h=1469598103934665603ull;for(uint32_t c=0;c<width;++c){h^=uint64_t(arena[uint64_t(c)*stride+row]);h*=1099511628211ull;}return mode==0?~0ull:mode==1?h&7ull:h;};
 auto compare=[&](uint32_t a,uint32_t b){auto ah=hash(a),bh=hash(b);if(ah!=bh)return ah<bh?-1:1;for(uint32_t c=0;c<width;++c){auto av=arena[uint64_t(c)*stride+a],bv=arena[uint64_t(c)*stride+b];if(av!=bv)return av<bv?-1:1;}return 0;};
 std::vector<uint32_t> expected;
 constexpr uint32_t classes=10,maxcapacity=page<<classes;Dev<GenericSortedHistoryRun> va(1),vb(1);Dev<uint64_t> mh(maxcapacity);Dev<uint32_t> mr(maxcapacity),flags(maxcapacity),prefix(maxcapacity),count(1),unique(1);Dev<cudaGraphConditionalHandle> handles(classes);
 GenericSortedCarryGraphShape graphshape{};CK(generic_sorted_carry_graph_shape(page,classes,&graphshape));if(graphshape.capacity!=maxcapacity||graphshape.hash_bytes!=uint64_t(maxcapacity)*8||graphshape.handle_bytes!=classes*sizeof(cudaGraphConditionalHandle))exit(17);size_t bytes=graphshape.scan_temporary_bytes;Dev<uint8_t> temporary(bytes);GenericSortedCarryWorkspace workspace{va.p,vb.p,mh.p,mr.p,flags.p,prefix.p,count.p,unique.p,temporary.p,bytes,handles.p};cudaGraph_t graph;auto undersized=workspace;undersized.scan_temporary_bytes=0;if(generic_sorted_carry_graph_create(&graph,pool,tiers.p,carry.p,hashes.p,rows.p,states.p,stride,width,error.p,classes,undersized)!=cudaErrorInvalidValue)exit(18);CK(generic_sorted_carry_graph_create(&graph,pool,tiers.p,carry.p,hashes.p,rows.p,states.p,stride,width,error.p,classes,workspace));cudaGraphExec_t exec;CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 // Five unequal overlap patterns trigger repeated carries and retained tiers.
 for(uint32_t batch=0;batch<5;++batch){std::vector<uint32_t> input;for(uint32_t i=0;i<257;++i)input.push_back((i*3+batch*113)%stride);std::sort(input.begin(),input.end(),[&](auto a,auto b){return compare(a,b)<0;});input.erase(std::unique(input.begin(),input.end(),[&](auto a,auto b){return compare(a,b)==0;}),input.end());
  expected.insert(expected.end(),input.begin(),input.end());std::vector<uint64_t> ih;for(auto row:input)ih.push_back(hash(row));Dev<uint64_t> ihash(ih.size());ihash.put(ih);Dev<uint32_t> irows(input.size());irows.put(input);uint32_t cls=0;while((uint64_t(page)<<cls)<input.size())++cls;
  root<<<1,1>>>(pool,carry.p,hashes.p,rows.p,ihash.p,irows.p,cls,input.size(),error.p);
  CK(cudaGraphLaunch(exec,0));CK(cudaDeviceSynchronize());if(error.get(1)[0])exit(9);
  if(carry.get(1)[0].valid)exit(10);
 }
 std::sort(expected.begin(),expected.end(),[&](auto a,auto b){return compare(a,b)<0;});expected.erase(std::unique(expected.begin(),expected.end(),[&](auto a,auto b){return compare(a,b)==0;}),expected.end());auto ht=tiers.get(1)[0];auto descriptors=desc.get(regions*64);auto allrows=rows.get(shape.entries);auto allhashes=hashes.get(shape.entries);std::vector<uint32_t> actual;
 for(uint32_t cls=0;cls<32;++cls)if(ht.present&(1u<<cls)){auto token=ht.tokens[cls];auto d=descriptors[token.slot];for(uint32_t i=0;i<d.count;++i){auto offset=uint64_t(token.slot)*page+i;auto row=allrows[offset];if(row>=stride||allhashes[offset]!=hash(row)||(i&&compare(allrows[offset-1],row)>=0))exit(11);actual.push_back(row);}}
 std::sort(actual.begin(),actual.end(),[&](auto a,auto b){return compare(a,b)<0;});actual.erase(std::unique(actual.begin(),actual.end(),[&](auto a,auto b){return compare(a,b)==0;}),actual.end());if(actual.size()!=expected.size())exit(12);for(size_t i=0;i<actual.size();++i)if(compare(actual[i],expected[i]))exit(13);
 CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));cleanup<<<1,1>>>(pool,tiers.p,error.p);CK(cudaDeviceSynchronize());if(error.get(1)[0])exit(14);for(auto word:bitmap.get(regions))if(word)exit(15);printf("SORTED_TIER_GRAPH_PAYLOAD_PASS device=%d bytes=%zu width=%u mode=%u unique=%zu\n",device,sizeof(State),width,mode,actual.size());
}
int main(){int count=0;CK(cudaGetDeviceCount(&count));if(count<2)return 5;for(int d=0;d<2;++d){CK(cudaSetDevice(d));for(uint32_t width:{2u,25u,129u})for(uint32_t mode=0;mode<3;++mode){fixture<uint8_t>(d,width,mode);fixture<int64_t>(d,width,mode);}}}

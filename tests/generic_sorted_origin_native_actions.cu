#include "generic_parent_action.cuh"
#include "generic_sorted_origin_exact.cuh"
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>struct Device{T* p;explicit Device(size_t n){CUDA_CHECK(cudaMalloc(&p,(n?n:1)*sizeof(T)));}~Device(){cudaFree(p);}void put(const std::vector<T>& v){if(v.size())CUDA_CHECK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(size_t n){std::vector<T> v(n);if(n)CUDA_CHECK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}};
template<class A>static void oracle(A action,uint32_t count,const std::vector<uint32_t>& physical,
 const std::vector<int64_t>& values,const std::vector<uint8_t>& active,const char* label){
 auto cmp=[&](uint32_t a,uint32_t b){for(uint32_t c=0;c<action.elements;++c){auto av=values[uint64_t(a)*action.elements+c],bv=values[uint64_t(b)*action.elements+c];if(av<bv)return -1;if(av>bv)return 1;}return 0;};
 std::vector<uint32_t> sorted_expected,unique;
 for(uint32_t i=0;i<count;++i)if(active[i])sorted_expected.push_back(i);
 std::sort(sorted_expected.begin(),sorted_expected.end(),[&](auto a,auto b){int order=cmp(a,b);return order<0||(!order&&a<b);});
 uint32_t prior=0xffffffffu;for(auto i:sorted_expected){if(prior==0xffffffffu||cmp(prior,i)!=0)unique.push_back(physical[i]);prior=i;}
 Device<uint64_t> hashes(count),out_hashes(count);
 Device<uint32_t> origins(count),sorted(count),flags(count),prefix(count),out(count),out_count(1),error(1);error.put({0});
 GenericSortedOriginShape shape{};CUDA_CHECK(generic_sorted_origin_shape(action,count,&shape));Device<uint8_t> temporary(shape.shared_temporary_bytes);
 CUDA_CHECK(generic_sorted_origin_exact(action,count,0,64,hashes.p,origins.p,sorted.p,flags.p,prefix.p,out_hashes.p,out.p,out_count.p,temporary.p,shape.shared_temporary_bytes,error.p));
 CUDA_CHECK(cudaDeviceSynchronize());
 if(error.get(1)[0]||out_count.get(1)[0]!=unique.size()||out.get(unique.size())!=unique){fprintf(stderr,"native action oracle failed %s error%u\n",label,error.get(1)[0]);exit(3);}
 auto actual_sorted=sorted.get(count);for(size_t i=0;i<sorted_expected.size();++i)if(actual_sorted[i]!=sorted_expected[i])exit(3);
 for(auto key:out_hashes.get(unique.size()))if(key!=~uint64_t(0))exit(3);
 // Reject invalid hash configuration before launching the action.
 if(generic_sorted_origin_exact(action,count,0,65,hashes.p,origins.p,sorted.p,flags.p,prefix.p,out_hashes.p,out.p,out_count.p,temporary.p,shape.shared_temporary_bytes,error.p)!=cudaErrorInvalidValue)exit(4);
 printf("NATIVE_ACTION_EXACT_SORT_PASS %s unique=%zu\n",label,unique.size());
}
template<class State,bool Packed>static void incoming(uint32_t width){
 constexpr uint32_t world=2,shards=3,shard=1,q=16,queues=world*shards,count=world*q;
 std::vector<State> states(uint64_t(queues)*width*q);std::vector<GenericRouteRecord> metadata(queues*q);
 std::vector<uint32_t> counts(queues,0),physical(count);std::vector<int64_t> expected(uint64_t(count)*width);std::vector<uint8_t> active(count);
 for(uint32_t source=0;source<world;++source){uint32_t queue=source*shards+shard;counts[queue]=source?9:11;
  for(uint32_t row=0;row<q;++row){uint32_t child=queue*q+row,logical=source*q+row;physical[logical]=child;active[logical]=row<counts[queue];auto& record=metadata[child];record.hash=~uint64_t(0);
   for(uint32_t c=0;c<width;++c){uint8_t byte=uint8_t((row/2+source+c)%13);State value=State(byte);states[uint64_t(queue)*width*q+uint64_t(c)*q+row]=value;expected[uint64_t(logical)*width+c]=int64_t(value);
    if constexpr(Packed){uint64_t shift=8*(c%8);if(c<8)record.parent|=uint64_t(byte)<<shift;else if(c<16){if(c<12)record.source|=uint32_t(byte)<<(8*(c%4));else record.generator|=uint32_t(byte)<<(8*(c%4));}else{if(c<20)record.shard|=uint32_t(byte)<<(8*(c%4));else record.reserved|=uint32_t(byte)<<(8*(c%4));}}
   }
  }
 }
 Device<State> ds(states.size());ds.put(states);Device<GenericRouteRecord> dm(metadata.size());dm.put(metadata);Device<uint32_t> dc(counts.size());dc.put(counts);
 IncomingAllAction<State,Packed,true> action{width,count,q,1,0,world,shard,shards,ds.p,ds.p,dm.p,dm.p,dc.p,dc.p,nullptr};
 oracle(action,count,physical,expected,active,Packed?"packed24":sizeof(State)==1?"incoming_byte":"incoming_int64");
}
template<class State>static void parent_origin(uint32_t width,bool matrix){
 constexpr uint32_t world=2,shards=3,shard=1,q=16,parents=8,count=world*q;
 std::vector<State> states(uint64_t(world)*parents*width);std::vector<GenericRouteRecord> metadata(world*shards*q);
 std::vector<uint32_t> counts(world*shards,0),frontiers(world,parents),permutations(2*width),physical(count);
 std::vector<uint64_t> cursors{100,200};std::vector<int64_t> expected(uint64_t(count)*width);std::vector<uint8_t> active(count);
 std::vector<int64_t> matrices{1,0,0,1,0,1,-1,0};std::vector<uint32_t> moduli{17,17};
 for(uint32_t g=0;g<2;++g)for(uint32_t c=0;c<width;++c)permutations[g*width+c]=(c+g)%width;
 for(uint32_t source=0;source<world;++source){uint32_t queue=source*shards+shard;counts[queue]=source?9:11;
  for(uint32_t p=0;p<parents;++p)for(uint32_t c=0;c<width;++c)states[uint64_t(source)*parents*width+uint64_t(c)*parents+p]=State((p/2+source+c)%13);
  for(uint32_t row=0;row<q;++row){uint32_t logical=source*q+row,child=queue*q+row,parent=row%parents,g=row%2;physical[logical]=child;active[logical]=row<counts[queue];
   metadata[child]={~uint64_t(0),cursors[source]+parent,source,g,shard,0};
   for(uint32_t c=0;c<width;++c){int64_t value=0;
    if(!matrix)value=states[uint64_t(source)*parents*width+uint64_t(permutations[g*width+c])*parents+parent];
    else{uint32_t rr=c/2,col=c%2;for(uint32_t k=0;k<2;++k)value+=matrices[g*4+rr*2+k]*int64_t(states[uint64_t(source)*parents*width+uint64_t(k*2+col)*parents+parent]);value%=17;if(value<0)value+=17;}
    expected[uint64_t(logical)*width+c]=value;
   }
  }
 }
 Device<State> ds(states.size());ds.put(states);Device<GenericRouteRecord> dm(metadata.size());dm.put(metadata);Device<uint32_t> dc(counts.size()),df(world),dp(permutations.size()),dmod(2);dc.put(counts);df.put(frontiers);dp.put(permutations);dmod.put(moduli);
 Device<uint64_t> cursor(world);cursor.put(cursors);Device<int64_t> matrix_data(matrices.size());matrix_data.put(matrices);
 IncomingAllAction<State,false,true> incoming{width,count,q,1,0,world,shard,shards,nullptr,nullptr,dm.p,dm.p,dc.p,dc.p,nullptr};
 ActionT<State> graph{uint32_t(matrix),width,matrix?2u:width,matrix?2u:1u,2,parents,parents,nullptr,dp.p,matrix_data.p,dmod.p};
 ParentOriginAction<State> action{incoming,graph,ds.p,cursor.p,df.p,parents,parents,0};
 oracle(action,count,physical,expected,active,matrix?"parent_matrix":sizeof(State)==1?"parent_permutation_byte":"parent_permutation_int64");
}
int main(){int devices=0;CUDA_CHECK(cudaGetDeviceCount(&devices));if(devices<2)return 5;for(int d=0;d<2;++d){CUDA_CHECK(cudaSetDevice(d));incoming<uint8_t,false>(25);incoming<int64_t,false>(25);incoming<uint8_t,true>(24);parent_origin<uint8_t>(25,false);parent_origin<int64_t>(129,false);parent_origin<int64_t>(4,true);printf("NATIVE_ACTION_EXACT_SORT_DEVICE_PASS device=%d\n",d);}}

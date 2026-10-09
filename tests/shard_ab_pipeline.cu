#include "../cuda/shard_ab_pipeline.h"
#include <cuda_runtime.h>
#include <array>
#include <cstdlib>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <stdexcept>
struct alignas(16) Key{uint32_t w[4];};
using State=std::array<uint32_t,4>;
struct Less{bool operator()(Key const&a,Key const&b)const{for(int i=3;i>=0;--i)if(a.w[i]!=b.w[i])return a.w[i]<b.w[i];return false;}};
bool eq(Key a,Key b){return !Less{}(a,b)&&!Less{}(b,a);}
void ck(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
void require(bool b,const char*m){if(!b)throw std::runtime_error(m);}
template<class T>struct D{T*p;size_t n;D(size_t c):n(c){ck(cudaMalloc(&p,n*sizeof(T)));}~D(){cudaFree(p);}
 void put(std::vector<T>const&v){require(v.size()<=n,"UPLOAD");ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T>get(size_t c){std::vector<T>v(c);ck(cudaMemcpy(v.data(),p,c*sizeof(T),cudaMemcpyDeviceToHost));return v;}};
Key key(unsigned id){return Key{{id,7,id*3,(id*2654435761u)%4294967291u}};}
State state(unsigned id){return State{{id,id*2,id*3,id*4}};}
void trial(unsigned shards,unsigned slots,bool skew=false){
 unsigned C=2048/shards;if(skew)C=32;
 cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));void*p=nullptr;
 uint64_t bytes=0;require(mgbfs_shard_ab_pipeline_query(shards,C,16,slots,&bytes)==0,"QUERY");
 require(mgbfs_shard_ab_pipeline_create(shards,C,16,slots,stream,&p)==0,"CREATE");
 const unsigned frames=24,N=200;std::vector<std::vector<Key>>all_keys(frames);std::vector<std::vector<State>>all_states(frames);
 std::vector<uint32_t>all_counts(frames,N);uint32_t zero=0;
 std::vector<Key>previous={key(3),key(17),key(222)},current={key(20),key(44),key(333)};
 std::reverse(previous.begin(),previous.end());std::reverse(current.begin(),current.end());
 D<Key>prev(previous.size()),curr(current.size()),input(N);prev.put(previous);curr.put(current);
 D<State>input_states(N);D<uint32_t>begin(1),count(1),fatal(1);begin.put({0});fatal.put({0});
 require(mgbfs_shard_ab_pipeline_begin(p,shards,0,1,prev.p,previous.size(),curr.p,current.size(),C,fatal.p)==0,"BEGIN");
 std::map<Key,State,Less>oracle;
 for(unsigned frame=0;frame<frames;++frame){
  std::vector<unsigned>ids;for(unsigned i=0;i<N;++i)ids.push_back((frame*79+i*17)%(std::getenv("MGBFS_TEST_AB_REPEATED_KEY")?13:1000));
  std::stable_sort(ids.begin(),ids.end(),[&](unsigned a,unsigned b){return (shards==1?0:key(a).w[3]>>(32-__builtin_ctz(shards)))<(shards==1?0:key(b).w[3]>>(32-__builtin_ctz(shards)));});
  for(auto id:ids){Key k=key(id);if(skew)k.w[3]=0;all_keys[frame].push_back(k);all_states[frame].push_back(state(id));oracle.emplace(k,state(id));}
  if(skew){std::vector<unsigned>order(N);for(unsigned i=0;i<N;++i)order[i]=i;
   std::sort(order.begin(),order.end(),[&](unsigned a,unsigned b){return Less{}(all_keys[frame][a],all_keys[frame][b]);});
   auto keys=all_keys[frame];auto states=all_states[frame];for(unsigned i=0;i<N;++i){all_keys[frame][i]=keys[order[i]];all_states[frame][i]=states[order[i]];}}
  ck(cudaMemcpyAsync(input.p,all_keys[frame].data(),N*sizeof(Key),cudaMemcpyHostToDevice,stream));
  ck(cudaMemcpyAsync(input_states.p,all_states[frame].data(),N*sizeof(State),cudaMemcpyHostToDevice,stream));
  ck(cudaMemcpyAsync(count.p,&all_counts[frame],4,cudaMemcpyHostToDevice,stream));
  require(mgbfs_shard_ab_pipeline_push(p,input.p,(uint8_t*)input_states.p,begin.p,count.p,count.p,N)==0,"PUSH");
  if(std::getenv("MGBFS_TEST_SERIALIZE_AB"))ck(cudaDeviceSynchronize());
 }
 void*out_keys=nullptr;uint8_t*out_states=nullptr;const uint32_t*out_count=nullptr;
 require(mgbfs_shard_ab_pipeline_finalize(p,&out_keys,&out_states,&out_count)==0,"FINALIZE");
 ck(cudaStreamSynchronize(stream));auto error=fatal.get(1)[0];
 if(skew)require(error==112,"SKEW_EXPLICIT_OVERFLOW");
 else{
  require(error==0,"FATAL");for(auto k:previous)oracle.erase(k);for(auto k:current)oracle.erase(k);
  uint32_t n;ck(cudaMemcpy(&n,out_count,4,cudaMemcpyDeviceToHost));require(n==oracle.size(),"FINAL_UNIQUE_COUNT");
  std::vector<Key>keys(n);std::vector<State>states(n);
  const uint32_t*device_counts=nullptr;const uint32_t*device_offsets=nullptr;
  require(mgbfs_shard_ab_pipeline_views(p,&device_counts,&device_offsets)==0,"VIEWS");
  std::vector<uint32_t>sizes(shards),offsets(shards+1);
  ck(cudaMemcpy(sizes.data(),device_counts,shards*4,cudaMemcpyDeviceToHost));
  ck(cudaMemcpy(offsets.data(),device_offsets,(shards+1)*4,cudaMemcpyDeviceToHost));
  require(offsets.back()==n,"PREFIX_TOTAL");
  for(unsigned j=0;j<shards;++j){require(offsets[j+1]==offsets[j]+sizes[j],"PREFIX_SHAPE");
   ck(cudaMemcpy(keys.data()+offsets[j],static_cast<Key*>(out_keys)+j*C,sizes[j]*sizeof(Key),cudaMemcpyDeviceToHost));
   ck(cudaMemcpy(states.data()+offsets[j],reinterpret_cast<State*>(out_states)+j*C,sizes[j]*sizeof(State),cudaMemcpyDeviceToHost));}

  std::map<Key,State,Less>actual;for(unsigned i=0;i<n;++i)require(actual.emplace(keys[i],states[i]).second,"OUTPUT_UNIQUE");require(actual.size()==oracle.size(),"OUTPUT_SIZE");for(auto const&x:oracle)require(actual.count(x.first)&&actual.at(x.first)==x.second,"FULL_KEY_STATE_PARITY");
  // Begin a fresh depth only after output consumers have drained.
  fatal.put({0});require(mgbfs_shard_ab_pipeline_begin(p,1,0,1,nullptr,0,nullptr,0,C,fatal.p)==0,"DEPTH_REUSE");
  require(mgbfs_shard_ab_pipeline_finalize(p,&out_keys,&out_states,&out_count)==0,"EMPTY_FINALIZE");
  ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(&n,out_count,4,cudaMemcpyDeviceToHost));require(n==0&&fatal.get(1)[0]==0,"EMPTY_DEPTH");
 }
 require(mgbfs_shard_ab_pipeline_destroy(p)==0,"DESTROY");ck(cudaStreamDestroy(stream));
 std::cout<<"SHARD_AB_PIPELINE_CPU_ORACLE_PASS shards="<<shards<<" slots="<<slots<<" bytes="<<bytes<<" skew="<<skew<<"\n";
}
int main(){try{int devices;ck(cudaGetDeviceCount(&devices));for(int d=0;d<devices;++d){ck(cudaSetDevice(d));
 for(unsigned s:{1u,2u,4u,8u,16u,32u,64u,128u,256u})for(unsigned k:{1u,2u,4u})trial(s,k);
 if(!std::getenv("MGBFS_TEST_AB_REPEATED_KEY"))trial(8,2,true);}return 0;}catch(std::exception const&e){std::cerr<<e.what()<<"\n";return 1;}}

#include "shard_ab_pipeline.h"
#include "state_commit.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <iostream>
struct alignas(16) Key{uint32_t w[4];};
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void req(bool b,const char*s){if(!b)throw std::runtime_error(s);}
template<class T>struct D{T*p;size_t n;D(size_t n):n(n){ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));}~D(){cudaFree(p);}void put(std::vector<T>v){req(v.size()<=n,"put bounds");ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(size_t m){std::vector<T>v(m);ck(cudaMemcpy(v.data(),p,m*sizeof(T),cudaMemcpyDeviceToHost));return v;}};
Key key(unsigned id){return {{id,0,0,(id%4)<<30}};}
bool less(Key a,Key b){for(int i=3;i>=0;--i)if(a.w[i]!=b.w[i])return a.w[i]<b.w[i];return false;}

void negative(unsigned kind){
 cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));void*p=nullptr;req(!mgbfs_shard_ab_indexed_create(4,128,16,2,s,&p),"create");
 D<MgbfsStateRingControl>ring(1);D<MgbfsOwnerControl>owner(1);D<MgbfsStateExtent>extent(1),next(2);D<uint32_t>layer(1),next_count(1),begin(1),rows(1);D<unsigned char>states(2048*16);D<Key>keys(4),prev(1),curr(1);
 ring.put({MgbfsStateRingControl{0,0,0,0,2048,64,0,0,0}});
 req(!mgbfs_shard_ab_indexed_bind(p,ring.p,owner.p,extent.p,states.p,layer.p,1024,next_count.p,next.p),"bind");
 req(!mgbfs_shard_ab_pipeline_begin(p,4,0,kind==1?2:1,prev.p,0,curr.p,0,128,(uint32_t*)((char*)ring.p+48)),"begin");
 std::vector<Key>k(4);for(unsigned i=0;i<4;++i)k[i]={{i,0,0,0}};
 unsigned expected=0;
 if(kind==0){k[0].w[0]=4294967291u;expected=102;}
 if(kind==1){k[0].w[3]=0x80000000u;expected=111;}
 if(kind==2){k[0].w[3]=0x40000000u;expected=111;}
 if(kind==3){expected=110;}
 keys.put(k);rows.put({kind==3?5u:4u});
 req(!mgbfs_shard_ab_pipeline_select(p,keys.p,begin.p,rows.p,rows.p,4),"select enqueue");ck(cudaStreamSynchronize(s));auto observed=ring.get(1)[0].fatal;std::cerr<<"NEGATIVE_DIAGNOSTIC kind="<<kind<<" expected="<<expected<<" observed="<<observed<<"\n";req(observed==expected||(kind==2&&observed==112),"malformed input rejected by order or derived-range guard");
 auto bytes=states.get(2048*16);req(std::all_of(bytes.begin(),bytes.end(),[](unsigned char b){return b==0;}),"no state mutation");
 req(!mgbfs_shard_ab_pipeline_destroy(p),"destroy");ck(cudaStreamDestroy(s));std::cout<<"DIRECT_INPUT_NEGATIVE_PASS kind="<<kind<<" fatal="<<observed<<"\n";
}
int main(){try{int n;ck(cudaGetDeviceCount(&n));for(int i=0;i<n;++i){ck(cudaSetDevice(i));for(unsigned kind=0;kind<4;++kind)negative(kind);}return 0;}catch(std::exception const&e){std::cerr<<e.what()<<"\n";return 1;}}

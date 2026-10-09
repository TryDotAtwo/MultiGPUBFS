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
void trial(unsigned stride,bool bad){
 cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));void*p=nullptr;
 req(!mgbfs_shard_ab_indexed_create(4,128,stride,2,s,&p),"create");
 D<MgbfsStateRingControl>ring(1);D<MgbfsOwnerControl>owner(1);D<MgbfsStateExtent>extent(1),next(2);D<uint32_t>layer(1),next_count(1),begin(1),rows(1),out_count(1);D<unsigned char>states(2048*stride);D<Key>keys(96),prev(1),curr(1),published(1024);D<MgbfsRegenerateOrigin>origins(96),requests(96);D<unsigned char>responses(96*stride);
 ring.put({MgbfsStateRingControl{0,0,0,0,2048,64,0,0,0}});prev.put({key(1)});curr.put({key(2)});
 req(!mgbfs_shard_ab_indexed_bind(p,ring.p,owner.p,extent.p,states.p,layer.p,1024,next_count.p,next.p),"bind");
 req(!mgbfs_shard_ab_pipeline_begin(p,4,0,1,prev.p,1,curr.p,1,128,(uint32_t*)((char*)ring.p+48)),"begin");
 std::set<unsigned>all;
 for(unsigned frame=0;frame<2;++frame){
  std::vector<Key>k;std::vector<MgbfsRegenerateOrigin>o;
  std::vector<unsigned>ids;for(unsigned i=0;i<96;++i)ids.push_back((i+frame*23)%70);std::stable_sort(ids.begin(),ids.end(),[](unsigned a,unsigned b){return less(key(a),key(b));});
  std::set<unsigned>expected;for(unsigned id:ids){k.push_back(key(id));o.push_back({0,0,0,id});if(id!=1&&id!=2&&!all.count(id))expected.insert(id);}keys.put(k);origins.put(o);rows.put({96});
  req(!mgbfs_shard_ab_pipeline_select(p,keys.p,begin.p,rows.p,rows.p,96),"select");
  req(mgbfs_shard_ab_pipeline_select(p,keys.p,begin.p,rows.p,rows.p,96)!=0,"pending push rejected");
  void*ok;unsigned char*os;const uint32_t*oc;req(mgbfs_shard_ab_pipeline_finalize(p,&ok,&os,&oc)!=0,"pending finalize rejected");
  req(!mgbfs_shard_ab_pipeline_requests(p,origins.p,requests.p,out_count.p,96),"requests");req(mgbfs_shard_ab_pipeline_requests(p,origins.p,requests.p,out_count.p,96)!=0,"duplicate requests rejected");
  ck(cudaStreamSynchronize(s));unsigned count=out_count.get(1)[0];req(count==expected.size(),"request count oracle");auto selected=requests.get(count);std::set<unsigned>actual;std::vector<unsigned char>payload(count*stride);
  for(unsigned i=0;i<count;++i){auto x=selected[i];req(x.source==0&&x.move==0&&x.reserved==0,"origin metadata");req(actual.insert(x.parent).second,"unique selection");for(unsigned j=0;j<stride;++j)payload[i*stride+j]=(x.parent*17+j)%251;}
  req(actual==expected,"selected origins oracle");responses.put(payload);
  if(bad){out_count.put({count+1});req(!mgbfs_shard_ab_pipeline_apply_responses(p,responses.p,out_count.p),"bad response enqueued");req(!mgbfs_shard_ab_pipeline_drain(p),"drain bad response");ck(cudaStreamSynchronize(s));req(ring.get(1)[0].fatal==114,"response mismatch rejected");auto bytes=states.get(2048*stride);req(std::all_of(bytes.begin(),bytes.end(),[](unsigned char b){return b==0;}),"no partial state writes");break;}
  auto e=extent.get(1)[0];req(!mgbfs_shard_ab_pipeline_apply_responses(p,responses.p,out_count.p),"apply");req(mgbfs_shard_ab_pipeline_apply_responses(p,responses.p,out_count.p)!=0,"duplicate apply rejected");req(!mgbfs_shard_ab_pipeline_drain(p),"drain response");ck(cudaStreamSynchronize(s));req(owner.get(1)[0].stage==2 && extent.get(1)[0].ready==1,"drained live metadata ready");req(!ring.get(1)[0].fatal,"fatal");auto bytes=states.get(2048*stride);for(unsigned i=0;i<count*stride;++i)req(bytes[e.begin*stride+i]==payload[i],"state response oracle");all.insert(expected.begin(),expected.end());
 }
 if(!bad){void*k;unsigned char*st;const uint32_t*n;req(!mgbfs_shard_ab_pipeline_finalize(p,&k,&st,&n),"finalize");req(!mgbfs_shard_ab_pipeline_prepare_publish(p,ring.p,owner.p,extent.p,layer.p,1024),"prepare");req(!mgbfs_shard_ab_pipeline_publish(p,ring.p,owner.p,extent.p,states.p,published.p,layer.p,1024,next_count.p,next.p,out_count.p),"publish");ck(cudaStreamSynchronize(s));req(!ring.get(1)[0].fatal,"publish fatal");req(layer.get(1)[0]==all.size(),"layer count");auto result=published.get(all.size());std::set<unsigned>actual;for(auto k:result)actual.insert(k.w[0]);req(actual==all,"published keys oracle");}
 req(!mgbfs_shard_ab_pipeline_destroy(p),"destroy");ck(cudaStreamDestroy(s));std::cout<<"REQUEST_ORACLE_PASS stride="<<stride<<" bad="<<bad<<"\n";
}
int main(){try{int n;ck(cudaGetDeviceCount(&n));for(int i=0;i<n;++i){ck(cudaSetDevice(i));for(unsigned stride:{16u,32u,128u}){trial(stride,false);trial(stride,true);}}return 0;}catch(std::exception const&e){std::cerr<<e.what()<<"\n";return 1;}}

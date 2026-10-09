#include "../cuda/shard_ab_job.h"
#include <cuda_runtime.h>
#include <array>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <climits>
struct alignas(16) Key{uint32_t w[4];};
struct Less{bool operator()(Key const&a,Key const&b)const{for(int i=3;i>=0;--i)if(a.w[i]!=b.w[i])return a.w[i]<b.w[i];return false;}};
bool eq(Key a,Key b){return !Less{}(a,b)&&!Less{}(b,a);}
using State=std::array<uint32_t,4>;
void ck(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
void require(bool b,const char*m){if(!b)throw std::runtime_error(m);}
template<class T>struct D{T*p;size_t n;D(size_t c):n(c){ck(cudaMalloc(&p,n*sizeof(T)));}~D(){cudaFree(p);}void put(std::vector<T>const& v){require(v.size()<=n,"UPLOAD_BOUND");ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(size_t c){std::vector<T>v(c);ck(cudaMemcpy(v.data(),p,c*sizeof(T),cudaMemcpyDeviceToHost));return v;}};
struct Slot{void*job=nullptr;cudaStream_t stream;D<Key>keys;D<State>states;D<uint32_t>count,fatal;
 Slot(unsigned cap):keys(cap),states(cap),count(1),fatal(1){ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));require(mgbfs_shard_ab_job_create(cap,16,&job)==0,"CREATE");}
 ~Slot(){cudaStreamSynchronize(stream);mgbfs_shard_ab_job_destroy(job);cudaStreamDestroy(stream);}
};
void test_device(int device){
 ck(cudaSetDevice(device));constexpr unsigned C=4096;
 uint64_t bytes=0;require(mgbfs_shard_ab_job_query(C,16,&bytes)==0&&bytes>uint64_t(C)*61,"QUERY");
 require(mgbfs_shard_ab_job_query(0,16,&bytes)!=0&&bytes==0,"QUERY_REJECT");
 Slot a(C),b(C);std::vector<Key>input;std::vector<State>states;
 std::map<Key,State,Less> oracle;
 for(unsigned i=0;i<2000;++i){Key k={{(i*71)%701,0,0,(i*13)%4}};State s={{i,i+1,i+2,i+3}};input.push_back(k);states.push_back(s);oracle.emplace(k,s);}
 std::vector<Key>previous,current;unsigned j=0;for(auto const&x:oracle){if(j%11==0)previous.push_back(x.first);else if(j%13==0)current.push_back(x.first);++j;}
 for(auto k:previous)oracle.erase(k);for(auto k:current)oracle.erase(k);
 D<Key>prev(previous.size()),curr(current.size());prev.put(previous);curr.put(current);D<uint32_t>added(1);added.put({1});
 for(Slot*p:{&a,&b}){p->keys.put(input);p->states.put(states);p->count.put({unsigned(input.size())});p->fatal.put({0});
  require(mgbfs_shard_ab_job_run_added(p->job,C,p->keys.p,(uint8_t*)p->states.p,p->count.p,prev.p,previous.size(),curr.p,current.size(),p->fatal.p,added.p,p->stream)==0,"HASH_RUN");
  require(mgbfs_shard_ab_job_run(p->job,p->keys.p,(uint8_t*)p->states.p,p->count.p,prev.p,previous.size(),curr.p,current.size(),p->fatal.p,p->stream)==0,"RUN");}
 for(Slot*p:{&a,&b}){ck(cudaStreamSynchronize(p->stream));require(p->fatal.get(1)[0]==0,"FATAL");unsigned n=p->count.get(1)[0];require(n==oracle.size(),"COUNT");
  auto keys=p->keys.get(n);auto out=p->states.get(n);unsigned i=0;for(auto const&x:oracle){require(eq(keys[i],x.first)&&out[i]==x.second,"KEY_STATE_IDENTITY");++i;}}
 // Reuse the same slot after publishing; add duplicates of prior portions.
 auto clean=a.keys.get(oracle.size());auto clean_states=a.states.get(oracle.size());
 auto combined=clean;auto combined_states=clean_states;
 combined.insert(combined.end(),input.begin(),input.begin()+1000);combined_states.insert(combined_states.end(),states.begin(),states.begin()+1000);
 a.keys.put(combined);a.states.put(combined_states);a.count.put({unsigned(combined.size())});
 require(mgbfs_shard_ab_job_run_added(a.job,C,a.keys.p,(uint8_t*)a.states.p,a.count.p,prev.p,previous.size(),curr.p,current.size(),a.fatal.p,added.p,a.stream)==0,"HASH_REUSE");
 require(mgbfs_shard_ab_job_run(a.job,a.keys.p,(uint8_t*)a.states.p,a.count.p,prev.p,previous.size(),curr.p,current.size(),a.fatal.p,a.stream)==0,"REUSE");
 ck(cudaStreamSynchronize(a.stream));require(a.fatal.get(1)[0]==0&&a.count.get(1)[0]==oracle.size(),"REUSE_COUNT");
 auto keys=a.keys.get(oracle.size());auto out=a.states.get(oracle.size());j=0;for(auto const&x:oracle){require(eq(keys[j],x.first)&&out[j]==x.second,"REUSE_STATE_IDENTITY");++j;}
 added.put({0});require(mgbfs_shard_ab_job_run_added(a.job,C,a.keys.p,(uint8_t*)a.states.p,a.count.p,prev.p,previous.size(),curr.p,current.size(),a.fatal.p,added.p,a.stream)==0,"NO_DIRTY");
 ck(cudaStreamSynchronize(a.stream));require(a.count.get(1)[0]==oracle.size(),"NO_DIRTY_COUNT");require(a.states.get(oracle.size())==out,"NO_DIRTY_STATES");added.put({1});
 a.count.put({0});a.fatal.put({0});require(mgbfs_shard_ab_job_run(a.job,a.keys.p,(uint8_t*)a.states.p,a.count.p,nullptr,0,nullptr,0,a.fatal.p,a.stream)==0,"EMPTY");
 ck(cudaStreamSynchronize(a.stream));require(a.count.get(1)[0]==0&&a.fatal.get(1)[0]==0,"EMPTY_RESULT");
 a.keys.put(input);a.states.put(states);a.count.put({C+1});a.fatal.put({0});
 require(mgbfs_shard_ab_job_run(a.job,a.keys.p,(uint8_t*)a.states.p,a.count.p,nullptr,0,nullptr,0,a.fatal.p,a.stream)==0,"OVERFLOW_ENQUEUE");
 ck(cudaStreamSynchronize(a.stream));require(a.fatal.get(1)[0]==101&&a.count.get(1)[0]==C+1,"OVERFLOW_TRANSACTION");
 auto preserved=a.keys.get(input.size());for(unsigned i=0;i<input.size();++i)require(eq(preserved[i],input[i]),"OVERFLOW_PRESERVES_BUFFER");
 // Full uint128 key domain, including the same value as sort padding.
 std::vector<Key>boundary;for(unsigned word=0;word<4;++word)for(unsigned long long v=4294967291ULL;v<=4294967295ULL;++v){Key k={{0,0,0,0}};k.w[word]=unsigned(v);boundary.push_back(k);}
 boundary.push_back(Key{{UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX}});boundary.push_back(boundary.back());
 std::vector<State>boundary_states;std::map<Key,State,Less>boundary_oracle;for(unsigned i=0;i<boundary.size();++i){State state={{i,i+1,i+2,i+3}};boundary_states.push_back(state);boundary_oracle.emplace(boundary[i],state);}
 for(unsigned path=0;path<2;++path){a.keys.put(boundary);a.states.put(boundary_states);a.count.put({unsigned(boundary.size())});a.fatal.put({0});
 if(path==0)require(!mgbfs_shard_ab_job_run_added(a.job,C,a.keys.p,(uint8_t*)a.states.p,a.count.p,nullptr,0,nullptr,0,a.fatal.p,added.p,a.stream),"FULL_DOMAIN_HASH");
 require(!mgbfs_shard_ab_job_run(a.job,a.keys.p,(uint8_t*)a.states.p,a.count.p,nullptr,0,nullptr,0,a.fatal.p,a.stream),"FULL_DOMAIN_SORT");ck(cudaStreamSynchronize(a.stream));require(!a.fatal.get(1)[0]&&a.count.get(1)[0]==boundary_oracle.size(),"FULL_DOMAIN_COUNT");
 auto result_keys=a.keys.get(boundary_oracle.size());auto result_states=a.states.get(boundary_oracle.size());unsigned at=0;for(auto const& entry:boundary_oracle){require(eq(result_keys[at],entry.first)&&result_states[at]==entry.second,"FULL_DOMAIN_STATE_IDENTITY");++at;}}
 std::cout<<"SHARD_AB_JOB_CPU_ORACLE_PASS device="<<device<<" unique="<<oracle.size()<<"\n";
}
int main(){try{int n;ck(cudaGetDeviceCount(&n));for(int i=0;i<n;++i)test_device(i);return 0;}catch(std::exception const&e){std::cerr<<e.what()<<"\n";return 1;}}

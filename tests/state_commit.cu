#include "state_commit.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <array>
#include <set>
#include <numeric>
struct alignas(16) Hash {uint32_t w[4];};
using Matrix=std::array<uint8_t,16>;
static Matrix child(Matrix a,unsigned move,unsigned modulus){unsigned row=move/2;int sign=move%2?-1:1;
  for(unsigned c=0;c<4;++c)a[row*4+c]=uint8_t((int(a[row*4+c])+sign*int(a[(row+1)*4+c])+int(modulus))%int(modulus));return a;}
static uint32_t code(const Matrix& a,unsigned m){uint32_t v=0;for(unsigned r=0;r<4;++r)for(unsigned c=r+1;c<4;++c)v=v*m+a[r*4+c];return v;}
static Hash hash(const Matrix& a,unsigned m){auto v=code(a,m);return {{v,0,0,(v%4)<<30}};}
static bool less(Hash a,Hash b){for(int w=3;w>=0;--w)if(a.w[w]!=b.w[w])return a.w[w]<b.w[w];return false;}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void req(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class T> struct Device {
  T* p;size_t n;explicit Device(size_t count):n(count){ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));}
  ~Device(){cudaFree(p);}
  void put(const std::vector<T>& v){req(v.size()<=n,"fixture capacity");ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
  std::vector<T> get(){std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
static void reservation(unsigned mode){
  Device<MgbfsStateRingControl> ring(1);Device<MgbfsOwnerControl> owner(1);Device<MgbfsStateExtent> extent(1);
  MgbfsStateRingControl r{4,6,0,1,8,4,0,0,0};MgbfsOwnerControl o{};o.stage=1;o.survivors=3;
  if(mode==1)r.head=0; // wrap padding + live rows exceed capacity
  if(mode==2)r.descriptor_tail=4;
  if(mode==3)o.survivors=0;
  if(mode==4){r.head=6;r.descriptor_head=r.descriptor_tail;o.survivors=8;}
  if(mode==5){r.head=UINT64_MAX-1;r.tail=UINT64_MAX-1;}
  ring.put({r});owner.put({o});
  req(mgbfs_state_reserve(ring.p,owner.p,extent.p,nullptr)==0,"reserve enqueue");ck(cudaDeviceSynchronize());
  auto rr=ring.get()[0];auto e=extent.get()[0];auto oo=owner.get()[0];
  if(mode==1||mode==2||mode==5){req(rr.fatal&&oo.error&&e.granted_rows==0,"fatal reserve");req(rr.tail==r.tail&&rr.descriptor_tail==r.descriptor_tail,"partial reservation");}
  else if(mode==3){req(!oo.error&&e.count==0&&e.granted_rows==0&&rr.tail==r.tail&&rr.descriptor_tail==r.descriptor_tail,"zero reserve");}
  else{unsigned n=mode==4?8:3;req(!oo.error&&e.sequence==8&&e.begin==0&&e.count==n&&e.granted_rows==n&&rr.tail==8+n,"wrap reserve");}
}
static void materialization(bool invalid){
  Device<MgbfsStateRingControl> ring(1);Device<MgbfsOwnerControl> owner(1);Device<MgbfsStateExtent> extent(1);
  Device<uint8_t> input(64),states(128);Device<uint64_t> refs(4);Device<uint32_t> selected(2);
  std::vector<uint8_t> data(64);for(unsigned i=0;i<64;++i)data[i]=uint8_t(i+1);input.put(data);
  ring.put({{0,4,0,1,8,4,0,0,0}});MgbfsOwnerControl o{};o.stage=2;o.survivors=2;owner.put({o});
  MgbfsStateExtent e{};e.sequence=2;e.begin=2;e.count=2;e.granted_rows=2;extent.put({e});
  refs.put({3,1,0,2});selected.put({0,invalid?4u:2u});
  req(mgbfs_state_materialize(input.p,4,refs.p,4,selected.p,2,16,states.p,ring.p,owner.p,extent.p,nullptr)==0,"materialize enqueue");
  ck(cudaDeviceSynchronize());auto out=states.get();
  if(invalid){req(owner.get()[0].error&&ring.get()[0].fatal&&!extent.get()[0].ready,"invalid materialize fatal");req(std::all_of(out.begin(),out.end(),[](uint8_t x){return x==0;}),"partial state copy");}
  else{req(!owner.get()[0].error&&extent.get()[0].ready==1,"state ready");for(unsigned i=0;i<16;++i){req(out[32+i]==data[48+i],"first state mapping");req(out[48+i]==data[i],"second state mapping");}}
}
// Verification harness only: CPU prepares candidates/descriptors and reads
// snapshots. It is NOT a production CPU data plane or performance benchmark.
static void full_layers(unsigned modulus){
  constexpr unsigned I=128,B=4,K=1024,CAP=2048;
  Matrix identity{};for(unsigned r=0;r<4;++r)identity[r*4+r]=1;
  std::vector<std::set<Matrix>> oracle(1,std::set<Matrix>{identity});std::set<Matrix> visited{identity};
  while(!oracle.back().empty()){std::set<Matrix> next;
    for(const auto& a:oracle.back())for(unsigned move=0;move<6;++move){auto b=child(a,move,modulus);if(visited.insert(b).second)next.insert(b);}
    oracle.push_back(next);
  }
  req(visited.size()==(modulus==2?64u:729u),"oracle group cardinality");
  Device<Hash> in(I),prev(CAP),curr(CAP),accepted(B*K);
  Device<uint32_t> lengths(B),selected(I);Device<uint64_t> refs(I);
  Device<Matrix> candidates(I),states(CAP);
  Device<MgbfsBucketJob> jobs(B);Device<MgbfsOwnerCounts> counts(B);Device<MgbfsOwnerControl> owner(1);
  Device<MgbfsStateRingControl> ring(1);Device<MgbfsStateExtent> extent(1);
  ring.put({{0,0,0,0,CAP,2048,0,0,0}});
  void* plan=nullptr;req(mgbfs_bounded_owner_create(I,B,K,&plan)==0,"BFS owner create");
  std::vector<Matrix> previous,front{identity};
  for(unsigned depth=0;!front.empty();++depth){
    std::vector<Hash> ph,ch;for(auto& x:previous)ph.push_back(hash(x,modulus));for(auto& x:front)ch.push_back(hash(x,modulus));
    std::sort(ph.begin(),ph.end(),less);std::sort(ch.begin(),ch.end(),less);prev.put(ph);curr.put(ch);
    lengths.put(std::vector<uint32_t>(B));std::vector<Matrix> next;
    auto range=[](const std::vector<Hash>& h,unsigned bucket){uint64_t begin=0;while(begin<h.size()&&(h[begin].w[3]>>30)<bucket)++begin;
      uint64_t end=begin;while(end<h.size()&&(h[end].w[3]>>30)==bucket)++end;return MgbfsOwnerRange{begin,end-begin};};
    for(unsigned base=0;base<front.size();base+=16){
      std::vector<Matrix> generated;for(unsigned p=base;p<std::min<unsigned>(base+16,unsigned(front.size()));++p)
        for(unsigned move=0;move<6;++move)generated.push_back(child(front[p],move,modulus));
      std::vector<uint64_t> order(generated.size());std::iota(order.begin(),order.end(),0);
      std::stable_sort(order.begin(),order.end(),[&](uint64_t a,uint64_t b){return less(hash(generated[a],modulus),hash(generated[b],modulus));});
      std::vector<Hash> sorted;for(auto x:order)sorted.push_back(hash(generated[x],modulus));
      std::vector<MgbfsBucketJob> descriptors;auto live=lengths.get();
      for(unsigned b=0;b<B;++b){auto incoming=range(sorted,b);if(incoming.count)descriptors.push_back({b,0,incoming,range(ph,b),range(ch,b),live[b],depth});}
      candidates.put(generated);refs.put(order);in.put(sorted);jobs.put(descriptors);
      req(mgbfs_bounded_owner_compare(plan,jobs.p,unsigned(descriptors.size()),unsigned(sorted.size()),in.p,prev.p,ph.size(),curr.p,ch.size(),accepted.p,lengths.p,B,B,0,depth,counts.p,owner.p,nullptr)==0,"BFS compare");
      req(mgbfs_state_reserve(ring.p,owner.p,extent.p,nullptr)==0,"BFS reserve");
      req(mgbfs_bounded_owner_commit(plan,jobs.p,unsigned(descriptors.size()),in.p,accepted.p,lengths.p,counts.p,owner.p,&extent.p->granted_rows,selected.p,nullptr)==0,"BFS commit");
      req(mgbfs_state_materialize(reinterpret_cast<uint8_t*>(candidates.p),unsigned(generated.size()),refs.p,unsigned(sorted.size()),selected.p,I,16,reinterpret_cast<uint8_t*>(states.p),ring.p,owner.p,extent.p,nullptr)==0,"BFS materialize");
      ck(cudaDeviceSynchronize());req(!owner.get()[0].error&&!ring.get()[0].fatal,"BFS fatal");auto e=extent.get()[0];req(e.ready==1,"BFS state ready");
      auto snapshot=states.get();for(uint64_t r=0;r<e.count;++r)next.push_back(snapshot[e.begin+r]);
    }
    std::set<Matrix> actual(next.begin(),next.end());req(actual.size()==next.size(),"cross-job state duplicates");
    req(depth+1<oracle.size()&&actual==oracle[depth+1],"full-state layer mismatch");
    previous=front;front=next;
  }
  mgbfs_bounded_owner_destroy(plan);
}
int main(){try{materialization(false);materialization(true);for(unsigned m=0;m<6;++m)reservation(m);full_layers(2);full_layers(3);std::puts("STATE_COMMIT_PASS");return 0;}
catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}

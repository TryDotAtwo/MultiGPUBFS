#include "state_commit.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <array>
#include <set>
#include <numeric>
extern "C" int mgbfs_state_publish_next_extent(MgbfsStateRingControl*,
    MgbfsOwnerControl*,const MgbfsStateExtent*,uint32_t*,
    MgbfsStateExtent*,uint32_t,void*);
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
static void rank_batch_reservation(unsigned mode){
  Device<MgbfsStateRingControl> ring(1);Device<MgbfsOwnerControl> owner(1);Device<MgbfsStateExtent> extent(1);
  Device<uint32_t> survivors(3),accepted(3),capacities(3),offsets(4),layer(1);
  MgbfsStateRingControl r{4,6,0,1,12,4,0,0,0};
  MgbfsOwnerControl o{};o.stage=1;
  std::vector<uint32_t> s{2,0,1},a{1,3,2},c{4,4,4};
  uint32_t layer_cap=10,request_cap=3,hash_first=0;
  if(mode==1)c[2]=2; // accepted capacity, but only shard 2 fails
  if(mode==2)layer_cap=4;
  if(mode==3){hash_first=1;request_cap=2;}
  if(mode==4)r.descriptor_tail=4;
  if(mode==5)s={0,0,0};
  if(mode==6){o.error=23;}
  if(mode==7){r.head=0;r.capacity=8;} // wrap plus live rows exceed capacity
  if(mode==8){hash_first=1;request_cap=3;}
  ring.put({r});owner.put({o});survivors.put(s);accepted.put(a);capacities.put(c);layer.put({2});
  offsets.put({99,99,99,99});
  req(mgbfs_state_reserve_rank_batch(ring.p,owner.p,extent.p,survivors.p,accepted.p,
      capacities.p,3,offsets.p,layer.p,layer_cap,request_cap,hash_first,nullptr)==0,
      "rank batch enqueue");
  ck(cudaDeviceSynchronize());auto rr=ring.get()[0];auto oo=owner.get()[0];auto e=extent.get()[0];
  if((mode>=1&&mode<=4)||mode==6||mode==7){
    req(rr.tail==r.tail&&rr.descriptor_tail==r.descriptor_tail&&layer.get()[0]==2,
        "rank batch partial mutation");
    req(e.count==0&&e.granted_rows==0&&oo.error,"rank batch failure not atomic");
  }else if(mode==5){
    req(!oo.error&&e.count==0&&rr.tail==r.tail&&layer.get()[0]==2,
        "empty rank batch changed ring");
    req(offsets.get()==std::vector<uint32_t>({0,0,0,0}),"empty offsets");
  }else{
    req(!oo.error&&oo.survivors==3&&e.sequence==6&&e.begin==6&&e.count==3,
        "rank batch extent");
    req(rr.tail==9&&rr.descriptor_tail==2&&layer.get()[0]==5,"rank batch counters");
    req(offsets.get()==std::vector<uint32_t>({0,2,2,3}),"rank batch offsets");
  }
}
static void next_extent_publication(){
  Device<MgbfsStateRingControl> ring(1);Device<MgbfsOwnerControl> owner(1);
  Device<MgbfsStateExtent> incoming(1),next(2);Device<uint32_t> count(1);
  ring.put({{4,10,0,3,8,4,0,0,0}});owner.put({MgbfsOwnerControl{}});
  auto publish=[&](MgbfsStateExtent e){
    incoming.put({e});
    req(mgbfs_state_publish_next_extent(ring.p,owner.p,incoming.p,count.p,
        next.p,2,nullptr)==0,"next extent enqueue");
    ck(cudaDeviceSynchronize());
  };
  MgbfsStateExtent first{};first.sequence=4;first.begin=4;first.count=2;
  first.descriptor=0;first.granted_rows=2;first.ready=1;
  publish(first);
  req(count.get()[0]==1&&next.get()[0].count==2&&
      next.get()[0].padding[1]==0,"first next extent");
  MgbfsStateExtent adjacent{};adjacent.sequence=6;adjacent.begin=6;
  adjacent.count=2;adjacent.descriptor=1;adjacent.granted_rows=2;
  adjacent.ready=1;
  publish(adjacent);
  req(count.get()[0]==1&&next.get()[0].count==4&&
      next.get()[0].padding[1]==1,"adjacent next extents merge");
  MgbfsStateExtent wrapped{};wrapped.sequence=8;wrapped.begin=0;
  wrapped.count=2;wrapped.descriptor=2;wrapped.granted_rows=2;
  wrapped.ready=1;
  publish(wrapped);
  auto entries=next.get();
  req(count.get()[0]==2&&entries[0].sequence==4&&entries[0].count==4&&
      entries[1].sequence==8&&entries[1].count==2,
      "wrapped next extent remains separate");
  ring.put({{4,12,0,4,8,4,0,0,0}});
  MgbfsStateExtent third{};third.sequence=11;third.begin=3;third.count=1;
  third.descriptor=3;third.granted_rows=1;third.ready=1;
  publish(third);
  req(ring.get()[0].fatal&&owner.get()[0].error&&count.get()[0]==2&&
      next.get()[1].count==2,"third extent must fail atomically");
}
static void shard_count_directory(unsigned mode){
  Device<MgbfsStateRingControl> ring(1);Device<MgbfsOwnerControl> owner(1);
  Device<uint32_t> high(6),candidate_count(1),selected(6),selected_count(1);
  Device<uint32_t> counts(4),offsets(5);
  MgbfsStateRingControl r{0,0,0,0,16,4,0,0,0};
  MgbfsOwnerControl o{};
  std::vector<uint32_t> words{0x80000000,0x80000001,0xa0000000,
      0xa0000001,0xe0000000,0xe0000001};
  std::vector<uint32_t> indices{0,2,3,4,0,0};
  uint32_t n=4;
  if(mode==1){n=0;}
  if(mode==2){indices[2]=2;} // duplicate selected index
  if(mode==3){indices[3]=6;} // out of candidate range
  if(mode==4){words[2]=0x00000000;} // wrong owner and broken order
  if(mode==5){indices[2]=1;} // selected order goes backwards
  if(mode==6){n=7;} // selected count exceeds capacity
  ring.put({r});owner.put({o});high.put(words);candidate_count.put({6});
  selected.put(indices);selected_count.put({n});
  counts.put({99,99,99,99});offsets.put({99,99,99,99,99});
  req(mgbfs_owner_shard_counts(high.p,candidate_count.p,selected.p,selected_count.p,
      6,1,2,4,counts.p,offsets.p,ring.p,owner.p,nullptr)==0,"shard counts enqueue");
  ck(cudaDeviceSynchronize());
  if(mode>=2){
    req(owner.get()[0].error&&ring.get()[0].fatal,"malformed shard selection not fatal");
    req(counts.get()==std::vector<uint32_t>({99,99,99,99}),"partial shard counts");
    req(offsets.get()==std::vector<uint32_t>({99,99,99,99,99}),"partial shard offsets");
  }else if(mode==1){
    req(!owner.get()[0].error,"empty shard selection fatal");
    req(counts.get()==std::vector<uint32_t>({0,0,0,0}),"empty shard counts");
    req(offsets.get()==std::vector<uint32_t>({0,0,0,0,0}),"empty shard offsets");
  }else{
    req(!owner.get()[0].error,"valid shard selection fatal");
    req(counts.get()==std::vector<uint32_t>({1,2,0,1}),"shard counts");
    req(offsets.get()==std::vector<uint32_t>({0,1,3,3,4}),"shard offsets");
  }
  req(ring.get()[0].tail==r.tail,"shard counting changed ring");
}
static void materialization(bool invalid,bool packed=false){
  Device<MgbfsStateRingControl> ring(1);Device<MgbfsOwnerControl> owner(1);Device<MgbfsStateExtent> extent(1);
  Device<uint8_t> input(64),states(128);Device<uint64_t> refs(4);Device<uint32_t> selected(2);
  std::vector<uint8_t> data(64);for(unsigned i=0;i<64;++i)data[i]=uint8_t(i+1);input.put(data);
  ring.put({{0,4,0,1,8,4,0,0,0}});MgbfsOwnerControl o{};o.stage=2;o.survivors=2;owner.put({o});
  MgbfsStateExtent e{};e.sequence=2;e.begin=2;e.count=2;e.granted_rows=2;extent.put({e});
  refs.put({3,1,0,2});selected.put(packed?std::vector<uint32_t>{2,invalid?3u:0u}:std::vector<uint32_t>{0,invalid?4u:2u});
  // Nonzero source-span offset catches accidental use of full-frame indices.
  int status=packed?mgbfs_state_materialize_packed(input.p+16,3,selected.p,2,16,states.p,ring.p,owner.p,extent.p,nullptr):
      mgbfs_state_materialize(input.p,4,refs.p,4,selected.p,2,16,states.p,ring.p,owner.p,extent.p,nullptr);
  req(status==0,"materialize enqueue");
  ck(cudaDeviceSynchronize());auto out=states.get();
  if(invalid){req(owner.get()[0].error&&ring.get()[0].fatal&&!extent.get()[0].ready,"invalid materialize fatal");req(std::all_of(out.begin(),out.end(),[](uint8_t x){return x==0;}),"partial state copy");}
  else{req(!owner.get()[0].error&&extent.get()[0].ready==1,"state ready");for(unsigned i=0;i<16;++i){req(out[32+i]==data[48+i],"first state mapping");req(out[48+i]==data[(packed?16:0)+i],"second state mapping");}}
}
static void rank_batch_materialization(unsigned mode){
  Device<MgbfsStateRingControl> ring(1);Device<MgbfsOwnerControl> owner(1);
  Device<MgbfsStateExtent> extent(1);Device<uint8_t> input(64),states(128);
  Device<uint32_t> source_rows(1),selected_count(1),sources(3);
  std::vector<uint8_t> data(64);for(unsigned i=0;i<64;++i)data[i]=uint8_t(i+1);
  input.put(data);ring.put({{0,4,0,1,8,4,0,0,0}});
  MgbfsOwnerControl o{};o.stage=2;o.survivors=2;owner.put({o});
  MgbfsStateExtent e{};e.sequence=2;e.begin=2;e.count=2;e.granted_rows=2;extent.put({e});
  source_rows.put({mode==3?5u:4u});selected_count.put({mode==2?3u:2u});
  sources.put({3,mode==1?4u:0u,1});
  req(mgbfs_state_materialize_rank_batch(input.p,source_rows.p,4,sources.p,
      selected_count.p,3,16,states.p,ring.p,owner.p,extent.p,nullptr)==0,
      "rank materialize enqueue");
  ck(cudaDeviceSynchronize());auto out=states.get();
  if(mode){
    req(owner.get()[0].error&&ring.get()[0].fatal&&!extent.get()[0].ready,
        "rank invalid source/count not fatal");
    req(std::all_of(out.begin(),out.end(),[](uint8_t x){return x==0;}),
        "rank partial state copy");
  }else{
    req(!owner.get()[0].error&&extent.get()[0].ready==1,"rank state ready");
    for(unsigned i=0;i<16;++i){
      req(out[32+i]==data[48+i],"rank first state");
      req(out[48+i]==data[i],"rank second state");
    }
  }
}
static void retire_prefix(){
  Device<MgbfsStateRingControl> ring(1);Device<MgbfsStateExtent> extent(1);
  ring.put({{0,8,0,1,8,4,0,0,0}});
  MgbfsStateExtent current{};current.count=8;current.granted_rows=8;current.ready=1;current.padding[1]=0;extent.put({current});
  req(mgbfs_state_retire_dense_prefix(ring.p,extent.p,3,nullptr)==0,"retire enqueue");
  ck(cudaDeviceSynchronize());
  auto r=ring.get()[0];auto e=extent.get()[0];
  req(!r.fatal&&r.head==3,"retire advances head");
  req(e.sequence==3&&e.begin==3&&e.count==5,"retire shrinks current extent");
  MgbfsOwnerControl owner{};owner.stage=1;owner.survivors=3;Device<MgbfsOwnerControl> o(1);o.put({owner});
  Device<MgbfsStateExtent> next(1);
  req(mgbfs_state_reserve(ring.p,o.p,next.p,nullptr)==0,"reuse enqueue");ck(cudaDeviceSynchronize());
  auto n=next.get()[0];req(!o.get()[0].error&&n.begin==0&&n.count==3,"reuses retired prefix");

  // Two live FIFO extents straddle the physical end. Retiring the first
  // descriptor must also make the one-record wrap padding reclaimable.
  ring.put({{6,14,0,2,10,4,0,0,0}});
  MgbfsStateExtent before{};before.sequence=6;before.begin=6;before.count=3;before.granted_rows=3;before.ready=1;before.padding[1]=0;extent.put({before});
  req(mgbfs_state_retire_dense_prefix(ring.p,extent.p,3,nullptr)==0,"retire before wrap");ck(cudaDeviceSynchronize());
  MgbfsStateExtent after{};after.sequence=10;after.begin=0;after.count=4;after.descriptor=1;after.granted_rows=4;after.ready=1;after.padding[1]=1;extent.put({after});
  req(mgbfs_state_retire_dense_prefix(ring.p,extent.p,1,nullptr)==0,"retire after wrap");ck(cudaDeviceSynchronize());
  r=ring.get()[0];e=extent.get()[0];req(!r.fatal&&r.head==11&&e.sequence==11&&e.count==3,"wrap padding reclaimed");
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
int main(){try{materialization(false);materialization(true);materialization(false,true);materialization(true,true);for(unsigned m=0;m<4;++m)rank_batch_materialization(m);retire_prefix();for(unsigned m=0;m<6;++m)reservation(m);for(unsigned m=0;m<9;++m)rank_batch_reservation(m);next_extent_publication();for(unsigned m=0;m<7;++m)shard_count_directory(m);full_layers(2);full_layers(3);std::puts("STATE_COMMIT_PASS");return 0;}
catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}

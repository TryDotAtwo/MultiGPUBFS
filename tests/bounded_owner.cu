#include "bounded_owner.h"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <set>
#include <random>
struct alignas(16) Key { uint32_t w[4]; };
static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void require(bool x, const char* s) { if(!x) throw std::runtime_error(s); }
static int create_owner(unsigned i,unsigned j,unsigned k,void** plan){
#ifdef MGBFS_TEST_BMMA
  return mgbfs_bounded_owner_create_backend(i,j,k,1,i,256,plan);
#else
  return mgbfs_bounded_owner_create(i,j,k,plan);
#endif
}
template<class T> struct Device {
  T* p; size_t n;
  explicit Device(size_t count):n(count) { ck(cudaMalloc(&p,n*sizeof(T))); ck(cudaMemset(p,0,n*sizeof(T))); }
  ~Device(){cudaFree(p);}
  void put(const std::vector<T>& v) { require(v.size()<=n,"test input capacity"); ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); }
  std::vector<T> get() {std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
static std::vector<Key> keys(std::initializer_list<uint32_t> v) {std::vector<Key> r; for(auto x:v) r.push_back({{x,0,0,0}});return r;}
static Key wide(uint32_t x){return {{x%3,(x/3)%3,(x/9)%3,x/27}};}
static bool same(Key a,Key b){return std::equal(a.w,a.w+4,b.w);}
static void sweep(unsigned seed,unsigned mode) {
  constexpr unsigned I=8192,J=4,K=2048;
  std::mt19937 rng(seed);std::vector<Key> input,pv,cv,av(J*K);
  std::vector<uint32_t> lengths(J),expected_indices;std::vector<MgbfsBucketJob> js;
  std::vector<std::vector<uint32_t>> expected(J);std::vector<MgbfsOwnerCounts> want(J);
  for(unsigned b=0;b<J;++b){
    std::set<uint32_t> ps,cs,as;std::vector<uint32_t> values;
    for(unsigned r=0;r<400;++r){ps.insert(rng()%900);cs.insert(rng()%900);as.insert(rng()%900);}
    unsigned n=seed==0?1:seed==1?257:2000;
    for(unsigned r=0;r<n;++r)values.push_back(seed==1?42:rng()%1100);
    std::sort(values.begin(),values.end());
    MgbfsBucketJob d{b,2,{input.size(),values.size()},{pv.size(),ps.size()},{cv.size(),cs.size()},unsigned(as.size()),5};
    for(auto x:ps)pv.push_back(wide(x));for(auto x:cs)cv.push_back(wide(x));
    unsigned ai=0;for(auto x:as)av[b*K+ai++]=wide(x);lengths[b]=ai;
    want[b].output_offset=expected_indices.size();
    std::set<uint32_t> seen;
    for(auto x:values){unsigned index=unsigned(input.size());input.push_back(wide(x));
      if(!seen.insert(x).second)++want[b].duplicates;
      else if(ps.count(x))++want[b].prev;
      else if(cs.count(x))++want[b].curr;
      else if(as.count(x))++want[b].accepted;
      else {expected_indices.push_back(index);expected[b].push_back(x);++want[b].survivors;}
    }
    std::set<uint32_t> merged=as;merged.insert(expected[b].begin(),expected[b].end());expected[b].assign(merged.begin(),merged.end());
    want[b].new_count=unsigned(expected[b].size());js.push_back(d);
  }
  void* plan=nullptr;require(create_owner(I,J,K,&plan)==0,"sweep create");
  Device<Key> in(I),prev(pv.size()),curr(cv.size()),accepted(J*K);
  Device<uint32_t> lens(J),grant(1),selected(I);Device<MgbfsBucketJob> jobs(J);
  Device<MgbfsOwnerCounts> counts(J);Device<MgbfsOwnerControl> control(1);
  in.put(input);prev.put(pv);curr.put(cv);accepted.put(av);lens.put(lengths);
  if(mode==2)js[2].prev.begin=UINT64_MAX;
  if(mode==3)js[2].accepted_count=K+1;
  if(mode==4){js[2].accepted_count=K;lengths[2]=K;lens.put(lengths);
    for(unsigned r=0;r<K;++r)av[2*K+r]=wide(10000+r);accepted.put(av);}
  jobs.put(js);
  grant.put({mode==1?0u:unsigned(expected_indices.size())});
  require(mgbfs_bounded_owner_compare(plan,jobs.p,J,unsigned(input.size()),in.p,prev.p,pv.size(),curr.p,cv.size(),accepted.p,lens.p,J,J,2,5,counts.p,control.p,nullptr)==0,"sweep compare");
  // Enqueue commit immediately without a host wait/readback between stages.
  require(mgbfs_bounded_owner_commit(plan,jobs.p,J,in.p,accepted.p,lens.p,counts.p,control.p,grant.p,selected.p,nullptr)==0,"sweep commit");
  ck(cudaDeviceSynchronize());auto c=control.get()[0];auto a=accepted.get();
  if(mode){require(c.error==(mode==1?4u:mode==2?1u:2u),"fatal code");
    require(lens.get()==lengths,"fatal changed lengths");
    for(unsigned r=0;r<J*K;++r)require(same(a[r],av[r]),"fatal changed persistent hashes");
  }else{
    require(c.error==0&&c.stage==2&&c.survivors==expected_indices.size(),"sweep control");
    auto actual=selected.get();require(std::equal(expected_indices.begin(),expected_indices.end(),actual.begin()),"sweep stable indices");
    auto got=counts.get();auto ls=lens.get();
    for(unsigned b=0;b<J;++b){auto x=got[b],y=want[b];
      require(x.duplicates==y.duplicates&&x.prev==y.prev&&x.curr==y.curr&&x.accepted==y.accepted&&x.survivors==y.survivors&&x.output_offset==y.output_offset,"sweep categories");
      require(ls[b]==expected[b].size(),"sweep length");
      for(unsigned r=0;r<ls[b];++r)require(same(a[b*K+r],wide(expected[b][r])),"sweep merged hashes");
    }
  }
  mgbfs_bounded_owner_destroy(plan);
}
int main() { try {
  // Dropping any one of the four rejection categories changes this literal result.
  void* plan=nullptr; require(create_owner(16,2,8,&plan)==0,"create");
  Device<Key> in(16), prev(2), curr(2), accepted(16);
  Device<uint32_t> lengths(2), grant(1), selected(16);
  Device<MgbfsBucketJob> jobs(2);
  Device<MgbfsOwnerCounts> counts(2);
  Device<MgbfsOwnerControl> control(1);
  in.put(keys({1,1,2,3,4,5,5,6,10,11,11,12}));
  prev.put(keys({1,10}));curr.put(keys({2,11}));
  auto old=std::vector<Key>(16);old[0]={{3,0,0,0}};accepted.put(old);lengths.put({1,0});
  std::vector<MgbfsBucketJob> js={{0,0,{0,8},{0,1},{0,1},1,9},{1,0,{8,4},{1,1},{1,1},0,9}};jobs.put(js);
  require(mgbfs_bounded_owner_compare(plan,jobs.p,2,12,in.p,prev.p,2,curr.p,2,accepted.p,lengths.p,2,2,0,9,counts.p,control.p,nullptr)==0,"compare enqueue");
  ck(cudaDeviceSynchronize());auto c=control.get()[0];require(c.error==0&&c.stage==1&&c.survivors==4,"compare result");
  auto cs=counts.get();require(cs[0].duplicates==2&&cs[0].prev==1&&cs[0].curr==1&&cs[0].accepted==1&&cs[0].survivors==3,"categories");
  require(cs[1].duplicates==1&&cs[1].prev==1&&cs[1].curr==1&&cs[1].survivors==1,"bucket categories");
  require(accepted.get()[1].w[0]==0&&lengths.get()[0]==1,"compare mutated persistent data");
  grant.put({4});
  require(mgbfs_bounded_owner_commit(plan,jobs.p,2,in.p,accepted.p,lengths.p,counts.p,control.p,grant.p,selected.p,nullptr)==0,"commit enqueue");
  ck(cudaDeviceSynchronize());c=control.get()[0];require(c.error==0&&c.stage==2,"commit result");
  auto a=accepted.get();require(a[0].w[0]==3&&a[1].w[0]==4&&a[2].w[0]==5&&a[3].w[0]==6&&a[8].w[0]==12,"accepted merge");
  auto s=selected.get();require(s[0]==4&&s[1]==5&&s[2]==7&&s[3]==11,"stable source indices");
  require(lengths.get()==std::vector<uint32_t>({4,1}),"published counts");
  // Same candidates in a later job cannot survive a second time.
  js[0].accepted_count=4;js[1].accepted_count=1;jobs.put(js);
  require(mgbfs_bounded_owner_compare(plan,jobs.p,2,12,in.p,prev.p,2,curr.p,2,accepted.p,lengths.p,2,2,0,9,counts.p,control.p,nullptr)==0,"repeat");
  ck(cudaDeviceSynchronize());require(control.get()[0].survivors==0,"cross job dedup");
  mgbfs_bounded_owner_destroy(plan);
  for(unsigned seed=0;seed<12;++seed)sweep(seed,0);
  for(unsigned mode=1;mode<=4;++mode)sweep(99,mode);
  std::puts("BOUNDED_OWNER_PASS");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}

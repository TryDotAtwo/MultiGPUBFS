#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "generic_sorted_run_merge.cuh"
#define CUDA_CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
template<class T> struct Device {
 T* p;size_t n;
 explicit Device(size_t count):n(count){CUDA_CHECK(cudaMalloc(&p,(count?count:1)*sizeof(T)));}
 ~Device(){cudaFree(p);}
 void put(const std::vector<T>& v){if(!v.empty())CUDA_CHECK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T> get(size_t count){std::vector<T> v(count);if(count)CUDA_CHECK(cudaMemcpy(v.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
static uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ull;x^=x>>27;x*=0x94d049bb133111ebull;return x^(x>>31);}
template<class State> static void run_case(int device,uint32_t width,uint32_t mode,uint32_t an,uint32_t bn){
 constexpr uint32_t stride=4093;uint32_t capacity=an+bn+37;
 std::vector<State> arena(uint64_t(stride)*width);
 for(uint32_t row=0;row<stride;++row)for(uint32_t c=0;c<width;++c){
  uint64_t raw=mix(uint64_t(row%19?row:0)*1315423911+c*1234567);
  State value;
  if(sizeof(State)==8)memcpy(&value,&raw,sizeof(value));else value=State(raw);
  if(row==1&&c==0)value=std::numeric_limits<State>::min();
  if(row==2&&c==0)value=std::numeric_limits<State>::max();
  arena[uint64_t(c)*stride+row]=value;
 }
 auto hash=[&](uint32_t row)->uint64_t{
  uint64_t h=0xcbf29ce484222325ull;
  for(uint32_t c=0;c<width;++c){h^=uint64_t(arena[uint64_t(c)*stride+row]);h*=0x100000001b3ull;}
  if(mode==0)return uint64_t(0);if(mode==1)return ~uint64_t(0);if(mode==2)return h&15ull;return h;
 };
 auto compare=[&](uint32_t a,uint32_t b){
  auto ah=hash(a),bh=hash(b);if(ah<bh)return -1;if(ah>bh)return 1;
  for(uint32_t c=0;c<width;++c){auto av=arena[uint64_t(c)*stride+a],bv=arena[uint64_t(c)*stride+b];if(av<bv)return -1;if(av>bv)return 1;}
  return 0;
 };
 std::vector<uint32_t> a(an),b(bn),merged(an+bn);
 for(uint32_t i=0;i<an;++i)a[i]=(i*17)%stride;
 for(uint32_t i=0;i<bn;++i)b[i]=(i*29)%stride;
 std::stable_sort(a.begin(),a.end(),[&](auto x,auto y){return compare(x,y)<0;});
 std::stable_sort(b.begin(),b.end(),[&](auto x,auto y){return compare(x,y)<0;});
 std::merge(a.begin(),a.end(),b.begin(),b.end(),merged.begin(),[&](auto x,auto y){return compare(x,y)<0;});
 std::vector<uint32_t> unique;
 for(auto row:merged)if(unique.empty()||compare(unique.back(),row)!=0)unique.push_back(row);
 std::vector<uint64_t> ah(an),bh(bn);for(uint32_t i=0;i<an;++i)ah[i]=hash(a[i]);for(uint32_t i=0;i<bn;++i)bh[i]=hash(b[i]);
 Device<State> ds(arena.size());ds.put(arena);
 Device<uint64_t> da(an),db(bn),dm(capacity),du(capacity);da.put(ah);db.put(bh);
 Device<uint32_t> dar(an),dbr(bn),dmr(capacity),dur(capacity),flags(capacity),prefix(capacity),count(1),ucount(1),error(1);
 dar.put(a);dbr.put(b);error.put({0});
 Device<GenericSortedHistoryRun> va(1),vb(1);
 va.put({{an?da.p:nullptr,an?dar.p:nullptr,an}});vb.put({{bn?db.p:nullptr,bn?dbr.p:nullptr,bn}});
 generic_sorted_run_validate_exact<<<8,128>>>(va.p,an,ds.p,stride,width,error.p);
 generic_sorted_run_validate_exact<<<8,128>>>(vb.p,bn,ds.p,stride,width,error.p);
 generic_sorted_run_merge<<<(capacity+255)/256,256>>>(va.p,vb.p,ds.p,stride,width,dm.p,dmr.p,an,bn,capacity,count.p,error.p);
 generic_sorted_run_unique_flags<<<8,128>>>(dm.p,dmr.p,count.p,capacity,ds.p,stride,width,flags.p,error.p);
 GenericSortedRunMergeShape shape{};
 CUDA_CHECK(generic_sorted_run_merge_shape(capacity,&shape));
 auto align=[](uint64_t bytes){return ((bytes+255)/256)*256;};
 if(shape.merged_hash_bytes!=uint64_t(capacity)*8||shape.merged_row_bytes!=uint64_t(capacity)*4||
   shape.flag_bytes!=uint64_t(capacity)*4||shape.prefix_bytes!=uint64_t(capacity)*4||
   shape.aligned_workspace_bytes!=align(uint64_t(capacity)*8)+3*align(uint64_t(capacity)*4)+
    align(shape.scan_temporary_bytes)+align(2*sizeof(GenericSortedHistoryRun)+12))exit(6);
 size_t temporary_bytes=0;
 CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,temporary_bytes,flags.p,prefix.p,capacity));
 if(temporary_bytes!=shape.scan_temporary_bytes)exit(6);
 Device<uint8_t> temporary(temporary_bytes);
 CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temporary.p,temporary_bytes,flags.p,prefix.p,capacity));
 generic_sorted_run_unique_scatter<<<8,128>>>(dm.p,dmr.p,flags.p,prefix.p,capacity,du.p,dur.p,ucount.p);
 CUDA_CHECK(cudaDeviceSynchronize());
 if(error.get(1)[0]||count.get(1)[0]!=merged.size()||ucount.get(1)[0]!=unique.size()||
  dmr.get(merged.size())!=merged||dur.get(unique.size())!=unique){fprintf(stderr,"merge/oracle mismatch device%d bytes%zu width%u mode%u a%u b%u error%u\n",device,sizeof(State),width,mode,an,bn,error.get(1)[0]);exit(3);}
 auto mh=dm.get(merged.size()),uh=du.get(unique.size());
 for(size_t i=0;i<merged.size();++i)if(mh[i]!=hash(merged[i]))exit(3);
 for(size_t i=0;i<unique.size();++i)if(uh[i]!=hash(unique[i]))exit(3);
 if(an+bn){
  error.put({0});
  generic_sorted_run_merge<<<1,256>>>(va.p,vb.p,ds.p,stride,width,dm.p,dmr.p,an,bn,an+bn-1,count.p,error.p);
  CUDA_CHECK(cudaDeviceSynchronize());if(error.get(1)[0]!=32||count.get(1)[0]!=0)exit(4);
 }
 // Reject a device view exceeding its admitted input credit before any input read.
 error.put({0});va.put({{nullptr,nullptr,1}});
 generic_sorted_run_merge<<<1,256>>>(va.p,vb.p,ds.p,stride,width,dm.p,dmr.p,0,bn,capacity,count.p,error.p);
 CUDA_CHECK(cudaDeviceSynchronize());if(error.get(1)[0]!=32||count.get(1)[0]!=0)exit(4);
 if(an>1&&compare(a.front(),a.back())<0){
  auto reverse_rows=a;auto reverse_hashes=ah;
  std::reverse(reverse_rows.begin(),reverse_rows.end());std::reverse(reverse_hashes.begin(),reverse_hashes.end());
  dar.put(reverse_rows);da.put(reverse_hashes);va.put({{da.p,dar.p,an}});error.put({0});
  generic_sorted_run_validate_exact<<<8,128>>>(va.p,an,ds.p,stride,width,error.p);
  CUDA_CHECK(cudaDeviceSynchronize());if(error.get(1)[0]!=256)exit(4);
  dar.put(a);da.put(ah);
 }
 if(an){
  a[0]=stride;dar.put(a);va.put({{da.p,dar.p,an}});error.put({0});
  generic_sorted_run_validate_exact<<<8,128>>>(va.p,an,ds.p,stride,width,error.p);
  CUDA_CHECK(cudaDeviceSynchronize());if(!(error.get(1)[0]&8u))exit(4);
 }
 printf("SORTED_MERGE_ORACLE bytes=%zu width=%u mode=%u a=%u b=%u unique=%zu\n",sizeof(State),width,mode,an,bn,unique.size());
}
static void zero_capacity_gate(){
 GenericSortedRunMergeShape shape{};CUDA_CHECK(generic_sorted_run_merge_shape(0,&shape));
 if(shape.scan_temporary_bytes||shape.aligned_workspace_bytes!=256)exit(7);
 Device<GenericSortedHistoryRun> view(1);view.put({{nullptr,nullptr,0}});
 Device<uint32_t> count(1),error(1);count.put({123});error.put({0});
 generic_sorted_run_merge<uint8_t><<<1,256>>>(view.p,view.p,nullptr,1,1,nullptr,nullptr,0,0,0,count.p,error.p);
 generic_sorted_run_unique_flags<uint8_t><<<1,128>>>(nullptr,nullptr,count.p,0,nullptr,1,1,nullptr,error.p);
 generic_sorted_run_unique_scatter<<<1,128>>>(nullptr,nullptr,nullptr,nullptr,0,nullptr,nullptr,count.p);
 CUDA_CHECK(cudaDeviceSynchronize());if(count.get(1)[0]||error.get(1)[0])exit(7);
 puts("SORTED_MERGE_ZERO_CAPACITY_PASS");
}
int main(){int devices=0;CUDA_CHECK(cudaGetDeviceCount(&devices));if(devices<2)return 5;
 for(int device=0;device<2;++device){CUDA_CHECK(cudaSetDevice(device));zero_capacity_gate();
  for(uint32_t width:{2u,17u,25u,129u})for(uint32_t mode=0;mode<4;++mode){
   run_case<uint8_t>(device,width,mode,0,0);run_case<int64_t>(device,width,mode,0,0);
   run_case<uint8_t>(device,width,mode,0,257);run_case<int64_t>(device,width,mode,257,0);
   run_case<uint8_t>(device,width,mode,63,256);run_case<int64_t>(device,width,mode,256,63);
   run_case<uint8_t>(device,width,mode,1025,997);run_case<int64_t>(device,width,mode,997,1025);
  }
  printf("SORTED_MERGE_PRIMITIVE_PASS device=%d\n",device);
 }
}

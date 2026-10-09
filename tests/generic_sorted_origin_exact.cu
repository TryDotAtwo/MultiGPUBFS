#include "generic_sorted_origin_exact.cuh"
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#define CUDA_CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>struct Device{T* p;explicit Device(size_t n){CUDA_CHECK(cudaMalloc(&p,(n?n:1)*sizeof(T)));}~Device(){cudaFree(p);}void put(const std::vector<T>& v){if(v.size())CUDA_CHECK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T> get(size_t n){std::vector<T> v(n);if(n)CUDA_CHECK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}};
static uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ull;x^=x>>27;x*=0x94d049bb133111ebull;return x^(x>>31);}
template<class State>struct TestAction{
 uint32_t elements,stride;const State* states;const uint64_t* hashes;const uint8_t* active;uint32_t malformed,mapping;
 __device__ uint32_t child(uint32_t i)const{return mapping?i*2+1:i;}
 __device__ bool valid(uint32_t child,uint32_t* error)const{if(malformed){atomicOr(error,128u);return false;}return active[child]!=0;}
 __device__ uint64_t hash(uint32_t child,uint64_t,uint32_t)const{return hashes[child];}
 __device__ int64_t value(uint32_t child,uint32_t c)const{return int64_t(states[uint64_t(c)*stride+child]);}
};
template<class State>static void fixture(int device,uint32_t width,uint32_t mode,uint32_t count,bool all_padding,bool mapping=false){
 uint32_t stride=count?(mapping?count*2+1:count):1;std::vector<State> states(uint64_t(stride)*width);
 std::vector<uint64_t> hashes(stride);std::vector<uint8_t> active(stride);
 for(uint32_t row=0;row<stride;++row){
  uint64_t hash=19;for(uint32_t c=0;c<width;++c){
   uint64_t bits=mix(uint64_t(row%19?row:0)*1315423911+c*1234567);State value;
   if(sizeof(State)==8)memcpy(&value,&bits,sizeof(value));else value=State(bits);
   if(row==1&&c==0)value=std::numeric_limits<State>::min();if(row==2&&c==0)value=std::numeric_limits<State>::max();
   states[uint64_t(c)*stride+row]=value;hash=mix(hash^uint64_t(int64_t(value))^c);
  }
  hashes[row]=mode==0?0:mode==1?~uint64_t(0):mode==2?hash&15ull:hash;
  active[row]=!all_padding&&row%7!=0;
 }
 auto map=[&](uint32_t i){return mapping?i*2+1:i;};
 auto exact_compare=[&](uint32_t a,uint32_t b){
  a=map(a);b=map(b);
  if(hashes[a]!=hashes[b])return hashes[a]<hashes[b]?-1:1;
  for(uint32_t c=0;c<width;++c){State av=states[uint64_t(c)*stride+a],bv=states[uint64_t(c)*stride+b];if(av<bv)return -1;if(av>bv)return 1;}return 0;
 };
 std::vector<uint32_t> ordered,expected;for(uint32_t i=0;i<count;++i)if(active[map(i)])ordered.push_back(i);
 std::sort(ordered.begin(),ordered.end(),[&](auto a,auto b){int cmp=exact_compare(a,b);return cmp<0||(!cmp&&a<b);});
 for(auto row:ordered)if(expected.empty()||exact_compare(expected.back(),row)!=0)expected.push_back(row);
 Device<State> ds(states.size());ds.put(states);Device<uint64_t> dh(stride),cached(count),out_hashes(count);dh.put(hashes);
 Device<uint8_t> mask(stride);mask.put(active);
 Device<uint32_t> input(count),sorted(count),flags(count),prefix(count),out_origins(count),out_count(1),error(1);error.put({0});out_count.put({123});
 TestAction<State> action{width,stride,ds.p,dh.p,mask.p,0,uint32_t(mapping)};GenericSortedOriginShape shape{};
 CUDA_CHECK(generic_sorted_origin_shape(action,count,&shape));
 auto align=[](uint64_t n){return ((n+255)/256)*256;};
 if(shape.hash_bytes!=uint64_t(count)*8||shape.origin_bytes!=uint64_t(count)*4||shape.flag_bytes!=uint64_t(count)*4||
  shape.aligned_workspace_bytes!=2*align(uint64_t(count)*8)+5*align(uint64_t(count)*4)+align(shape.shared_temporary_bytes)+256||
  shape.shared_temporary_bytes!=std::max(shape.sort_temporary_bytes,shape.scan_temporary_bytes))exit(6);
 Device<uint8_t> temporary(shape.shared_temporary_bytes);
 CUDA_CHECK(generic_sorted_origin_exact(action,count,0,64,cached.p,input.p,sorted.p,flags.p,prefix.p,out_hashes.p,out_origins.p,out_count.p,temporary.p,shape.shared_temporary_bytes,error.p));
 CUDA_CHECK(cudaDeviceSynchronize());
 if(error.get(1)[0]||out_count.get(1)[0]!=expected.size()||out_origins.get(expected.size())!=([&](){auto mapped=expected;for(auto& i:mapped)i=map(i);return mapped;})()){fprintf(stderr,"origin oracle mismatch device%d width%u mode%u count%u error%u\n",device,width,mode,count,error.get(1)[0]);exit(3);}
 auto actual_sorted=sorted.get(count);for(size_t i=0;i<ordered.size();++i)if(actual_sorted[i]!=ordered[i])exit(3);
 for(size_t i=ordered.size();i<count;++i)if(actual_sorted[i]!=0xffffffffu)exit(3);
 auto actual_hashes=out_hashes.get(expected.size());for(size_t i=0;i<expected.size();++i)if(actual_hashes[i]!=hashes[map(expected[i])])exit(3);
 if(count){
  action.malformed=1;error.put({0});out_count.put({123});
  CUDA_CHECK(generic_sorted_origin_exact(action,count,0,64,cached.p,input.p,sorted.p,flags.p,prefix.p,out_hashes.p,out_origins.p,out_count.p,temporary.p,shape.shared_temporary_bytes,error.p));
  CUDA_CHECK(cudaDeviceSynchronize());if(error.get(1)[0]!=128||out_count.get(1)[0]!=0)exit(4);
 }
 printf("EXACT_ORIGIN_ORACLE bytes=%zu width=%u mode=%u count=%u all_padding=%u unique=%zu\n",sizeof(State),width,mode,count,all_padding,expected.size());
}
int main(){int devices=0;CUDA_CHECK(cudaGetDeviceCount(&devices));if(devices<2)return 5;
 for(int device=0;device<2;++device){CUDA_CHECK(cudaSetDevice(device));
  for(uint32_t width:{2u,17u,25u,129u})for(uint32_t mode=0;mode<4;++mode){
   fixture<uint8_t>(device,width,mode,0,false);fixture<int64_t>(device,width,mode,0,false);
   fixture<uint8_t>(device,width,mode,65,false);fixture<int64_t>(device,width,mode,65,false);
   fixture<uint8_t>(device,width,mode,4097,false);fixture<int64_t>(device,width,mode,4097,false);
   fixture<uint8_t>(device,width,mode,65,false,true);fixture<int64_t>(device,width,mode,65,false,true);
   fixture<uint8_t>(device,width,mode,65,true);fixture<int64_t>(device,width,mode,65,true);
  }
  printf("EXACT_ORIGIN_PRIMITIVE_PASS device=%d\n",device);
 }
}

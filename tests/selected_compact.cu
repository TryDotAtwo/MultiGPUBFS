#include "mgbfs_cuda.h"
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

void trial(unsigned n,bool major,bool bad,bool device_base){
 unsigned moves=3,parents=17,stride=(n+15)&~15u,children=parents*moves;std::vector<unsigned char>g(moves*n*n,0),a(parents*stride,0);
 for(unsigned m=0;m<moves;++m)for(unsigned j=0;j<n;++j)g[(m*n+j)*n+(j+m)%n]=1;
 for(unsigned p=0;p<parents;++p)for(unsigned j=0;j<n;++j)a[p*stride+j]=(p*3+j)%n;
 void*plan=nullptr;char error[512];unsigned weights[]={1,1,1};int status=major?mgbfs_generate_create_macro_variant(n,moves,2,32,g.data(),weights,5,&plan,error,512):mgbfs_generate_create_variant(n,moves,2,32,g.data(),5,&plan,error,512);req(!status,error);
 D<unsigned char>pa(parents*stride),out(64*stride);pa.put(a);D<uint64_t>refs(children+3);D<uint32_t>base(1);base.put({3});D<uint32_t>requests(64),count(1),fatal(1);std::vector<uint64_t>rf(children);std::vector<unsigned>rq;for(unsigned i=0;i<children;++i)rf[i]=children-1-i;for(unsigned i=0;i<64;++i)rq.push_back((i*7)%children);if(device_base){auto shifted=rf;shifted.insert(shifted.begin(),3,999999);refs.put(shifted);}else refs.put(rf);requests.put(rq);count.put({64});
 if(bad){rq[63]=children;requests.put(rq);}
 req(!(device_base?mgbfs_generate_selected_compact_device_base(plan,pa.p,parents,refs.p,base.p,children+3,requests.p,count.p,64,out.p,fatal.p,nullptr):mgbfs_generate_selected_compact(plan,pa.p,parents,refs.p,0,children,requests.p,count.p,64,out.p,fatal.p,nullptr)),"enqueue");ck(cudaDeviceSynchronize());auto result=out.get(64*stride);
 if(bad){req(fatal.get(1)[0]==2,"bad row rejected");req(std::all_of(result.begin(),result.end(),[](unsigned char v){return v==0;}),"no partial writes");}
 else{req(!fatal.get(1)[0],"fatal");for(unsigned i=0;i<64;++i){unsigned child=rf[rq[i]],parent=major?child%parents:child/moves,move=major?child/parents:child%moves;for(unsigned j=0;j<stride;++j)req(result[i*stride+j]==(j<n?a[parent*stride+(j+move)%n]:0),"CPU selected permutation oracle");}}
 mgbfs_generate_destroy(plan);std::cout<<"SELECTED_COMPACT_ORACLE_PASS n="<<n<<" major="<<major<<" bad="<<bad<<"\n";
}
int main(){try{int n;ck(cudaGetDeviceCount(&n));for(int d=0;d<n;++d){ck(cudaSetDevice(d));for(unsigned width:{14u,32u,128u})for(bool major:{false,true})for(bool bad:{false,true})for(bool device_base:{false,true})trial(width,major,bad,device_base);}return 0;}catch(std::exception const&e){std::cerr<<e.what()<<"\n";return 1;}}

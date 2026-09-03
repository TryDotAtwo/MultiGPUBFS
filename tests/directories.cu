#include "directories.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <vector>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void req(bool x,const char*s){if(!x)throw std::runtime_error(s);}
template<class T>struct D{T*p;size_t n;D(size_t n):n(n){ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));}~D(){cudaFree(p);}void put(std::vector<T>v){ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(){std::vector<T>v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}};
struct alignas(16) Key{uint32_t w[4];};
int main(){try{
 D<Key> keys(8),out(8);D<uint32_t>count(1),fatal(1),counts(4);D<MgbfsOwnerRange>dir(4);D<MgbfsBucketJob>jobs(2);
 keys.put({{{1,0,0,0}},{{2,0,0,2u<<30}},{{3,0,0,2u<<30}},{{4,0,0,3u<<30}},{{5,0,0,3u<<30}}});count.put({5});
 req(mgbfs_bucket_directory(keys.p,count.p,8,4,dir.p,fatal.p,nullptr)==0,"directory enqueue");ck(cudaDeviceSynchronize());
 auto r=dir.get();req(!fatal.get()[0]&&r[0].begin==0&&r[0].count==1&&r[1].begin==1&&r[1].count==0&&r[2].begin==1&&r[2].count==2&&r[3].begin==3&&r[3].count==2,"directory ranges");
 counts.put({1,0,2,1});std::vector<Key>a(8);a[0].w[0]=10;a[4].w[0]=20;a[5].w[0]=21;a[6].w[0]=30;keys.put(a);
 req(mgbfs_compact_hash_layer(keys.p,counts.p,4,2,out.p,8,dir.p,count.p,fatal.p,nullptr)==0,"compact enqueue");ck(cudaDeviceSynchronize());
 auto o=out.get();req(count.get()[0]==4&&o[0].w[0]==10&&o[1].w[0]==20&&o[2].w[0]==21&&o[3].w[0]==30,"compact order");
 std::vector<MgbfsBucketJob>j(2);j[0].bucket=2;j[1].bucket=0;jobs.put(j);
 req(mgbfs_bind_owner_jobs(jobs.p,2,counts.p,4,nullptr)==0,"bind enqueue");ck(cudaDeviceSynchronize());auto got=jobs.get();req(got[0].accepted_count==2&&got[1].accepted_count==1,"fresh counts");
 req(mgbfs_compact_hash_layer(keys.p,counts.p,4,2,out.p,3,dir.p,count.p,fatal.p,nullptr)==0,"overflow enqueue");ck(cudaDeviceSynchronize());req(fatal.get()[0]!=0,"missing capacity failure");
 std::puts("DIRECTORIES_PASS");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}

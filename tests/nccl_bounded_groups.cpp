#include <cassert>
#include <algorithm>
#include <cstdio>
#include <map>
#include <tuple>
#include <vector>
#include "../cuda/nccl_transport.cpp"
namespace test {
struct E {int kind,peer,group; size_t off,count;};
std::vector<E> sent,received;
std::vector<unsigned char> a[6];
size_t stride[3]; int depth=0,groups=0,calls=0,peak=0,fail=0;
E record(const void* b,size_t n,int peer,bool send) {
 int found=-1;size_t offset=0;
 for(int k=send?0:3;k<(send?3:6);k++){
  auto address=reinterpret_cast<uintptr_t>(b),base=reinterpret_cast<uintptr_t>(a[k].data());
  if(address>=base && address<base+a[k].size()){found=k%3;offset=address-base;break;}
 }
 assert(found>=0);assert(depth==1);calls++;peak=std::max(peak,calls);
 assert(offset>=size_t(peer)*stride[found]);return {found,peer,groups,offset-size_t(peer)*stride[found],n};
}
}
extern "C" ncclResult_t ncclGroupStart(){assert(test::depth==0);test::depth=1;test::groups++;test::calls=0;return ncclSuccess;}
extern "C" ncclResult_t ncclGroupEnd(){assert(test::depth==1);test::depth=0;return test::fail==2?ncclSystemError:ncclSuccess;}
extern "C" ncclResult_t ncclSend(const void* b,size_t n,ncclDataType_t,int peer,ncclComm_t,cudaStream_t){test::sent.push_back(test::record(b,n,peer,true));return test::fail==1?ncclSystemError:ncclSuccess;}
extern "C" ncclResult_t ncclRecv(void* b,size_t n,ncclDataType_t,int peer,ncclComm_t,cudaStream_t){test::received.push_back(test::record(b,n,peer,false));return test::fail==3?ncclSystemError:ncclSuccess;}
extern "C" ncclResult_t ncclCommGetAsyncError(ncclComm_t,ncclResult_t* r){*r=ncclSuccess;return ncclSuccess;}
extern "C" ncclResult_t ncclCommAbort(ncclComm_t){return ncclSuccess;}
int main(){
 using Key=std::tuple<int,size_t,size_t,int>;
 int cases=0;
 for(unsigned world:{1u,2u,8u,128u})for(unsigned shards:{1u,128u,256u})
 for(unsigned width:{0u,16u})for(unsigned state_bytes:{1u,8u})for(unsigned used:{0u,2u}){
  unsigned cap=4,payload=width*state_bytes;
  test::stride[0]=shards*4;test::stride[1]=size_t(shards)*cap*32;test::stride[2]=size_t(shards)*cap*payload;
  for(int k=0;k<6;k++)test::a[k].resize(std::max(size_t(1),world*test::stride[k%3]));
  std::map<std::pair<unsigned,unsigned>,std::vector<Key>> sends,recvs;
  for(unsigned rank=0;rank<(world<=8?world:1);rank++){
   auto* comm=new Comm;comm->value=reinterpret_cast<ncclComm_t>(1);comm->rank=rank;comm->world=world;
   test::sent.clear();test::received.clear();test::groups=test::peak=0;
   int result=mgbfs_nccl_exchange_bounded_triplets(comm,rank,world,shards,cap,used,payload,state_bytes,test::a[0].data(),test::a[1].data(),test::a[2].data(),test::a[3].data(),test::a[4].data(),test::a[5].data(),nullptr);
   assert(result==0);assert(test::depth==0);
   if(!used && world>1)assert(test::groups==1);
   if(test::peak>256){std::fprintf(stderr,"UNBOUNDED_GROUP world=%u shards=%u width=%u calls=%d\n",world,shards,width,test::peak);return 2;}
   const size_t per_peer=1+(used?size_t(shards)*(1+width):0);
   assert(test::sent.size()==(world-1)*per_peer);assert(test::sent.size()==test::received.size());
   for(auto e:test::sent)sends[{rank,e.peer}].emplace_back(e.kind,e.off,e.count,e.group);
   for(auto e:test::received)recvs[{e.peer,rank}].emplace_back(e.kind,e.off,e.count,e.group);
   comm->value=nullptr;delete comm;
  }
  if(world<=8)assert(sends==recvs);
  cases++;
 }

 // Error paths must close the active group; cancellation stops between groups.
 test::stride[0]=256*4;test::stride[1]=256*4*32;test::stride[2]=256*4*16;
 for(int k=0;k<6;k++)test::a[k].resize(2*test::stride[k%3]);
 for(int mode:{1,2,3,4}){
  Comm comm;comm.value=reinterpret_cast<ncclComm_t>(1);comm.rank=0;comm.world=2;
  test::groups=test::peak=0;test::fail=mode;
  if(mode==4)comm.cancel_requested=[](void*)->int{return test::groups>=1;};
  int code=mgbfs_nccl_exchange_bounded_triplets(&comm,0,2,256,4,2,16,1,test::a[0].data(),test::a[1].data(),test::a[2].data(),test::a[3].data(),test::a[4].data(),test::a[5].data(),nullptr);
  assert(code==(mode==1?3:mode==2?5:mode==3?4:7));assert(test::depth==0);assert(test::groups==1);comm.value=nullptr;
 }
 std::printf("PASS %d geometry cases and 4 error/cancel cases; peer payloads matched; group cap 256\n",cases);
}

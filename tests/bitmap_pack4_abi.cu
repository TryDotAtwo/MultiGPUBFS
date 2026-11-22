#include "../cuda/bitmap_pack4_owner.cuh"
#include "../cuda/bitmap_pack4_route.cuh"
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

int main(int argc,char**argv){
 if(argc!=2){std::fprintf(stderr,"expected path to CUDA library\n");return 2;}
 void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
 if(!lib){std::fprintf(stderr,"dlopen failed: %s\n",dlerror());return 2;}
 using Query=uint32_t(*)(uint32_t);
 auto query=reinterpret_cast<Query>(dlsym(lib,"mgbfs_pack4_bfs_abi_query"));
 if(!query){std::fprintf(stderr,"PACK4_ABI_MISSING\n");return 1;}
 const uint32_t expected[]={1,sizeof(MgbfsPack4Dictionary),sizeof(MgbfsPack4RouteArgs),sizeof(MgbfsPack4OwnerArgs),sizeof(MgbfsPack4SeedArgs)};
 for(uint32_t i=0;i<sizeof(expected)/sizeof(*expected);++i){
  const auto value=query(i);
  if(value!=expected[i]){std::fprintf(stderr,"PACK4_ABI_MISMATCH field=%u actual=%u expected=%u\n",i,value,expected[i]);return 1;}
 }
 for(const char* name:{"mgbfs_pack4_route","mgbfs_pack4_owner_accept","mgbfs_pack4_seed"}){
  if(!dlsym(lib,name)){std::fprintf(stderr,"PACK4_ABI_SYMBOL_MISSING %s\n",name);return 1;}
 }
 if(query(999)!=0){std::fprintf(stderr,"PACK4_ABI_UNKNOWN_FIELD\n");return 1;}
 std::puts("PACK4_ABI_SHARED_LIBRARY_PASS");
 dlclose(lib);
}

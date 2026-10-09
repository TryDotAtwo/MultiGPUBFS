// Diagnostic interposer only. No CUDA/NCCL state, cancellation, allocation or abort ownership.
#include <dlfcn.h>
#include <unistd.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <pthread.h>
namespace {
std::atomic<uintptr_t> owner{0};
std::atomic<const char*> active_api{nullptr};
thread_local unsigned nesting=0;
void observe(const char* api,const char* phase) {
  const auto caller=static_cast<uintptr_t>(pthread_self());
  uintptr_t unset=0;owner.compare_exchange_strong(unset,caller);
  if(owner.load()!=caller)return;
  if(std::strcmp(phase,"enter")==0) {
    if(nesting++==0)active_api.store(api,std::memory_order_release);
  } else if(std::strcmp(phase,"exit")==0 && nesting) {
    if(--nesting==0)active_api.store(nullptr,std::memory_order_release);
  }
}
void mark(const char* api,const char* phase,int code) {
  observe(api,phase);
  if(std::getenv("MGBFS_FAILURE_STAGE_QUIET"))return;
  char text[256]; struct timespec now{}; clock_gettime(CLOCK_MONOTONIC,&now);
  const char* rank=std::getenv("RANK");
  int n=std::snprintf(text,sizeof(text),"MGBFS_FAILURE_STAGE rank=%s api=%s phase=%s code=%d t_ns=%lld\n",
      rank?rank:"unset",api,phase,code,(long long)now.tv_sec*1000000000LL+now.tv_nsec);
  if(n>0 && n<(int)sizeof(text)) { ssize_t ignored=write(STDERR_FILENO,text,(size_t)n); (void)ignored; }
}
}
extern "C" int ncclCommAbort(void* comm) {
  static auto next=reinterpret_cast<int(*)(void*)>(dlsym(RTLD_NEXT,"ncclCommAbort"));
  if(!next) _exit(98);
  mark("ncclCommAbort","enter",0);int result=next(comm);mark("ncclCommAbort","exit",result);return result;
}
extern "C" int cudaDeviceSynchronize() {
  static auto next=reinterpret_cast<int(*)()>(dlsym(RTLD_NEXT,"cudaDeviceSynchronize"));
  if(!next) _exit(98);
  mark("cudaDeviceSynchronize","enter",0);int result=next();mark("cudaDeviceSynchronize","exit",result);return result;
}

extern "C" int ncclGroupStart() {
  static auto next=reinterpret_cast<int(*)()>(dlsym(RTLD_NEXT,"ncclGroupStart"));
  if(!next) _exit(98);
  mark("ncclGroupStart","enter",0);int result=next();mark("ncclGroupStart","exit",result);return result;
}
extern "C" int ncclGroupEnd() {
  static auto next=reinterpret_cast<int(*)()>(dlsym(RTLD_NEXT,"ncclGroupEnd"));
  if(!next) _exit(98);
  mark("ncclGroupEnd","enter",0);int result=next();mark("ncclGroupEnd","exit",result);return result;
}
extern "C" int ncclAllReduce(const void* send,void* recv,size_t count,int datatype,int op,void* comm,void* stream) {
  static auto next=reinterpret_cast<int(*)(const void*,void*,size_t,int,int,void*,void*)>(dlsym(RTLD_NEXT,"ncclAllReduce"));
  if(!next) _exit(98);
  mark("ncclAllReduce","enter",0);int result=next(send,recv,count,datatype,op,comm,stream);mark("ncclAllReduce","exit",result);return result;
}

// Read-only snapshot called by the existing cancellation mirror, not NCCL.
extern "C" const char* mgbfs_failure_stage_snapshot() {
  const char* api=active_api.load(std::memory_order_acquire);
  return api?api:"none";
}

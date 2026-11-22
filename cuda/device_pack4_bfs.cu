#define MGBFS_RESIDENT_IMPLEMENTATION 1
#include "device_pack4_bfs.cuh"
#include <cstring>

namespace {
__global__ void resident_smoke_kernel(uint32_t* result) {
  if(!blockIdx.x&&!threadIdx.x)*result=0x6d676266u;
}
}
// Startup-only: verifies that a real kernel from the loaded library runs on
// this device before a communicator/window or large BFS allocation exists.
// All CUDA error codes remain visible to the caller.
extern "C" int mgbfs_device_pack4_bfs_smoke_v1() {
  uint32_t* device=nullptr;uint32_t host=0;
  auto result=cudaMalloc(reinterpret_cast<void**>(&device),sizeof(host));
  if(result!=cudaSuccess)return 1000+int(result);
  resident_smoke_kernel<<<1,1>>>(device);
  result=cudaGetLastError();
  if(result==cudaSuccess)result=cudaMemcpy(&host,device,sizeof(host),cudaMemcpyDeviceToHost);
  const auto released=cudaFree(device);
  if(result!=cudaSuccess)return 1000+int(result);
  if(released!=cudaSuccess)return 1000+int(released);
  return host==0x6d676266u?0:6;
}
extern "C" uint32_t mgbfs_device_pack4_bfs_abi_query(uint32_t item) {
  switch(item) {
    case 0:return 1;
    case 1:return sizeof(mgbfs_resident::Args);
    case 2:return sizeof(mgbfs_resident::Report);
    case 3:return sizeof(mgbfs_device_pack4::Extent);
    case 4:return sizeof(mgbfs_resident::Control);
    case 5:return sizeof(mgbfs_resident::Capture);
    case 6:return sizeof(mgbfs_resident::Payload)*mgbfs_resident::max_packet_records;
    case 7:return mgbfs_resident::max_lanes;
    case 8:return 2; // payload identity modes 0, 1, 2
    default:return 0;
  }
}
static int validate(const mgbfs_resident::Args* a,uint64_t window_bytes) {
  if(!a||a->abi_version!=1||a->struct_bytes!=sizeof(*a)||
     !a->world||a->world>mgbfs_resident::max_world||a->rank>=a->world||
     a->owner>=a->world||!a->shards||!a->generators||a->banks!=3||
     a->payload_identity>2||!a->job_generation||!a->domain.total||
     !a->domain.symbols||!a->domain.counts||!mgbfs_pack4_valid(a->dictionary)||
     a->dictionary.elements!=a->domain.elements||!a->permutations||
     !a->physical_of_logical||(!a->extents&&a->extent_count)||
     (!a->extent_owner&&a->extent_count)||!a->shard_locks||!a->visited||
     !a->rank_count||!a->layer_sizes||!a->layer_capacity||!a->rank_counts||
     !a->terminal||bool(a->capture)!=bool(a->capture_count)||
     (a->capture&&!a->capture_capacity))return 1;
  MgbfsDeviceMailboxLayoutV1 expected{};
  const uint32_t payload_bytes=a->mailbox.requested_payload_bytes;
  if(payload_bytes<sizeof(mgbfs_resident::Payload)||
     payload_bytes>sizeof(mgbfs_resident::Payload)*mgbfs_resident::max_packet_records||
     payload_bytes%sizeof(mgbfs_resident::Payload))return 2;
  if(!a->mailbox.lanes||a->mailbox.lanes>mgbfs_resident::max_lanes)return 2;
  if(mgbfs_device_mailbox_layout_v1(a->world,a->mailbox.lanes,
       a->mailbox.slots_per_ring,payload_bytes,UINT64_MAX,
       sizeof(expected),&expected)!=MGBFS_MAILBOX_LAYOUT_OK||
     std::memcmp(&expected,&a->mailbox,sizeof(expected))||
     expected.total_bytes>UINT64_MAX-sizeof(mgbfs_resident::Control)||
     window_bytes<sizeof(mgbfs_resident::Control)+expected.total_bytes)return 2;
  const uint64_t expected_len=a->domain.total/a->world+(a->owner<a->domain.total%a->world);
  if(a->owner_len!=expected_len)return 3;
  return 0;
}
extern "C" int mgbfs_device_pack4_bfs_init_v1(void* window,uint64_t window_bytes,
                                                const mgbfs_resident::Args* args,void* raw_stream) {
  int status=validate(args,window_bytes);if(status)return status;
  if(!window||reinterpret_cast<uintptr_t>(window)%64)return 4;
  mgbfs_resident::initialize<<<1,1,0,static_cast<cudaStream_t>(raw_stream)>>>(
      static_cast<mgbfs_resident::Control*>(window),*args);
  const auto result=cudaGetLastError();
  return result==cudaSuccess?0:1000+int(result);
}

extern "C" int mgbfs_device_pack4_lsa_smoke_v1(
    const void* raw_window,const void* raw_comm,const mgbfs_resident::Args* args,
    uint64_t window_bytes,uint32_t* report,void* raw_stream) {
  int status=validate(args,window_bytes);if(status)return status;
  if(!raw_window||!raw_comm||!report||args->mailbox.slots_per_ring!=1)return 4;
  auto window=*static_cast<const ncclWindow_t*>(raw_window);
  auto comm=*static_cast<const ncclDevComm*>(raw_comm);
  if(comm.lsaSize!=int(args->world)||comm.lsaRank!=int(args->rank))return 6;
  auto value=*args;
  void* params[]={&window,&comm,&value,&report};
  auto result=cudaLaunchCooperativeKernel(reinterpret_cast<void*>(mgbfs_resident::peer_smoke),
                                           dim3(1),dim3(mgbfs_resident::threads),
                                           params,0,static_cast<cudaStream_t>(raw_stream));
  return result==cudaSuccess?0:1000+int(result);
}
extern "C" int mgbfs_device_pack4_bfs_launch_v1(
    const void* raw_window,const void* raw_comm,const mgbfs_resident::Args* args,
    uint64_t window_bytes,mgbfs_resident::Report* report,void* raw_stream) {
  int status=validate(args,window_bytes);if(status)return status;
  if(!raw_window||!raw_comm||!report)return 4;
  auto window=*static_cast<const ncclWindow_t*>(raw_window);
  auto comm=*static_cast<const ncclDevComm*>(raw_comm);
  if(comm.lsaSize!=int(args->world)||comm.lsaRank!=int(args->rank))return 6;
  auto value=*args;
  void* params[]={&window,&comm,&value,&report};
  const auto kernel=args->mailbox.lanes<=mgbfs_resident::lanes
      ? reinterpret_cast<void*>(mgbfs_resident::run<mgbfs_resident::lanes>)
      : reinterpret_cast<void*>(mgbfs_resident::run<mgbfs_resident::max_lanes>);
  auto result=cudaLaunchCooperativeKernel(kernel,
                                           dim3(1),dim3(32*(args->mailbox.lanes+2)),
                                           params,0,static_cast<cudaStream_t>(raw_stream));
  return result==cudaSuccess?0:1000+int(result);
}

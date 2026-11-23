#define MGBFS_BATCH_HASH_BFS_IMPLEMENTATION 1
#include "device_batch_hash_bfs.cuh"
#include <limits>

extern "C" uint32_t mgbfs_device_batch_hash_query_v1(uint32_t item) {
  switch(item) {
    case 0:return 1;
    case 1:return sizeof(mgbfs_resident_hash_batch::Args);
    case 2:return sizeof(mgbfs_device_batch::Window);
    case 3:return sizeof(mgbfs_device_batch::Input);
    case 4:return sizeof(mgbfs_device_batch::HashSpan);
    default:return 0;
  }
}
extern "C" int mgbfs_device_batch_hash_geometry_v1(uint32_t* blocks,uint32_t* threads) {
  if(!blocks||!threads)return 1;
  int device=0,sms=0,active=0;
  auto status=cudaGetDevice(&device);
  if(status==cudaSuccess)status=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);
  if(status==cudaSuccess)status=cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active,mgbfs_resident_hash_batch::run,128,0);
  if(status!=cudaSuccess)return 1000+int(status);
  if(sms<1||active<1)return 2;
  *blocks=uint32_t(sms);*threads=128;return 0;
}
extern "C" int mgbfs_device_batch_hash_launch_v1(
    const void* raw_window,const void* raw_comm,const mgbfs_resident_hash_batch::Args* args,
    uint64_t window_bytes,mgbfs_resident::Report* report,void* stream) {
  if(!raw_window||!raw_comm||!args||!report)return 1;
  const auto& a=args->base;
  if(a.abi_version!=1||a.struct_bytes!=sizeof(a)||!a.world||a.world>mgbfs_device_batch::max_world||
     a.rank>=a.world||a.owner>=a.world||!a.shards||!a.generators||a.banks!=3||
     !a.job_generation||
     !mgbfs_pack4_valid(a.dictionary)||(!a.extents&&a.extent_count)||(!a.extent_owner&&a.extent_count)||
     !a.visited||!a.rank_count||!a.layer_sizes||!a.layer_capacity||!a.rank_counts||
     !a.permutations||!a.physical_of_logical||!a.terminal||!args->send_capacity||!args->receive_capacity||
     !args->raw||!args->positions||!args->block_prefix||!args->offsets||!args->receive||!args->grouped||
     !args->shard_counts||!args->shard_starts||!args->spans||!args->parent_prefix||!args->error||
     args->send_capacity>(UINT64_MAX-sizeof(mgbfs_device_batch::Window))/sizeof(mgbfs_device_batch::Payload)||
     window_bytes<sizeof(mgbfs_device_batch::Window)+args->send_capacity*sizeof(mgbfs_device_batch::Payload))return 2;
  const auto window=*static_cast<const ncclWindow_t*>(raw_window);
  const auto comm=*static_cast<const ncclDevComm*>(raw_comm);
  if(comm.lsaRank!=int(a.rank)||comm.lsaSize!=int(a.world))return 3;
  uint32_t blocks=0,threads=0;
  const int geometry=mgbfs_device_batch_hash_geometry_v1(&blocks,&threads);
  if(geometry)return geometry;
  auto value=*args;
  void* params[]={const_cast<void*>(static_cast<const void*>(&window)),const_cast<void*>(static_cast<const void*>(&comm)),&value,&report};
  auto status=cudaLaunchCooperativeKernel(reinterpret_cast<void*>(mgbfs_resident_hash_batch::run),
      dim3(blocks),dim3(threads),params,0,static_cast<cudaStream_t>(stream));
  return status==cudaSuccess?0:1000+int(status);
}

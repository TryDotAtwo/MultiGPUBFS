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
    case 5:return 1; // pack4 hash segment and seed support
    case 6:return 1; // cooperative virtual-bucket owners
    case 7:return 1; // optional no-reader completed-layer table rebuild
    default:return 0;
  }
}
namespace {
__global__ void seed_hash_pack4(MgbfsSegment segment,uint64_t* counts,uint64_t hash,
                                uint64_t word,uint32_t* error) {
  if(blockIdx.x||threadIdx.x)return;
  const mgbfs_device_hash::Extent extent{segment,counts};
  if(!error||!mgbfs_device_hash::valid_extent(extent,0,1,8)||counts[0]||counts[1]||counts[2]) {
    if(error)*error=1;return;
  }
  const uint64_t slot=mgbfs_shared_table_bucket(hash,segment.slots_count);
  if(segment.slots[slot]!=mgbfs_segmented_exact::empty){*error=1;return;}
  static_cast<uint64_t*>(segment.states)[0]=word;
  if(segment.position_item_bytes==4)
    reinterpret_cast<uint32_t*>(segment.positions)[0]=uint32_t(slot);
  else segment.positions[0]=slot;
  segment.slots[slot]=hash&mgbfs_segmented_exact::prefix;
  counts[0]=1;*error=0;
}
}
extern "C" int mgbfs_device_batch_hash_seed_pack4_v1(
    MgbfsSegment segment,uint64_t* counts,uint64_t hash,uint64_t word,uint32_t* error,void* stream) {
  if(!counts||!error)return 1;
  seed_hash_pack4<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(segment,counts,hash,word,error);
  const auto status=cudaGetLastError();
  return status==cudaSuccess?0:1000+int(status);
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
     !a.job_generation||a.reserved0>1||
     !mgbfs_pack4_valid(a.dictionary)||(!a.extents&&a.extent_count)||(!a.extent_owner&&a.extent_count)||
     !a.visited||!a.rank_count||!a.layer_sizes||!a.layer_capacity||!a.rank_counts||
     !a.permutations||!a.physical_of_logical||!a.terminal||!args->send_capacity||!args->receive_capacity||
     !args->raw||!args->positions||!args->block_prefix||!args->offsets||!args->receive||!args->grouped_origins||
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

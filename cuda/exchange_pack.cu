#include "mgbfs_cuda.h"
#include <cuda_runtime.h>
#include "dense_frame_layout.h"
#include "owner_partition.h"
#include <cstdint>
#include <cstring>
namespace {
struct alignas(16) Key { uint32_t w[4]; };
struct HeaderWords { uint32_t words[16]; };
static_assert(sizeof(HeaderWords) == 64);
__global__ void write_prefix(HeaderWords header, uint32_t* out) {
  out[threadIdx.x] = mgbfs_frame_prefix_word(header.words, threadIdx.x);
}
__global__ void gather_frame(MgbfsDenseFrameLayout layout, uint32_t stride,
    uint32_t begin, uint32_t source_count, const uint32_t* hashes,
    const uint64_t* refs, const uint32_t* states, uint32_t* output,
    uint32_t* fatal) {
  const uint64_t step = uint64_t(blockDim.x) * gridDim.x;
  for (uint64_t word = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       word < layout.total_bytes / 4; word += step) {
    uint32_t value = 0;
    if (!mgbfs_dense_frame_word(layout, stride, begin, source_count,
                               hashes, refs, states, word, &value)) {
      atomicExch(fatal, 1u);
    }
    output[word] = value;
  }
}
__global__ void gather_macro_frame(MgbfsMacroFrameLayout layout, uint32_t stride,
    uint32_t begin, uint32_t source_count, uint32_t source_depth,
    uint32_t weight, const uint32_t* hashes, const uint64_t* refs,
    const uint32_t* states, uint32_t* output, uint32_t* fatal) {
  const uint64_t step = uint64_t(blockDim.x) * gridDim.x;
  for (uint64_t word = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       word < layout.total_bytes / 4; word += step) {
    uint32_t value = 0;
    if (!mgbfs_macro_frame_word(layout, stride, begin, source_count,
                                source_depth, weight, hashes, refs, states,
                                word, &value)) {
      atomicExch(fatal, 1u);
    }
    output[word] = value;
  }
}
__global__ void validate_macro_refs(const uint4* refs, uint32_t count,
    uint32_t source_depth, uint32_t target_depth, uint32_t max_weight,
    uint64_t max_state_ref, uint32_t* fatal) {
  const uint64_t step = uint64_t(blockDim.x) * gridDim.x;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += step) {
    const uint4 ref = refs[i];
    const uint64_t state_ref = uint64_t(ref.z) | (uint64_t(ref.w) << 32);
    if (state_ref >= max_state_ref || ref.x != source_depth ||
        ref.y == 0 || ref.y > max_weight ||
        uint64_t(ref.x) + ref.y != target_depth) {
      atomicExch(fatal, 1u);
    }
  }
}
__global__ void split(const Key* keys,uint32_t count,uint32_t* output){
  if(threadIdx.x||blockIdx.x)return;uint32_t lo=0,hi=count;
  while(lo<hi){uint32_t mid=lo+(hi-lo)/2;if((keys[mid].w[3]>>31)==0)lo=mid+1;else hi=mid;}
  output[0]=lo;output[1]=count-lo;
}
__global__ void split_n(const uint32_t* keys,uint32_t count,uint32_t world,uint32_t* output){
  const uint32_t owner=threadIdx.x;
  if(owner<world) output[owner]=mgbfs_owner_boundary(keys,count,owner+1,world)-
      mgbfs_owner_boundary(keys,count,owner,world);
}
__global__ void split_n_device(const uint32_t* keys,const uint32_t* count,
    uint32_t capacity,uint32_t world,uint32_t* output){
  const uint32_t owner=threadIdx.x;
  if(owner>=world)return;
  const uint32_t n=*count;
  if(n>capacity){output[owner]=owner==0?UINT32_MAX:0;return;}
  output[owner]=mgbfs_owner_boundary(keys,n,owner+1,world)-
      mgbfs_owner_boundary(keys,n,owner,world);
}
__global__ void owner_window_from_counts(uint32_t world,uint32_t capacity,
    uint32_t logical_owner,const uint32_t* counts,
    const uint32_t* routed_count,uint32_t* begin,uint32_t* rows){
  if(threadIdx.x||blockIdx.x)return;
  uint64_t total=0,offset=0;
  for(uint32_t owner=0;owner<world;++owner){
    if(owner==logical_owner)offset=total;
    total+=counts[owner];
  }
  if(counts[0]==UINT32_MAX||total>capacity||
     *routed_count>capacity||total!=*routed_count){
    *begin=0;
    *rows=UINT32_MAX;
    return;
  }
  *begin=uint32_t(offset);
  *rows=counts[logical_owner];
}
__global__ void gather_device(const uint4* source,const uint64_t* refs,
    uint4* output,const uint32_t* count,uint32_t capacity,uint32_t chunks,
    uint32_t source_count,uint32_t* owner_counts){
  const uint32_t n=*count;
  if(n>capacity)return;
  const uint64_t step=uint64_t(blockDim.x)*gridDim.x;
  for(uint64_t p=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
      p<uint64_t(capacity)*chunks;p+=step){
    const uint32_t row=p/chunks;
    if(row>=n)continue;
    const uint64_t ref=refs[row];
    if(ref>=source_count){atomicExch(owner_counts,UINT32_MAX);continue;}
    output[p]=source[ref*chunks+p%chunks];
  }
}
} // namespace
extern "C" int mgbfs_frame_write_header(const uint8_t* host_header,
    uint8_t* device_prefix, void* raw_stream) {
  if (!host_header || !device_prefix || uintptr_t(device_prefix) % 256) return 1;
  HeaderWords header{};
  std::memcpy(header.words, host_header, 64);
  write_prefix<<<1,64,0,static_cast<cudaStream_t>(raw_stream)>>>(
      header, reinterpret_cast<uint32_t*>(device_prefix));
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
extern "C" int mgbfs_exchange_pack_frame(uint32_t stride,
    const uint8_t* source_states, uint32_t source_count,
    const void* sorted_hashes, const uint64_t* sorted_refs,
    uint32_t sorted_count, uint32_t begin, uint32_t count,
    uint8_t* output, uint64_t output_capacity, uint32_t* fatal,
    void* raw_stream) {
  MgbfsDenseFrameLayout layout{};
  if (mgbfs_dense_frame_layout(count, stride, &layout) ||
      begin > sorted_count || count > sorted_count - begin ||
      layout.total_bytes > output_capacity || !fatal) return 1;
  if (!count) return 0;
  if (!source_states || !sorted_hashes || !sorted_refs || !output ||
      uintptr_t(source_states) % 16 || uintptr_t(sorted_hashes) % 16 ||
      uintptr_t(sorted_refs) % 8 || uintptr_t(output) % 256) return 1;
  const uint64_t needed = (layout.total_bytes / 4 + 255) / 256;
  const uint32_t blocks = uint32_t(needed > 65535 ? 65535 : needed);
  gather_frame<<<blocks, 256, 0, static_cast<cudaStream_t>(raw_stream)>>>(
      layout, stride, begin, source_count, static_cast<const uint32_t*>(sorted_hashes),
      sorted_refs, reinterpret_cast<const uint32_t*>(source_states),
      reinterpret_cast<uint32_t*>(output), fatal);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
extern "C" int mgbfs_macro_exchange_pack_frame(uint32_t stride,
    uint32_t source_depth, uint32_t weight, const uint8_t* source_states,
    uint32_t source_count, const void* sorted_hashes, const uint64_t* sorted_refs,
    uint32_t sorted_count, uint32_t begin, uint32_t count,
    uint8_t* output, uint64_t output_capacity, uint32_t* fatal,
    void* raw_stream) {
  MgbfsMacroFrameLayout layout{};
  if (!weight || source_depth > UINT32_MAX - weight ||
      mgbfs_macro_frame_layout(count, stride, &layout) ||
      begin > sorted_count || count > sorted_count - begin ||
      layout.total_bytes > output_capacity || !fatal) return 1;
  if (!count) return 0;
  if (!source_states || !sorted_hashes || !sorted_refs || !output ||
      uintptr_t(source_states) % 16 || uintptr_t(sorted_hashes) % 16 ||
      uintptr_t(sorted_refs) % 8 || uintptr_t(output) % 256) return 1;
  const uint64_t needed = (layout.total_bytes / 4 + 255) / 256;
  const uint32_t blocks = uint32_t(needed > 65535 ? 65535 : needed);
  gather_macro_frame<<<blocks, 256, 0, static_cast<cudaStream_t>(raw_stream)>>>(
      layout, stride, begin, source_count, source_depth, weight,
      static_cast<const uint32_t*>(sorted_hashes), sorted_refs,
      reinterpret_cast<const uint32_t*>(source_states),
      reinterpret_cast<uint32_t*>(output), fatal);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
extern "C" int mgbfs_macro_validate_refs(const void* refs, uint32_t count,
    uint32_t source_depth, uint32_t target_depth, uint32_t max_weight,
    uint64_t max_state_ref, uint32_t* fatal, void* raw_stream) {
  if (!fatal || !max_weight || target_depth <= source_depth ||
      uint64_t(target_depth) > uint64_t(source_depth) + max_weight ||
      (count && (!refs || uintptr_t(refs) % 16))) return 1;
  if (!count) return 0;
  const uint32_t blocks = count / 256 + (count % 256 != 0);
  validate_macro_refs<<<blocks, 256, 0, static_cast<cudaStream_t>(raw_stream)>>>(
      static_cast<const uint4*>(refs), count, source_depth, target_depth,
      max_weight, max_state_ref, fatal);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
namespace {
__global__ void gather(const uint4* source,const uint64_t* refs,uint4* output,uint32_t count,uint32_t chunks,uint32_t source_count,uint32_t* owner_counts){
  uint64_t p=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(p>=uint64_t(count)*chunks)return;uint32_t row=p/chunks,chunk=p%chunks;uint64_t ref=refs[row];if(ref>=source_count){atomicExch(owner_counts,UINT32_MAX);return;}output[uint64_t(row)*chunks+chunk]=source[ref*chunks+chunk];
}
}
extern "C" int mgbfs_exchange_pack(uint32_t stride,uint32_t capacity,const uint8_t* source_states,uint32_t source_count,const void* sorted_hashes,const uint64_t* sorted_refs,uint32_t count,uint8_t* packed_states,uint32_t* owner_counts,void* raw_stream){
  if(!stride||stride%16||!capacity||count>capacity||!source_states||!sorted_hashes||!sorted_refs||!packed_states||!owner_counts)return 1;auto stream=static_cast<cudaStream_t>(raw_stream);split<<<1,1,0,stream>>>(static_cast<const Key*>(sorted_hashes),count,owner_counts);if(count)gather<<<(uint64_t(count)*(stride/16)+255)/256,256,0,stream>>>(reinterpret_cast<const uint4*>(source_states),sorted_refs,reinterpret_cast<uint4*>(packed_states),count,stride/16,source_count,owner_counts);return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_exchange_pack_n(uint32_t world,uint32_t stride,uint32_t capacity,
    const uint8_t* source_states,uint32_t source_count,const void* sorted_hashes,
    const uint64_t* sorted_refs,uint32_t count,uint8_t* packed_states,
    uint32_t* owner_counts,void* raw_stream){
  if(!world||(world&(world-1))||world>8||!stride||stride%16||!capacity||count>capacity||
      !source_states||!sorted_hashes||!sorted_refs||!packed_states||!owner_counts)return 1;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  split_n<<<1,8,0,stream>>>(static_cast<const uint32_t*>(sorted_hashes),count,world,owner_counts);
  if(count)gather<<<(uint64_t(count)*(stride/16)+255)/256,256,0,stream>>>(
      reinterpret_cast<const uint4*>(source_states),sorted_refs,
      reinterpret_cast<uint4*>(packed_states),count,stride/16,source_count,owner_counts);
  return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_exchange_pack_device_n(uint32_t world,uint32_t stride,
    uint32_t capacity,const uint8_t* source_states,uint32_t source_count,
    const void* sorted_hashes,const uint64_t* sorted_refs,const uint32_t* count,
    uint8_t* packed_states,uint32_t* owner_counts,void* raw_stream){
  if(!world||(world&(world-1))||world>8||!stride||stride%16||!capacity||
      !source_states||!sorted_hashes||!sorted_refs||!count||!packed_states||
      !owner_counts)return 1;
  auto stream=static_cast<cudaStream_t>(raw_stream);
  split_n_device<<<1,8,0,stream>>>(static_cast<const uint32_t*>(sorted_hashes),
      count,capacity,world,owner_counts);
  const uint64_t words=uint64_t(capacity)*(stride/16);
  const uint32_t blocks=uint32_t((words+255)/256>65535?65535:(words+255)/256);
  gather_device<<<blocks,256,0,stream>>>(reinterpret_cast<const uint4*>(source_states),
      sorted_refs,reinterpret_cast<uint4*>(packed_states),count,capacity,
      stride/16,source_count,owner_counts);
  return cudaGetLastError()==cudaSuccess?0:2;
}
extern "C" int mgbfs_owner_window_from_counts(uint32_t world,
    uint32_t packed_capacity,uint32_t logical_owner,
    const uint32_t* owner_counts,const uint32_t* routed_count,
    uint32_t* begin,uint32_t* rows,
    void* raw_stream){
  if(!world||(world&(world-1))||world>8||!packed_capacity||
      logical_owner>=world||!owner_counts||!routed_count||!begin||!rows)return 1;
  owner_window_from_counts<<<1,1,0,static_cast<cudaStream_t>(raw_stream)>>>(
      world,packed_capacity,logical_owner,owner_counts,routed_count,begin,rows);
  return cudaGetLastError()==cudaSuccess?0:2;
}

#include "mgbfs_cuda.h"
#include <cuda_runtime.h>
#include "dense_frame_layout.h"
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
__global__ void split(const Key* keys,uint32_t count,uint32_t* output){
  if(threadIdx.x||blockIdx.x)return;uint32_t lo=0,hi=count;
  while(lo<hi){uint32_t mid=lo+(hi-lo)/2;if((keys[mid].w[3]>>31)==0)lo=mid+1;else hi=mid;}
  output[0]=lo;output[1]=count-lo;
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
namespace {
__global__ void gather(const uint4* source,const uint64_t* refs,uint4* output,uint32_t count,uint32_t chunks,uint32_t source_count,uint32_t* owner_counts){
  uint64_t p=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(p>=uint64_t(count)*chunks)return;uint32_t row=p/chunks,chunk=p%chunks;uint64_t ref=refs[row];if(ref>=source_count){atomicExch(owner_counts,UINT32_MAX);return;}output[uint64_t(row)*chunks+chunk]=source[ref*chunks+chunk];
}
}
extern "C" int mgbfs_exchange_pack(uint32_t stride,uint32_t capacity,const uint8_t* source_states,uint32_t source_count,const void* sorted_hashes,const uint64_t* sorted_refs,uint32_t count,uint8_t* packed_states,uint32_t* owner_counts,void* raw_stream){
  if(!stride||stride%16||!capacity||count>capacity||!source_states||!sorted_hashes||!sorted_refs||!packed_states||!owner_counts)return 1;auto stream=static_cast<cudaStream_t>(raw_stream);split<<<1,1,0,stream>>>(static_cast<const Key*>(sorted_hashes),count,owner_counts);if(count)gather<<<(uint64_t(count)*(stride/16)+255)/256,256,0,stream>>>(reinterpret_cast<const uint4*>(source_states),sorted_refs,reinterpret_cast<uint4*>(packed_states),count,stride/16,source_count,owner_counts);return cudaGetLastError()==cudaSuccess?0:2;
}

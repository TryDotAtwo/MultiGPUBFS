#include "../mgbfs_cuda.h"
#include "../dense_frame_layout.h"
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
  constexpr uint32_t count = 2, stride = 16;
  MgbfsMacroFrameLayout layout{};
  if (mgbfs_macro_frame_layout(count, stride, &layout) || layout.total_bytes != 768) return 1;
  const std::array<uint8_t, 48> states = {
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
      16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
      32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47};
  const std::array<uint32_t, 8> hashes = {1,2,3,4,5,6,7,8};
  const std::array<uint64_t, 2> refs = {2,0};
  uint8_t *d_states=nullptr, *d_output=nullptr;
  uint32_t *d_hashes=nullptr, *d_fatal=nullptr;
  uint64_t *d_refs=nullptr;
  if (cudaMalloc(&d_states, states.size()) != cudaSuccess ||
      cudaMalloc(&d_hashes, sizeof(hashes)) != cudaSuccess ||
      cudaMalloc(&d_refs, sizeof(refs)) != cudaSuccess ||
      cudaMalloc(&d_output, layout.total_bytes) != cudaSuccess ||
      cudaMalloc(&d_fatal, sizeof(uint32_t)) != cudaSuccess) return 2;
  cudaMemcpy(d_states, states.data(), states.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_hashes, hashes.data(), sizeof(hashes), cudaMemcpyHostToDevice);
  cudaMemcpy(d_refs, refs.data(), sizeof(refs), cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0xff, layout.total_bytes);
  cudaMemset(d_fatal, 0, sizeof(uint32_t));
  if (mgbfs_macro_exchange_pack_frame(stride, 2, 3, d_states, 3, d_hashes,
      d_refs, count, 0, count, d_output, layout.total_bytes, d_fatal, nullptr)) return 3;
  if (cudaDeviceSynchronize() != cudaSuccess) return 4;
  std::vector<uint8_t> output(layout.total_bytes);
  uint32_t fatal=1;
  cudaMemcpy(output.data(), d_output, output.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(&fatal, d_fatal, sizeof(fatal), cudaMemcpyDeviceToHost);
  if (fatal) return 5;
  const auto read32 = [&](size_t offset) {
    return uint32_t(output[offset]) | uint32_t(output[offset+1])<<8 |
           uint32_t(output[offset+2])<<16 | uint32_t(output[offset+3])<<24;
  };
  std::array<uint32_t, 12> state_words{};
  std::memcpy(state_words.data(), states.data(), states.size());
  for (uint64_t word=0;word<layout.total_bytes/4;++word) {
    uint32_t expected=0;
    if (!mgbfs_macro_frame_word(layout, stride, 0, 3, 2, 3,
          hashes.data(), refs.data(), state_words.data(), word, &expected) ||
        read32(word*4)!=expected) return 13;
  }
  for (uint32_t i=0;i<8;++i) if (read32(i*4) != hashes[i]) return 6;
  if (read32(256)!=2 || read32(260)!=3 || read32(264)!=2 || read32(268)!=0 ||
      read32(272)!=2 || read32(276)!=3 || read32(280)!=0 || read32(284)!=0) return 7;
  for (size_t i=0;i<16;++i) {
    if (output[512+i]!=states[32+i] || output[528+i]!=states[i]) return 8;
  }
  for (size_t i=32;i<256;++i)
    if (output[i] || output[256+i] || output[512+i]) return 9;
  const uint64_t bad_ref=3;
  cudaMemcpy(d_refs, &bad_ref, sizeof(bad_ref), cudaMemcpyHostToDevice);
  cudaMemset(d_fatal, 0, sizeof(uint32_t));
  if (mgbfs_macro_exchange_pack_frame(stride, 2, 3, d_states, 3, d_hashes,
      d_refs, count, 0, count, d_output, layout.total_bytes, d_fatal, nullptr)) return 10;
  cudaDeviceSynchronize();
  cudaMemcpy(&fatal, d_fatal, sizeof(fatal), cudaMemcpyDeviceToHost);
  if (fatal != 1) return 11;
  if (mgbfs_macro_exchange_pack_frame(stride, 2, 0, d_states, 3, d_hashes,
      d_refs, count, 0, count, d_output, layout.total_bytes, d_fatal, nullptr) != 1) return 12;
  cudaFree(d_states); cudaFree(d_hashes); cudaFree(d_refs);
  cudaFree(d_output); cudaFree(d_fatal);
  std::puts("MACRO_EXCHANGE_PACK_PASS");
  return 0;
}

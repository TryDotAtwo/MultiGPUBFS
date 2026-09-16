#include "../experiments/library_owner/control_transfer.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
void require(bool condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}
void checked(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
int main() {
  try {
    // This first constructor also supplies an executable RED before GPU setup.
    void* handle = nullptr;
    require(mgbfs_control_transfer_create_v1(nullptr, &handle) == 0 && handle,
            "CONTROL_TRANSFER_CREATE");
    require(mgbfs_control_transfer_destroy_v1(handle) == 0, "DEFAULT_DESTROY");
    cudaStream_t stream;
    checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    require(mgbfs_control_transfer_create_v1(stream, &handle) == 0, "NONBLOCKING_CREATE");
    void* allocation = nullptr;
    checked(cudaMalloc(&allocation, 256));
    auto* control = static_cast<MgbfsOwnerControl*>(allocation);
    auto* extent = reinterpret_cast<MgbfsStateExtent*>(control + 1);
    auto* ring = reinterpret_cast<MgbfsStateRingControl*>(extent + 1);
    auto* count = reinterpret_cast<uint32_t*>(ring + 1);
    for (uint32_t i = 0; i < 32; ++i) {
      MgbfsOwnerControl source{};
      source.stage = 1 + i % 2; source.survivors = 100 + i; source.error = i;
      MgbfsStateExtent expected_extent{};
      expected_extent.begin = 500 + i; expected_extent.granted_rows = 100 + i;
      expected_extent.count = 100 + i; expected_extent.ready = i % 2;
      MgbfsStateRingControl expected_ring{};
      expected_ring.head = 200 + i; expected_ring.tail = 800 + i; expected_ring.fatal = i + 7;
      uint32_t expected_count = i + 41;
      require(mgbfs_control_transfer_upload_v1(handle, &source, control) == 0, "UPLOAD");
      checked(cudaMemcpyAsync(extent, &expected_extent, 64, cudaMemcpyHostToDevice, stream));
      checked(cudaMemcpyAsync(ring, &expected_ring, 64, cudaMemcpyHostToDevice, stream));
      checked(cudaMemcpyAsync(count, &expected_count, 4, cudaMemcpyHostToDevice, stream));
      MgbfsControlSnapshotV1 result{};
      require(mgbfs_control_transfer_read_v1(handle, control, extent, ring,
                                            i % 2 ? nullptr : count, &result) == 0, "READ");
      require(result.control.stage == source.stage && result.control.survivors == source.survivors &&
              result.control.error == i, "CONTROL_FIELDS_INCLUDING_FATAL");
      require(result.extent.begin == expected_extent.begin && result.extent.count == 100 + i &&
              result.extent.granted_rows == 100 + i && result.extent.ready == i % 2, "EXTENT_FIELDS");
      require(result.ring.head == expected_ring.head && result.ring.tail == expected_ring.tail &&
              result.ring.fatal == i + 7, "RING_FIELDS_INCLUDING_FATAL");
      require(result.count == (i % 2 ? 0 : expected_count) && result.reserved == 0, "OPTIONAL_COUNT_RESET");
    }
    MgbfsOwnerControl source{};
    require(mgbfs_control_transfer_upload_v1(handle, &source, control) == 0, "PENDING_UPLOAD");
    require(mgbfs_control_transfer_upload_v1(handle, &source, control) != 0, "REJECT_OVERWRITE_PENDING_DMA");
    MgbfsControlSnapshotV1 result{}; result.count = 99;
    require(mgbfs_control_transfer_read_v1(handle, control, extent, ring, count, &result) != 0 &&
            result.count == 0, "POISONED_READ_CLEARS_OUTPUT");
    require(mgbfs_control_transfer_destroy_v1(handle) == 0, "DESTROY_DRAINS_PENDING_DMA");
    checked(cudaFree(allocation));
    checked(cudaStreamDestroy(stream));
    std::cout << "CONTROL_TRANSFER_PASS\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

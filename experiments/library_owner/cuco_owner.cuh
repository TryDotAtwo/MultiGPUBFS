#pragma once
#include "owner_abi.h"
#include "cuco_index.hpp"
#include "cuco_pool_allocator.hpp"
#include "cuco_workspace_lease.hpp"
#include "owner_source_gather.cuh"
#include <cuco/static_set.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <rmm/device_buffer.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>

namespace mgbfs {
namespace cuco_owner_detail {
inline void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
inline unsigned grid(uint32_t rows) { return std::min(4096u, (rows + 255u) / 256u); }

// At most half the slots can be populated, even if every offered row is unique.
// A full open-addressing table must never be used as a capacity detector.
template<class Ref>
__global__ void insert_rows(Ref set, unsigned storage, uint32_t first, uint32_t rows) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
       i += gridDim.x * blockDim.x)
    set.insert(key_index(storage, first + i));
}
template<class Ref>
__global__ void first_rows(Ref set, uint32_t rows, uint32_t* representatives,
                           uint32_t* minima, uint32_t* error) {
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += gridDim.x * blockDim.x) {
    auto found = set.find(key_index(3, row));
    if (found == set.end()) {
      representatives[row] = UINT32_MAX;
      atomicExch(error, 1u);
    } else {
      uint64_t const index = *found;
      uint64_t const representative = index & ((uint64_t{1} << 62) - 1);
      if ((index >> 62) != 3 || representative >= rows) {
        representatives[row] = UINT32_MAX;
        atomicExch(error, 1u);
      } else {
        representatives[row] = static_cast<uint32_t>(representative);
        atomicMin(minima + representative, row);
      }
    }
  }
}
template<class Ref>
__global__ void survivor_flags(Ref persistent, uint32_t rows,
    uint32_t const* representatives, uint32_t const* minima, uint8_t* flags) {
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += gridDim.x * blockDim.x) {
    uint32_t const representative = representatives[row];
    flags[row] = representative != UINT32_MAX && minima[representative] == row &&
                 !persistent.contains(key_index(3, row));
  }
}
static __global__ void append_keys(uint32_t const* selected, uint32_t rows,
    uint32_t const* candidate, size_t candidate_stride,
    uint32_t* accepted, size_t accepted_stride, uint32_t first) {
  unsigned const word = blockIdx.y;
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += gridDim.x * blockDim.x)
    accepted[word * accepted_stride + first + row] =
        candidate[word * candidate_stride + selected[row]];
}
}

// One fixed temporary allocation set for serialized shard jobs. Persistent
// accepted keys and membership tables are deliberately NOT stored here.
struct CucoWorkspace {
  uint32_t incoming;
  size_t stride, scratch_bytes{0};
  rmm::cuda_stream_view stream;
  int device;
  CucoWorkspaceLease lease;
  rmm::device_buffer candidates, minima, representatives, flags, selected,
                     sources, control, scratch;
  CucoWorkspace(uint32_t rows, rmm::cuda_stream_view s,
                rmm::device_async_resource_ref resource) : incoming(rows), stream(s) {
    if (!rows || rows > INT32_MAX) throw std::runtime_error("OWNER_CAPACITY");
    cuco_owner_detail::check(cudaGetDevice(&device));
    stride = (static_cast<size_t>(rows) + 63) & ~size_t{63};
    auto allocate = [&](size_t bytes) { return rmm::device_buffer{bytes, s, resource}; };
    candidates = allocate(stride * 16);
    minima = allocate(static_cast<size_t>(rows) * 4);
    representatives = allocate(static_cast<size_t>(rows) * 4);
    flags = allocate(rows);
    selected = allocate(static_cast<size_t>(rows) * 4);
    sources = allocate(static_cast<size_t>(rows) * 4);
    control = allocate(8);
    cuco_owner_detail::check(cub::DeviceSelect::Flagged(nullptr, scratch_bytes,
        thrust::counting_iterator<uint32_t>{0}, static_cast<uint8_t*>(flags.data()),
        static_cast<uint32_t*>(selected.data()), static_cast<uint32_t*>(control.data()),
        static_cast<int>(rows), s.value()));
    scratch = allocate(scratch_bytes);
  }
};

// Experimental single-stream owner. No default allocator, table growth or
// fallback. The caller supplies a fixed pool, drains before seal/destruction,
// and does not overlap compares or consumers of the borrowed result. History
// planes remain immutable until seal; accepted planes live through destruction.
// Compare includes one explicit count readback and a final result-ready wait.
// This is a correctness reference, not yet an overlap/performance claim.
class CucoOwner {
  using Allocator = CucoPoolAllocator<uint64_t, rmm::device_async_resource_ref,
                                     rmm::cuda_stream_view>;
  using Set = cuco::static_set<uint64_t, cuco::extent<size_t>,
      cuda::thread_scope_device, IndexKeyEqual,
      cuco::linear_probing<1, IndexKeyHasher>, Allocator, cuco::storage<1>>;
 public:
  CucoOwner(MgbfsLibraryKeysV1 previous, MgbfsLibraryKeysV1 current,
      uint32_t capacity, uint32_t incoming_capacity, rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref resource,
      std::shared_ptr<CucoWorkspace> workspace = {})
      : stream_(stream), resource_(resource), capacity_(capacity),
        incoming_capacity_(incoming_capacity), shared_workspace_(bool(workspace)),
        workspace_(workspace ? std::move(workspace) :
            std::make_shared<CucoWorkspace>(incoming_capacity, stream, resource)),
        candidates_(workspace_->candidates), minima_(workspace_->minima),
        representatives_(workspace_->representatives), flags_(workspace_->flags),
        selected_(workspace_->selected), sources_(workspace_->sources),
        control_(workspace_->control), scratch_(workspace_->scratch) {
    validate(previous);
    validate(current);
    if (!capacity || !incoming_capacity || capacity > INT32_MAX || incoming_capacity > INT32_MAX)
      throw std::runtime_error("OWNER_CAPACITY");
    cuco_owner_detail::check(cudaGetDevice(&device_));
    if (workspace_->incoming != incoming_capacity || workspace_->device != device_ ||
        workspace_->stream.value() != stream_.value())
      throw std::runtime_error("WORKSPACE_CONFIGURATION");
    workspace_->lease.check_idle();
    accepted_stride_ = aligned_words(capacity);
    candidate_stride_ = workspace_->stride;
    accepted_ = buffer(accepted_stride_ * 16);
    IndexKeyViews views{};
    for (unsigned word = 0; word < 4; ++word) {
      views.planes[0][word] = previous.words[word];
      views.planes[1][word] = current.words[word];
      views.planes[2][word] = data(accepted_) + word * accepted_stride_;
      views.planes[3][word] = data(candidates_) + word * candidate_stride_;
    }
    // Individual planes use signed-32-bit row bounds; the combined table size
    // uses size_t so previous + current + maximum accepted cannot wrap u32.
    size_t const persistent_rows = static_cast<size_t>(previous.rows) + current.rows + capacity;
    persistent_ = make_set(persistent_rows, views);
    transient_ = make_set(incoming_capacity, views);
    scratch_bytes_ = workspace_->scratch_bytes;
    insert(*persistent_, 0, 0, previous.rows);
    insert(*persistent_, 1, 0, current.rows);
  }
  CucoOwner(CucoOwner const&) = delete;
  CucoOwner& operator=(CucoOwner const&) = delete;
  ~CucoOwner() { workspace_->lease.abort(this); }

  MgbfsLibrarySurvivorsV1 compare(uint64_t epoch, MgbfsLibraryCandidatesV1 input) {
    try {
      check_device();
      if (poisoned_ || sealed_ || pending_ || (has_epoch_ && epoch <= last_epoch_))
        throw std::runtime_error("OWNER_ORDER");
      validate(input.keys);
      auto const rows = input.keys.rows;
      if (rows > incoming_capacity_) throw std::runtime_error("OWNER_INPUT_CAPACITY");
      if (rows && !input.source_indices) throw std::runtime_error("OWNER_NULL_SOURCE");
      if (shared_workspace_) workspace_->lease.acquire(this, epoch);
      staged_count_ = 0;
      if (rows) {
        // All old readers precede this clear on the exclusive owner stream.
        // Only the transient set ever holds tag-3 indices.
        transient_->clear_async(cuda::stream_ref{stream_.value()});
        for (unsigned word = 0; word < 4; ++word)
          cuco_owner_detail::check(cudaMemcpyAsync(data(candidates_) + word * candidate_stride_,
              input.keys.words[word], static_cast<size_t>(rows) * 4,
              cudaMemcpyDeviceToDevice, stream_.value()));
        cuco_owner_detail::check(cudaMemsetAsync(minima_.data(), 0xff,
            static_cast<size_t>(rows) * 4, stream_.value()));
        cuco_owner_detail::check(cudaMemsetAsync(control_.data(), 0, 8, stream_.value()));
        insert(*transient_, 3, 0, rows);
        cuco_owner_detail::first_rows<<<cuco_owner_detail::grid(rows), 256, 0, stream_.value()>>>(
            transient_->ref(cuco::find), rows, data(representatives_), data(minima_), data(control_) + 1);
        cuco_owner_detail::check(cudaGetLastError());
        cuco_owner_detail::survivor_flags<<<cuco_owner_detail::grid(rows), 256, 0, stream_.value()>>>(
            persistent_->ref(cuco::contains), rows, data(representatives_), data(minima_),
            static_cast<uint8_t*>(flags_.data()));
        cuco_owner_detail::check(cudaGetLastError());
        size_t bytes = scratch_bytes_;
        cuco_owner_detail::check(cub::DeviceSelect::Flagged(scratch_.data(), bytes,
            thrust::counting_iterator<uint32_t>{0}, static_cast<uint8_t*>(flags_.data()),
            data(selected_), data(control_), static_cast<int>(rows), stream_.value()));
        gather_owner_sources<<<cuco_owner_detail::grid(rows), 256, 0, stream_.value()>>>(
            data(selected_), data(control_), rows, input.source_indices, data(sources_));
        cuco_owner_detail::check(cudaGetLastError());
        uint32_t counts[2]{};
        cuco_owner_detail::check(cudaMemcpyAsync(counts, control_.data(), sizeof(counts),
            cudaMemcpyDeviceToHost, stream_.value()));
        stream_.synchronize();
        if (counts[1] || counts[0] > rows) throw std::runtime_error("OWNER_INDEX_CORRUPTION");
        if (counts[0] > capacity_ - count_) throw std::runtime_error("OWNER_CAPACITY");
        staged_count_ = counts[0];
      }
      pending_ = true;
      pending_epoch_ = epoch;
      return {data(sources_), epoch, staged_count_, 0};
    } catch (...) { workspace_->lease.abort(this); poisoned_ = true; throw; }
  }

  void commit(uint64_t epoch, uint32_t granted) {
    try {
      check_device();
      if (poisoned_ || sealed_ || !pending_ || epoch != pending_epoch_)
        throw std::runtime_error("OWNER_ORDER");
      if (granted < staged_count_ || staged_count_ > capacity_ - count_)
        throw std::runtime_error("OWNER_CREDIT");
      if (staged_count_) {
        cuco_owner_detail::append_keys<<<dim3(cuco_owner_detail::grid(staged_count_), 4),
            256, 0, stream_.value()>>>(data(selected_), staged_count_, data(candidates_),
            candidate_stride_, data(accepted_), accepted_stride_, count_);
        cuco_owner_detail::check(cudaGetLastError());
        // Never insert the transient representative. Copy completes before the
        // next kernel can expose a stable accepted index to membership probes.
        insert(*persistent_, 2, count_, staged_count_);
        count_ += staged_count_;
      }
      pending_ = false;
      has_epoch_ = true;
      last_epoch_ = epoch;
      if (shared_workspace_) workspace_->lease.commit(this, epoch);
    } catch (...) { workspace_->lease.abort(this); poisoned_ = true; throw; }
  }

  // The caller has completed ALL GPU readers, not only accepted-key writes.
  void complete(uint64_t epoch) {
    try {
      check_device();
      if (poisoned_ || pending_ || !has_epoch_ || last_epoch_ != epoch)
        throw std::runtime_error("OWNER_ORDER");
      if (shared_workspace_) workspace_->lease.complete(this, epoch);
    } catch (...) { workspace_->lease.abort(this); poisoned_ = true; throw; }
  }

  MgbfsLibraryKeysV1 export_committed() {
    try {
      check_device();
      if (poisoned_ || pending_) throw std::runtime_error("OWNER_ORDER");
      MgbfsLibraryKeysV1 keys{};
      keys.rows = count_;
      for (unsigned word = 0; word < 4; ++word)
        keys.words[word] = data(accepted_) + word * accepted_stride_;
      return keys;
    } catch (...) { poisoned_ = true; throw; }
  }

  void seal() {
    try {
      check_device();
      if (poisoned_ || pending_ || sealed_) throw std::runtime_error("OWNER_ORDER");
      if (shared_workspace_) workspace_->lease.check_idle();
      transient_.reset();
      persistent_.reset();
      sealed_ = true;
    } catch (...) { poisoned_ = true; throw; }
  }

 private:
  static size_t aligned_words(uint32_t rows) { return (static_cast<size_t>(rows) + 63) & ~size_t{63}; }
  static uint32_t* data(rmm::device_buffer& value) { return static_cast<uint32_t*>(value.data()); }
  static void validate(MgbfsLibraryKeysV1 keys) {
    if (keys.reserved || keys.rows > INT32_MAX) throw std::runtime_error("OWNER_SCHEMA");
    for (auto word : keys.words)
      if (keys.rows && !word) throw std::runtime_error("OWNER_NULL_KEYS");
  }
  void check_device() const {
    int device;
    cuco_owner_detail::check(cudaGetDevice(&device));
    if (device != device_) throw std::runtime_error("OWNER_DEVICE");
  }
  rmm::device_buffer buffer(size_t bytes) { return rmm::device_buffer{bytes, stream_, resource_}; }
  std::unique_ptr<Set> make_set(size_t maximum_rows, IndexKeyViews views) {
    if (maximum_rows > std::numeric_limits<size_t>::max() / 2)
      throw std::runtime_error("OWNER_TABLE_CAPACITY");
    return std::make_unique<Set>(cuco::extent<size_t>{std::max(size_t{2}, maximum_rows * 2)},
        cuco::empty_key<uint64_t>{empty_index}, IndexKeyEqual{views},
        cuco::linear_probing<1, IndexKeyHasher>{IndexKeyHasher{views}},
        cuco::cuda_thread_scope<cuda::thread_scope_device>{}, cuco::storage<1>{},
        Allocator{resource_}, cuda::stream_ref{stream_.value()});
  }
  void insert(Set& set, unsigned storage, uint32_t first, uint32_t rows) {
    if (!rows) return;
    cuco_owner_detail::insert_rows<<<cuco_owner_detail::grid(rows), 256, 0, stream_.value()>>>(
        set.ref(cuco::insert), storage, first, rows);
    cuco_owner_detail::check(cudaGetLastError());
  }
  rmm::cuda_stream_view stream_;
  rmm::device_async_resource_ref resource_;
  uint32_t capacity_, incoming_capacity_, count_{0}, staged_count_{0};
  size_t accepted_stride_{0}, candidate_stride_{0}, scratch_bytes_{0};
  int device_{0};
  uint64_t pending_epoch_{0}, last_epoch_{0};
  bool pending_{false}, has_epoch_{false}, poisoned_{false}, sealed_{false};
  bool shared_workspace_;
  std::shared_ptr<CucoWorkspace> workspace_;
  rmm::device_buffer accepted_;
  rmm::device_buffer &candidates_, &minima_, &representatives_, &flags_,
                     &selected_, &sources_, &control_, &scratch_;
  // Sets must be destroyed before the immutable views they reference.
  std::unique_ptr<Set> persistent_, transient_;
};
}

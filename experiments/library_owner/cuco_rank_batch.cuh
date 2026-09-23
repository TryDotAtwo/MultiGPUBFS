#pragma once
#include "cuco_owner.cuh"
#include "../../cuda/state_commit.h"
#include <memory>
#include <vector>

namespace mgbfs {
struct CucoRankDeviceBatch {
  const uint32_t* high_words;
  const uint32_t* valid_rows;
  const uint32_t* selected;
  const uint32_t* selected_count;
  const uint32_t* source_indices;
  const uint32_t* accepted_counts;
  const uint32_t* accepted_capacities;
  uint32_t* shard_counts;
  uint32_t* shard_offsets;
};

/* One stream-ordered rank batch across distinct persistent shard tables.
 * Hot-path counts and capacity decisions stay on the device. The caller
 * enqueues shard counts and StateRing reservation between compare and commit,
 * then keeps all result readers ordered before complete().
 */
class CucoRankBatch {
 public:
  CucoRankBatch(std::vector<MgbfsLibraryKeysV1> previous,
      std::vector<MgbfsLibraryKeysV1> current,
      std::vector<uint32_t> accepted_capacities, uint32_t incoming_capacity,
      uint32_t logical_owner, uint32_t world, rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref resource);
  CucoRankBatch(CucoRankBatch const&) = delete;
  CucoRankBatch& operator=(CucoRankBatch const&) = delete;
  ~CucoRankBatch();
  CucoRankDeviceBatch compare(uint64_t epoch, MgbfsLibraryCandidatesV1 input,
      const uint32_t* valid_rows, MgbfsOwnerControl* owner,
      MgbfsStateRingControl* ring);
  void commit(uint64_t epoch, MgbfsOwnerControl* owner,
      MgbfsStateRingControl* ring, const MgbfsStateExtent* extent);
  void complete(uint64_t epoch);
  MgbfsLibraryKeysV1 export_shard(uint32_t shard, uint32_t rows) const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}

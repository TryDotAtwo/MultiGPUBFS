#pragma once
#include <cudf/table/table.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/stream_compaction.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <memory>
#include <vector>
#include <optional>

namespace mgbfs {
// Experimental single-shard owner. All library allocations must use the caller's
// already-installed fixed RMM pool, which outlives this object. History columns
// are borrowed and must remain alive/immutable through the last owner operation.
// One serialized writer and one stream per owner. Library host synchronizations
// are not hidden: this backend must earn its place in an end-to-end benchmark.
class CudfOwner {
 public:
  CudfOwner(cudf::table_view history, cudf::size_type capacity,
            rmm::cuda_stream_view stream,
            std::optional<cudf::table_view> current = std::nullopt)
      : capacity_(capacity), stream_(stream) {
    if (capacity < 0) throw std::runtime_error("OWNER_CAPACITY");
    schema(history, 4);
    if (current) schema(*current, 4);
    cuda_ok(cudaGetDevice(&device_));
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (int c = 0; c < 4; ++c) {
      columns.push_back(cudf::make_numeric_column(
          cudf::data_type{cudf::type_id::UINT32}, capacity,
          cudf::mask_state::UNALLOCATED, stream_));
    }
    accepted_ = std::make_unique<cudf::table>(std::move(columns));
    if (history.num_rows()) history_ = index(history);
    if (current && current->num_rows()) current_history_ = index(*current);
  }
  CudfOwner(CudfOwner const&) = delete;
  CudfOwner& operator=(CudfOwner const&) = delete;

  // Input: four UINT32 hash planes plus UINT32 source ordinal. Result owns its
  // compact keys/ordinals; borrowed view is valid until the next compare/destruct.
  cudf::table_view compare(cudf::table_view incoming) {
    try {
      check_device();
      if (poisoned_ || pending_ || sealed_) throw std::runtime_error("OWNER_ORDER");
      schema(incoming, 5);
      staged_ = cudf::stable_distinct(incoming, {0,1,2,3},
          cudf::duplicate_keep_option::KEEP_FIRST, cudf::null_equality::EQUAL,
          cudf::nan_equality::ALL_EQUAL, stream_);
      if (history_) staged_ = exclude(staged_->view(), *history_);
      if (current_history_) staged_ = exclude(staged_->view(), *current_history_);
      if (count_) {
        if (!next_) next_ = index(accepted_view());
        staged_ = exclude(staged_->view(), *next_);
      }
      if (staged_->num_rows() > capacity_ - count_)
        throw std::runtime_error("OWNER_CAPACITY");
      pending_ = true;
      return staged_->view();
    } catch (...) { poisoned_ = true; throw; }
  }

  // Caller first reserves actual StateRing/materialization/archive credits.
  // This enqueues publication on stream_; caller must wait for its completion
  // event before publishing host metadata or releasing any external resources.
  void commit(cudf::size_type granted) {
    try {
      check_device();
      if (poisoned_ || !pending_ || sealed_) throw std::runtime_error("OWNER_ORDER");
      auto const rows = staged_->num_rows();
      if (granted < rows || rows > capacity_ - count_)
        throw std::runtime_error("OWNER_CREDIT");
      if (rows) {
        // Invalidate the index before appending. It is rebuilt only when another
        // batch probes this shard, not for every unrelated shard or whole layer.
        next_.reset();
        for (int c = 0; c < 4; ++c) {
          cuda_ok(cudaMemcpyAsync(
              accepted_->get_column(c).mutable_view().data<uint32_t>() + count_,
              staged_->view().column(c).data<uint32_t>(),
              static_cast<std::size_t>(rows) * sizeof(uint32_t),
              cudaMemcpyDeviceToDevice, stream_.value()));
        }
        count_ += rows;
      }
      pending_ = false;
    } catch (...) { poisoned_ = true; throw; }
  }

  // Stream-ordered appended count, NOT evidence of completed device execution.
  cudf::size_type accepted_count() const { return count_; }

  // Caller has drained all operations and external result consumers. After
  // releasing these indexes, borrowed history may be overwritten for rotation.
  void seal() {
    try {
      check_device();
      if (poisoned_ || pending_ || sealed_) throw std::runtime_error("OWNER_ORDER");
      history_.reset();
      current_history_.reset();
      next_.reset();
      staged_.reset();
      sealed_ = true;
    } catch (...) { poisoned_ = true; throw; }
  }

  // Borrow committed append-order keys; stream completion remains caller-owned.
  cudf::table_view export_committed() {
    try {
      check_device();
      if (poisoned_ || pending_) throw std::runtime_error("OWNER_ORDER");
      return accepted_view();
    } catch (...) { poisoned_ = true; throw; }
  }

 private:
  static void cuda_ok(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  }
  static void schema(cudf::table_view table, int columns) {
    if (table.num_columns() != columns) throw std::runtime_error("OWNER_SCHEMA");
    for (auto const& col : table) {
      if (col.type().id() != cudf::type_id::UINT32 || col.nullable())
        throw std::runtime_error("OWNER_SCHEMA");
    }
  }
  void check_device() const {
    int device;
    cuda_ok(cudaGetDevice(&device));
    if (device != device_) throw std::runtime_error("OWNER_DEVICE");
  }
  std::unique_ptr<cudf::filtered_join> index(cudf::table_view keys) {
    return std::make_unique<cudf::filtered_join>(keys, cudf::null_equality::EQUAL,
        cudf::set_as_build_table::RIGHT, stream_);
  }
  cudf::table_view accepted_view() const {
    std::vector<cudf::column_view> columns;
    for (auto const& col : accepted_->view()) {
      columns.emplace_back(col.type(), count_, col.data<uint32_t>(), nullptr, 0, 0);
    }
    return cudf::table_view(columns);
  }
  std::unique_ptr<cudf::table> exclude(cudf::table_view input,
                                       cudf::filtered_join const& history) {
    auto rows = history.anti_join(input.select({0,1,2,3}), stream_);
    cudf::column_view map{cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(rows->size()), rows->data(), nullptr, 0, 0, {}};
    return cudf::gather(input, map, cudf::out_of_bounds_policy::DONT_CHECK, stream_);
  }

  cudf::size_type capacity_, count_{0};
  rmm::cuda_stream_view stream_;
  int device_{0};
  bool pending_{false}, poisoned_{false}, sealed_{false};
  std::unique_ptr<cudf::table> accepted_, staged_;
  // Destroy indexes before their referenced storage.
  std::unique_ptr<cudf::filtered_join> history_, current_history_, next_;
};
}

#pragma once
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <stdexcept>

namespace mgbfs {
// Experimental single-shard owner; GPU conformance tests precede implementation.
class CudfOwner {
 public:
  CudfOwner(cudf::table_view, cudf::size_type, rmm::cuda_stream_view) {}
  cudf::table_view compare(cudf::table_view) {
    throw std::runtime_error("OWNER_NOT_IMPLEMENTED");
  }
  void commit(cudf::size_type) { throw std::runtime_error("OWNER_NOT_IMPLEMENTED"); }
  cudf::size_type accepted_count() const { return 0; }
};
}

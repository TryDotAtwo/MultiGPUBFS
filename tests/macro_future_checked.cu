#include "mgbfs_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {
struct alignas(16) Key { uint32_t word[4]; };
bool ok(cudaError_t status) { return status == cudaSuccess; }
template <class T> bool alloc(T** out, size_t rows) {
  return ok(cudaMalloc(out, sizeof(T) * rows));
}
}

int main() {
  void* plan = nullptr;
  char error[256]{};
  if (mgbfs_future_merge_create(16, 4, 1, &plan, error, sizeof(error))) {
    std::fprintf(stderr, "FUTURE_CREATE: %s\n", error);
    return 1;
  }
  void* settle_plan = nullptr;
  if (mgbfs_macro_settle_create(4, 4, 4, &settle_plan, error, sizeof(error))) {
    std::fprintf(stderr, "SETTLE_CREATE: %s\n", error);
    return 13;
  }
  uint8_t *future_states = nullptr, *source_states = nullptr;
  Key *future_hashes = nullptr, *incoming_hashes = nullptr;
  MgbfsFrontierState* state = nullptr;
  uint64_t* refs = nullptr;
  uint32_t *count = nullptr, *input_fatal = nullptr;
  Key *history = nullptr, *survivors = nullptr;
  uint32_t *history_counts = nullptr, *survivor_count = nullptr;
  uint64_t* survivor_refs = nullptr;
  MgbfsMacroSettleState* settled_state = nullptr;
  if (!alloc(&future_states, 64) || !alloc(&source_states, 16) ||
      !alloc(&future_hashes, 4) || !alloc(&incoming_hashes, 1) ||
      !alloc(&state, 1) || !alloc(&refs, 1) || !alloc(&count, 1) ||
      !alloc(&input_fatal, 1) || !alloc(&history, 16) ||
      !alloc(&history_counts, 4) || !alloc(&survivors, 4) ||
      !alloc(&survivor_refs, 4) || !alloc(&survivor_count, 1) ||
      !alloc(&settled_state, 1)) return 2;
  const uint8_t source[16] = {3, 1, 4, 1, 5, 9};
  const Key key = {{9, 0, 0, 0}};
  const uint64_t identity_ref = 0;
  const uint32_t one = 1;
  const uint32_t zero = 0;
  if (!ok(cudaMemset(future_states, 0xa5, 64)) ||
      !ok(cudaMemset(future_hashes, 0xa5, 64)) ||
      !ok(cudaMemset(state, 0, sizeof(*state))) ||
      !ok(cudaMemcpy(source_states, source, 16, cudaMemcpyHostToDevice)) ||
      !ok(cudaMemcpy(incoming_hashes, &key, 16, cudaMemcpyHostToDevice)) ||
      !ok(cudaMemcpy(refs, &identity_ref, 8, cudaMemcpyHostToDevice)) ||
      !ok(cudaMemcpy(count, &one, 4, cudaMemcpyHostToDevice)) ||
      !ok(cudaMemcpy(input_fatal, &zero, 4, cudaMemcpyHostToDevice)) ||
      !ok(cudaMemset(history, 0, 16 * 16)) ||
      !ok(cudaMemset(history_counts, 0, 4 * 4)) ||
      !ok(cudaMemset(settled_state, 0, sizeof(*settled_state)))) return 3;
  if (mgbfs_future_merge_run_bounded_checked(plan, future_states, future_hashes,
      state, 0, source_states, 1, incoming_hashes, refs, count, 1,
      input_fatal, nullptr) ||
      mgbfs_macro_settle_run_frontier(settle_plan, future_hashes, refs, state,
          history, history_counts, survivors, survivor_refs, survivor_count,
          settled_state, 1, nullptr) || !ok(cudaDeviceSynchronize())) return 4;
  MgbfsFrontierState published{};
  Key actual_key{};
  uint8_t actual_state[16]{};
  if (!ok(cudaMemcpy(&published, state, sizeof(published), cudaMemcpyDeviceToHost)) ||
      !ok(cudaMemcpy(&actual_key, future_hashes, 16, cudaMemcpyDeviceToHost)) ||
      !ok(cudaMemcpy(actual_state, future_states, 16, cudaMemcpyDeviceToHost))) return 5;
  if (published.count != 1 || published.fatal || actual_key.word[0] != 9) return 6;
  MgbfsMacroSettleState settled{};
  uint32_t survivor_rows = 0;
  if (!ok(cudaMemcpy(&settled, settled_state, sizeof(settled), cudaMemcpyDeviceToHost)) ||
      !ok(cudaMemcpy(&survivor_rows, survivor_count, 4, cudaMemcpyDeviceToHost))) return 14;
  if (settled.fatal || settled.count != 1 || settled.last_epoch != 1 ||
      survivor_rows != 1) return 15;
  for (int i = 0; i < 16; ++i) if (actual_state[i] != source[i]) return 7;
  const Key poison_key = {{2, 0, 0, 0}};
  const uint8_t poison_state[16] = {8};
  if (!ok(cudaMemcpy(incoming_hashes, &poison_key, 16, cudaMemcpyHostToDevice)) ||
      !ok(cudaMemcpy(source_states, poison_state, 16, cudaMemcpyHostToDevice)) ||
      !ok(cudaMemcpy(input_fatal, &one, 4, cudaMemcpyHostToDevice))) return 8;
  if (mgbfs_future_merge_run_bounded_checked(plan, future_states, future_hashes,
      state, 1, source_states, 1, incoming_hashes, refs, count, 1,
      input_fatal, nullptr) ||
      mgbfs_macro_settle_run_frontier(settle_plan, future_hashes, refs, state,
          history, history_counts, survivors, survivor_refs, survivor_count,
          settled_state, 2, nullptr) || !ok(cudaDeviceSynchronize())) return 9;
  if (!ok(cudaMemcpy(&published, state, sizeof(published), cudaMemcpyDeviceToHost)) ||
      !ok(cudaMemcpy(&actual_key, future_hashes, 16, cudaMemcpyDeviceToHost)) ||
      !ok(cudaMemcpy(actual_state, future_states, 16, cudaMemcpyDeviceToHost))) return 10;
  if (published.fatal != 4 || published.count != 1 || actual_key.word[0] != 9) return 11;
  if (!ok(cudaMemcpy(&settled, settled_state, sizeof(settled), cudaMemcpyDeviceToHost)) ||
      !ok(cudaMemcpy(&survivor_rows, survivor_count, 4, cudaMemcpyDeviceToHost))) return 16;
  if (settled.fatal != 4 || settled.last_epoch != 1 || survivor_rows != 0) return 17;
  for (int i = 0; i < 16; ++i) if (actual_state[i] != source[i]) return 12;
  cudaFree(input_fatal); cudaFree(count); cudaFree(refs); cudaFree(state);
  cudaFree(settled_state); cudaFree(survivor_count); cudaFree(survivor_refs);
  cudaFree(survivors); cudaFree(history_counts); cudaFree(history);
  cudaFree(incoming_hashes); cudaFree(future_hashes);
  cudaFree(source_states); cudaFree(future_states);
  mgbfs_future_merge_destroy(plan);
  mgbfs_macro_settle_destroy(settle_plan);
  std::puts("MACRO_FUTURE_CHECKED_PASS");
  return 0;
}

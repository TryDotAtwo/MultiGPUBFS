#include "graph_window.h"
#include <cuda_runtime.h>
#include <memory>
#include <new>
#include <initializer_list>

namespace {
struct Window {
  int device{-1};
  cudaEvent_t fork{}, joins[2]{};
  cudaStream_t owner{}, generation{}, exchange{};
  cudaGraphExec_t executable{};
  bool capturing{false};
  uint64_t launches{}, full_windows{}, batches{}, updates{}, rebuilds{};
  ~Window() {
    if (capturing) {
      cudaGraph_t abandoned{};
      cudaStreamEndCapture(owner, &abandoned);
      if (abandoned) cudaGraphDestroy(abandoned);
    }
    if (executable) cudaGraphExecDestroy(executable);
    if (fork) cudaEventDestroy(fork);
    for (auto event : joins) if (event) cudaEventDestroy(event);
  }
  cudaError_t context() const {
    int current{-1};
    auto status = cudaGetDevice(&current);
    return status != cudaSuccess ? status :
           current == device ? cudaSuccess : cudaErrorInvalidDevice;
  }
};
struct Graph {
  cudaGraph_t handle{};
  ~Graph() { if (handle) cudaGraphDestroy(handle); }
};
}

extern "C" int mgbfs_batch_graph_create_v1(void** out) {
  if (!out) return cudaErrorInvalidValue;
  *out = nullptr;
  std::unique_ptr<Window> window(new (std::nothrow) Window);
  if (!window) return cudaErrorMemoryAllocation;
  auto status = cudaGetDevice(&window->device);
  if (status != cudaSuccess) return status;
  for (auto event : {&window->fork, &window->joins[0], &window->joins[1]}) {
    status = cudaEventCreateWithFlags(event, cudaEventDisableTiming);
    if (status != cudaSuccess) return status;
  }
  *out = window.release();
  return cudaSuccess;
}

extern "C" int mgbfs_batch_graph_begin_v1(void* handle, void* owner,
                                         void* generation, void* exchange) {
  if (!handle || !owner || !generation || !exchange ||
      owner == generation || owner == exchange || generation == exchange)
    return cudaErrorInvalidValue;
  auto& w = *static_cast<Window*>(handle);
  if (w.capturing) return cudaErrorIllegalState;
  auto status = w.context();
  if (status != cudaSuccess) return status;
  w.owner = static_cast<cudaStream_t>(owner);
  w.generation = static_cast<cudaStream_t>(generation);
  w.exchange = static_cast<cudaStream_t>(exchange);
  status = cudaStreamBeginCapture(w.owner, cudaStreamCaptureModeThreadLocal);
  if (status != cudaSuccess) return status;
  w.capturing = true;
  status = cudaEventRecord(w.fork, w.owner);
  if (status == cudaSuccess) status = cudaStreamWaitEvent(w.generation, w.fork, 0);
  if (status == cudaSuccess) status = cudaStreamWaitEvent(w.exchange, w.fork, 0);
  return status;
}

extern "C" int mgbfs_batch_graph_submit_v1(void* handle, uint32_t batches) {
  if (!handle || batches == 0 || batches > 32) return cudaErrorInvalidValue;
  auto& w = *static_cast<Window*>(handle);
  if (!w.capturing) return cudaErrorIllegalState;
  auto status = w.context();
  if (status != cudaSuccess) return status;
  status = cudaEventRecord(w.joins[0], w.generation);
  if (status == cudaSuccess) status = cudaEventRecord(w.joins[1], w.exchange);
  if (status == cudaSuccess) status = cudaStreamWaitEvent(w.owner, w.joins[0], 0);
  if (status == cudaSuccess) status = cudaStreamWaitEvent(w.owner, w.joins[1], 0);
  Graph graph;
  // End even an invalidated capture to restore streams before unwinding.
  auto ended = cudaStreamEndCapture(w.owner, &graph.handle);
  w.capturing = false;
  if (status != cudaSuccess) return status;
  if (ended != cudaSuccess) return ended;
  if (w.executable) {
    cudaGraphExecUpdateResultInfo info{};
    status = cudaGraphExecUpdate(w.executable, graph.handle, &info);
    if (status == cudaSuccess && info.result == cudaGraphExecUpdateSuccess) {
      ++w.updates;
    } else if (status == cudaErrorGraphExecUpdateFailure) {
      // The failed update also sets the thread's last-error slot. Consume the
      // handled topology failure before later kernel launch wrappers inspect it.
      auto last = cudaGetLastError();
      if (last != cudaSuccess && last != cudaErrorGraphExecUpdateFailure)
        return last;
      // Physical frontier wraps / final short windows can change topology.
      // Destroying an in-flight exec frees it asynchronously after completion.
      status = cudaGraphExecDestroy(w.executable);
      w.executable = nullptr;
      if (status != cudaSuccess) return status;
    } else {
      return status == cudaSuccess ? cudaErrorUnknown : status;
    }
  }
  if (!w.executable) {
    status = cudaGraphInstantiateWithFlags(&w.executable, graph.handle, 0);
    if (status != cudaSuccess) return status;
    ++w.rebuilds;
  }
  status = cudaGraphLaunch(w.executable, w.owner);
  if (status == cudaSuccess) {
    ++w.launches; w.batches += batches;
    if (batches == 32) ++w.full_windows;
  }
  return status;
}

extern "C" int mgbfs_batch_graph_stats_v1(void* handle, uint64_t* launches,
    uint64_t* full_windows, uint64_t* batches, uint64_t* updates, uint64_t* rebuilds) {
  if (!handle || !launches || !full_windows || !batches || !updates || !rebuilds)
    return cudaErrorInvalidValue;
  auto& w = *static_cast<Window*>(handle);
  *launches = w.launches; *full_windows = w.full_windows; *batches = w.batches;
  *updates = w.updates; *rebuilds = w.rebuilds;
  return cudaSuccess;
}
extern "C" void mgbfs_batch_graph_destroy_v1(void* handle) {
    delete static_cast<Window*>(handle);
}
extern "C" void mgbfs_batch_graph_cancel_v1(void* handle) {
  if (!handle) return;
  auto& w = *static_cast<Window*>(handle);
  if (w.capturing) {
    Graph abandoned;
    cudaStreamEndCapture(w.owner, &abandoned.handle);
    w.capturing = false;
  }
}

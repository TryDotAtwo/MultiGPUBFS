#include "graph_window.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

void check(int status) {
  if (status) throw std::runtime_error(cudaGetErrorString(static_cast<cudaError_t>(status)));
}
void require(bool value, char const* label) {
  if (!value) throw std::runtime_error(label);
}
__global__ void produce(unsigned* value, unsigned input) { *value = input; }
__global__ void transform(unsigned* value) { *value *= 2; }
__global__ void consume(unsigned const* value, unsigned* sum) { *sum += *value; }

int main() {
  cudaStream_t owner{}, generation{}, exchange{}, archive{};
  for (auto p : {&owner, &generation, &exchange, &archive})
    check(cudaStreamCreateWithFlags(p, cudaStreamNonBlocking));
  cudaEvent_t generated{}, packed{}, consumed{}, archived{};
  for (auto p : {&generated, &packed, &consumed, &archived})
    check(cudaEventCreateWithFlags(p, cudaEventDisableTiming));
  unsigned *value{}, *sum{}, *host{};
  check(cudaMalloc(&value, sizeof(unsigned))); check(cudaMalloc(&sum, sizeof(unsigned)));
  check(cudaHostAlloc(&host, sizeof(unsigned), 0));
  check(cudaMemsetAsync(sum, 0, sizeof(unsigned), owner));
  check(cudaStreamSynchronize(owner));
  void* window{}; check(mgbfs_batch_graph_create_v1(&window));
  unsigned expected{};
  for (unsigned index=0; index<3; ++index) {
    unsigned const batches = index == 2 ? 5 : 32;
    // A real archive copy/event remains outside capture and observable on CPU.
    check(cudaMemcpyAsync(host, sum, sizeof(unsigned), cudaMemcpyDeviceToHost, archive));
    check(cudaEventRecord(archived, archive));
    check(mgbfs_batch_graph_begin_v1(window, owner, generation, exchange));
    check(cudaStreamWaitEvent(owner, archived, cudaEventWaitExternal));
    for (unsigned batch=0; batch<batches; ++batch) {
      unsigned const input=index*32+batch+1; expected += 2*input;
      produce<<<1,1,0,generation>>>(value,input);
      check(cudaEventRecord(generated,generation));
      check(cudaStreamWaitEvent(owner,generated,0));
      transform<<<1,1,0,owner>>>(value);
      check(cudaEventRecord(packed,owner));
      check(cudaStreamWaitEvent(exchange,packed,0));
      consume<<<1,1,0,exchange>>>(value,sum);
      check(cudaEventRecord(consumed,exchange));
      check(cudaStreamWaitEvent(generation,consumed,0));
      check(cudaStreamWaitEvent(owner,consumed,0));
    }
    check(mgbfs_batch_graph_submit_v1(window,batches));
    check(cudaStreamSynchronize(owner));
    unsigned actual{}; check(cudaMemcpy(&actual,sum,sizeof(actual),cudaMemcpyDeviceToHost));
    require(actual==expected,"WINDOW_RESULT");
  }
  uint64_t launches{}, full_windows{}, batches{}, updates{}, rebuilds{};
  check(mgbfs_batch_graph_stats_v1(window,&launches,&full_windows,&batches,&updates,&rebuilds));
  require(full_windows==2 && batches==69,"WINDOW_BATCH_ACCOUNTING");
  require(launches==3 && updates>=1 && rebuilds>=2,"WINDOW_UPDATE_AND_SHORT_TAIL");
  // Cancellation must end capture before auxiliary streams or payloads drop.
  check(mgbfs_batch_graph_begin_v1(window,owner,generation,exchange));
  produce<<<1,1,0,generation>>>(value,999);
  mgbfs_batch_graph_cancel_v1(window);
  cudaGetLastError(); // EndCapture may report an intentionally unjoined branch.
  check(cudaStreamSynchronize(generation));
  unsigned cancelled_result{};
  check(cudaMemcpy(&cancelled_result,sum,sizeof(cancelled_result),cudaMemcpyDeviceToHost));
  require(cancelled_result==expected,"CANCEL_EXECUTED_PAYLOAD");
  mgbfs_batch_graph_destroy_v1(window);
  check(cudaFree(value)); check(cudaFree(sum)); check(cudaFreeHost(host));
  for (auto e : {generated,packed,consumed,archived}) check(cudaEventDestroy(e));
  for (auto s : {owner,generation,exchange,archive}) check(cudaStreamDestroy(s));
  std::cout << "BATCH_GRAPH_32_MULTISTREAM_UPDATE_EXTERNAL_ARCHIVE_PASS\n";
}

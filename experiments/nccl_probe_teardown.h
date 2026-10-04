#pragma once
// Probe-only lifetime policy; callbacks retain the existing bounded poller.
template<class Finalize, class Progress, class Destroy>
auto nccl_probe_teardown(Finalize finalize, Progress progress, Destroy destroy) {
  const auto result=progress(finalize());
  if(result!=0)return result;
  return destroy(); // No NCCL query may use the handle after this call.
}

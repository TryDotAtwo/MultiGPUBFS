# T4 v14: device communicator versus registration

Exact source f29ae949b9051a9cc7a273d9f2bb48ed7c8a6b3e, same single notebook
trydotatwo/mgbfs-native-rank-owner-t4. Two independent rank processes, actual
2xT4 with P2P both ways, pinned CUDA 12.9 and NCCL 2.29.7. No suppressions,
API-error filtering or timeout extensions. Raw results downloaded to
test_results/kaggle_window_device_v14_20261001/lsa-bfs-gate/.

| Stage | plain | memcheck | racecheck | initcheck | synccheck |
|---|---|---|---|---|---|
| allocation + window registration | 0/0 | 97/97 | 0/0 | 7/7 | 0/0 |
| registration + device communicator (16 barriers) | 0/0 | 97/97 | 0/0 | 0/0 | 0/0 |

Both requested device-communicator markers were reached in the second stage.
This is a reduced diagnostic, not full BFS acceptance. The registration-only
initcheck result contradicts any claim of uniformly working registration under
this tool: v12 passed; v14 failed despite the later device-communicator case
passing on the same devices. No universal cause or fix is established.

Failing rank 1 reports dev_runtime.cc:1037 CUDA 'unspecified launch failure'
during registration's internal NCCL device-runtime setup. Rank 0 subsequently
reports bootstrap/system errors and incomplete shadow-pool cleanup. Both
processes terminate without a forced supervisor timeout and sanitizer reports
zero detected errors, but the nonzero application exits make this a FAIL.
The official v2.29.7-1 source's physical line 1037 is cudaStreamSynchronize:
https://raw.githubusercontent.com/NVIDIA/nccl/v2.29.7-1/src/dev_runtime.cc
That identifies a failure-reporting boundary, not the earlier failing operation.

Next full BFS replay enables NCCL_DEBUG=INFO and reports native status callsites
without changing legacy CUDA_STATUS error strings or healthy execution decisions.
All full BFS sanitizer gates remain open until genuine full runs pass unfiltered.

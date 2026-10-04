# Kaggle v91: initcheck open, healthy full-state timelines pass

Notebook `trydotatwo/mgbfs-lsa-full-bfs-gate-t4` v91 completed.
Runtime source `6d9a2009010c7306e50769c8db611897c608264e`.
Summary SHA256 `08d86c09fefda16d3b4567e5ffbe31e0025c9e80e25356023f8a6b5d6c514c28`.
Two physical T4s, sm75, P2P allowed both directions. Experimental NCCL
`minimum_arch_guard_posix`, upstream `b91894bd5b190c874d98a017f93f5daa515b65d0`.
Library SHA256 `2a515273bf8f77622961cd38954817a1cd2725f65b06c31526bffd408147aefd`.

Both DENSE/HASH_FIRST U4/F2 timeline replays pass the full-state oracle and
authenticated two-rank archive/group commit verifier. Unlike v90, pyarrow is
installed and verifier succeeds during the original replay, exit codes `[0,0]`.
Each contains 64 states; this is not a large-frontier throughput measurement.

All 18 unfiltered initcheck replays FAIL, without supervisor timeout. Both
ranks return 1 in each case, before search. Among 36 rank error records,
30 report window_register CUDA_STATUS_7 and 6 device_comm_create CUDA_STATUS_8.
The native detail preserves submitted success vs eventual terminal NCCL error:
`progress=9 submitted=0 terminal=1 detail=unhandled cuda error`.
Vendor logs report unspecified launch failure at dev_runtime.cc:1037 and611.
Zero sanitizer error summaries do NOT override the failed application exits.
Root cause and full initcheck acceptance remain OPEN.

Small summaries and 40 rank logs retained at `build/kaggle-v91-observation`;
state archives were not downloaded. Next diagnostic uses the existing reduced
independent-process NCCL window/device-communicator probe, no BFS and no second
concurrent notebook. Typed U4/F3 all-owner stress remains prepared, not passed.

Output retrieval caveat: installed `KaggleApi.kernels_output` parses a numeric
version suffix but does not set it on ApiListKernelSessionOutputRequest.
`kernels_list_files` also parses but discards the numeric suffix. Downloading
`.../91` after v92 starts therefore does not establish v91 provenance. The
summaries/logs above were downloaded before v92 was submitted and have the
source/digest checks above. Subsequent SQLite retrieval returned no files; no
v91 timeline analysis is claimed. Page token alone did not recover those
outputs. The SDK exposes version_label, but setting it to '91' returned404;
its semantics have not been established. No third-party credential or signed
download URL is retained in this report.

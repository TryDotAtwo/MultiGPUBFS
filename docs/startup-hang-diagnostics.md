# Native startup hang diagnostics

Version 0.2.0.dev11 adds independent per-GPU `cudaMemGetInfo` and
`mgbfs_generic_hardware_info` (`cudaGetDeviceProperties`) probes and a bounded
Python control-thread probe. These run in isolated processes even if another
probe fails. They do not change CUDA visibility, retry a real BFS, or reset GPUs.

Native persistent inventory records entry checkpoints before the Rust control
thread, cudaSetDevice, cudaMemGetInfo, hardware query and reusable-buffer
accounting. Standalone graph-info records definition read, device selection,
memory query, hardware query and memory admission. Logging is enabled only by
MGBFS_DIAGNOSTIC_TRACE_FILE in diagnostic children. The flushed append-only
native-startup.log survives timeout and is included in the checksummed ZIP.
A checkpoint means the operation is about to start, not that it finished.

CUDA/NCCL primitive success does not establish full native startup success.
Native CLI requires EAGER module/data loading; independently loaded ctypes
primitives can use another loading policy. A timeout is not proof of OOM or
driver incompatibility. The Python thread probe is not Rust-thread acceptance;
the native control-thread checkpoints distinguish that boundary separately.

Install in the server job environment:

```sh
python -m pip install --upgrade --force-reinstall "multigpubfs @ https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/native-pip-20261014-v11/multigpubfs-0.2.0.dev11-py3-none-linux_x86_64.whl#sha256=ff30bbdde91bbf66654cf55e8f39fff4b7811105d2319d02e559a3844daf9dd4"
multigpubfs doctor
```

Start a fresh Python process/job after upgrading. Return diagnostics.zip from
that invocation. No privileged commands or administrator access are required.
The original server-specific hang has not been reproduced here.

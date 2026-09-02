# REF-012: Docker Rust-to-CUDA environment smoke

Date: 2026-08-27  
Status: pass after one build failure

## User-required stack boundary

All GPU build/run work is containerized. Rust owns the executable, validation,
and stdout artifact. C++ is confined to a CUDA translation unit containing the
kernel and minimal CUDA Runtime C ABI.

## Host and container evidence

Host inventory before container selection:

```text
GPU                 NVIDIA GeForce RTX 3070 Laptop GPU
VRAM                8192 MiB
compute capability  8.6
driver              572.70
Docker Desktop      4.67.0
Docker Engine       29.3.1, linux/amd64
```

The existing local image `gpu-dev-cutlass-nsight:2026-05-24` was reused instead
of downloading another CUDA development stack. A GPU container smoke confirmed
CUDA 12.8.1 and nvcc 12.8.93. The base lacked Rust, so the project builder adds
Ubuntu's Rust 1.75 packages in a cached layer. The final runner does not contain
the Rust compiler.

Project image:

```text
tag     multigpubfs-gpu:dev
digest  sha256:f31f597c5b8943dc1e6540f3784fc941b4ebc619a92c01757dc6d946f7b39693
size    6,902,394,311 bytes (Docker inspect logical image size)
```

The image is intentionally a development/profiling runner based on the reused
Nsight/CUTLASS image, not a minimal production image.

## Build and run

```powershell
docker build -f docker/Dockerfile.gpu -t multigpubfs-gpu:dev .
docker run --rm --gpus all multigpubfs-gpu:dev
```

Verified stdout:

```json
{"status":"pass","host":"rust","gpu_code":"cuda_c_abi","gpu":"NVIDIA GeForce RTX 3070 Laptop GPU","compute_capability":"8.6","elements":1048576}
```

Rust constructed and validated all 1,048,576 output elements. CUDA allocated
device buffers, copied input, launched the kernel, and returned output through
the narrow C ABI.

`cuobjdump` found `libmultigpubfs_cuda.1.sm_86.cubin` and the affine kernel in
that cubin. This proves a native `sm_86` image is present; execution is not
dependent only on PTX JIT fallback.

## Recorded failures and exclusions

1. The first Docker build failed before compilation because the project asked
   for CMake 3.24 while the reused image contains 3.22.1. No used feature needed
   3.24; lowering the declared minimum to 3.22 made the next build pass.
2. Host CuPy 13.5.1 enumerated the GPU but failed its first generated operation
   because it searched for `nvrtc64_112_0.dll`. It is an inconsistent host CUDA
   11.x package and is excluded from the container baseline.
3. Host WMI hardware queries again returned access denied. GPU identity came
   from NVML/`nvidia-smi` and CUDA device properties instead.

No throughput claim is made from this smoke. It validates the container, native
architecture, FFI boundary, kernel execution, and independent Rust checking.

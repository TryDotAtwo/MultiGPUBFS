#!/usr/bin/env bash
set -euo pipefail
mkdir -p test_results/native
export CARGO_HOME="${CARGO_HOME:-/src/build/cargo-home}"
export MGBFS_CUDA_LIB_DIR="$(pwd)/build/native-cuda"
export LD_LIBRARY_PATH="$MGBFS_CUDA_LIB_DIR:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
{
  rustc --version
  cargo --version
  nvcc --version
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
  git -C "${CUTLASS_ROOT:-/opt/cutlass}" rev-parse HEAD
} 2>&1 | tee test_results/native/environment.log
cargo test --locked 2>&1 | tee test_results/native/cpu-tests.log
cmake -S cuda -B build/native-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_CUDA_ARCHITECTURES=${MGBFS_CUDA_ARCHITECTURES:-75;86}" \
  "-DCUTLASS_ROOT=${CUTLASS_ROOT:-/opt/cutlass}" 2>&1 | tee test_results/native/configure.log
cmake --build build/native-cuda 2>&1 | tee test_results/native/build.log
cargo test --locked -p mgbfs-cuda --features cuda 2>&1 | tee test_results/native/gpu-tests.log
cargo test --locked -p mgbfs-runtime --features cuda --test dense_device -- --test-threads=1 2>&1 | tee test_results/native/gpu-feedback-tests.log

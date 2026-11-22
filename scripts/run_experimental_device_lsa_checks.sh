#!/usr/bin/env bash
# Research-only bounded two-GPU LSA prototypes; no production install or system changes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NCCL_ROOT="$(printenv NCCL_ROOT || true)"
if [ -z "$NCCL_ROOT" ] || [ "$#" -ne 1 ]; then
  printf 'Usage: NCCL_ROOT=/path/to/nccl bash %s /path/to/results\n' "$0" >&2
  exit 2
fi
OUT="$1"
ARCH="$(printenv MGBFS_TEST_ARCH || true)"
if [ -z "$ARCH" ]; then ARCH=sm_75; fi
NVCC="$(printenv NVCC || true)"
if [ -z "$NVCC" ]; then NVCC=nvcc; fi
mkdir -p "$OUT"
printf 'source=' > "$OUT/environment.txt"
git -C "$ROOT" rev-parse HEAD >> "$OUT/environment.txt"
"$NVCC" --version >> "$OUT/environment.txt"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv >> "$OUT/environment.txt"
sha256sum "$NCCL_ROOT/lib/libnccl.so.2" >> "$OUT/environment.txt"
for name in experimental_device_ring_p2p_gpu \
  experimental_device_ring_nccl_lsa_multiprocess_gpu \
  experimental_device_bfs_lsa_gpu \
  experimental_device_pack4_bfs_lsa_gpu \
  experimental_device_hash_bfs_lsa_gpu \
  experimental_device_pack4_warps_lsa_gpu; do
  source="$ROOT/tests/$name.cu"
  bin="$OUT/$name"
  "$NVCC" -std=c++17 -O2 -lineinfo "-arch=$ARCH" --cudart shared \
    "-I$NCCL_ROOT/include" "$source" "-L$NCCL_ROOT/lib" -l:libnccl.so.2 \
    -Xlinker "-rpath,$NCCL_ROOT/lib" -o "$bin"
  sha256sum "$source" "$bin" >> "$OUT/environment.txt"
  printf 'RUN %s\n' "$name"
  NCCL_DEBUG=WARN timeout 180s "$bin" > "$OUT/$name.log" 2>&1
  tail -1 "$OUT/$name.log"
done
printf 'PASS: standalone experiments only; no production backend was activated.\n'

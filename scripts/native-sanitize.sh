#!/usr/bin/env bash
set -euo pipefail
log_dir="${MGBFS_SANITIZER_LOG_DIR:-test_results/native-sanitizer}"
mkdir -p "$log_dir"
for binary in "$@"; do
  name=$(basename "$binary")
  if [[ "$name" == dense_device-* ]]; then
    inventory=$("$binary" --list)
    if [[ "$inventory" != *"gpu_feedback_small_full_depth_sanitizer_fixture: test"* || "$inventory" != *"gpu_feedback_exhausts_exact_layers_without_cpu_supplied_frontiers: test"* ]]; then
      echo "DENSE_DEVICE_TEST_INVENTORY_MISMATCH" >&2
      exit 64
    fi
  fi
  for tool in memcheck racecheck initcheck synccheck; do
    args=(--test-threads=1 --nocapture)
    if [[ "$name" == ping_pong-* ]]; then
      args+=(--skip full_u4_pipelined_sweep)
    fi
    if [[ "$name" == dense_device-* ]]; then
      if [[ "$tool" == racecheck ]]; then
        args+=(--skip gpu_feedback_exhausts_exact_layers_without_cpu_supplied_frontiers)
      else
        args+=(--skip gpu_feedback_small_full_depth_sanitizer_fixture)
      fi
    fi
    compute-sanitizer --error-exitcode 99 --tool "$tool" "$binary" "${args[@]}" 2>&1 | tee "$log_dir/${name}-${tool}.log"
  done
done

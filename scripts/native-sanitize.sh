#!/usr/bin/env bash
set -euo pipefail
mkdir -p test_results/native-sanitizer
for binary in "$@"; do
  name=$(basename "$binary")
  for tool in memcheck racecheck initcheck synccheck; do
    compute-sanitizer --error-exitcode 99 --tool "$tool" "$binary" --test-threads=1 2>&1 | tee "test_results/native-sanitizer/${name}-${tool}.log"
  done
done

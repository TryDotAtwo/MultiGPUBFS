#!/usr/bin/env bash
set -euo pipefail
binary="${1:-build/macro-settle/mgbfs-macro-settle-test}"
for tool in memcheck racecheck initcheck synccheck; do
  compute-sanitizer --error-exitcode 99 --tool "$tool" "$binary"
done

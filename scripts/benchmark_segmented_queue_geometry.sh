#!/usr/bin/env bash
# Controlled benchmark harness. Time limits here bound the experiment only;
# they are not a production BFS time limit.
set -euo pipefail
: "${MGBFS_QUEUE_ROOT:?missing MGBFS_QUEUE_ROOT}"
: "${MGBFS_QUEUE_ARCHIVE:?missing MGBFS_QUEUE_ARCHIVE}"
: "${MGBFS_QUEUE_MODE:?missing MGBFS_QUEUE_MODE}"
case "$MGBFS_QUEUE_MODE" in gpu|stub) ;; *) echo BAD_MODE >&2; exit 64;; esac
if [[ -e "$MGBFS_QUEUE_ROOT" || -e "$MGBFS_QUEUE_ARCHIVE" ]]; then
  echo OUTPUT_ALREADY_EXISTS >&2
  exit 64
fi
if [[ "$MGBFS_QUEUE_MODE" == gpu ]]; then
  : "${MGBFS_QUEUE_PYTHON:?missing MGBFS_QUEUE_PYTHON}"
  : "${MGBFS_QUEUE_SCRIPT:?missing MGBFS_QUEUE_SCRIPT}"
  : "${MGBFS_QUEUE_REPO:?missing MGBFS_QUEUE_REPO}"
  : "${MGBFS_EXECUTABLE:?missing MGBFS_EXECUTABLE}"
  : "${MGBFS_BENCH_CUDA_LIBRARY:?missing MGBFS_BENCH_CUDA_LIBRARY}"
  [[ -x "$MGBFS_QUEUE_PYTHON" && -f "$MGBFS_QUEUE_SCRIPT" && -d "$MGBFS_QUEUE_REPO" && -x "$MGBFS_EXECUTABLE" && -f "$MGBFS_BENCH_CUDA_LIBRARY" ]] || { echo MISSING_INPUT >&2; exit 65; }
fi
mkdir -p "$MGBFS_QUEUE_ROOT" "$(dirname "$MGBFS_QUEUE_ARCHIVE")"
mgbfs_finish(){
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$MGBFS_QUEUE_ROOT/exit-code.txt"
  if ! tar -czf "$MGBFS_QUEUE_ARCHIVE" -C "$MGBFS_QUEUE_ROOT" .; then
    echo ARCHIVE_FAILED >&2
    exit 70
  fi
  echo "BENCH_ARCHIVE $MGBFS_QUEUE_ARCHIVE code=$rc"
  exit "$rc"
}
trap mgbfs_finish EXIT
export MGBFS_SEGMENT_BATCH_RECOVERY=0
unset MGBFS_SEGMENT_MAX_RANK_STATES MGBFS_SEGMENT_STORAGE_BUDGET_BYTES MGBFS_SEGMENT_BITMAP_OWNER_PAGE MGBFS_SEGMENT_TEST_HOST_TIMEOUT_MS MGBFS_SEGMENT_LAYER_STATS || true
if [[ "$MGBFS_QUEUE_MODE" == gpu ]]; then
  exec 9>/workspace/mgbfs-benchmark-exclusive.lock
  flock -n 9 || { echo GPU_LOCK_BUSY >&2; exit 20; }
  export MGBFS_BENCH_REPO="$MGBFS_QUEUE_REPO"
  {
    date -u +%FT%TZ
    git -C "$MGBFS_QUEUE_REPO" rev-parse HEAD
    sha256sum "$MGBFS_EXECUTABLE" "$MGBFS_BENCH_CUDA_LIBRARY" "$MGBFS_QUEUE_SCRIPT" "$0"
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
    nvidia-smi topo -m
  } > "$MGBFS_QUEUE_ROOT/environment.txt" 2>&1
fi
mgbfs_idle(){
  if [[ "$MGBFS_QUEUE_MODE" == stub ]]; then return 0; fi
  local busy
  for _ in $(seq 1 20); do
    busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" || return 1
    [ -z "$(printf '%s' "$busy" | tr -d '[:space:]')" ] && return 0
    sleep 1
  done
  printf 'GPU_BUSY %s\n' "$busy" >&2
  return 1
}
mgbfs_case(){
  local label="$1" backend="$2" strategy="$3" n="$4" batch="$5"
  local root="$MGBFS_QUEUE_ROOT/$label" rc=0
  mkdir "$root"
  echo RUNNING > "$root/status.txt"
  mgbfs_idle || { echo GPU_BUSY > "$root/status.txt"; return 21; }
  if [[ "$MGBFS_QUEUE_MODE" == stub ]]; then
    echo "stub partial result $label" > "$root/run.log"
    if [[ "${MGBFS_QUEUE_STUB_FAILURE:-}" == "$label" ]]; then rc=33; fi
  else
    timeout --kill-after=15s 1200s "$MGBFS_QUEUE_PYTHON" -u "$MGBFS_QUEUE_SCRIPT" \
      --backend "$backend" --strategy "$strategy" --n "$n" --batch "$batch" --base "$root" \
      > "$root/run.log" 2>&1 || rc=$?
  fi
  if (( rc != 0 )); then printf 'FAILED %s\n' "$rc" > "$root/status.txt"; return "$rc"; fi
  mgbfs_idle || { echo POST_GPU_BUSY > "$root/status.txt"; return 22; }
  echo COMPLETE > "$root/status.txt"
  echo "PASS $label"
}
# Small end-to-end smoke and alternating balanced S12 experiments.
mgbfs_case smoke-small permutation_bitmap atomic 9 8192
mgbfs_case smoke-large permutation_bitmap atomic 9 32768
mgbfs_case bitmap-small-1 permutation_bitmap atomic 12 8192
mgbfs_case bitmap-large-1 permutation_bitmap atomic 12 32768
mgbfs_case bitmap-large-2 permutation_bitmap atomic 12 32768
mgbfs_case bitmap-small-2 permutation_bitmap atomic 12 8192
mgbfs_case bitmap-small-3 permutation_bitmap atomic 12 8192
mgbfs_case bitmap-large-3 permutation_bitmap atomic 12 32768
mgbfs_case hash-small-1 segmented24 warp 12 8192
mgbfs_case hash-large-1 segmented24 warp 12 32768
echo ALL_CASES_COMPLETE

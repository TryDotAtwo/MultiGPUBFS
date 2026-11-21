#!/usr/bin/env bash
# Bounded test harness only. Production BFS retains its own time policy.
set -uo pipefail
: "${MGBFS_ACCEPT_MODE:?missing acceptance mode}"
: "${MGBFS_ACCEPT_ROOT:?missing new evidence directory}"
: "${MGBFS_ACCEPT_ARCHIVE:?missing archive path}"
: "${MGBFS_ACCEPT_HARNESS:?missing harness path}"
: "${MGBFS_ACCEPT_SOURCE:?missing source path}"
: "${MGBFS_ACCEPT_PYTHON:=python3}"

if [[ -e "$MGBFS_ACCEPT_ROOT" || -e "$MGBFS_ACCEPT_ARCHIVE" ]]; then
  echo "ACCEPTANCE_OUTPUT_ALREADY_EXISTS" >&2
  exit 64
fi
if [[ ! -f "$MGBFS_ACCEPT_HARNESS" || ! -d "$MGBFS_ACCEPT_SOURCE" ]]; then
  echo "ACCEPTANCE_INPUT_MISSING" >&2
  exit 65
fi
mkdir -p "$MGBFS_ACCEPT_ROOT"
{
  date -u +%FT%TZ
  uname -a
  if [[ "$MGBFS_ACCEPT_MODE" == gpu ]] && command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
  fi
} > "$MGBFS_ACCEPT_ROOT/system.txt" 2>&1

rc=0
"$MGBFS_ACCEPT_PYTHON" "$MGBFS_ACCEPT_HARNESS" \
  --mode "$MGBFS_ACCEPT_MODE" \
  --source "$MGBFS_ACCEPT_SOURCE" \
  --output "$MGBFS_ACCEPT_ROOT/evidence" \
  > "$MGBFS_ACCEPT_ROOT/harness.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$MGBFS_ACCEPT_ROOT/exit-code.txt"
tar -czf "$MGBFS_ACCEPT_ARCHIVE" -C "$MGBFS_ACCEPT_ROOT" . || exit 70
echo "ACCEPTANCE_ARCHIVE $MGBFS_ACCEPT_ARCHIVE status=$rc"
exit "$rc"

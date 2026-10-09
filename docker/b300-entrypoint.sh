#!/bin/sh
set -eu
export PATH=/opt/mgbfs/build/venv/bin:$PATH
if [ $# -gt 0 ]; then
  case "$1" in
    --*) exec python /opt/mgbfs/src/scripts/b300_production.py "$@" ;;
    *) exec "$@" ;;
  esac
fi
if [ -n "${MGBFS_WORK_DEADLINE_UNIX:-}" ]; then
  exec python /opt/mgbfs/src/scripts/b300_production.py --root "${MGBFS_RUN_ROOT:?}" --repo-id "${MGBFS_HF_REPO:?}" --deadline-unix "$MGBFS_WORK_DEADLINE_UNIX" --token-file "${MGBFS_HF_TOKEN_FILE:-/run/secrets/hf_token}"
fi
exec sleep infinity

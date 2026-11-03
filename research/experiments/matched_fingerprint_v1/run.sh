#!/bin/sh
set -eu
TASK_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
TASK_ROOT="$(CDPATH= cd -- "$TASK_SCRIPT_DIR/../../.." && pwd)"
TASK_OLD_DYLD="$(printenv DYLD_LIBRARY_PATH || true)"
export DYLD_LIBRARY_PATH="$TASK_ROOT/.venv/lib:$TASK_OLD_DYLD"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec "$TASK_ROOT/.venv/bin/python" "$TASK_SCRIPT_DIR/run.py" "$@"

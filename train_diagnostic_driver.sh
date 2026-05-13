#!/usr/bin/env bash
# Driver for Diagnostic (B2) and Diagnostic (C) training runs.
# Usage: bash train_diagnostic_driver.sh {B2|C} [epochs]
#
# Defensive: per-subprocess isolation, no ulimit -v (collides with CUDA VA),
# TF GPU memory growth on, per-epoch checkpoint inside the python script.

set -u
cd "$(dirname "$0")"
if [ -z "${1:-}" ]; then
  echo "usage: $0 (B2|C) [epochs]" >&2
  exit 64
fi
variant="$1"
epochs="${2:-50}"

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

echo "$(date -Iseconds) train driver: variant=$variant epochs=$epochs"
free -m | awk 'NR==1 || NR==2'

source .venv/bin/activate
python -u train_diagnostic.py --variant "$variant" --epochs "$epochs"
rc=$?
echo "$(date -Iseconds) train driver done rc=$rc"
free -m | awk 'NR==2'
exit $rc

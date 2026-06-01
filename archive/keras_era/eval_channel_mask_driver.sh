#!/usr/bin/env bash
# Defensive driver for Diagnostic (B1) inference-time channel mask.
# 20 GB virtual memory cap; TF memory growth on; run in subprocess.

set -u
cd "$(dirname "$0")"
# ulimit -v removed: TF/CUDA reserves huge VA on init, collides with any
# practical cap. Defense is per-subprocess + cgroup + set_memory_growth.
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

echo "$(date -Iseconds) B1 driver starting; ulimit -v = $(ulimit -v) KB"
free -m | awk 'NR==1 || NR==2'

source .venv/bin/activate
python -u eval_channel_mask.py
rc=$?
echo "$(date -Iseconds) B1 driver done rc=$rc"
free -m | awk 'NR==2'
exit $rc

#!/usr/bin/env bash
# Chained pipeline for gameplan step 4: more-data run on runB2.
#
# Sequence:
#   1. Wait for cache build (build_runB2_pool_cache.py) to finish
#   2. Train runB2 on the 7000-event cache
#   3. Eval at EB sweep
#
# All defensive plumbing from train_diagnostic.py / driver applies.

set -u
cd "$(dirname "$0")"
LOG=step4_pipeline.log
exec >> "$LOG" 2>&1

echo "$(date -Iseconds) ====================================================="
echo "$(date -Iseconds) step4 pipeline starting"

# ---- Step 1: wait for cache build ----
echo "$(date -Iseconds) Step 1: waiting for cache build to complete"
# Poll for ANY ds_pool_runB2_train_*.npz file (cache build will name it via hash)
# OR poll for the build python process to exit.
while pgrep -f "build_runB2_pool_cache.py" > /dev/null; do
  sleep 60
  pid=$(pgrep -f "build_runB2_pool_cache.py" | head -1)
  if [ -n "$pid" ]; then
    rss_kb=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null || echo 0)
    echo "$(date -Iseconds)   cache build pid=$pid still running, RSS=$((rss_kb/1024))MB"
  fi
done
echo "$(date -Iseconds) Step 1 done: cache build process exited"

# Find the new train cache (the latest runB2 cache file)
TRAIN_NPZ=$(ls -t cache/ds_pool_runB2_train_*.npz 2>/dev/null | head -1)
VAL_NPZ=$(ls -t cache/ds_pool_runB2_val_*.npz 2>/dev/null | head -1)
# Fallback: subagent may have used a different naming
if [ -z "$TRAIN_NPZ" ]; then
  TRAIN_NPZ=$(ls -t cache/ds_pool_run41_train_*.npz 2>/dev/null | grep -v "13767833f56c8d8d" | head -1)
fi
if [ -z "$VAL_NPZ" ]; then
  VAL_NPZ=cache/ds_pool_run41_val_e36bcb8add5ec9f0.npz
fi
echo "$(date -Iseconds) Detected caches:"
echo "$(date -Iseconds)   TRAIN_NPZ=$TRAIN_NPZ"
echo "$(date -Iseconds)   VAL_NPZ=$VAL_NPZ"

if [ ! -f "$TRAIN_NPZ" ]; then
  echo "$(date -Iseconds) FATAL: no train cache found — aborting"
  exit 1
fi

# Wait a sec for filesystem flush, then check sizes
sleep 5
ls -la "$TRAIN_NPZ" "$VAL_NPZ"

# ---- Step 2: train ----
echo "$(date -Iseconds) Step 2: training runB2 on $TRAIN_NPZ"
free -m | awk 'NR==1 || NR==2'

source .venv/bin/activate
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

python -u train_diagnostic.py \
  --variant B2 \
  --epochs 50 \
  --tag runB2_7k \
  --train-cache "$TRAIN_NPZ" \
  --val-cache   "$VAL_NPZ" \
  > train_B2_7k.log 2>&1
TRAIN_RC=$?
echo "$(date -Iseconds) Step 2 done: rc=$TRAIN_RC"
free -m | awk 'NR==2'

if [ $TRAIN_RC -ne 0 ]; then
  echo "$(date -Iseconds) FATAL: training rc=$TRAIN_RC — aborting before eval"
  exit $TRAIN_RC
fi

# ---- Step 3: eval ----
FINAL=models_test/cnn_breakpoint_runB2_7k_final.keras
echo "$(date -Iseconds) Step 3: evaluating $FINAL"
if [ ! -f "$FINAL" ]; then
  echo "$(date -Iseconds) WARN: $FINAL not found, looking for alternates"
  FINAL=$(ls -t models_test/cnn_breakpoint_runB2_7k*final*.keras 2>/dev/null | head -1)
  echo "$(date -Iseconds) Using: $FINAL"
fi

# Standard eval at EB=200 (canonical) + the eb-sweep variant if available
python -u eval_one_model.py \
  --tag runB2_7k \
  --model "$FINAL" \
  --zero-lo 15 --zero-hi 22 \
  > eval_runB2_7k.log 2>&1
echo "$(date -Iseconds) eval_one_model rc=$?"

# Optional EB sweep (already wrote eval_runB2_7k_eb_sweep.py)
if [ -f eval_runB2_7k_eb_sweep.py ]; then
  python -u eval_runB2_7k_eb_sweep.py > eval_runB2_7k_eb_sweep.log 2>&1
  echo "$(date -Iseconds) eb_sweep rc=$?"
fi

echo "$(date -Iseconds) ====================================================="
echo "$(date -Iseconds) step4 pipeline complete"
free -m | awk 'NR==2'

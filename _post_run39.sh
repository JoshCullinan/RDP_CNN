#!/bin/bash
# Post-training pipeline for run #39 (or any run that bumped N_INPUT_CHANNELS).
# Runs sequentially: test cache regen, eval, summary print.
# Does NOT commit — review summary first, then run git commit manually.
set -e
cd /home/joshcullinan/RDP_CNN

RUN_NUM="${1:-39}"
MODEL="models_test/cnn_breakpoint_run${RUN_NUM}_final.keras"
[ -f "$MODEL" ] || { echo "ERROR: $MODEL not found"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
echo "=== Step 1: regen UnseenTestSet cache at new channel count ==="
nohup systemd-run --user --scope --quiet -p MemoryMax=20G \
  bash -c "source .venv/bin/activate && python3 -u cache_test_set.py" \
  > "/tmp/cache_test_run${RUN_NUM}_${TS}.log" 2>&1 &
CACHE_PID=$!
echo "cache pid: $CACHE_PID  (logging to /tmp/cache_test_run${RUN_NUM}_${TS}.log)"
wait $CACHE_PID || { echo "cache regen FAILED"; tail -30 "/tmp/cache_test_run${RUN_NUM}_${TS}.log"; exit 1; }
echo "cache regen complete."

echo ""
echo "=== Step 2: eval $MODEL on UnseenTestSet ==="
EVAL_LOG="eval_run${RUN_NUM}_${TS}.log"
nohup systemd-run --user --scope --quiet -p MemoryMax=20G \
  bash -c "source .venv/bin/activate && python3 -u eval_run29.py $MODEL run${RUN_NUM}" \
  > "$EVAL_LOG" 2>&1 &
EVAL_PID=$!
echo "eval pid: $EVAL_PID  (logging to $EVAL_LOG)"
wait $EVAL_PID || { echo "eval FAILED"; tail -30 "$EVAL_LOG"; exit 1; }

echo ""
echo "=== Step 3: summary ==="
echo "Eval log: $EVAL_LOG"
grep -E "honest|F1|threshold|edge" "$EVAL_LOG" | tail -30

echo ""
echo "Done. Review the eval log and append entry to cell-experiment-log if results look right."

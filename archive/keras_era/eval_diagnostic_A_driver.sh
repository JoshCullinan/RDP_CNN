#!/usr/bin/env bash
# Defensive driver for Diagnostic (A) eval.
#
# Runs each model in a FRESH python subprocess so the OS reclaims all RAM/GPU
# between iterations. Caps per-process virtual memory at 20 GB via ulimit so
# python hits MemoryError before WSL2 panics.

set -u
cd "$(dirname "$0")"

# Note: ulimit -v was tried (20 GB, 26 GB) and rejected. TF/CUDA reserves
# huge VIRTUAL address space on init (often 20-40 GB), independent of
# physical RAM use, so any practical -v cap collides with CUDA. Physical
# safety comes from:
#   1. per-subprocess isolation (OS reclaims VA on exit)
#   2. WSL2 cgroup at 28 GB (kernel OOM-kills the subprocess, not WSL2)
#   3. set_memory_growth on GPU (CUDA only takes physical RAM as needed)
#   4. drop-X-after-predict in eval_one_model.py
# Peak physical per subprocess: ~15-16 GB (12 GB cache + 3 GB CUDA).

# TF: don't pre-grab GPU memory.
export TF_FORCE_GPU_ALLOW_GROWTH=true
# TF: less verbose.
export TF_CPP_MIN_LOG_LEVEL=2

models=(
  "run38       models_test/cnn_breakpoint_run38_final.keras"
  "run41       models_test/cnn_breakpoint_run41_final.keras"
  "run42c      models_test/cnn_breakpoint_run42c_final.keras"
  "run42c_diag models_test/cnn_breakpoint_run42c_diag_final.keras"
  "run43       models_test/cnn_breakpoint_run43_final.keras"
)

echo "$(date -Iseconds) DRIVER starting; ulimit -v = $(ulimit -v) KB"
free -m | awk 'NR==1 || NR==2'

source .venv/bin/activate

for entry in "${models[@]}"; do
  tag=$(echo "$entry" | awk '{print $1}')
  model=$(echo "$entry" | awk '{print $2}')
  echo ""
  echo "===================================================="
  echo "$(date -Iseconds) START $tag"
  free -m | awk 'NR==2'
  python -u eval_one_model.py --tag "$tag" --model "$model"
  rc=$?
  echo "$(date -Iseconds) END $tag rc=$rc"
  free -m | awk 'NR==2'
  if [[ $rc -ne 0 && $rc -ne 2 ]]; then
    echo "FATAL: $tag exited rc=$rc — continuing to next model"
  fi
done

echo ""
echo "===================================================="
echo "$(date -Iseconds) DRIVER done — assembling summary"
python - <<'PY'
import json
from pathlib import Path
d = Path('results_diagnostic_A_partial')
rows = []
for tag in ['run38','run41','run42c','run42c_diag','run43']:
    p = d / f'{tag}.json'
    if not p.exists():
        rows.append((tag, 'MISSING', 'MISSING'))
        continue
    r = json.loads(p.read_text())
    rows.append((tag, f"{r['best_full_eb200']['f1']:.4f}", f"{r['best_sub_eb200']['f1']:.4f}"))
print(f"{'model':<14s} {'F1_full_honest':>15s} {'F1_sub_honest':>15s}")
for tag, f, s in rows:
    print(f"  {tag:<14s} {f:>15s} {s:>15s}")
with open('results_diagnostic_A_summary.json','w') as f:
    json.dump({tag: dict(full=fu, sub=su) for tag,fu,su in rows}, f, indent=2)
print("\nWrote results_diagnostic_A_summary.json")
PY

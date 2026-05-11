#!/usr/bin/env bash
# WSL-side memory/GPU/process logger.
# Writes append-only CSVs to /mnt/c/Users/joshc/wsl_monitor/ which survive WSL2 crash.
# Polls every INTERVAL seconds. Run in background; kill via the PID echoed at start.
#
# Usage:
#   bash /home/joshcullinan/RDP_CNN/monitor/wsl_log.sh &
#   echo $! > /tmp/wsl_log.pid
#
# Stop:
#   kill $(cat /tmp/wsl_log.pid)

set -u
INTERVAL=${INTERVAL:-3}
OUT_DIR=/mnt/c/Users/joshc/wsl_monitor
mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
MEM_LOG="$OUT_DIR/wsl_mem_${TS}.csv"
GPU_LOG="$OUT_DIR/wsl_gpu_${TS}.csv"
TOP_LOG="$OUT_DIR/wsl_top_${TS}.csv"
DMESG_LOG="$OUT_DIR/wsl_dmesg_${TS}.log"

echo "ts,MemTotal_kB,MemAvailable_kB,MemFree_kB,Buffers_kB,Cached_kB,SwapTotal_kB,SwapFree_kB,Committed_AS_kB,AnonPages_kB,Slab_kB,KernelStack_kB,PageTables_kB" > "$MEM_LOG"
echo "ts,gpu_mem_used_MiB,gpu_mem_free_MiB,gpu_util_pct,gpu_temp_C,gpu_power_W" > "$GPU_LOG"
echo "ts,pid,rss_MB,vsz_MB,cmd" > "$TOP_LOG"

echo "WSL logger PID=$$ writing to $OUT_DIR (interval ${INTERVAL}s)"

# Best-effort kernel ring-buffer dump on start (will catch any pre-existing OOM messages)
dmesg -T > "$DMESG_LOG" 2>/dev/null || true

while true; do
  NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  # /proc/meminfo
  awk -v ts="$NOW" 'BEGIN{
      keys["MemTotal"]=1; keys["MemAvailable"]=1; keys["MemFree"]=1;
      keys["Buffers"]=1; keys["Cached"]=1; keys["SwapTotal"]=1; keys["SwapFree"]=1;
      keys["Committed_AS"]=1; keys["AnonPages"]=1; keys["Slab"]=1;
      keys["KernelStack"]=1; keys["PageTables"]=1;
    }
    {
      gsub(":","",$1);
      if ($1 in keys) { v[$1]=$2 }
    }
    END {
      printf("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", ts,
        v["MemTotal"], v["MemAvailable"], v["MemFree"],
        v["Buffers"], v["Cached"], v["SwapTotal"], v["SwapFree"],
        v["Committed_AS"], v["AnonPages"], v["Slab"],
        v["KernelStack"], v["PageTables"]);
    }' /proc/meminfo >> "$MEM_LOG"

  # GPU
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -v ts="$NOW" -F', *' '{printf("%s,%s,%s,%s,%s,%s\n", ts, $1, $2, $3, $4, $5)}' \
      >> "$GPU_LOG"
  fi

  # Top RSS process
  ps -eo pid,rss,vsz,comm --sort=-rss --no-headers | head -3 \
    | awk -v ts="$NOW" '{printf("%s,%s,%.1f,%.1f,%s\n", ts, $1, $2/1024, $3/1024, $4)}' \
    >> "$TOP_LOG"

  # Append any new dmesg lines (OOM kills, kernel panics)
  dmesg -T 2>/dev/null | tail -5 >> "$DMESG_LOG"

  sleep "$INTERVAL"
done

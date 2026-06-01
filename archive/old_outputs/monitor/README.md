# Persistent crash monitor for run #28+

Two pollers write to `C:\Users\joshc\wsl_monitor\` (`/mnt/c/Users/joshc/wsl_monitor/` from
WSL). That path is on the **Windows** filesystem, so the logs survive a full WSL2 crash.

## Why two pollers

- **WSL side** (`monitor/wsl_log.sh`) — sees `/proc/meminfo`, swap, GPU, top processes, dmesg.
  Captures what the Linux kernel sees right up to the moment Vmmem dies.
- **Windows side** (`C:\Users\joshc\wsl_monitor\win_log.ps1`) — sees Vmmem.exe working set,
  Windows page file, system-wide available memory. **This is the only vantage point that
  survives a full WSL2 crash**, because if Vmmem dies the WSL logger dies with it.

If WSL2 goes catastrophic, look at the *last* line of `win_vmmem_*.csv` — that's the
moment Vmmem either ballooned past Windows' tolerance or got terminated. Cross-reference
with `win_sys_*.csv` (was Windows itself starved?) and `win_events_*.csv` (any Hyper-V
warning?).

## Start (must do BEFORE launching the notebook run)

### Windows side — start FIRST, in a Windows PowerShell window:

```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\joshc\wsl_monitor\win_log.ps1
```

Leave that window open. It logs every 3s. Ctrl+C to stop.

### WSL side — start from this repo:

```bash
nohup bash /home/joshcullinan/RDP_CNN/monitor/wsl_log.sh \
  > /tmp/wsl_log.stdout 2>&1 &
echo $! > /tmp/wsl_log.pid
```

Stop:

```bash
kill $(cat /tmp/wsl_log.pid)
```

## Files produced

| File | Source | Granularity |
|---|---|---|
| `wsl_mem_<ts>.csv` | WSL `/proc/meminfo` | every 3s |
| `wsl_gpu_<ts>.csv` | WSL `nvidia-smi` | every 3s |
| `wsl_top_<ts>.csv` | WSL `ps --sort=-rss` (top 3) | every 3s |
| `wsl_dmesg_<ts>.log` | WSL `dmesg -T` rolling tail | every 3s |
| `win_vmmem_<ts>.csv` | Windows Vmmem.exe metrics | every 3s |
| `win_sys_<ts>.csv` | Windows system memory + page file | every 3s |
| `win_events_<ts>.csv` | Hyper-V / System event log filtered for WSL/Vmmem | snapshot at start |

## Post-crash forensics

After a crash, in PowerShell:

```powershell
Get-WinEvent -LogName 'System' -MaxEvents 100 | Where-Object { $_.Message -match 'WSL|Vmmem|Hyper-V|memory' }
Get-WinEvent -LogName 'Microsoft-Windows-Hyper-V-Worker-Admin' -MaxEvents 50
```

Also tail the most recent `win_vmmem_*.csv` — the last few rows show the runup.

## Reducing pressure (separate from monitoring)

- `MAX_SEQ_LEN` halving from 32000 → 16000 cuts X memory by ~half. Most sequences are
  ≤10000 already; the tail is what's pushing budget.
- `BATCH_SIZE` is already 2; can't easily go lower.
- fp16 X storage already in place (cell-11).
- Check if Windows page file is on a fast drive; default is system-managed and adequate.

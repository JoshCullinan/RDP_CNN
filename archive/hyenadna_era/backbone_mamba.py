"""M1.1 — Mamba backbone skeleton (PyTorch).

SUPERSEDED 2026-05-19 by backbone_hyenadna.py. Kept as a record of the
Mamba investigation and to make a future revisit easier if a machine
with system CUDA toolkit becomes available.

The slow-path (sequential scan, no custom CUDA kernels) measured
**18.9 s/forward at L=30k, batch=2** on the RTX 3070 — 9× over the
master-plan budget of <2 s and ~3000 hours of wall time to pretrain MLM
over the 1.3M-sequence corpus. The fast path requires causal-conv1d /
mamba-ssm packages, both of which build their own CUDA kernels at
install time and require system `nvcc`; this dev box has TF's bundled
CUDA wheels only, no system toolkit, so the build fails.

backbone_hyenadna.py uses HF HyenaDNA-small-32k instead: pure PyTorch
(FFT-based long convs, no custom CUDA), 0.3 s/forward at L=30k, plus
pretrained on the human genome.

Original docstring follows:

`build_backbone(seq_len, d_model=256, n_layers=8)` returns a
`SequenceBackbone` that maps `(B, L, 5)` one-hot input → `(B, L, d_model)`
per-position embeddings. The model is HF's MambaModel with a
linear-projection head for the 5-channel one-hot input (vs the usual
discrete-token vocab embedding).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn

# HF Mamba — pure-PyTorch implementation, no custom CUDA needed (it'll
# use the slow_forward path if the optimized causal_conv1d kernel isn't
# available, which is fine for forward-pass smoke testing).
from transformers import MambaConfig, MambaModel


@dataclass
class BackboneConfig:
    seq_len: int = 30_000
    d_model: int = 256
    n_layers: int = 8
    input_channels: int = 5
    state_size: int = 16  # SSM state dim; HF default
    conv_kernel: int = 4  # depthwise conv inside each block; HF default


class SequenceBackbone(nn.Module):
    """One-hot → per-position embedding via stacked Mamba blocks.

    Input:  (B, L, C=5) float — nucleotide one-hot, gap = channel 4
    Output: (B, L, d_model) float — per-position contextual embeddings

    Wraps HuggingFace MambaModel. We bypass MambaModel's nn.Embedding
    (which expects integer token IDs) by feeding `inputs_embeds` directly,
    after a 5→d_model linear projection.
    """

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_channels, cfg.d_model)

        mamba_cfg = MambaConfig(
            vocab_size=cfg.input_channels,  # unused — we pass inputs_embeds
            hidden_size=cfg.d_model,
            num_hidden_layers=cfg.n_layers,
            state_size=cfg.state_size,
            conv_kernel=cfg.conv_kernel,
            use_cache=False,
            use_mambapy=True,  # use mambapy backend instead of slow path
        )
        self.mamba = MambaModel(mamba_cfg)
        # Tie the proj's bias init to 0 — the residual stream learns offsets.
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C). Project to (B, L, d_model) then feed Mamba.
        h = self.input_proj(x)
        out = self.mamba(inputs_embeds=h, return_dict=True)
        # MambaModel returns last_hidden_state of shape (B, L, hidden_size).
        return out.last_hidden_state

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_backbone(seq_len: int = 30_000, d_model: int = 256,
                   n_layers: int = 8) -> SequenceBackbone:
    return SequenceBackbone(BackboneConfig(
        seq_len=seq_len, d_model=d_model, n_layers=n_layers
    ))


# ---------- smoke test ------------------------------------------------------

def smoke_test(seq_len: int = 30_000, batch: int = 2,
               d_model: int = 256, n_layers: int = 8,
               device: str | None = None) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}  seq_len={seq_len}  batch={batch}  "
          f"d_model={d_model}  n_layers={n_layers}", flush=True)

    model = build_backbone(seq_len=seq_len, d_model=d_model, n_layers=n_layers)
    model.to(device).eval()
    print(f"[smoke] params={model.n_params:,}", flush=True)

    # Random one-hot input. Real data has exactly one channel set per
    # position (or zero, for padding past content_end); a random uniform
    # is denser than that but stresses the forward path the same way.
    x = torch.zeros(batch, seq_len, 5, device=device)
    idx = torch.randint(0, 5, (batch, seq_len), device=device)
    x.scatter_(2, idx.unsqueeze(-1), 1.0)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        y = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"[smoke] forward: {elapsed:.2f} s  output.shape={tuple(y.shape)}",
          flush=True)

    info = {"params": model.n_params, "output_shape": tuple(y.shape),
            "elapsed_s": elapsed, "device": device}
    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 2**20
        free_mb, total_mb = torch.cuda.mem_get_info()
        free_mb /= 2**20
        total_mb /= 2**20
        info.update({
            "gpu_peak_mb": peak_mb,
            "gpu_free_mb_after": free_mb,
            "gpu_total_mb": total_mb,
        })
        print(f"[smoke] GPU peak {peak_mb:.0f} MB / total {total_mb:.0f} MB; "
              f"free after {free_mb:.0f} MB", flush=True)

    return info


if __name__ == "__main__":
    smoke_test()

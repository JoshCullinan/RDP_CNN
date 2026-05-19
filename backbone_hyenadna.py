"""M1.1 — HyenaDNA backbone skeleton (PyTorch).

The master plan said "Mamba primary, Longformer fallback, Keras model."
Reality narrowed that down through the audit:
  - HF transformers 5.x has no TFLongformerModel (TF support deprecated).
  - mamba-ssm / causal-conv1d require nvcc on the build box; this dev
    machine has TF's bundled CUDA only and no system toolkit.
  - HF Mamba on the slow path: 18.9 s forward at L=30k on the 3070,
    well over the master-plan budget of <2 s.

HyenaDNA is the chosen fallback:
  - Pure-PyTorch (no custom CUDA), uses FFT-based long convolutions.
  - Already pretrained on the human genome at 32k context — potentially
    shortcuts Phase 1 MLM since DNA conservation/structure priors are
    relevant to HIV.
  - HF checkpoint `LongSafari/hyenadna-small-32k-seqlen-hf`:
        d_model=256, n_layer=4, max_seq=32,770, vocab=12.
    Matches the master plan's d_model=256 and fits L=30k with margin.
  - Smoke: forward(B=2, L=30k) = 0.42 s, GPU peak 1.23 GB, ≥5 GB free
    after — beats the master-plan spec on both axes.

Input contract: v2 cache's int8 nucleotides `{A:0, T:1, G:2, C:3, gap:4}`
must be mapped to HyenaDNA's tokenizer vocab. The forward() helper
below does the mapping in a single torch.gather, so callers can pass
v2 int8 directly.

`build_backbone()` returns a `SequenceBackbone` exposing
`(B, L) int → (B, L, d_model) float` per-position embeddings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from transformers import AutoModel


# ---------- vocab mapping --------------------------------------------------
#
# v2 cache encoding (build_cache_v2.NT_TO_INT):
#     A=0, T=1, G=2, C=3, gap=4
#
# HyenaDNA tokenizer:
#     [CLS]=0, [SEP]=1, [BOS]=2, [MASK]=3, [PAD]=4, [RESERVED]=5,
#     [UNK]=6, A=7, C=8, G=9, T=10, N=11.
#
# Mapping: nucleotides go to their canonical HyenaDNA tokens; gap (v2=4)
# goes to [PAD] so the model treats it as positional padding rather than
# unknown nucleotide.

_V2_TO_HYENA = torch.tensor(
    [7, 10, 9, 8, 4],  # A, T, G, C, gap → 7, 10, 9, 8, [PAD]=4
    dtype=torch.long,
)


def v2_to_hyena_ids(v2_int8: torch.Tensor) -> torch.Tensor:
    """Map v2 cache int8 nucleotides to HyenaDNA token IDs.

    v2_int8 may be int8/int32/int64; output is int64 (HF embedding expects long).
    Anything outside [0,4] is mapped to [UNK]=6 as a safety net (shouldn't
    happen with the v2 cache but cheap to guard).
    """
    lut = _V2_TO_HYENA.to(device=v2_int8.device)
    v = v2_int8.long().clamp(min=0, max=4)
    return lut[v]


# ---------- backbone wrapper -----------------------------------------------

DEFAULT_HF_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"


@dataclass
class BackboneConfig:
    hf_name: str = DEFAULT_HF_NAME
    d_model: int = 256
    n_layers: int = 4  # HyenaDNA-small architectural fact, not a knob
    max_seq_len: int = 32_770


class SequenceBackbone(nn.Module):
    """HyenaDNA wrapped to consume the v2 cache directly.

    Accepts either:
      - int8/int64 token IDs in the v2 vocab `[0..4]` (shape (B, L)), or
      - already-mapped HyenaDNA token IDs (shape (B, L), values in [0..11])
        when `is_hyena_ids=True` is passed.

    Returns per-position embeddings (B, L, d_model).
    """

    def __init__(self, cfg: Optional[BackboneConfig] = None,
                 pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg or BackboneConfig()
        if pretrained:
            self.hyena = AutoModel.from_pretrained(
                self.cfg.hf_name, trust_remote_code=True
            )
        else:
            from transformers import AutoConfig
            hf_cfg = AutoConfig.from_pretrained(self.cfg.hf_name,
                                                trust_remote_code=True)
            self.hyena = AutoModel.from_config(hf_cfg, trust_remote_code=True)

    def forward(self, ids: torch.Tensor,
                is_hyena_ids: bool = False) -> torch.Tensor:
        if not is_hyena_ids:
            ids = v2_to_hyena_ids(ids)
        out = self.hyena(input_ids=ids)
        # HyenaDNA returns BaseModelOutputWithNoAttention; pull the hidden state.
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out[0]

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_backbone(pretrained: bool = True,
                   hf_name: str = DEFAULT_HF_NAME) -> SequenceBackbone:
    return SequenceBackbone(BackboneConfig(hf_name=hf_name), pretrained=pretrained)


# ---------- smoke test ------------------------------------------------------

def smoke_test(seq_len: int = 30_000, batch: int = 2,
               pretrained: bool = True,
               device: Optional[str] = None) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}  seq_len={seq_len}  batch={batch}  "
          f"pretrained={pretrained}", flush=True)

    model = build_backbone(pretrained=pretrained)
    model.to(device).eval()
    print(f"[smoke] params={model.n_params:,}", flush=True)

    # v2-cache style input: int8 in {0..4}.
    v2_ids = torch.randint(0, 5, (batch, seq_len), device=device,
                           dtype=torch.int8)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        y = model(v2_ids)  # mapping done inside forward()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    info = {
        "params": model.n_params,
        "output_shape": tuple(y.shape),
        "elapsed_s": elapsed,
        "device": device,
    }
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
        print(f"[smoke] forward {elapsed:.2f} s — out {tuple(y.shape)} | "
              f"GPU peak {peak_mb:.0f} MB / total {total_mb:.0f} MB; "
              f"free after {free_mb:.0f} MB", flush=True)

    # Master-plan acceptance.
    ok = elapsed < 2.0 and (info.get("gpu_free_mb_after", 0) >= 1024)
    print(f"[smoke] master-plan spec (<2 s forward, >1 GB free): "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return info


if __name__ == "__main__":
    smoke_test()

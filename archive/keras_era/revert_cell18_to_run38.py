#!/usr/bin/env python3
"""Revert cell-18 to run #38 architecture (residual dilated stack with
LayerNorm, n_filters=128). Run #41 holds architecture constant at the
known-good run #38 baseline; the experiment only varies data + split.
"""
import json
from pathlib import Path

NB = Path('CNN.ipynb')
RUN38_CELL18 = Path('/tmp/run38_cell18.py').read_text()

nb = json.loads(NB.read_text())
for c in nb['cells']:
    if c.get('id') == 'cell-18':
        c['source'] = RUN38_CELL18.splitlines(keepends=True)
        break
NB.write_text(json.dumps(nb, indent=1))
print(f"Reverted cell-18 to run #38 architecture ({len(RUN38_CELL18)} chars).")

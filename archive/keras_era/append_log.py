#!/usr/bin/env python3
"""Append a new entry to cell-experiment-log. Call:
  python3 append_log.py <entry_path>
"""
import json, sys
from pathlib import Path
NB = Path('/home/joshcullinan/RDP_CNN/CNN.ipynb')
text = Path(sys.argv[1]).read_text()
nb = json.loads(NB.read_text())
for c in nb['cells']:
    if c.get('id') == 'cell-experiment-log':
        existing = ''.join(c['source']).rstrip()
        c['source'] = (existing + '\n\n' + text + '\n').splitlines(keepends=True)
        break
NB.write_text(json.dumps(nb, indent=1) + '\n')
print('appended.')

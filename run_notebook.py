#!/usr/bin/env python3
"""Execute CNN.ipynb in-place with nbclient. Used for background runs."""
import sys
import time
import nbformat
from nbclient import NotebookClient


def main():
    print(f"[{time.strftime('%H:%M:%S')}] reading notebook", flush=True)
    nb = nbformat.read('CNN.ipynb', as_version=4)
    print(f"[{time.strftime('%H:%M:%S')}] {len(nb.cells)} cells; starting execution", flush=True)

    client = NotebookClient(
        nb,
        timeout=21600,  # 6h — LayerNorm runs are ~2.5h+
        kernel_name='python3',
        allow_errors=False,
    )

    start = time.time()
    try:
        client.execute()
        elapsed = time.time() - start
        print(f"[{time.strftime('%H:%M:%S')}] execution complete in {elapsed:.0f}s", flush=True)
    except Exception as e:
        elapsed = time.time() - start
        print(f"[{time.strftime('%H:%M:%S')}] EXECUTION ERROR after {elapsed:.0f}s: {type(e).__name__}: {e}", flush=True)
        nbformat.write(nb, 'CNN.ipynb')
        sys.exit(1)

    print(f"[{time.strftime('%H:%M:%S')}] writing notebook", flush=True)
    nbformat.write(nb, 'CNN.ipynb')
    print(f"[{time.strftime('%H:%M:%S')}] done", flush=True)


if __name__ == '__main__':
    main()

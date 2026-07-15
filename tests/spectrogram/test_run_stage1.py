import json
from pathlib import Path
from spectrogram.run_stage1 import write_preregistration

def test_prereg_written(tmp_path):
    p = tmp_path / "prereg.json"
    write_preregistration(p)
    d = json.loads(Path(p).read_text())
    assert d["primary_metric"].startswith("mean held-out-CRF")
    assert "decision_rule" in d and "power_statement" in d

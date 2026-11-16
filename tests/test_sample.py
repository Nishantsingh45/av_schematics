"""The provided sample drawing must reproduce its answer key exactly.

Run with:  python -m pytest   (or)   python tests/test_sample.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avconn.evaluate import score  # noqa: E402
from avconn.pipeline import extract_file  # noqa: E402

SAMPLE = ROOT / "AI ML Assignment" / "Sample"


def test_sample_matches_answer_key():
    result = extract_file(SAMPLE / "Sample Line Schematics Drawing.pdf")
    truth = json.loads((SAMPLE / "sample-output.json").read_text(encoding="utf-8"))
    s = score([c.as_output() for c in result.connections], truth)
    assert s["strict"]["f1"] == 1.0, s


def test_rack_layout_has_no_connections():
    rack = ROOT / "AI ML Assignment" / "Line Schematic Drawings" / "7619_ORC Chamber_ Chambers-Line Schematics-02-01-2026.pdf"
    result = extract_file(rack)
    assert result.connections == []
    assert result.pages[0].status == "no_cables"


if __name__ == "__main__":
    test_sample_matches_answer_key()
    test_rack_layout_has_no_connections()
    print("ok")

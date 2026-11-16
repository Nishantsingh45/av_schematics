"""Drawing patterns from the real drawings, rebuilt as a tiny 2-page PDF:
an off-page link split across pages, a dashed cable, a dashed zone border,
a chain of identical devices, and a port label printed twice on top of itself.

Run with:  python -m pytest   (or)   python tests/test_synthetic.py
"""
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avconn.evaluate import score  # noqa: E402
from avconn.pipeline import extract_file  # noqa: E402

FS = 6  # font size


def _block(page, x, y, name, left=(), right=(), twice=False):
    """A device block; returns {port name: outer end of its connector line}."""
    h = 14 * max(len(left), len(right), 1) + 22
    page.draw_rect(pymupdf.Rect(x, y, x + 140, y + h), color=(0, 0, 0), width=0.8)
    page.insert_text((x + (140 - pymupdf.get_text_length(name, fontsize=FS)) / 2, y + h - 6), name, fontsize=FS)
    ends = {}
    for i, p in enumerate(left):
        yy = y + 10 + 14 * i
        for _ in range(2 if twice else 1):
            page.insert_text((x + 4, yy + 2), p, fontsize=FS)
        page.draw_line((x - 15, yy), (x, yy), color=(0, 0, 0), width=0.7)
        ends[p] = (x - 15, yy)
    for i, p in enumerate(right):
        yy = y + 10 + 14 * i
        page.insert_text((x + 136 - pymupdf.get_text_length(p, fontsize=FS), yy + 2), p, fontsize=FS)
        page.draw_line((x + 140, yy), (x + 155, yy), color=(0, 0, 0), width=0.7)
        ends[p] = (x + 155, yy)
    return ends


def _dot(page, p):
    page.draw_circle(p, 1.8, color=None, fill=(0, 0, 0))


def _cable(page, pts):
    page.draw_polyline(pts, color=(0, 0, 1), width=0.7)


def _dashed(page, pts, colour=(0, 0, 1)):
    """A dashed line exported the way the real drawings do it: one fill path of small dashes."""
    shape = page.new_shape()
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / n, (y1 - y0) / n
        nx, ny = -uy * 0.36, ux * 0.36
        s = 0.0
        while s < n - 0.5:
            e = min(s + 2.2, n)
            a, b = (x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)
            shape.draw_polyline([(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny), (b[0] + ux * 0.2, b[1] + uy * 0.2),
                                 (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny), (a[0] + nx, a[1] + ny)])
            s += 4.4
    shape.finish(color=None, fill=colour, closePath=False)
    shape.commit()


def _link(page, centre, ref):
    page.draw_circle(centre, 10, color=(0, 0, 1), width=0.7)
    page.insert_text((centre[0] - pymupdf.get_text_length(ref, fontsize=FS) / 2, centre[1] + 2), ref, fontsize=FS)


def _make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    p1 = doc.new_page(width=800, height=600)
    a = _block(p1, 100, 100, "DEV A", right=["HDMI OP1", "LAN OP2"])
    c = _block(p1, 450, 200, "DEV C", left=["LAN IP1"], twice=True)
    # A -> off-page circle 5 (continues on page 2)
    _dot(p1, a["HDMI OP1"])
    _cable(p1, [a["HDMI OP1"], (380, a["HDMI OP1"][1])])
    _link(p1, (390, a["HDMI OP1"][1]), "5")
    # dashed cable A -> C
    _dot(p1, a["LAN OP2"])
    _dot(p1, c["LAN IP1"])
    _dashed(p1, [a["LAN OP2"], (330, a["LAN OP2"][1]), (330, c["LAN IP1"][1]), c["LAN IP1"]])
    # dashed zone border around everything (must not become a cable)
    _dashed(p1, [(40, 40), (760, 40), (760, 560), (40, 560), (40, 40)], colour=(0, 0, 0))
    # chain of three identical lights
    lights = [_block(p1, 80 + 230 * i, 400, "ETC LIGHT X", left=["DMX IP1"], right=["DMX THRU OP1"]) for i in range(3)]
    for k in range(2):
        s, t = lights[k]["DMX THRU OP1"], lights[k + 1]["DMX IP1"]
        _dot(p1, s)
        _dot(p1, t)
        _cable(p1, [s, t])

    p2 = doc.new_page(width=800, height=600)
    b = _block(p2, 400, 100, "DEV B", left=["HDMI IP1"])
    _link(p2, (200, b["HDMI IP1"][1]), "5")
    _dot(p2, b["HDMI IP1"])
    _cable(p2, [(210, b["HDMI IP1"][1]), b["HDMI IP1"]])
    doc.save(path)


def _rows(result):
    return Counter((c.source_device, c.source_port, c.target_device, c.target_port) for c in result.connections)


def test_synthetic_patterns():
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "synthetic.pdf"
        _make_pdf(pdf)
        result = extract_file(pdf)
    assert _rows(result) == Counter({
        ("DEV A", "HDMI OP1", "DEV B", "HDMI IP1"): 1,                    # joined across pages
        ("DEV A", "LAN OP2", "DEV C", "LAN IP1"): 1,                      # dashed cable; label printed twice
        ("ETC LIGHT X", "DMX THRU OP1", "ETC LIGHT X", "DMX IP1"): 2,     # one row per cable in the chain
    }), _rows(result)
    joined = [c for c in result.connections if c.source_port == "HDMI OP1"][0]
    assert "joined_offpage" in joined.flags


def test_score_counts_identical_rows():
    row = {"source_device": "L", "source_port": "THRU OP1", "target_device": "L", "target_port": "IP1"}
    s = score([row], [row, row, row])
    assert s["strict"]["matched"] == 1 and s["strict"]["recall"] == round(1 / 3, 3), s
    assert len(s["missing"]) == 2, s


if __name__ == "__main__":
    test_synthetic_patterns()
    test_score_counts_identical_rows()
    print("ok")

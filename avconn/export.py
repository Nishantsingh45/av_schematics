"""Write results: the connections JSON, a report, and a debug picture per page."""
from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from .pipeline import FileResult, PageResult

_PALETTE = [(0.90, 0.10, 0.10), (0.10, 0.45, 0.90), (0.10, 0.65, 0.20), (0.85, 0.50, 0.00),
            (0.60, 0.20, 0.80), (0.00, 0.65, 0.65), (0.85, 0.15, 0.55), (0.45, 0.35, 0.10)]


def write_outputs(result: FileResult, out_dir: Path, debug: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = result.path.stem
    main = out_dir / f"{stem}.json"
    if result.not_processed:
        # Nothing was read: an empty list would wrongly say "this drawing has no connections".
        main.unlink(missing_ok=True)
    else:
        main.write_text(json.dumps([c.as_output() for c in result.connections], indent=2), encoding="utf-8")

    report = {
        "file": result.path.name,
        "connections": len(result.connections),
        "pages": [{
            "page": p.page, "reader": p.reader, "status": p.status, "note": p.note, "stats": p.stats,
            "connections": [dict(c.as_output(), confidence=c.confidence, flags=c.flags, cable=c.cable)
                            for c in p.connections],
        } for p in result.pages],
    }
    (out_dir / f"{stem}.report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if debug and result.path.suffix.lower() == ".pdf":
        with pymupdf.open(result.path) as doc:
            for p in result.pages:
                if p.elements is not None:
                    _debug_picture(doc[p.page - 1], p, out_dir / f"{stem}.p{p.page}.debug.png")
    return main


def _debug_picture(page, p: PageResult, out_png: Path) -> None:
    """Draw what the tool understood on top of the page: blocks, cables, cable ends."""
    u = p.elements.unit
    shape = page.new_shape()
    for blk in p.blocks:
        b = blk.box
        shape.draw_rect(pymupdf.Rect(b.x0, b.y0, b.x1, b.y1))
        shape.finish(color=(0.0, 0.6, 0.0), width=0.35 * u, stroke_opacity=0.6, closePath=False)
    for i, nt in enumerate(p.nets):
        col = _PALETTE[i % len(_PALETTE)]
        for s in nt.segments:
            shape.draw_line((s.x0, s.y0), (s.x1, s.y1))
        shape.finish(color=col, width=0.45 * u, stroke_opacity=0.45, closePath=False)
        for t in nt.terminals:
            shape.draw_circle((t.x, t.y), 0.7 * u)
            shape.finish(color=col, fill=col if t.block is not None else None, width=0.2 * u, fill_opacity=0.8)
    shape.commit()
    zoom = max(1.0, min(4.0, 3600 / max(page.rect.width, page.rect.height)))
    page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(out_png)

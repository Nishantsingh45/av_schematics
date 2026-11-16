"""Reader for PDF pages that carry real drawing data (text + vector paths).

Nothing here depends on a particular drawing style: no colours, no fixed sizes.
Shapes are classified by geometry only:
  dot  = small filled closed curve
  box  = closed outline that hugs its bounding rectangle (plain or rounded)
  wire = any other visible stroke (curves inside a stroke are line hops)
  dashed wire = a row of small identical filled dashes that does not close on itself
  off-page marker = small outlined circle with a short text (the link number) inside
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

import pymupdf

from .elements import Box, Dot, Marker, PageElements, Segment, TextLine

_WHITE = 0.97  # strokes/fills this light are invisible on white paper


def _is_white(c) -> bool:
    return c is not None and min(c) >= _WHITE


def has_vector_content(page: "pymupdf.Page") -> bool:
    """True if the page has real text and line work (not just a scanned picture)."""
    if len(page.get_text("words")) < 3:
        return False
    n_strokes = 0
    for d in page.get_drawings():
        if d.get("color") is not None:
            n_strokes += 1
            if n_strokes >= 3:
                return True
    return False


def _read_texts(page) -> list[TextLine]:
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            # Some exports store every device name twice, once fully transparent.
            spans = [s for s in line["spans"] if s.get("alpha", 255) > 0 and s["text"].strip()]
            if not spans:
                continue
            text = " ".join("".join(s["text"] for s in spans).split())
            x0 = min(s["bbox"][0] for s in spans)
            y0 = min(s["bbox"][1] for s in spans)
            x1 = max(s["bbox"][2] for s in spans)
            y1 = max(s["bbox"][3] for s in spans)
            rotated = abs(line["dir"][1]) > 0.1
            out.append(TextLine(x0, y0, x1, y1, text, rotated))
    # Some exports print the same label several times exactly on top of itself.
    unique, placed = [], defaultdict(list)
    for t in out:
        if any(abs(t.x0 - o.x0) + abs(t.y0 - o.y0) < 0.3 * min(t.h, o.h) for o in placed[t.text]):
            continue
        placed[t.text].append(t)
        unique.append(t)
    return unique


def _marker_ref(texts: list[TextLine], rect) -> str:
    """The short text written inside a circle (an off-page link number), or ''."""
    cx, cy, r = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2, max(rect.width, rect.height) / 2
    inside = [t for t in texts if math.hypot(t.xc - cx, t.yc - cy) <= 0.6 * r and t.w <= 1.8 * r
              and 0 < len(t.text.replace(" ", "")) <= 6]
    return min(inside, key=lambda t: math.hypot(t.xc - cx, t.yc - cy)).text if inside else ""


def _piece_boxes(items) -> list[tuple[float, float, float, float]]:
    """Bounding box of each separate piece (sub-path) of a path made of lines and curves."""
    groups, cur, prev = [], [], None
    for it in items:
        if it[0] not in ("l", "c"):
            return []
        if cur and abs(it[1] - prev) > 1e-3:
            groups.append(cur)
            cur = []
        cur.extend(p for p in it[1:] if hasattr(p, "x"))
        prev = it[2] if it[0] == "l" else it[4]
    if cur:
        groups.append(cur)
    return [(min(p.x for p in g), min(p.y for p in g), max(p.x for p in g), max(p.y for p in g)) for g in groups]


def _extend(a: tuple, b: tuple, d: float) -> tuple:
    """Point b moved a further distance d away from a."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    return (b[0] + dx / n * d, b[1] + dy / n * d)


def _simplify(pts: list[tuple], tol: float) -> list[tuple]:
    """Drop points that lie on the straight line between their neighbours."""
    out = [pts[0]]
    for b, c in zip(pts[1:-1], pts[2:]):
        a = out[-1]
        off = abs((c[0] - a[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (c[1] - a[1])) / (math.dist(a, c) or 1.0)
        if off > tol:
            out.append(b)
    out.append(pts[-1])
    return out


def _dashed_lines(items, unit: float) -> list[tuple[list[tuple], float]]:
    """A dashed line drawn as a row of small filled dashes -> its centre line.

    Returns (points, dash spacing) for each line in the path. A row that closes on
    itself is a dashed outline (zone or room border), not a cable, and is dropped.
    """
    boxes = _piece_boxes(items)
    if len(boxes) < 6:
        return []
    # Compare dashes by their size (box diagonal), which barely changes when a dash is tilted.
    sizes = [math.hypot(b[2] - b[0], b[3] - b[1]) for b in boxes]
    thins = [min(b[2] - b[0], b[3] - b[1]) for b in boxes]
    L, S, T = statistics.median(max(b[2] - b[0], b[3] - b[1]) for b in boxes), statistics.median(sizes), statistics.median(thins)
    if not (0 < T < 0.75 * S and S <= 0.8 * unit):
        return []
    if sum(1 for s in sizes if abs(s - S) <= 0.3 * S) < 0.7 * len(boxes):
        return []  # pieces of different sizes: artwork, not a dash pattern
    centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    gaps = [math.dist(a, b) for a, b in zip(centres, centres[1:])]
    step = statistics.median(gaps)
    if step <= 0 or sum(1 for g in gaps if 0.5 * step <= g <= 1.5 * step) < 0.7 * len(gaps):
        return []  # dashes are evenly spaced along a line; artwork pieces are not
    runs, cur = [], [centres[0]]
    for c, g in zip(centres[1:], gaps):
        if g > 2.5 * step:  # a jump: the next dashes belong to another line in the same path
            runs.append(cur)
            cur = []
        cur.append(c)
    runs.append(cur)
    lines = []
    for pts in runs:
        if len(pts) < 3 or math.dist(pts[0], pts[-1]) <= 2.5 * step:
            continue
        pts = [_extend(pts[1], pts[0], L / 2), *pts[1:-1], _extend(pts[-2], pts[-1], L / 2)]
        lines.append((_simplify(pts, 0.1 * unit), step))
    return lines


def _is_boxlike(items, rect, unit: float) -> bool:
    """Closed path whose straight parts run along its own bounding rectangle."""
    if rect.width < 2 * unit or rect.height < 1.5 * unit:
        return False
    first = items[0][1]
    last = items[-1][-1] if items[-1][0] != "l" else items[-1][2]
    try:
        closed = abs(first.x - last.x) < 0.1 * unit and abs(first.y - last.y) < 0.1 * unit
    except AttributeError:
        return False
    if not closed:
        return False
    tol = 0.1 * unit
    on_border = 0.0
    for it in items:
        if it[0] != "l":
            continue
        p, q = it[1], it[2]
        for same in (
            abs(p.y - rect.y0) < tol and abs(q.y - rect.y0) < tol,
            abs(p.y - rect.y1) < tol and abs(q.y - rect.y1) < tol,
            abs(p.x - rect.x0) < tol and abs(q.x - rect.x0) < tol,
            abs(p.x - rect.x1) < tol and abs(q.x - rect.x1) < tol,
        ):
            if same:
                on_border += abs(p.x - q.x) + abs(p.y - q.y)
                break
    return on_border >= 0.55 * 2 * (rect.width + rect.height)


def read_page(page: "pymupdf.Page") -> PageElements:
    texts = _read_texts(page)
    heights = [t.h for t in texts if not t.rotated]
    unit = statistics.median(heights) if heights else 8.0
    el = PageElements(width=page.rect.width, height=page.rect.height, unit=unit, texts=texts)

    seen_boxes = set()
    path_id = 0
    dashed: list[tuple[list[tuple], float]] = []

    def add_box(r) -> None:
        if r.width < 2 * unit or r.height < 1.5 * unit:
            return
        key = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
        if key not in seen_boxes:
            seen_boxes.add(key)
            el.boxes.append(Box(r.x0, r.y0, r.x1, r.y1))

    for d in page.get_drawings():
        items = d["items"]
        if not items:
            continue
        stroke, fill, rect = d.get("color"), d.get("fill"), d["rect"]
        stroked = d["type"] in ("s", "fs") and stroke is not None and not _is_white(stroke)
        filled = d["type"] in ("f", "fs") and fill is not None and not _is_white(fill)
        if not (stroked or filled):
            continue  # invisible (e.g. white masks behind cable tags)
        kinds = {it[0] for it in items}

        if kinds == {"c"}:
            size = max(rect.width, rect.height)
            round_ = min(rect.width, rect.height) >= 0.8 * size
            if filled and 0 < size <= 1.5 * unit and min(rect.width, rect.height) >= 0.6 * size:
                el.dots.append(Dot((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2, size / 2))
            elif stroked and not filled and round_ and 1.5 * unit <= size <= 8 * unit:
                ref = _marker_ref(texts, rect)
                if ref:
                    el.markers.append(Marker((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2, size / 2, ref))
            continue  # any other pure-curve shape is a symbol (icon), not a wire

        if kinds & {"re", "qu"}:
            for it in items:
                if it[0] == "re":
                    add_box(it[1])
                elif it[0] == "qu":
                    add_box(it[1].rect)
            continue

        if _is_boxlike(items, rect, unit):
            add_box(rect)
            continue

        if not stroked:
            # Filled artwork (icons), or a dashed line exported as a row of filled dashes.
            dashed.extend(_dashed_lines(items, unit))
            continue
        # A line hop over a crossing is a small curve inside the stroke: keep its
        # chord so the cable stays continuous.
        pieces = [(it[1], it[2] if it[0] == "l" else it[4]) for it in items if it[0] in ("l", "c")]
        path_id += 1
        for k, (p, q) in enumerate(pieces):
            starts = k == 0 or abs(pieces[k - 1][1] - p) > 1e-3
            stops = k == len(pieces) - 1 or abs(q - pieces[k + 1][0]) > 1e-3
            el.segments.append(Segment(p.x, p.y, q.x, q.y, path_id, starts, stops))

    # A dashed line is a cable only if it reaches a connection dot; zone borders and
    # other dashed decoration never do. Snap its ends onto the dots it reaches.
    for pts, step in dashed:
        pts, reached = list(pts), 0
        for i in (0, -1):
            x, y = pts[i]
            d = min(el.dots, key=lambda d: math.hypot(d.x - x, d.y - y), default=None)
            if d is not None and math.hypot(d.x - x, d.y - y) <= d.r + step:
                pts[i] = (d.x, d.y)
                reached += 1
        if not reached:
            continue
        path_id += 1
        for k, (p, q) in enumerate(zip(pts, pts[1:])):
            el.segments.append(Segment(p[0], p[1], q[0], q[1], path_id, k == 0, k == len(pts) - 2))
    return el

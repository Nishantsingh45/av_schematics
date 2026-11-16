"""Find device blocks and read them, following the block anatomy in the brief:

      DEVICE TYPE            <- label outside the box (above or below)
   +-----------+-----------+
  -| PORT NAME | PORT NAME |- <- port rows; a short connector line leaves the side
   +-----------+-----------+
   |     PRODUCT NAME      |  <- spans both columns (top or bottom of the box)
   +-----------------------+

A box is a device block because of this structure (text inside + cables or
connector lines leaving its left/right side), not because of colour or shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .elements import Box, PageElements, TextLine


@dataclass
class Block:
    box: Box
    name: str = ""          # product name (brand + model) as drawn
    secondary: str = ""     # extra product label printed under the box, if any
    type_label: str = ""    # generic role, e.g. DISPLAY
    marks: dict = field(default_factory=lambda: {"L": [], "R": []})  # y of each connector line / dot
    inside: list = field(default_factory=list)

    @property
    def display_name(self) -> str:
        base = self.name or self.type_label
        return f"{base} ({self.secondary})" if self.secondary and base else base


def _join(lines: list[TextLine]) -> str:
    return " ".join(t.text for t in sorted(lines, key=lambda t: (round(t.y0, 1), t.x0)))


def stack_group(seed: TextLine, pool: list[TextLine], unit: float) -> list[TextLine]:
    """The seed line plus the lines stacked directly above/below it (one multi-line label)."""
    group, rest = [seed], [t for t in pool if t is not seed]
    grew = True
    while grew:
        grew = False
        gy0, gy1 = min(t.y0 for t in group), max(t.y1 for t in group)
        gx0, gx1 = min(t.x0 for t in group), max(t.x1 for t in group)
        for t in rest:
            gap = max(t.y0 - gy1, gy0 - t.y1)
            overlap = min(t.x1, gx1) - max(t.x0, gx0)
            if gap < 0.8 * unit and overlap > 0:
                group.append(t)
                rest.remove(t)
                grew = True
                break
    return group


def first_word(s: str) -> str:
    return s.split()[0].upper() if s.split() else ""


def find_blocks(el: PageElements) -> list[Block]:
    eps, unit = el.eps, el.unit
    horizontal = [s for s in el.segments if abs(s.y0 - s.y1) <= eps]
    cands: list[Block] = []
    for b in el.boxes:
        blk = Block(box=b)
        for s in horizontal:
            if not (b.y0 - eps <= s.y0 <= b.y1 + eps):
                continue
            lo, hi = min(s.x0, s.x1), max(s.x0, s.x1)
            if abs(hi - b.x0) <= eps and lo < b.x0 - eps:
                blk.marks["L"].append(s.y0)
            elif abs(lo - b.x1) <= eps and hi > b.x1 + eps:
                blk.marks["R"].append(s.y0)
        for d in el.dots:
            if b.y0 - eps <= d.y <= b.y1 + eps:
                if abs(d.x - b.x0) <= d.r + eps:
                    blk.marks["L"].append(d.y)
                elif abs(d.x - b.x1) <= d.r + eps:
                    blk.marks["R"].append(d.y)
        if not (blk.marks["L"] or blk.marks["R"]):
            continue
        blk.inside = [t for t in el.texts if not t.rotated and b.contains_text(t, eps)]
        if not blk.inside:
            continue
        cands.append(blk)

    # A box that contains other device blocks is a zone/room outline, not a device.
    blocks = [c for c in cands if not any(o is not c and c.box.contains_box(o.box, eps) for o in cands)]

    in_any = {id(t) for blk in blocks for t in blk.inside}
    outside = [t for t in el.texts if not t.rotated and id(t) not in in_any]
    for blk in blocks:
        b, mid = blk.box, blk.box.xc
        d = 0.25 * unit
        blk.name = _join([t for t in blk.inside if t.x0 < mid - d and t.x1 > mid + d])

        above = [t for t in outside if 0 <= b.y0 - t.y1 <= 2.5 * unit and t.x1 > b.x0 and t.x0 < b.x1]
        if above:
            seed = max(above, key=lambda t: t.y1)
            blk.type_label = _join(stack_group(seed, outside, unit))
        below = [t for t in outside if 0 <= t.y0 - b.y1 <= 3.5 * unit and b.x0 <= t.xc <= b.x1]
        if below:
            seed = min(below, key=lambda t: t.y0)
            label = _join(stack_group(seed, outside, unit))
            # A label under the box that repeats the brand is a second product line;
            # otherwise it is the device type (some styles print the type underneath).
            if blk.name and first_word(label) == first_word(blk.name):
                blk.secondary = label
            elif not blk.type_label:
                blk.type_label = label
    return blocks


def port_label(blk: Block, side: str, y: float, unit: float) -> str:
    """Text of the port row at height y on the given side ('L' or 'R') of the block."""
    mid, d = blk.box.xc, 0.25 * unit
    window = 1.2 * unit
    # Marks closer than one line of text are one port (e.g. a connector line and a dot drawn
    # slightly off it): two port names cannot sit that close without overlapping.
    rows: list[list[float]] = []
    for v in sorted(set(round(v, 1) for v in blk.marks[side])):
        if rows and v - rows[-1][-1] < 0.9 * unit:
            rows[-1].append(v)
        else:
            rows.append([v])
    ys = [sum(r) / len(r) for r in rows]
    gaps = [b - a for a, b in zip(ys, ys[1:]) if b - a > 0.5 * unit]
    if gaps:
        window = min(window, 0.55 * min(gaps))
    rows = []
    for t in blk.inside:
        if t.x0 < mid - d and t.x1 > mid + d:
            continue  # spans both columns: product name, not a port
        if (side == "L") != (t.xc < mid):
            continue
        if abs(t.yc - y) <= window:
            rows.append(t)
    return _join(rows)

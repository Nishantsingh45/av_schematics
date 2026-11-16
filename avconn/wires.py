"""Trace cables: join strokes into nets and find where each net ends.

A net is a set of strokes that are electrically one cable:
  - strokes sharing an end point,
  - strokes glued by a dot (cable attachment or junction),
  - a stroke ending exactly on another stroke (T-junction).
Two strokes that merely cross are NOT joined (drawings mark crossings with a hop).

A net ends either at a port (on the left/right side of a device block) or at a
free end (icon-style device such as a speaker or sensor, or an off-page marker).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .blocks import Block, first_word, stack_group, _join
from .elements import PageElements, Segment, TextLine, point_segment_distance


@dataclass
class Terminal:
    x: float
    y: float
    block: Block | None = None   # None => free end (icon / off-page marker)
    side: str = ""               # 'L' or 'R' for ports
    has_dot: bool = False
    label: str = ""              # for free ends: text found next to the end
    link: str = ""               # for free ends at an off-page circle: the link number
    out: tuple = (0.0, 0.0)      # for free ends: direction the cable runs as it arrives here


@dataclass
class Net:
    segments: list[Segment] = field(default_factory=list)
    terminals: list[Terminal] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)  # cable IDs printed on the cable
    n_dots: int = 0


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def _inside_a_block(s: Segment, blocks: list[Block], eps: float) -> bool:
    """Divider lines drawn inside a device box are decoration, not cables."""
    for blk in blocks:
        b = blk.box
        if b.contains_point(s.x0, s.y0, eps) and b.contains_point(s.x1, s.y1, eps):
            return True
    return False


def _text_on_segment(t: TextLine, s: Segment, eps: float) -> bool:
    """True if a (horizontal or vertical) stroke runs through the text: a cable tag."""
    if abs(s.y0 - s.y1) <= eps:
        lo, hi = min(s.x0, s.x1), max(s.x0, s.x1)
        return t.y0 + 0.2 * t.h <= s.y0 <= t.y1 - 0.2 * t.h and min(hi, t.x1) - max(lo, t.x0) >= 0.5 * t.w
    if abs(s.x0 - s.x1) <= eps:
        lo, hi = min(s.y0, s.y1), max(s.y0, s.y1)
        return t.x0 + 0.2 * t.w <= s.x0 <= t.x1 - 0.2 * t.w and min(hi, t.y1) - max(lo, t.y0) >= 0.5 * t.h
    return False


def build_nets(el: PageElements, blocks: list[Block]) -> list[Net]:
    eps, unit = el.eps, el.unit
    segs = [s for s in el.segments if s.length > eps * 0.1 and not _inside_a_block(s, blocks, eps)]
    pts = [(c[0], c[1]) for s in segs for c in ((s.x0, s.y0), (s.x1, s.y1))]
    n, nd = len(pts), len(el.dots)
    loc = _UnionFind(n + nd)   # same physical point
    net = _UnionFind(n + nd)   # same cable

    def both(a: int, b: int) -> None:
        loc.union(a, b)
        net.union(a, b)

    cell = max(eps, 1e-6)
    grid = defaultdict(list)
    for i, (x, y) in enumerate(pts):
        grid[(int(x // cell), int(y // cell))].append(i)

    def near(x: float, y: float, radius: float):
        k = int(math.ceil(radius / cell))
        cx, cy = int(x // cell), int(y // cell)
        for gx in range(cx - k, cx + k + 1):
            for gy in range(cy - k, cy + k + 1):
                for j in grid.get((gx, gy), ()):
                    if math.hypot(pts[j][0] - x, pts[j][1] - y) <= radius:
                        yield j

    path_of = [s.path for s in segs for _ in (0, 1)]
    is_end = [e for s in segs for e in (s.a_end, s.b_end)]
    for i, (x, y) in enumerate(pts):          # coincident points
        for j in near(x, y, eps):
            # Two different lines that merely bend at the same spot (parallel
            # cables sharing a track) are not joined; a line END touching is.
            if j > i and (path_of[i] == path_of[j] or is_end[i] or is_end[j]):
                both(i, j)
    for i in range(len(segs)):                # the two ends of a stroke are one cable
        net.union(2 * i, 2 * i + 1)

    through = defaultdict(int)                # strokes passing through a dot (junction)
    for k, d in enumerate(el.dots):
        node, radius = n + k, d.r + eps
        for j in near(d.x, d.y, radius):
            both(node, j)
        for i, s in enumerate(segs):
            if min(s.x0, s.x1) - radius <= d.x <= max(s.x0, s.x1) + radius and \
               min(s.y0, s.y1) - radius <= d.y <= max(s.y0, s.y1) + radius:
                dist, t = point_segment_distance(d.x, d.y, s)
                end_gap = min(t, 1 - t) * s.length
                if dist <= radius and end_gap > radius:
                    net.union(node, 2 * i)
                    through[node] += 1

    members = defaultdict(list)
    for i in range(n + nd):
        members[loc.find(i)].append(i)

    def degree(group: list[int]) -> int:
        return sum(1 for m in group if m < n) + 2 * sum(through.get(m, 0) for m in group if m >= n)

    def port_of(group: list[int]):
        """(block, side, y) if this point lies on the left or right side of a device block."""
        probes = [(pts[m][0], pts[m][1], eps) for m in group if m < n]
        probes += [(el.dots[m - n].x, el.dots[m - n].y, el.dots[m - n].r + eps) for m in group if m >= n]
        for x, y, tol in probes:
            for blk in blocks:
                b = blk.box
                if not (b.y0 - eps <= y <= b.y1 + eps):
                    continue
                if abs(x - b.x0) <= tol:
                    return blk, "L", y
                if abs(x - b.x1) <= tol:
                    return blk, "R", y
        return None

    ports = {root: port_of(group) for root, group in members.items()}
    port_points = [(root, p[0]) for root, p in ports.items() if p]

    def devices_on(net_root: int) -> set:
        return {id(blk) for root, blk in port_points if net.find(root) == net_root}

    # T-junctions: a dangling end that lands on the middle of another stroke.
    joined = set()
    for root, group in members.items():
        ends = [m for m in group if m < n]
        if len(ends) != 1 or degree(group) != 1:
            continue
        x, y = pts[ends[0]]
        for i, s in enumerate(segs):
            if i == ends[0] // 2:
                continue
            if min(s.x0, s.x1) - eps <= x <= max(s.x0, s.x1) + eps and \
               min(s.y0, s.y1) - eps <= y <= max(s.y0, s.y1) + eps:
                dist, t = point_segment_distance(x, y, s)
                own = segs[ends[0] // 2]
                cos = abs((own.x1 - own.x0) * (s.x1 - s.x0) + (own.y1 - own.y0) * (s.y1 - s.y0)) / (own.length * s.length)
                # A real T meets the other stroke at an angle. Two strokes lying on
                # top of each other (parallel cables sharing a track) are not joined.
                if dist <= eps and min(t, 1 - t) * s.length > eps and cos < 0.5:
                    # A cable never runs from a device back into the same device, so a line
                    # end that would join such a cable only touches it (no dot = no junction).
                    a, b = net.find(ends[0]), net.find(2 * i)
                    if a != b and devices_on(a) & devices_on(b):
                        continue
                    net.union(ends[0], 2 * i)
                    joined.add(root)
                    break

    nets: dict[int, Net] = defaultdict(Net)
    for i, s in enumerate(segs):
        nets[net.find(2 * i)].segments.append(s)
    for k in range(nd):
        nets[net.find(n + k)].n_dots += 1

    # Terminals
    seen_ports = set()
    for root, group in members.items():
        net_id = net.find(group[0])
        has_dot = any(m >= n for m in group)
        port = ports[root]
        if port:
            blk, side, y = port
            key = (net_id, id(blk), side, round(y / (0.6 * unit)))
            if key not in seen_ports:
                seen_ports.add(key)
                x = blk.box.x0 if side == "L" else blk.box.x1
                nets[net_id].terminals.append(Terminal(x, y, blk, side, has_dot))
        elif degree(group) == 1 and root not in joined:
            ends = [m for m in group if m < n]
            if ends:
                x, y = pts[ends[0]]
                ox, oy = pts[ends[0] ^ 1]          # the other end of the same stroke
                d = math.hypot(x - ox, y - oy) or 1.0
                nets[net_id].terminals.append(Terminal(x, y, None, "", has_dot, out=((x - ox) / d, (y - oy) / d)))

    # Keep only real cables. Evidence of a cable: a dot, two ports, or a port whose
    # connector line continues into more strokes. This drops unused connector
    # lines (a single stroke with nothing attached), title-block rules and other
    # loose strokes.
    live = []
    for nt in nets.values():
        n_ports = sum(1 for t in nt.terminals if t.block is not None)
        is_cable = nt.n_dots > 0 or n_ports >= 2 or (n_ports == 1 and len(nt.segments) >= 2)
        if is_cable and len(nt.terminals) >= 2:
            live.append(nt)

    _attach_text(el, blocks, live)
    return live


def _attach_text(el: PageElements, blocks: list[Block], nets: list[Net]) -> None:
    """Cable tags (text printed on a cable) and labels for free ends."""
    eps, unit = el.eps, el.unit
    in_block = {id(t) for blk in blocks for t in blk.inside}
    loose = [t for t in el.texts if id(t) not in in_block]
    on_wire = set()
    for nt in nets:
        for t in loose:
            if any(_text_on_segment(t, s, eps) for s in nt.segments):
                on_wire.add(id(t))
                if t.text not in nt.tags:
                    nt.tags.append(t.text)

    def is_connector_label(t: TextLine) -> bool:
        """Connector type (RJ45, HDMI, ...) written on a port's connector line beside a block."""
        for blk in blocks:
            b = blk.box
            for side, beside in (("L", 0 <= b.x0 - t.x1 <= 5.5 * unit), ("R", 0 <= t.x0 - b.x1 <= 5.5 * unit)):
                if beside and any(abs(t.yc - y) <= 1.2 * unit for y in blk.marks[side]):
                    return True
        return False

    def in_marker(t: TextLine) -> bool:
        return any(math.hypot(t.xc - m.x, t.yc - m.y) <= m.r for m in el.markers)

    cands = [t for t in loose if not t.rotated and id(t) not in on_wire and not is_connector_label(t)
             and not in_marker(t)]
    for nt in nets:
        for term in nt.terminals:
            if term.block is not None:
                continue
            # A cable that stops at an off-page circle continues wherever the same number appears.
            at = [m for m in el.markers if math.hypot(term.x - m.x, term.y - m.y) <= m.r + 0.6 * unit]
            if at:
                m = min(at, key=lambda m: math.hypot(term.x - m.x, term.y - m.y))
                term.label = term.link = m.ref
            else:
                term.label = _free_end_label(term, cands, unit)


def _free_end_label(term: Terminal, cands: list[TextLine], unit: float) -> str:
    x, y = term.x, term.y
    ox, oy = term.out

    def side_gap(t: TextLine) -> float:
        return max(t.x0 - x, x - t.x1, 0.0)

    def ahead(t: TextLine) -> bool:
        """Beyond the end, in the direction the cable was running (not along the cable itself)."""
        dx, dy = t.xc - x, t.yc - y
        return dx * ox + dy * oy > 0.5 * math.hypot(dx, dy)

    # 1. A name written on the same line just past the end (a labelled dot or connector).
    beside = [t for t in cands if abs(t.yc - y) <= 0.8 * unit and side_gap(t) <= 2.5 * unit and ahead(t)]
    # 2. Otherwise the name printed under the end (icons such as speakers).
    below = [t for t in cands if -0.5 * unit <= t.y0 - y <= 6 * unit and abs(t.xc - x) <= 8 * unit]
    if beside:
        seed = min(beside, key=side_gap)
    elif below:
        seed = min(below, key=lambda t: (round(t.y0), abs(t.xc - x)))
    else:
        def gap(t: TextLine) -> float:
            dx = max(t.x0 - x, 0, x - t.x1)
            dy = max(t.y0 - y, 0, y - t.y1)
            return math.hypot(dx, dy)
        near = [t for t in cands if gap(t) <= 6 * unit]
        if not near:
            return ""
        seed = min(near, key=gap)
    group = stack_group(seed, cands, unit)
    name = _join(group)
    bottom = max(t.y1 for t in group)
    rest = [t for t in cands if t not in group and 0 <= t.y0 - bottom <= 2.5 * unit
            and t.x1 > min(g.x0 for g in group) and t.x0 < max(g.x1 for g in group)]
    if rest:
        second = _join(stack_group(min(rest, key=lambda t: t.y0), [t for t in cands if t not in group], unit))
        if first_word(second) == first_word(name):
            name = f"{name} ({second})"
    return name

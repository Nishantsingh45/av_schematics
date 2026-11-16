"""Shared intermediate format.

Every reader (vector PDF today, image reader later) turns a page into these
basic elements. Everything downstream (blocks, wires, semantics) only sees
this format, so it never needs to know where the page came from.

All tolerances are derived from `unit` (the page's median text-line height),
never from fixed point/pixel values, so the same logic works at any page scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextLine:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    rotated: bool = False

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class Segment:
    """One straight stroke. Curved pieces (line hops) are stored as their chord."""
    x0: float
    y0: float
    x1: float
    y1: float
    path: int = -1        # strokes drawn as one continuous line share a path id
    a_end: bool = True    # (x0, y0) is where that line starts
    b_end: bool = True    # (x1, y1) is where that line stops

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)


@dataclass(frozen=True)
class Dot:
    """Small filled circle: marks where a cable attaches or where cables join."""
    x: float
    y: float
    r: float


@dataclass(frozen=True)
class Marker:
    """Off-page link: a small outlined circle with a link number inside. The two circles
    that carry the same number are the two ends of one cable (usually on different pages)."""
    x: float
    y: float
    r: float
    ref: str


@dataclass(frozen=True)
class Box:
    """Closed rectangular (or rounded-rectangular) outline."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    def contains_point(self, x: float, y: float, tol: float = 0.0) -> bool:
        return self.x0 - tol <= x <= self.x1 + tol and self.y0 - tol <= y <= self.y1 + tol

    def contains_text(self, t: TextLine, tol: float = 0.0) -> bool:
        return self.contains_point(t.x0, t.y0, tol) and self.contains_point(t.x1, t.y1, tol)

    def contains_box(self, o: "Box", tol: float = 0.0) -> bool:
        return self.contains_point(o.x0, o.y0, tol) and self.contains_point(o.x1, o.y1, tol)


@dataclass
class PageElements:
    width: float
    height: float
    unit: float  # median text-line height = the page's own length scale
    texts: list[TextLine] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    dots: list[Dot] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    source: str = "vector"

    @property
    def eps(self) -> float:
        """Snap tolerance: two points closer than this are the same point."""
        return 0.15 * self.unit


def point_segment_distance(px: float, py: float, s: Segment) -> tuple[float, float]:
    """Distance from a point to a segment, and the position along it (0..1)."""
    dx, dy = s.x1 - s.x0, s.y1 - s.y0
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.hypot(px - s.x0, py - s.y0), 0.0
    t = max(0.0, min(1.0, ((px - s.x0) * dx + (py - s.y0) * dy) / L2))
    return math.hypot(px - (s.x0 + t * dx), py - (s.y0 + t * dy)), t

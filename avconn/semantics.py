"""Turn traced nets into source -> target connections.

Direction evidence, strongest first:
  1. the port name says so (OP/OUT = output, IP/IN = input);
  2. the block layout: in these drawings outputs leave the right side of a
     block and inputs enter on the left (true for every sample connection);
  3. position on the page (left-most end first) - reported as uncertain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .blocks import port_label
from .wires import Net, Terminal

_OUT = re.compile(r"^(OP|OUT|OUTPUT)\d*$")
_IN = re.compile(r"^(IP|IN|INPUT)\d*$")


@dataclass
class Connection:
    source_device: str
    source_port: str
    target_device: str
    target_port: str
    confidence: float = 1.0
    flags: list[str] = field(default_factory=list)
    cable: str = ""

    def as_output(self) -> dict:
        return {
            "source_device": self.source_device,
            "source_port": self.source_port,
            "target_device": self.target_device,
            "target_port": self.target_port,
        }

    def key(self) -> tuple:
        return (self.source_device, self.source_port, self.target_device, self.target_port)


def name_role(port: str) -> str:
    for token in re.split(r"[^A-Za-z0-9]+", port.upper()):
        if _OUT.match(token):
            return "out"
        if _IN.match(token):
            return "in"
    return ""


@dataclass
class End:
    """One end of a cable, already named (device + port) on its own page."""
    device: str
    port: str
    term: Terminal
    by_name: str   # 'out' / 'in' / ''
    by_side: str   # 'out' / 'in' / ''


def _describe(term: Terminal, unit: float) -> End:
    if term.block is None:
        return End(f"OFF-PAGE {term.link}" if term.link else term.label, "", term, "", "")
    port = port_label(term.block, term.side, term.y, unit)
    return End(term.block.display_name, port, term, name_role(port), "out" if term.side == "R" else "in")


def _order(a: End, b: End) -> tuple[End, End, bool]:
    """Return (source, target, certain)."""
    for attr in ("by_name", "by_side"):
        ra, rb = getattr(a, attr), getattr(b, attr)
        if ra and rb and ra != rb:
            return (a, b, True) if ra == "out" else (b, a, True)
        if bool(ra) != bool(rb):  # only one end gives evidence
            known, other = (a, b) if ra else (b, a)
            role = ra or rb
            return (known, other, True) if role == "out" else (other, known, True)
    first, second = sorted((a, b), key=lambda e: (e.term.x, e.term.y))
    return first, second, False


_PENALTY = {"direction_uncertain": 0.25, "device_name_missing": 0.4, "port_label_missing": 0.2,
            "multi_drop": 0.15, "offpage_unmatched": 0.3, "joined_offpage": 0.0}


def cable_ends(nets: list[Net], unit: float) -> list[list[End]]:
    """The named ends of every cable on a page (one list per cable)."""
    return [[_describe(t, unit) for t in nt.terminals] for nt in nets]


def connections_for_cable(ends: list[End], cable: str = "", extra_flags: tuple = ()) -> list[Connection]:
    """Connections made by ONE physical cable. Each cable gives its own rows, so identical
    devices chained one after another (A -> A -> A) keep one row per cable."""
    if len(ends) < 2:
        return []
    if len(ends) == 2:
        pairs = [(ends[0], ends[1], [])]
    else:
        # One cable feeding several ends: connect each output to every other end.
        sources = [e for e in ends if (e.by_name or e.by_side) == "out"] or [min(ends, key=lambda e: (e.term.x, e.term.y))]
        pairs = [(s, e, ["multi_drop"]) for s in sources for e in ends if e not in sources]
    out, seen = [], set()
    for a, b, extra in pairs:
        src, dst, certain = _order(a, b)
        flags = list(extra) + list(extra_flags)
        if not certain:
            flags.append("direction_uncertain")
        for e in (src, dst):
            if not e.device:
                flags.append("device_name_missing")
            if e.term.block is not None and not e.port:
                flags.append("port_label_missing")
            if e.term.link:
                flags.append("offpage_unmatched")
        conf = 1.0 - sum(_PENALTY.get(f, 0.1) for f in flags)
        c = Connection(src.device, src.port, dst.device, dst.port,
                       confidence=round(max(conf, 0.05), 2), flags=flags, cable=cable)
        if c.key() not in seen:  # the same pair twice from one multi-drop cable
            seen.add(c.key())
            out.append(c)
    return out

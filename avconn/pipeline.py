"""Run the whole pipeline on one file (any PDF or image) and collect results."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from . import llm_reader, vector_reader
from .blocks import Block, find_blocks
from .elements import PageElements
from .semantics import Connection, End, cable_ends, connections_for_cable
from .wires import Net, build_nets

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class PageResult:
    page: int
    reader: str                      # 'vector' | 'llm' | 'none'
    status: str                      # 'ok' | 'no_cables' | 'low_confidence' | 'not_processed'
    connections: list[Connection] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    note: str = ""
    # kept for the debug picture
    elements: PageElements | None = None
    blocks: list[Block] = field(default_factory=list)
    nets: list[Net] = field(default_factory=list)
    ends: list[list[End]] = field(default_factory=list)   # named ends of each cable, for off-page joining


@dataclass
class FileResult:
    path: Path
    pages: list[PageResult] = field(default_factory=list)

    @property
    def not_processed(self) -> bool:
        """True if no page of this file could be read at all (e.g. a picture with no picture reader)."""
        return bool(self.pages) and all(p.status == "not_processed" for p in self.pages)

    @property
    def connections(self) -> list[Connection]:
        # One row per cable: two identical rows are two real cables, so nothing is merged here.
        return [c for p in self.pages for c in p.connections]


def analyse_elements(el: PageElements, page_no: int) -> PageResult:
    blocks = find_blocks(el)
    nets = build_nets(el, blocks)
    ends = cable_ends(nets, el.unit)
    conns = [c for nt, e in zip(nets, ends) for c in connections_for_cable(e, ", ".join(nt.tags))]
    n_ends = sum(len(n.terminals) for n in nets)
    n_resolved = sum(1 for n in nets for t in n.terminals if (t.block is not None) or t.label)
    stats = {
        "device_blocks": len(blocks),
        "blocks_with_product_name": sum(1 for b in blocks if b.name),
        "dots": len(el.dots),
        "cables": len(nets),
        "cable_ends": n_ends,
        "cable_ends_resolved": n_resolved,
        "connections": len(conns),
        "mean_confidence": round(sum(c.confidence for c in conns) / len(conns), 2) if conns else None,
    }
    # Self-check: does the page look like something this reader understood?
    if not nets:
        status, note = "no_cables", "No cables found (e.g. a rack layout or a non-schematic page)."
    elif n_resolved < 0.8 * n_ends or (blocks and stats["blocks_with_product_name"] < 0.6 * len(blocks)):
        status, note = "low_confidence", "Many cable ends or block names could not be read; layout may be unfamiliar."
    else:
        status, note = "ok", ""
    return PageResult(page_no, el.source, status, conns, stats, note, el, blocks, nets, ends)


def join_offpage_links(result: FileResult) -> None:
    """A cable that runs into an off-page circle continues from the circle with the same
    number (on another page, or elsewhere on the same page). Join the pieces into one cable.

    Only numbers that appear exactly twice in the file are joined; any other off-page end
    is kept as 'OFF-PAGE <n>' and flagged offpage_unmatched.
    """
    where = defaultdict(list)                      # link number -> [(page index, cable index)]
    for pi, p in enumerate(result.pages):
        for ci, ends in enumerate(p.ends):
            for e in ends:
                if e.term.link:
                    where[e.term.link].append((pi, ci))
    used = {ref for ref, locs in where.items() if len(locs) == 2 and locs[0] != locs[1]}
    if not used:
        return

    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            a = parent[a]
        return a

    for ref in used:
        a, b = where[ref]
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for loc in parent:
        groups[find(loc)].append(loc)

    replaced = {}                                   # (page, cable) -> new connections ([] = removed)
    for locs in groups.values():
        locs.sort()
        ends = [e for pi, ci in locs for e in result.pages[pi].ends[ci] if e.term.link not in used]
        tags = [", ".join(result.pages[pi].nets[ci].tags) for pi, ci in locs]
        conns = connections_for_cable(ends, " + ".join(t for t in tags if t), ("joined_offpage",))
        for loc in locs:
            replaced[loc] = []
        replaced[locs[0]] = conns                   # report the joined cable on its first page
    for pi, p in enumerate(result.pages):
        if not any(loc[0] == pi for loc in replaced):
            continue
        p.connections = [c for ci, (nt, e) in enumerate(zip(p.nets, p.ends))
                         for c in (replaced[(pi, ci)] if (pi, ci) in replaced
                                   else connections_for_cable(e, ", ".join(nt.tags)))]
        p.stats["connections"] = len(p.connections)


def _llm_page(image_bytes: bytes, media_type: str, page_no: int, use_llm: bool) -> PageResult:
    if not use_llm:
        return PageResult(page_no, "none", "not_processed",
                          note="Picture input: needs the LLM reader (run with --llm auto and set ANTHROPIC_API_KEY).")
    try:
        data = llm_reader.read_image(image_bytes, media_type)
    except llm_reader.LLMUnavailable as e:
        return PageResult(page_no, "none", "not_processed", note=f"Picture input, LLM reader unavailable: {e}")
    conns = [Connection(c["source_device"], c["source_port"], c["target_device"], c["target_port"],
                        confidence=0.5, flags=["llm_read"]) for c in data["connections"]]
    note = data.get("notes", "")
    if not data["is_wiring_schematic"]:
        status = "no_cables"
    elif not conns:  # a schematic, but the reader could not read any connection from it
        status = "low_confidence"
    elif not data.get("text_legible", True):  # connections found, but names were partly guessed
        status = "low_confidence"
        for c in conns:
            c.flags.append("names_not_fully_legible")
        note = "Some device or port names are not fully legible and were partly inferred. " + note
    else:
        status = "ok"
    return PageResult(page_no, "llm", status, conns, {"connections": len(conns)}, note)


def _image_media_type(data: bytes) -> str | None:
    """Media type from the file's first bytes. The extension can be wrong (a PNG saved as .jpg)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return None


def extract_file(path: str | Path, use_llm: bool = False) -> FileResult:
    path = Path(path)
    result = FileResult(path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        data = path.read_bytes()
        media = _image_media_type(data)
        if media is None:  # other formats (BMP, TIFF, WebP, ...): convert to PNG
            data, media = pymupdf.Pixmap(str(path)).tobytes("png"), "image/png"
        result.pages.append(_llm_page(data, media, 1, use_llm))
        return result

    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            if vector_reader.has_vector_content(page):
                result.pages.append(analyse_elements(vector_reader.read_page(page), i))
            else:  # scanned page: render it and treat it as a picture
                zoom = min(3.0, 2500 / max(page.rect.width, page.rect.height))
                png = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
                result.pages.append(_llm_page(png, "image/png", i, use_llm))
    join_offpage_links(result)
    return result

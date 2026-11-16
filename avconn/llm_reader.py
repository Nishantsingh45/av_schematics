"""Optional reader for picture input (JPG/PNG, scanned PDF pages): a vision LLM.

Used only when a page has no real drawing data to parse. Needs the `anthropic`
package and credentials (ANTHROPIC_API_KEY). The rest of the tool works without it.

Two calls per picture:
  1. read  - the whole page plus enlarged, overlapping tiles of it, so the model sees the
             overall layout and can follow lines up close where they run together or cross;
  2. check - the list from step 1 is re-checked against finer, more enlarged tiles, with
             attention to neighbouring ports (IP1/IP2, OP1/OP2) where cables get swapped.
On test.jpg (a screenshot of Zoom Room.pdf) one whole-page call got 5-6 of 9 right;
read + check got 9/9 ports right in both runs. Step 2 is skipped when step 1 finds nothing.
"""
from __future__ import annotations

import base64
import json
import math
import os

import pymupdf

DEFAULT_MODEL = "claude-sonnet-5-5"


def model_name() -> str:
    return os.environ.get("AVCONN_LLM_MODEL") or DEFAULT_MODEL

_PROMPT = """This image is a page from an AV (audio/video) system drawing set.

First decide whether it is a wiring line schematic (device blocks joined by cables).
Rack layouts, title pages and other pages with no cables are NOT wiring schematics:
for those return is_wiring_schematic=false and an empty connections list.

How to read a device block:
- A box with two columns of port names inside. A short connector line leaves the
  box next to each port, with the connector type written on it (HDMI, RJ45, ...).
- The product name (brand + model) is written inside the box, across both columns.
- A generic device type (DISPLAY, NETWORK SWITCH, ...) is written just outside the box.
- Some devices are only an icon (speaker, sensor) with the product name beside it.

Rules:
- Report one connection per cable that actually joins two devices. Follow each
  cable from end to end; cables that only cross each other are not connected.
  A port with no cable attached is not a connection.
- source = the output end, target = the input end. Port names containing OP/OUT
  are outputs and IP/IN are inputs; otherwise ports on the right side of a block
  are outputs and ports on the left side are inputs.
- Device = the product name exactly as printed (not the generic device type). If a
  second product line is printed under the block, append it in parentheses.
- Port = the port name exactly as printed. Use "" for icon devices with no port.
- Never guess text you cannot read. If a name is illegible write "?" and say so in notes.
- Report every cable you can follow with confidence from one device to the other, even when a port
  name on it is unreadable: then write "?" for that port (not the connector type). Leave out only
  cables you cannot follow.
- text_legible: true only if every device and port name you report is copied from text you could
  read character by character. Set it to false if any name is partly guessed, completed from what
  the product probably is, or replaced by a description (e.g. a connector type instead of the port
  name printed on the drawing).
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "is_wiring_schematic": {"type": "boolean"},
        "connections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_device": {"type": "string"},
                    "source_port": {"type": "string"},
                    "target_device": {"type": "string"},
                    "target_port": {"type": "string"},
                },
                "required": ["source_device", "source_port", "target_device", "target_port"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
        "text_legible": {"type": "boolean"},
    },
    "required": ["is_wiring_schematic", "connections", "notes", "text_legible"],
    "additionalProperties": False,
}


class LLMUnavailable(RuntimeError):
    pass


def status() -> tuple[bool, str]:
    """(ready, explanation) for the picture reader. Never prints the key itself."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "the 'anthropic' package is not installed (pip install anthropic)"
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return False, "ANTHROPIC_API_KEY is not set (put it in .env or in the shell)"
    return True, f"model {model_name()}"


def available() -> bool:
    return status()[0]


_TILES_NOTE = """
The first image is the whole page. The following images are ENLARGED, OVERLAPPING tiles of the
same page (labelled by row and column). Use the overview to see which devices exist and where cables
go overall; use the zoomed tiles to follow each cable exactly where lines run close together, cross
(a small hop/arc means they cross WITHOUT connecting) or turn. Report each cable once."""

_CHECK_PROMPT = """You are checking a list of connections that was read from this AV wiring schematic.
The first image is the whole page; the others are enlarged, overlapping tiles of it.

For EVERY connection in the list, follow the cable in the zoomed tiles from one port to the other and
confirm both ends. Pay special attention to neighbouring ports on the same device (e.g. ...IP1 and
...IP2, ...OP1 and ...OP2): make sure each cable lands on the right one. A small hop/arc where lines
cross means they are NOT connected. Fix any connection that is wrong, add any cable that is missing,
remove any that does not exist - but only when the zoomed tiles CLEARLY show it. If a cable is hard
to follow, keep it exactly as listed. Keep the same naming rules. Return the full list.

Connections to check:
"""


def zoom_tiles(image_bytes: bytes, cell: int, long_side: int, max_tiles: int, overlap: float = 0.15) -> list:
    """Overlapping tiles of a picture, each enlarged so its long side is about `long_side` px.

    Tiles cover about `cell` x `cell` source pixels; bigger pictures get bigger tiles so there
    are never more than `max_tiles`. Plain interpolation: nothing is invented, only magnified.
    """
    pix = pymupdf.Pixmap(image_bytes)
    w, h = pix.width, pix.height
    cell = max(cell, math.sqrt(w * h / max_tiles))
    cols, rows = max(1, math.ceil(w / cell)), max(1, math.ceil(h / cell))
    while cols * rows > max_tiles:
        cell *= 1.1
        cols, rows = max(1, math.ceil(w / cell)), max(1, math.ceil(h / cell))
    tw, th = w / cols, h / rows
    tiles = []
    with pymupdf.open() as doc:
        page = doc.new_page(width=w, height=h)
        page.insert_image(page.rect, pixmap=pix)
        for r in range(rows):
            for c in range(cols):
                rect = pymupdf.Rect((c - overlap) * tw, (r - overlap) * th,
                                    (c + 1 + overlap) * tw, (r + 1 + overlap) * th) & page.rect
                z = long_side / max(rect.width, rect.height)
                png = page.get_pixmap(matrix=pymupdf.Matrix(z, z), clip=rect).tobytes("png")
                tiles.append((f"Zoomed tile (row {r + 1} of {rows}, column {c + 1} of {cols}):", png, "image/png"))
    return tiles


def _listing(conns: list[dict]) -> str:
    return "\n".join(f"- {c['source_device']} | {c['source_port']} -> {c['target_device']} | {c['target_port']}"
                     for c in conns)


def read_image(image_bytes: bytes, media_type: str) -> dict:
    """Is the page a wiring schematic, and its connections (read, then checked up close)."""
    overview = [("Whole page (overview):", image_bytes, media_type)]
    first = ask_json(overview + zoom_tiles(image_bytes, 700, 1500, 8), _PROMPT + _TILES_NOTE, _SCHEMA)
    if not first["is_wiring_schematic"] or not first["connections"]:
        return first
    checked = ask_json(overview + zoom_tiles(image_bytes, 450, 1800, 12),
                       _CHECK_PROMPT + _listing(first["connections"]) + "\n\n" + _PROMPT, _SCHEMA)
    checked["is_wiring_schematic"] = True
    return checked


def ask_json(images: list, prompt: str, schema: dict, max_tokens: int = 16000, effort: str = "high") -> dict:
    """Images (each a (caption, bytes, media type) tuple) + one instruction -> an answer that
    follows the given JSON schema."""
    content = []
    for caption, data, media_type in images:
        if caption:
            content.append({"type": "text", "text": caption})
        content.append({"type": "image", "source": {"type": "base64", "media_type": media_type,
                                                    "data": base64.standard_b64encode(data).decode("utf-8")}})
    content.append({"type": "text", "text": prompt})
    try:
        import anthropic
    except ImportError as e:
        raise LLMUnavailable("the 'anthropic' package is not installed (pip install anthropic)") from e
    try:
        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=model_name(),
            max_tokens=max_tokens,
            # If the model declines a request, let the API retry it on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.AuthenticationError as e:
        raise LLMUnavailable("no valid Anthropic credentials (set ANTHROPIC_API_KEY)") from e
    except anthropic.APIError as e:
        raise LLMUnavailable(f"API call failed: {e}") from e
    if response.stop_reason == "refusal":
        raise LLMUnavailable("the model declined to process this image")
    if response.stop_reason == "max_tokens":
        raise LLMUnavailable("the answer was cut off (drawing too large for one request)")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)

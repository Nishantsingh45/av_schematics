# Approach: connection extraction from AV line schematics

**Result.** On the 8 vector PDFs the tool reproduces 318 of 329 hand-checked connections exactly
(96.7 %) and all 329 when direction is ignored; every remaining difference is a cable whose direction
the drawing does not state. Pictures are read by a vision model using zoomed tiles of the image: 9/9
on a clear screenshot, and no guessed connections on the unreadable `9.jpg`.

## 1. The problem and the data

A connection is one cable between two device ports, output as `source_device / source_port →
target_device / target_port`. What the drawings contain decided the approach:

| Finding | Consequence |
|---|---|
| All 8 PDFs carry real vector data (positioned text, lines, dots, boxes) | Read geometry and text exactly; no OCR needed |
| Two files are rack elevations (`15.jpg`, one 7619 PDF) | The tool must recognise non-schematics and return no connections |
| `9.jpg` shows a whole sheet in 1258×1154 px; text is about 3 px tall | Port names are unreadable by any method; this is the limit for pictures |
| Two drawings span 2–3 pages, linked by numbered circles | A cable's two halves must be joined |
| One labelled example (8 connections), at least three drawing styles | Rules must rest on structure, not colours, names or coordinates |

## 2. Approach and techniques

Each page is read from the most exact source it has: vector pages as geometry, pictures with a vision model.

**PDF processing** (PyMuPDF), in order:
1. **Vector parsing**: every text span with its position and every drawn path.
2. **Text cleaning**: drop transparent duplicate text and labels printed several times on top of each other.
3. **Shape classification**: small filled circles = connection dots, rectangles = boxes, numbered circles = page links.
4. **Dash-to-line reconstruction**: dashed cables, stored as hundreds of tiny dashes, are rebuilt into one line.
5. **Block parsing** by the brief's anatomy: a box with text and connector lines on its sides is a device; port rows per side, product name across both columns.
6. **Connectivity graph** (union-find): shared ends, dots and T-junctions join lines into cables; crossings and hops do not.
7. **Direction rules and page linking**: OP/OUT = source, IP/IN = target, else the box side; circles with the same number join a cable's halves.
8. **Self-check**: a status per page, a confidence and flags per row.

All tolerances scale with the page's text height, so nothing is tied to a file, colour or coordinate.

**Image processing** (pictures and scanned pages):
1. **Format detection** from the file's bytes, not its extension.
2. **Tiling and zoom**: overlapping crops of the page, enlarged by plain interpolation (it magnifies, never invents).
3. **Read**: the vision model gets the whole page for layout plus the zoomed tiles for detail, and returns JSON in the required format.
4. **Re-check**: a finer zoom; every connection is verified, especially neighbouring ports such as `IP1`/`IP2`.
5. **Legibility check**: unreadable text must be `?`; if a name was not read exactly, the page is marked `low_confidence`.

**Why no trained model, and where ML fits next.** With one labelled drawing there is nothing to train or
validate on, and the PDFs already hold exact geometry that a model would only approximate. Where the
input is pixels, a pre-trained vision model is used zero-shot with structured output, tiling and a
self-check. The next step for ML is built in: the PDF pipeline is a free labeller. Rendering a vector
PDF as an image gives a picture whose every device, port and cable is already known. That makes an
automatically labelled training set for an image model (a detector for device blocks and ports, plus
line segmentation) to read scanned and photographed drawings without an API.

**Tools.** Python, PyMuPDF, and Claude (Anthropic) as the vision model; AI coding assistance was used during development.

## 3. Assumptions

- **Device** = the product name printed in the block (with any `(n)`); a same-brand line under the box is added in brackets, as in the sample.
- **Port** = the text of the row the cable reaches; `""` for icon devices such as speakers, as in the sample.
- **Direction** from OP/OUT and IP/IN, else the right side of a box is the output; otherwise flagged `direction_uncertain`.
- **One row per physical cable**; dashed lines are cables; numbered circles are page links.

## 4. Results and observations

Only the sample has an official answer key. The others were built by tracing every cable on zoomed
renders, cross-checked against the drawing's dot and cable-ID counts where possible
(`data/ground_truth/`). `evaluate` scores strict (all fields and direction) and relaxed (direction ignored).

| Drawing | Cables | Exact | Ignoring direction |
|---|---|---|---|
| Sample (official key) | 8 | 8 | 8 |
| Prada / Zimmerman / ERIN (2 pages) | 32 / 25 / 63 | all | all |
| Zoom Room, Page-2, Page-4 | 23 | 23 | 23 |
| 7619 ORC Chamber (3 pages) | 178 | 167 | 178 |
| **All vector PDFs** | **329** | **318 (96.7 %)** | **329** |
| Rack layouts (`15.jpg`, 7619 rack PDF) | 0 | correctly empty | correctly empty |
| `test.jpg` (added by us: a screenshot of `Zoom Room.pdf`) | 9 | 9 (4 of 4 runs) | 9 |
| `9.jpg` (low resolution) | 24 by hand | none reported | none reported |

- **All 11 misses on 7619 are direction only**: both ends inputs, both outputs, or neither says.
  Device-specific rules (such as "power plates are the source") were deliberately not added.
- **Hand-checking found what the sample hid.** The first version scored 8/8 on the sample but only
  278 of 312 rows overall, with 33 cables missing. Each cause was fixed with a general rule (section 5).
- **Zoom is what makes pictures work.** One whole-page pass read every name but swapped neighbouring
  ports where cables cross (5–6 of 9). Zoomed tiles plus the re-check gave 9/9 in every run.
- **`9.jpg` is beyond reliable reading**: even by hand, no port name is legible. The tool says so instead of guessing.

## 5. Challenges and limitations

| Imperfect input | How it is handled |
|---|---|
| Hidden or repeated text (transparent copies, `DATA14 DATA14`) | Opacity filter; one copy per position |
| Chains of identical devices | One row per cable (ERIN: 22 cables recovered) |
| Dashed cables stored as tiny dashes | Rebuilt into lines (7619: 11 recovered); zone borders excluded because they close on themselves |
| Labels and hops drawn over lines; drawings without junction dots | Hops stay continuous; crossings never join; no junction may connect a device to itself |
| Nearby text that is not the name (room headings, connector types) | Text on the cable's own line first; connector types only next to connector lines |
| Pictures: neighbouring ports swapped where cables cross | Zoomed tiles plus a re-check pass |

Tried and dropped: a pixel-level tracer for pictures broke cables at corners and labels (0/9 against
9/9 for zoomed reading); sharpening and AI upscaling cannot add real detail and can invent letters.

**Limitations and next steps**
- The rules come from about three drawing styles; a new style may need a new rule, and the per-page status makes that visible.
- Answer keys other than the sample's are hand-made.
- Picture input needs a paid API (two calls per picture) and was measured on one picture with a known answer; scanned PDFs are untested.
- Next: more labelled drawings, the auto-labelled image dataset described in section 2, and majority voting for pictures.

Setup and commands are in `README.md`.

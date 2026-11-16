# avconn — connection extraction from AV line schematics (v1)

Reads an AV line schematic (PDF or image) and writes every cable connection as JSON:

```json
[{"source_device": "INTEL NUC11PAHI7", "source_port": "HDMI OP1",
  "target_device": "CRESTRON DM-TX-4KZ-302-C", "target_port": "HDMI IP1"}]
```

The approach, assumptions, results and observations are in [APPROACH.md](APPROACH.md).

## Setup

Python 3.10+.

```
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill it in. The tool reads `.env` on every run.

| Variable | Needed for | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Picture input only (JPG/PNG files, scanned PDF pages) | none |
| `AVCONN_LLM_MODEL` | Optional: which vision model reads pictures | `claude-sonnet-5-5` |

PDFs with real drawing data need no key. Without a key, picture files are
reported as `not_processed` and get no `.json`.

## Run

```
python -m avconn extract "AI ML Assignment" --out results     # a file or a folder
python -m avconn evaluate --pred results --truth data/ground_truth
python tests/test_sample.py      # the provided sample must match its answer key exactly
python tests/test_synthetic.py   # a made-up 2-page drawing: page links, dashed cables, chains
```

For each drawing, `extract` writes into `results/`:

| File | Contents |
|---|---|
| `<name>.json` | The connections, in the format above |
| `<name>.report.json` | Per page: which reader was used, self-check status and counts, and a confidence and flags for each connection |
| `<name>.p<N>.debug.png` | Only with `--debug`: the page with detected blocks (green), cables (one colour each) and cable ends drawn on top |

`evaluate` compares every `data/ground_truth/<name>.json` with `results/<name>.json`
and prints precision / recall / F1, strict (names and direction must match) and
relaxed (direction ignored, quantity suffixes like `(2)` dropped). Every row counts:
a chain of ten identical cables must appear as ten rows. The provided sample's key is
the official one; the other keys in `data/ground_truth` were made by hand-tracing the drawings.

## How it works

1. **Route each page.** A page with real text and line data is parsed exactly
   (`vector_reader.py`, PyMuPDF). A page that is only a picture goes to a vision
   LLM (`llm_reader.py`) in two calls: it reads the page with enlarged tiles of it,
   then re-checks every connection against finer tiles.
2. **Basic elements** (`elements.py`): text with positions, line strokes, dots,
   closed boxes. Everything after this step sees only these elements.
3. **Device blocks** (`blocks.py`): a box is a device because of its structure
   (text inside, connector lines or dots on its left/right side), following the
   block anatomy in the brief. The product name is the text spanning both port
   columns; the device type is the label just outside the box.
4. **Cables** (`wires.py`): strokes are joined into cables through shared ends,
   dots and T-junctions; plain crossings are not joined. Dashed cables count too.
   A cable ends at a port, at a free end (icon devices such as speakers), or at an
   off-page circle.
5. **Direction and names** (`semantics.py`): output → input from the port name
   (OP/OUT vs IP/IN), else from the block side (right = output, left = input).
   Each cable gives its own row.
6. **Page links** (`pipeline.py`): two off-page circles with the same number are
   the two ends of one cable, so the two halves are joined into one connection.
7. **Self-check** (`pipeline.py`): pages with no cables are reported as
   `no_cables`; pages where many ends or names could not be read as `low_confidence`.

No file names, colours, coordinates or page sizes are hard-coded. All tolerances
scale with the page's own text size.

## Flags in the report

| Flag | Meaning |
|---|---|
| `direction_uncertain` | Neither port name nor block side decided source vs target |
| `port_label_missing` | The cable reaches a block side where no port text was found |
| `device_name_missing` | No product name or label was found for one end |
| `multi_drop` | One cable reaches more than two ends |
| `llm_read` | Read by the vision LLM from a picture, not parsed from drawing data |
| `names_not_fully_legible` | Picture input: some names were partly inferred, not read character by character |
| `joined_offpage` | The cable continues through an off-page circle; both halves were joined |
| `offpage_unmatched` | The cable ends at an off-page circle whose partner was not found |

## Known limits

- Only the provided sample has an official answer key; the others are hand-checked.
- Where both ends of a cable are inputs or outputs, or neither says, the drawing does
  not state the direction; such rows are flagged `direction_uncertain`.
- Picture input depends on the vision LLM. On a clear screenshot of `Zoom Room.pdf`
  it matched the answer key 9/9; on a low-resolution picture (`9.jpg`) it reports no
  connections rather than guess.

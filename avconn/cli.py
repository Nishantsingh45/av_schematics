"""Command line:

    python -m avconn extract  <file-or-folder> [--out results] [--llm auto|off] [--debug]
    python -m avconn evaluate [--pred results] [--truth data/ground_truth]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import llm_reader
from .evaluate import evaluate_dirs
from .export import write_outputs
from .pipeline import IMAGE_SUFFIXES, extract_file


def load_env_file() -> Path | None:
    """Read KEY=VALUE lines from a .env file (current folder, else the project folder).

    Values already set in the shell win. Returns the file that was read, if any.
    """
    for folder in (Path.cwd(), Path(__file__).resolve().parents[1]):
        f = folder / ".env"
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ.setdefault(key, value)
        return f
    return None


def _inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    wanted = IMAGE_SUFFIXES | {".pdf"}
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in wanted)


def cmd_extract(args) -> int:
    files = _inputs(Path(args.input))
    if not files:
        print(f"No PDF or image files found at {args.input}")
        return 1
    ready, why = llm_reader.status()
    use_llm = args.llm == "auto" and ready
    if args.llm == "off":
        print("Picture reader (JPG/PNG, scanned pages): OFF (--llm off)")
    else:
        print(f"Picture reader (JPG/PNG, scanned pages): {'ON, ' + why if ready else 'OFF - ' + why}")
    out_dir = Path(args.out)
    print(f"{'file':58} {'pages':>5} {'conns':>5}  status")
    for f in files:
        result = extract_file(f, use_llm=use_llm)
        write_outputs(result, out_dir, debug=args.debug)
        status = ", ".join(sorted({p.status for p in result.pages}))
        count = "-" if result.not_processed else len(result.connections)
        print(f"{f.name[:58]:58} {len(result.pages):>5} {count:>5}  {status}")
        for p in result.pages:
            if p.note:
                print(f"      p{p.page}: {p.note}")
    print(f"\nResults written to {out_dir.resolve()}")
    return 0


def cmd_evaluate(args) -> int:
    results = evaluate_dirs(Path(args.pred), Path(args.truth))
    if not results:
        print(f"No answer keys found in {args.truth}")
        return 1
    print(f"{'drawing':50} {'pred':>4} {'exp':>4}  {'strict P/R/F1':>18}  {'relaxed P/R/F1':>18}")
    for name, r in results.items():
        if "error" in r:
            print(f"{name[:50]:50} {r['error']}")
            continue
        s, x = r["strict"], r["relaxed"]
        print(f"{name[:50]:50} {r['predicted']:>4} {r['expected']:>4}  "
              f"{s['precision']:.2f}/{s['recall']:.2f}/{s['f1']:.2f}        "
              f"{x['precision']:.2f}/{x['recall']:.2f}/{x['f1']:.2f}")
        for m in r["missing"]:
            print(f"      missing: {m}")
        for m in r["extra"]:
            print(f"      extra:   {m}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="avconn", description="Extract connections from AV line schematics.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="extract connections from a file or folder")
    e.add_argument("input")
    e.add_argument("--out", default="results")
    e.add_argument("--llm", choices=["auto", "off"], default="auto",
                   help="vision LLM for picture input (auto = use it if credentials are set)")
    e.add_argument("--debug", action="store_true",
                   help="also save a debug picture per PDF page (detected blocks, cables, cable ends)")
    e.set_defaults(func=cmd_extract)
    v = sub.add_parser("evaluate", help="score results against answer keys")
    v.add_argument("--pred", default="results")
    v.add_argument("--truth", default="data/ground_truth")
    v.add_argument("--json", help="also write the scores to this file")
    v.set_defaults(func=cmd_evaluate)
    args = parser.parse_args(argv)
    load_env_file()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

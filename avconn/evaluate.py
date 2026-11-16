"""Score predicted connections against hand-checked answer keys.

Two levels:
  strict  - device names, port names and direction must all match exactly
            (ignoring upper/lower case and extra spaces);
  relaxed - direction is ignored and quantity suffixes like "(2)" are dropped.

Every row is one cable, so rows are counted, not just compared as a set: a chain of
identical devices with 11 identical cables must produce 11 rows to score fully.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


def _norm(s: str) -> str:
    return " ".join(s.upper().split())


def _loose(s: str) -> str:
    return _norm(re.sub(r"\(\s*\d+\s*\)", " ", s))


def _strict(conns: list[dict]) -> Counter:
    return Counter((_norm(c["source_device"]), _norm(c["source_port"]), _norm(c["target_device"]), _norm(c["target_port"]))
                   for c in conns)


def _relaxed(conns: list[dict]) -> Counter:
    return Counter(frozenset([(_loose(c["source_device"]), _loose(c["source_port"])),
                              (_loose(c["target_device"]), _loose(c["target_port"]))]) for c in conns)


def _prf(pred: Counter, truth: Counter) -> dict:
    tp, n_pred, n_truth = sum((pred & truth).values()), sum(pred.values()), sum(truth.values())
    p = tp / n_pred if n_pred else (1.0 if not n_truth else 0.0)
    r = tp / n_truth if n_truth else (1.0 if not n_pred else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "matched": tp}


def score(pred: list[dict], truth: list[dict]) -> dict:
    sp, st = _strict(pred), _strict(truth)
    return {
        "predicted": len(pred), "expected": len(truth),
        "strict": _prf(sp, st),
        "relaxed": _prf(_relaxed(pred), _relaxed(truth)),
        "missing": sorted(" | ".join(t) for t in (st - sp).elements()),
        "extra": sorted(" | ".join(t) for t in (sp - st).elements()),
    }


def evaluate_dirs(pred_dir: Path, truth_dir: Path) -> dict:
    results = {}
    for truth_file in sorted(truth_dir.glob("*.json")):
        pred_file = pred_dir / truth_file.name
        if not pred_file.exists():
            results[truth_file.stem] = {"error": "no prediction file"}
            continue
        results[truth_file.stem] = score(json.loads(pred_file.read_text(encoding="utf-8")),
                                         json.loads(truth_file.read_text(encoding="utf-8")))
    return results

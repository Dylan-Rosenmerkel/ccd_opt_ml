"""
freeze_choice_v2_taskB.py -- task B of the v2 training plan: the change (relative) model frozen from the session-D
scorecard (lvl1/scorecard_v2.py) BEFORE session E is scored.

Rule, fixed in advance: the cell is "new" (amplifiers not in training: the level-1 pass mark is on them); per target,
* direction: the highest share of clear changes called the right way on the between-packet RANDOM pairs (what the
  session-F funnel compares), ties broken by the n -> n+1 pairs;
* size: the lowest RMS size error on the same pairs (direction-only models excluded).
Candidates: every model that sees the traces; not the references (no change, settings change, settings + amplifier).
The choice for every other pair type and the trained-amplifier cell is listed next to it (information only).

    .venv/bin/python lvl1/freeze_choice_v2_taskB.py --scorecard <dir>/scorecard.json --out lvl1/frozen_choice_v2_taskB.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path

REFERENCES = ("no change", "settings change", "settings + amplifier")
RULE_CELL, RULE_PAIRS, TIEBREAK = "new", "random pairs", "n -> n+1"


def ok(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def best(rel, cell, pairs, target, key, higher):
    c = []
    for name, per_type in rel.get(cell, {}).items():
        if name.startswith(REFERENCES):
            continue
        r = per_type.get(pairs, {}).get(target)
        if r and ok(r.get(key)):
            tb = per_type.get(TIEBREAK, {}).get(target, {}).get(key)
            c.append((r[key], tb if ok(tb) else (-1e9 if higher else 1e9), name, r))
    if not c:
        return None
    c.sort(key=lambda x: (x[0], x[1]), reverse=higher)
    v, _, name, r = c[0]
    return {"model": name, key: v, "always_up": r.get("clear_up"), "n_clear": r.get("n_clear"),
            "vs_settings": r.get("direction_vs_settings" if key == "direction" else "size_vs_settings"),
            "runner_up": c[1][2] if len(c) > 1 else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scorecard", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = json.loads(Path(args.scorecard).read_text())
    rel = s["relative"]
    targets = sorted({t for cell in rel.values() for m in cell.values() for p in m.values() for t in p})
    out = {"frozen": datetime.datetime.now().isoformat(timespec="seconds"), "scorecard": args.scorecard,
           "rule": f"cell '{RULE_CELL}', pair type '{RULE_PAIRS}' (tie-break '{TIEBREAK}'): per target, the highest "
                   "direction on clear changes and the lowest RMS size error; trace models only (no references)",
           "info": s["info"], "targets": {}}
    for t in targets:
        out["targets"][t] = {
            "direction": best(rel, RULE_CELL, RULE_PAIRS, t, "direction", True),
            "size": best(rel, RULE_CELL, RULE_PAIRS, t, "size_error", False),
            "other_pair_types_and_cells (information only)": {
                f"{cell} / {pt}": {"direction": (best(rel, cell, pt, t, "direction", True) or {}).get("model"),
                                   "size": (best(rel, cell, pt, t, "size_error", False) or {}).get("model")}
                for cell in rel for pt in ("within packet (image)", "within packet (strip)", "n -> n+1", "n -> n+2",
                                           "random pairs")}}
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    for t, v in out["targets"].items():
        d, z = v["direction"], v["size"]
        print(f"{t:8s} direction: {d['model'] if d else '-'} {100 * d['direction']:.0f} % (always up "
              f"{100 * (d['always_up'] or float('nan')):.0f} %)  |  size: "
              + (f"{z['model']} {z['size_error']:.1f} %" if z else "-") if d else f"{t}: no candidates")


if __name__ == "__main__":
    main()

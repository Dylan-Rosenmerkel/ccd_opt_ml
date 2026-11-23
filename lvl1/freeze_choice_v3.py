"""
freeze_choice_v3.py -- freeze the v3 (voltage-only) model choice from the session-G scorecard BEFORE session H is
scored: per target (sigma_e, lambda, gain), per cell (trained amplifiers, new amplifier), the 3 best ABSOLUTE models
(typical error %) and the 3 best CHANGE models (size error % on random pairs; direction-only models are ranked by
direction on clear changes separately).  The physics models are always kept (the physically motivated reference),
as are the settings-only references and the baselines.  Writes the frozen JSON and the model lists for
scorecard_v2.py --abs-models / --rel-models.

    .venv/bin/python lvl1/freeze_choice_v3.py --scorecard processed_v3n/lvl1_results/scorecard_G/scorecard.json \\
        --out lvl1/frozen_choice_v3.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path

T = ("sigma_e", "lambda", "gain")
TOP = 3
BASELINES_ABS = ("mean", "amplifier mean")
PHYSICS_ABS = ("trace: physics",)
PHYSICS_REL = ("trace: physics difference (sigma_e)", "trace: physics difference + correction (sigma_e)",
               "absolute trace: physics, differenced")
SETTINGS_ABS = ("settings: gp", "settings: trees", "settings + amplifier: gp", "settings + amplifier: trees")
SETTINGS_REL = ("settings change: gp", "settings change: trees", "settings + amplifier: gp", "settings + amplifier: trees")
DIRECTION_ONLY = ("trace: direction classifier [A, B, B-A]", "settings + trace: direction classifier")
ENSEMBLE_MEMBERS = {"ensemble: trace models": ("trace change: trees", "trace: combined trees [A, B, B-A]",
                                               "trace: physics difference + correction (sigma_e)"),
                    "ensemble: settings + trace": ("settings + trace: combined trees", "settings + trace: residual trees",
                                                   "trace: physics difference + correction (sigma_e)")}


def ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scorecard", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = json.loads(Path(args.scorecard).read_text())
    frozen = {"frozen": datetime.datetime.now().isoformat(timespec="seconds"), "source": args.scorecard,
              "rule": f"per target and cell: top {TOP} absolute models by typical error % (baselines excluded); top {TOP} "
                      "change models by size error % on random pairs (no-change and direction-only excluded), plus the "
                      "top direction-only model by direction on clear changes; physics models, settings references and "
                      "baselines always kept",
              "absolute": {}, "relative": {}, "direction": {}}
    abs_set, rel_set = set(BASELINES_ABS) | set(PHYSICS_ABS) | set(SETTINGS_ABS), {"no change"} | set(PHYSICS_REL) | set(SETTINGS_REL)
    for cell in ("trained", "new"):
        for kind in ("absolute", "relative", "direction"):
            frozen[kind][cell] = {}
        for t in T:
            a = sorted((v[t]["error"], k) for k, v in s["absolute"][cell].items()
                       if k not in BASELINES_ABS and t in v and ok(v[t]["error"]))[:TOP]
            r = sorted((v["random pairs"][t]["size_error"], k) for k, v in s["relative"][cell].items()
                       if k != "no change" and k not in DIRECTION_ONLY and t in v["random pairs"]
                       and ok(v["random pairs"][t].get("size_error")))[:TOP]
            dr = sorted((-v["random pairs"][t]["direction"], k) for k, v in s["relative"][cell].items()
                        if k in DIRECTION_ONLY and t in v["random pairs"] and ok(v["random pairs"][t].get("direction")))[:1]
            frozen["absolute"][cell][t] = [{"model": k, "error_pct": round(e, 2)} for e, k in a]
            frozen["relative"][cell][t] = [{"model": k, "size_error_pct": round(e, 2),
                                            "spearman": round(s["relative"][cell][k]["random pairs"][t]["spearman"], 3)}
                                           for e, k in r]
            frozen["direction"][cell][t] = [{"model": k, "direction": round(-e, 3)} for e, k in dr]
            abs_set |= {k for _, k in a}
            rel_set |= {k for _, k in r} | {k for _, k in dr}
    for e, members in ENSEMBLE_MEMBERS.items():
        if e in rel_set:
            rel_set |= set(members)
    # "absolute X, differenced" change models come from the absolute model X
    for k in list(rel_set):
        if k.startswith("absolute ") and k.endswith(", differenced"):
            abs_set.add(k[len("absolute "):-len(", differenced")])
    frozen["abs_models"] = sorted(abs_set)
    frozen["rel_models"] = sorted(m for m in rel_set if not m.startswith("ensemble:") and not m.startswith("absolute "))
    frozen["ensembles"] = sorted(m for m in rel_set if m.startswith("ensemble:"))
    Path(args.out).write_text(json.dumps(frozen, indent=1) + "\n")
    for kind in ("absolute", "relative"):
        for cell in ("trained", "new"):
            for t in T:
                print(f"{kind:8s} {cell:7s} {t:7s} " + " | ".join(
                    f"{x['model']} {x.get('error_pct', x.get('size_error_pct'))}" for x in frozen[kind][cell][t]))
    print("abs models:", frozen["abs_models"])
    print("rel models:", frozen["rel_models"])


if __name__ == "__main__":
    main()

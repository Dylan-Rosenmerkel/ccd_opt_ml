"""
freeze_choice_v2.py -- task A protocol step 2 of the v2 training plan: per target, the model, feature set and source
with the lowest session-D cross-validation error, written down (frozen) before session E is scored.

Candidates: every model x feature set of the train_lvl1 reports given (the tier ablation of each source, plus the
--feature-tiers auto runs), except the no-trace baselines (mean, settings) and the references told the amplifier
(physics_hier, settings_amp*): a frozen model may use the traces and/or the strip only.  The choice also lists, per
target, the best model of every source (the plan scores E "between the three sources") and the runner-up.

    .venv/bin/python lvl1/freeze_choice_v2.py --reports <src>=<report.json> ... --out lvl1/frozen_choice_v2.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

TARGETS = ("sigma_e", "lambda", "gain")
NOT_CANDIDATES = ("mean", "settings", "physics_hier", "settings_amp", "settings_amp_trees")


def split_key(k: str):
    return (k[:-1].split(" [", 1) if k.endswith("]") and " [" in k else (k, None))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reports", nargs="+", required=True, help="SOURCE=report.json (several per source allowed)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cands, info = [], {}
    for spec in args.reports:
        src, path = spec.split("=", 1)
        rep = json.loads(Path(path).read_text())
        info.setdefault(src, []).append({"report": path, "n_rows": rep["info"]["n_rows"], "n_packets": rep["info"]["n_packets"],
                                         "exclude_amps": rep["info"].get("exclude_amps"),
                                         "exclude_features": rep["info"].get("exclude_features")})
        default_spec = rep["info"].get("feature_tiers")
        for key, r in rep["results"].items():
            model, fset = split_key(key)
            if model in NOT_CANDIDATES or "scores" not in r:
                continue
            fset = fset or r.get("feature_set") or default_spec or "default"
            for t in TARGETS:
                s = r["scores"][t]
                cands.append({"source": src, "model": model, "features": fset, "target": t, "error": s["error"],
                              "crps": s.get("crps"), "rank": s.get("rank"), "report": path})
    choice = {"frozen": datetime.datetime.now().isoformat(timespec="seconds"),
              "rule": "per target: lowest session-D setting-grouped CV error (RMS of log error) over model x feature set x "
                      "source; no baselines, no amplifier-labelled references",
              "inputs": info, "targets": {}}
    for t in TARGETS:
        c = sorted((x for x in cands if x["target"] == t), key=lambda x: x["error"])
        best = c[0]
        per_source = {}
        for x in c:
            per_source.setdefault(x["source"], x)
        choice["targets"][t] = {"choice": best, "runner_up": c[1] if len(c) > 1 else None, "best_per_source": per_source}
    Path(args.out).write_text(json.dumps(choice, indent=1))
    for t, v in choice["targets"].items():
        b = v["choice"]
        print(f"{t:8s} {b['source']:18s} {b['model']} [{b['features']}]  CV error {100 * b['error']:.1f} %   "
              + "  per source: " + ", ".join(f"{s} {x['model']} [{x['features']}] {100 * x['error']:.1f}"
                                            for s, x in v["best_per_source"].items()))


if __name__ == "__main__":
    main()

"""
compare_runs.py -- the same models in several train_lvl1 runs (e.g. with and without an amplifier, with and without the
fingerprint features), scored on the SAME rows: A+B cross-validation, session C, leave one amplifier out (and its
"shape": the error after removing each amplifier's own mean offset, i.e. after one calibration image).

Each run directory holds one train_lvl1 run per feature set (<run>/<set>/predictions.csv, as written by the run.sh
scripts of 2026-11-04); every model is read at the set its A+B cross-validation chose (lvl1/explain_models.py).

    .venv/bin/python lvl1/compare_runs.py RUN_DIR ... [--channels 0 1 3] [--out FILE.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lvl1"))

import explain_models as em  # noqa: E402

#: models whose lambda / gain are another model's (the physics family shares them); used when a run lacks the model
ALIAS = {("physics_white", "lambda"): "physics", ("physics_white", "gain"): "physics", ("physics_joint", "gain"): "physics"}


def rms(r) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    return float(100 * np.sqrt(np.mean(r ** 2))) if len(r) else float("nan")


def scores(run: Path, model: str, target: str, spec: str, channels) -> dict | None:
    f = run / spec / "predictions.csv"
    if not f.exists():
        return None
    p = pd.read_csv(f)
    p = p[(p.board == 101) & p.channel.isin(channels)]
    m = model if f"{model}: log {target}" in p.columns else ALIAS.get((model, target))
    if m is None or f"{m}: log {target}" not in p.columns:
        return None
    held = p["held out"].astype(str).str.lower().eq("true")
    r = p[f"{m}: log {target}"] - p[f"true log {target}"]
    out = {"cv": rms(r[~held]), "c": rms(r[held])}
    col = f"loao {m}: log {target}"
    if col in p.columns:
        lr = (p[col] - p[f"true log {target}"])[~held]
        out["new_amp"] = rms(lr)
        out["new_amp_shape"] = rms(lr - lr.groupby(p.loc[~held, "channel"]).transform("mean"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 1, 3])
    parser.add_argument("--out")
    args = parser.parse_args()
    sets = em.chosen_sets()
    runs = [Path(r) for r in args.runs]
    names = [r.name.replace("2026-11-04_", "") for r in runs]
    result = {}
    for t in em.TARGETS:
        print(f"\n{t} (board 101, channels {args.channels}); per run: CV / session C / new amplifier (after one image)")
        print(f"{'model [set]':26s}" + "".join(f"{n:>30s}" for n in names))
        for m in em.BASELINES + em.TRACE_MODELS:
            s = sets[m][t]
            cells = []
            for run, n in zip(runs, names):
                sc = scores(run, m, t, s, args.channels)
                result.setdefault(f"{m}|{t}", {"set": s})[n] = sc
                cells.append(f"{'-':>30s}" if sc is None else
                             f"{sc['cv']:7.1f} {sc['c']:6.1f} {sc.get('new_amp', np.nan):6.1f} ({sc.get('new_amp_shape', np.nan):4.1f})")
            print(f"{m + ' [' + s + ']':26s}" + "".join(cells))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

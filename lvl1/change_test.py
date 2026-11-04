"""
change_test.py -- can the CHANGE of sigma_e, lambda and gain between nominal and a new setting be predicted from the
change of the trace (own trace at the new setting minus the reference trace at nominal, same amplifier, same packet)?

Target per row (one amplifier in one packet): d = log y(new setting) - log y(nominal), where y(nominal) is the same
amplifier's label interpolated in time between the nominal packets (L1_01, 07, 14, 32, 43), so the night's drift
mostly cancels.  Nominal packets themselves are not targets.  Inputs: the trace-feature differences diff.* (own minus
reference) of the P+A tiers; the reference model sees the settings change delta.* instead; "no change" predicts d = 0.

Scored (board 101 without ch2): A+B cross-validation (packets held out whole), session C (trained on A+B), and a new
amplifier (trained on the other amplifiers and the other packet folds).  Per target: RMS error of d (%), the fraction of
rows where the direction (up / down) is right among clear changes, and the Spearman correlation.

    .venv/bin/python lvl1/change_test.py [--out processed/lvl1_results/2026-11-04_change_test/results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

import train_lvl1 as tl  # noqa: E402
from ml import models  # noqa: E402
from ml.evaluate import grouped_folds  # noqa: E402

LABELS = {"sigma_e": "label.sigma_e", "lambda": "label.lambda_e_per_bin", "gain": "label.gain_adu_per_e"}
NOMINAL = ["L1_01", "L1_07", "L1_14", "L1_32", "L1_43"]
#: a change counts as "clear" (for the direction score) above this size (log units), about twice the label scatter
CLEAR = {"sigma_e": 0.05, "lambda": 0.15, "gain": 0.012}
TIERS = ROOT / "lvl1" / "feature_study" / "real_AB" / "evidence_tiers.json"
PLAN = ROOT / "lvl1" / "packet_plan" / "plan.json"


def load(exclude_channels=(2,)):
    d = pd.read_csv(ROOT / "processed" / "dataset.csv", low_memory=False)
    d = d[(d.board == 101) & ~d.channel.isin(exclude_channels)].copy()
    d = d[d["label.ok"].astype(str).str.lower().eq("true") & (d["trace.n_usable"] > 0)]
    d["t"] = pd.to_datetime(d["date_start"]).astype("int64") / 1e9
    ses = {p["packet_id"]: str(p.get("session")) for p in json.loads(PLAN.read_text())["packets"]}
    d["session"] = d.packet_id.map(ses)
    for t, col in LABELS.items():
        d[f"log_{t}"] = np.log(d[col])
        ref = np.full(len(d), np.nan)
        for ch, g in d.groupby("channel"):
            nom = g[g.packet_id.isin(NOMINAL)].sort_values("t")
            ref[d.index.get_indexer(g.index)] = np.interp(g.t, nom.t, np.log(nom[col]))
        d[f"d_{t}"] = d[f"log_{t}"] - ref
    return d[~d.packet_id.isin(NOMINAL)].reset_index(drop=True)


def inputs(d: pd.DataFrame):
    tiers = tl.load_tiers(TIERS)
    keep = [f for f, ts in tiers.items() if ts & {"P", "A1", "A2"}]
    diff_cols = [f"diff.{f}" for f in keep if f"diff.{f}" in d.columns]
    set_cols = [c for c in d.columns if c.startswith("delta.")]
    return d[diff_cols].to_numpy(float), d[set_cols].to_numpy(float)


def fit_predict(kind: str, X: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray, groups) -> np.ndarray:
    if kind == "no change":
        return np.zeros((len(te), Y.shape[1]))
    xtr, xte, _ = tl.prep_columns(X[tr], X[te])
    if xtr.shape[1] == 0:
        return np.tile(Y[tr].mean(0), (len(te), 1))
    learner = {"ridge": lambda: models.BayesRidge(), "trees": lambda: models.GBT(),
               "gp": lambda: models.GP(), "nn": lambda: models.MLPEnsemble(n_models=5, epochs=600)}[kind]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "nn":
            return learner.fit(xtr, Y[tr], groups=groups[tr]).predict(xte)[0]
        return learner.fit(xtr, Y[tr]).predict(xte)[0]


def score(d_true: np.ndarray, d_pred: np.ndarray, target: str) -> dict:
    ok = np.isfinite(d_true) & np.isfinite(d_pred)
    a, b = d_true[ok], d_pred[ok]
    clear = np.abs(a) > CLEAR[target]
    return {"error": float(100 * np.sqrt(np.mean((b - a) ** 2))), "n": int(ok.sum()), "n_clear": int(clear.sum()),
            "direction": float(np.mean(np.sign(b[clear]) == np.sign(a[clear]))) if clear.any() else None,
            "spearman": float(spearmanr(a, b)[0]) if np.std(b) > 0 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(ROOT / "processed/lvl1_results/2026-11-04_change_test/results.json"))
    args = parser.parse_args()
    d = load()
    Xtr_diff, Xset = inputs(d)
    T = list(LABELS)
    Y = d[[f"d_{t}" for t in T]].to_numpy(float)
    packets, amp = d.packet_id.to_numpy(), d.channel.to_numpy()
    ab, c = np.flatnonzero(d.session != "C"), np.flatnonzero(d.session == "C")
    folds = [ab[f] for f in grouped_folds(packets[ab], 5, seed=0)]
    runs = [("no change", None), ("settings change (trees)", ("trees", Xset)), ("settings change (gp)", ("gp", Xset)),
            ("trace change: ridge", ("ridge", Xtr_diff)), ("trace change: trees", ("trees", Xtr_diff)),
            ("trace change: gp", ("gp", Xtr_diff)), ("trace change: nn", ("nn", Xtr_diff))]
    results = {"_info": {"rows_AB": int(len(ab)), "rows_C": int(len(c)), "n_trace_inputs": int(Xtr_diff.shape[1]),
                         "true_change_rms": {t: float(100 * np.sqrt(np.nanmean(Y[ab, i] ** 2))) for i, t in enumerate(T)}}}
    for name, spec in runs:
        kind, X = ("no change", Xset) if spec is None else spec
        res = {}
        pred = np.full(Y.shape, np.nan)                                   # A+B cross-validation
        for te in folds:
            pred[te] = fit_predict(kind, X, Y, np.setdiff1d(ab, te), te, packets)
        res["cv"] = {t: score(Y[ab, i], pred[ab, i], t) for i, t in enumerate(T)}
        pc = fit_predict(kind, X, Y, ab, c, packets)                     # session C
        res["c"] = {t: score(Y[c, i], pc[:, i], t) for i, t in enumerate(T)}
        pl = np.full(Y.shape, np.nan)                                     # new amplifier
        for a in np.unique(amp[ab]):
            for te in folds:
                tst = te[amp[te] == a]
                trn = np.setdiff1d(ab[amp[ab] != a], te)
                if len(tst) and len(trn):
                    pl[tst] = fit_predict(kind, X, Y, trn, tst, packets)
        res["new_amp"] = {t: score(Y[ab, i], pl[ab, i], t) for i, t in enumerate(T)}
        results[name] = res
        print(f"{name:26s} " + "  ".join(
            f"{t}: CV {res['cv'][t]['error']:5.1f} C {res['c'][t]['error']:5.1f} new {res['new_amp'][t]['error']:5.1f}"
            f" dir {res['new_amp'][t]['direction'] if res['new_amp'][t]['direction'] is not None else float('nan'):.2f}"
            for t in T), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results["_info"]))


if __name__ == "__main__":
    main()

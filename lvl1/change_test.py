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
DATA = ROOT / "processed"   # processed data folder; set with --data
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
HOLDOUT = "C"                 # the prospective-test session (v1: C; v2: E)
EXCLUDE_CHANNELS = (2,)       # v1: board 101 ch2 excluded; v2: none
EXCLUDE_FRACTIONAL = False    # drop rows whose delayInt gives N / N+1 sample windows (label.fractional_window)


def planned_id(packet_id: str, planned) -> str | None:
    """The planned id a processed id belongs to: equal, or continued by a non-digit (L2_02b, L1_33_retake)."""
    if packet_id in planned:
        return packet_id
    for q in planned:
        if packet_id.startswith(q) and not packet_id[len(q)].isdigit():
            return q
    return None


def configure(data=None, plan=None, holdout=None, exclude_channels=None, exclude_fractional=None):
    """Point the change scripts at another dataset / plan (v2: --plan lvl1/packet_plan_v2/plan.json --holdout E
    --exclude-channels).  With a plan other than v1's, the nominal packets are the plan's "nominal" packets."""
    global DATA, PLAN, HOLDOUT, EXCLUDE_CHANNELS, NOMINAL, EXCLUDE_FRACTIONAL
    if data is not None:
        DATA = Path(data).resolve()
    if plan is not None:
        PLAN = Path(plan).resolve()
        pk = json.loads(PLAN.read_text())["packets"]
        NOMINAL = [p["packet_id"] for p in pk if str(p.get("purpose", "")).startswith("nominal")]
    if holdout is not None:
        HOLDOUT = holdout
    if exclude_channels is not None:
        EXCLUDE_CHANNELS = tuple(exclude_channels)
    if exclude_fractional is not None:
        EXCLUDE_FRACTIONAL = bool(exclude_fractional)


def add_config_args(parser):
    parser.add_argument("--data", default=None, help="processed data folder (default processed/)")
    parser.add_argument("--plan", default=None, help="packet plan (default lvl1/packet_plan/plan.json)")
    parser.add_argument("--holdout", default=None, help="prospective-test session (default C)")
    parser.add_argument("--exclude-channels", type=int, nargs="*", default=None, help="board-101 channels left out (default 2)")
    parser.add_argument("--exclude-fractional", action="store_true", default=None,
                        help="drop rows with fractional (N / N+1) CDS windows: their image labels are biased")


def configure_from(args):
    configure(args.data, args.plan, args.holdout, args.exclude_channels, args.exclude_fractional)


def load(exclude_channels=None):
    exclude_channels = EXCLUDE_CHANNELS if exclude_channels is None else exclude_channels
    d = pd.read_csv(DATA / "dataset.csv", low_memory=False)
    d = d[(d.board == 101) & ~d.channel.isin(exclude_channels)].copy()
    d = d[d["label.ok"].astype(str).str.lower().eq("true") & (d["trace.n_usable"] > 0)]
    if EXCLUDE_FRACTIONAL and "label.fractional_window" in d:
        d = d[~d["label.fractional_window"].astype(str).str.lower().eq("true")]
    d["t"] = pd.to_datetime(d["date_start"]).astype("int64") / 1e9
    ses = {p["packet_id"]: str(p.get("session")) for p in json.loads(PLAN.read_text())["packets"]}
    d["planned_id"] = [planned_id(p, ses) for p in d.packet_id]
    d["session"] = d.planned_id.map(ses)
    nominal_rows = d.planned_id.isin(NOMINAL)
    # The reference is nominal (v1, most of v2) or, in v2, the previous packet's settings: then the change is taken
    # against the label of the latest earlier packet imaged at the reference setting (same amplifier).
    nom_theta = d.loc[nominal_rows, "theta_id"].mode()
    nom_theta = nom_theta.iloc[0] if len(nom_theta) else None
    nonnominal_ref = ("reference_id" in d) & (d.get("reference_id") != nom_theta) if nom_theta is not None else np.zeros(len(d), bool)
    d["nominal_reference"] = ~np.asarray(nonnominal_ref, bool)
    for t, col in LABELS.items():
        d[f"log_{t}"] = np.log(d[col])
        ref = np.full(len(d), np.nan)
        for ch, g in d.groupby("channel"):
            nom = g[g.planned_id.isin(NOMINAL)].sort_values("t")
            idx = d.index.get_indexer(g.index)
            ref[idx] = np.interp(g.t, nom.t, np.log(nom[col]))
            for j, (i, row) in zip(idx, g.iterrows()):
                if d.at[i, "nominal_reference"]:
                    continue
                prev = g[(g.theta_id == row.reference_id) & (g.t < row.t)].sort_values("t")
                ref[j] = np.log(prev[col].iloc[-1]) if len(prev) else np.nan
        d[f"d_{t}"] = d[f"log_{t}"] - ref
    d = d[np.isfinite(d[[f"d_{t}" for t in LABELS]]).all(axis=1) | nominal_rows]
    nominal_rows = d.planned_id.isin(NOMINAL)
    return d[~nominal_rows].reset_index(drop=True)


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
    parser.add_argument("--out", default=None, help="default <data>/lvl1_results/change_test/results.json")
    add_config_args(parser)
    args = parser.parse_args()
    configure_from(args)
    args.out = args.out or str(DATA / "lvl1_results" / "change_test" / "results.json")
    d = load()
    Xtr_diff, Xset = inputs(d)
    T = list(LABELS)
    Y = d[[f"d_{t}" for t in T]].to_numpy(float)
    packets, amp = d.packet_id.to_numpy(), d.channel.to_numpy()
    ab, c = np.flatnonzero(d.session != HOLDOUT), np.flatnonzero(d.session == HOLDOUT)
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
        if len(c):                                                        # the prospective-test session
            pc = fit_predict(kind, X, Y, ab, c, packets)
            res["c"] = {t: score(Y[c, i], pc[:, i], t) for i, t in enumerate(T)}
        else:
            res["c"] = {t: {"error": float("nan"), "n": 0, "n_clear": 0, "direction": None, "spearman": None} for t in T}
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

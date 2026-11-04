"""
change_test2.py -- follow-up of change_test.py: why is the SIZE of a change poorly predicted, and do models suited to
few examples and change targets do better?

1. Diagnostics: how much of the change error is one outlier packet (L1_18, a 38-minute stall), and how large the label
   noise of the change itself is (the nominal-to-nominal scatter after interpolation).
2. Inputs: plain differences (own - reference) vs log-ratios log(own / reference) for positive features (changes are
   multiplicative); all P+A features vs the 8 protected noise estimators (P).
3. Models for small data: Lasso and PLS (few effective inputs), a SAAS GP (sparse, Bayesian), trees, and for sigma_e a
   physics difference model, d log sigma_e = d log w - d log g with d log g learned.
4. Gain: is the change of the gain predictable, per packet (common to the amplifiers) and per amplifier.

Scored on a new amplifier (trained on the other amplifiers and the other packet folds), board 101 without ch2, A+B.

    .venv/bin/python lvl1/change_test2.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

import change_test as ct  # noqa: E402
import train_lvl1 as tl  # noqa: E402
from ml import models  # noqa: E402
from ml.evaluate import grouped_folds  # noqa: E402

T = list(ct.LABELS)
WHITE = "fit: white noise per skip (ADU)"


def feature_blocks(d):
    tiers = tl.load_tiers(ct.TIERS)
    pa = [f for f, ts in tiers.items() if ts & {"P", "A1", "A2"} and f"trace.{f}" in d.columns and f"ref.{f}" in d.columns]
    p = [f for f, ts in tiers.items() if "P" in ts and f"trace.{f}" in d.columns]
    def diff(fs):
        return (d[[f"trace.{f}" for f in fs]].to_numpy(float) - d[[f"ref.{f}" for f in fs]].to_numpy(float))
    def logratio(fs):
        a, b = d[[f"trace.{f}" for f in fs]].to_numpy(float), d[[f"ref.{f}" for f in fs]].to_numpy(float)
        pos = (np.nanmin(a, 0) > 0) & (np.nanmin(b, 0) > 0)
        out = np.where(pos, np.log(np.where(pos, a, 1) / np.where(pos, b, 1)), a - b)
        return out
    return {"diff P+A": diff(pa), "log-ratio P+A": logratio(pa), "log-ratio P": logratio(p),
            "settings change": d[[c for c in d.columns if c.startswith("delta.")]].to_numpy(float)}


def learner(kind):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import LassoCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    class Sk:
        def __init__(self, make):
            self.make = make
        def fit(self, X, Y):
            self.m = [self.make().fit(X, Y[:, t]) for t in range(Y.shape[1])]
            return self
        def predict(self, X):
            return np.column_stack([np.ravel(m.predict(X)) for m in self.m]), None
    if kind == "lasso":
        return Sk(lambda: make_pipeline(StandardScaler(), LassoCV(cv=4, alphas=30, max_iter=5000)))
    if kind == "pls":
        return Sk(lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=2, scale=False)))
    if kind == "trees":
        return models.GBT()
    if kind == "gp":
        return models.GP()
    if kind == "saas":
        from ml.bogp import BoTorchGP
        return BoTorchGP("saas", warmup=96, num_samples=48, thinning=8)
    raise ValueError(kind)


def predict(kind, X, Y, tr, te):
    xtr, xte, _ = tl.prep_columns(X[tr], X[te])
    if xtr.shape[1] == 0:
        return np.tile(Y[tr].mean(0), (len(te), 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = learner(kind).fit(xtr, Y[tr]).predict(xte)
    return np.asarray(out[0] if isinstance(out, tuple) else out)


def new_amp_predictions(kind, X, Y, ab, amp, folds):
    pl = np.full(Y.shape, np.nan)
    for a in np.unique(amp[ab]):
        for te in folds:
            tst = te[amp[te] == a]
            trn = np.setdiff1d(ab[amp[ab] != a], te)
            if len(tst) and len(trn):
                pl[tst] = predict(kind, X, Y, trn, tst)
    return pl


def scores(y, p, packets):
    ok = np.isfinite(y) & np.isfinite(p)
    keep = ok & (packets != "L1_18")
    return (100 * np.sqrt(np.mean((p[ok] - y[ok]) ** 2)), 100 * np.sqrt(np.mean((p[keep] - y[keep]) ** 2)),
            100 * np.median(np.abs(p[ok] - y[ok])), spearmanr(y[ok], p[ok])[0] if np.std(p[ok]) > 0 else np.nan)


def main():
    d = ct.load()
    blocks = feature_blocks(d)
    Y = d[[f"d_{t}" for t in T]].to_numpy(float)
    packets, amp = d.packet_id.to_numpy(), d.channel.to_numpy()
    ab = np.flatnonzero(d.session != "C")
    folds = [ab[f] for f in grouped_folds(packets[ab], 5, seed=0)]

    print("1. diagnostics (A+B, no ch2)")
    for i, t in enumerate(T):
        y = Y[ab, i]
        print(f"   {t:8s} true change RMS {100*np.sqrt(np.mean(y**2)):5.1f} %, without L1_18 "
              f"{100*np.sqrt(np.mean(y[packets[ab] != 'L1_18']**2)):5.1f} %, median |change| {100*np.median(np.abs(y)):4.1f} %")
    nom = ct.NOMINAL
    import pandas as pd
    full = pd.read_csv(ROOT / "processed" / "dataset.csv", low_memory=False)
    full = full[(full.board == 101) & (full.channel != 2) & full.packet_id.isin(nom)]
    for t, col in ct.LABELS.items():
        sd = np.median([np.std(np.diff(np.log(g.sort_values("date_start")[col]))) / np.sqrt(2) for _, g in full.groupby("channel")])
        print(f"   {t:8s} label noise of a change (nominal to nominal) about {100*sd*np.sqrt(2):4.1f} %")

    print("\n2-3. new amplifier: RMS / RMS without L1_18 / median |error| / Spearman")
    runs = [("no change", None, None)]
    for kind in ("trees", "gp", "lasso", "pls", "saas"):
        for inp in ("diff P+A", "log-ratio P+A", "log-ratio P", "settings change"):
            if kind == "saas" and inp in ("diff P+A",):
                continue
            runs.append((f"{kind}, {inp}", kind, inp))
    res = {}
    for name, kind, inp in runs:
        p = np.zeros(Y.shape) if kind is None else new_amp_predictions(kind, blocks[inp], Y, ab, amp, folds)
        res[name] = p
        print(f"   {name:28s} " + "  ".join(
            "{}: {:5.1f} {:5.1f} {:4.1f} {:5.2f}".format(t, *scores(Y[ab, i], p[ab, i], packets[ab])) for i, t in enumerate(T)), flush=True)

    # physics difference model for sigma_e: d log sigma_e = d log w - d log g, d log g from the best gain model
    i_s, i_g = T.index("sigma_e"), T.index("gain")
    dw = np.log(d[f"trace.{WHITE}"].to_numpy(float) / d[f"ref.{WHITE}"].to_numpy(float))
    for gname in ("no change", "trees, log-ratio P+A"):
        ps = dw - res[gname][:, i_g]
        print(f"   physics difference (gain from {gname}): sigma_e " + "{:5.1f} {:5.1f} {:4.1f} {:5.2f}".format(
            *scores(Y[ab, i_s], ps[ab], packets[ab])))
        # with a learned correction of the residual (trees on the log-ratio P features)
    r = Y[:, i_s] - dw
    Xr = blocks["log-ratio P+A"]
    pc = new_amp_predictions("trees", Xr, np.column_stack([r, r, r]), ab, amp, folds)[:, 0]
    print("   physics difference + learned correction: sigma_e " + "{:5.1f} {:5.1f} {:4.1f} {:5.2f}".format(
        *scores(Y[ab, i_s], dw[ab] + pc[ab], packets[ab])))

    print("\n4. gain: is the change common to the amplifiers of a packet?")
    g = d.assign(dg=Y[:, i_g]).iloc[ab].groupby("packet_id").dg
    common = g.transform("mean").to_numpy()
    y = Y[ab, i_g]
    print(f"   share of the gain-change variance that is common to the packet's amplifiers: "
          f"{1 - np.var(y - common) / np.var(y):.2f}")
    print(f"   predicting an amplifier's gain change by the other amplifiers' mean in the same packet: "
          f"{100*np.sqrt(np.mean((y - (g.transform('sum').to_numpy() - y) / (g.transform('count').to_numpy() - 1))**2)):.2f} % "
          f"(no change: {100*np.sqrt(np.mean(y**2)):.2f} %)")


if __name__ == "__main__":
    main()

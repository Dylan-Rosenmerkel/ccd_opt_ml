"""
change_handpicked.py -- a few hand-picked numbers from the difference curves (new trace minus reference trace) to
predict the change of sigma_e, lambda and gain.

For each target, on the TRAINING rows only (so the test never chooses the features): correlate every point of every
difference curve (16 x 256, standardized per point) with the change of the target, smooth along the curve (+-4 points),
and keep the K best bands, at most one per curve.  Each band becomes one feature: the mean of the difference curve over
those 9 points.  The change of the trace's white noise (log ratio) is always added.  Models: Huber regression (robust
linear, few inputs), GP, boosted trees.  Scored like change_models.py: a new amplifier (trained on the other amplifiers
and packet folds) and session C; board 101 without ch2.

    .venv/bin/python lvl1/change_handpicked.py [--k 6]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

import change_models as cm  # noqa: E402
import train_lvl1 as tl  # noqa: E402
from ml import models, real_inputs  # noqa: E402
from ml.evaluate import grouped_folds  # noqa: E402

T = cm.T
HALF = 4
WHITE = "fit: white noise per skip (ADU)"


def smooth(R):
    k = np.ones(2 * HALF + 1) / (2 * HALF + 1)
    return np.array([np.convolve(r, k, mode="same") for r in R])


class Handpicked:
    """Choose K bands of the difference curves for one target on training rows; turn any rows into band means."""

    def __init__(self, k):
        self.k = k

    def fit(self, Dc, y):
        self.mu, self.sd = Dc.mean(0), Dc.std(0) + 1e-9
        Z = (Dc - self.mu) / self.sd
        yz = (y - y.mean()) / (y.std() + 1e-12)
        R = smooth(np.einsum("n,nij->ij", yz, Z) / len(y))
        best_point = np.argmax(np.abs(R), 1)
        strength = np.abs(R[np.arange(len(R)), best_point])
        self.curves = np.argsort(-strength)[: self.k]
        self.points = best_point[self.curves]
        return self

    def transform(self, Dc):
        Z = (Dc - self.mu) / self.sd
        cols = [Z[:, c, max(0, p - HALF): p + HALF + 1].mean(1) for c, p in zip(self.curves, self.points)]
        return np.column_stack(cols)


def learn(kind, X, y, Xte):
    xtr, xte, _ = tl.prep_columns(X, Xte)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "huber":
            from sklearn.linear_model import HuberRegressor
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(StandardScaler(), HuberRegressor(alpha=1.0, max_iter=2000)).fit(xtr, y).predict(xte)
        m = {"gp": models.GP, "trees": models.GBT}[kind]()
        return m.fit(xtr, y[:, None]).predict(xte)[0][:, 0]


def predict(D, tr, te, kind, k, picks):
    Dc = D["CB"] - D["CA"]
    w = np.log(D["d"][f"trace.{WHITE}"].to_numpy(float) / D["d"][f"ref.{WHITE}"].to_numpy(float))
    out = np.zeros((len(te), 3))
    for i, t in enumerate(T):
        hp = Handpicked(k).fit(Dc[tr], D["Y"][tr, i])
        picks[t].update(zip(hp.curves.tolist(), hp.points.tolist()))
        Xtr = np.column_stack([hp.transform(Dc[tr]), w[tr]])
        Xte = np.column_stack([hp.transform(Dc[te]), w[te]])
        out[:, i] = learn(kind, Xtr, D["Y"][tr, i], Xte)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--k", type=int, default=6, help="bands per target")
    parser.add_argument("--out", default=str(ROOT / "processed/lvl1_results/2026-11-04_change_models/handpicked.json"))
    args = parser.parse_args()
    D = cm.build()
    ab, c = np.flatnonzero(D["session"] != "C"), np.flatnonzero(D["session"] == "C")
    folds = [ab[f] for f in grouped_folds(D["packet"][ab], 5, seed=0)]
    names = [str(n) for n in real_inputs.load(ROOT / "processed/model_inputs/curves.npz")["curve_names"]]
    results = {}
    for kind in ("huber", "gp", "trees"):
        picks = {t: collections.Counter() for t in T}
        pl = np.full(D["Y"].shape, np.nan)
        for a in np.unique(D["amp"][ab]):
            for te in folds:
                tst = te[D["amp"][te] == a]
                trn = np.setdiff1d(ab[D["amp"][ab] != a], te)
                if len(tst) and len(trn):
                    pl[tst] = predict(D, trn, tst, kind, args.k, picks)
        pc = predict(D, ab, c, kind, args.k, picks)
        name = f"hand-picked difference bands: {kind}"
        results[name] = {"new_amp": cm.score(D["Y"][ab], pl[ab]), "c": cm.score(D["Y"][c], pc),
                         "most_picked": {t: [[names[cv], int(p), n] for (cv, p), n in picks[t].most_common(6)] for t in T}}
        r = results[name]
        print(f"{name:40s} new amp: " + "  ".join(
            f"{t} {r['new_amp'][t]['error']:5.1f} (med {r['new_amp'][t]['median']:4.1f}, dir {r['new_amp'][t]['direction']:.2f})"
            for t in T) + "   C: " + "  ".join(f"{t} {r['c'][t]['error']:5.1f}" for t in T), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    for t in T:
        print(t, "most often picked:", results["hand-picked difference bands: huber"]["most_picked"][t][:4])


if __name__ == "__main__":
    main()

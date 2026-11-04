"""
change_pairs.py -- predict the change of sigma_e, lambda and gain between ANY two packets of the same amplifier
(setting 1 vs setting 2), from the change of its own trace between them.

Every pair (i, j) of packets of one amplifier gives one example: target log y_j - log y_i (both labels measured from
images, no interpolated nominal), inputs the log-ratio of the own-trace features (P+A tiers) between the two packets, or
the settings change theta_j - theta_i for the reference.  Board 101 without ch2, sessions A+B, nominal packets included.

Testing on a new amplifier: the packets are split into 5 folds; a test pair has both packets in the test fold and
belongs to the held-out amplifier; training pairs come from the other amplifiers with neither packet in the test fold.

    .venv/bin/python lvl1/change_pairs.py [--max-hours 1]
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

import change_test as ct  # noqa: E402
import train_lvl1 as tl  # noqa: E402
from ml import models  # noqa: E402
from ml.evaluate import grouped_folds  # noqa: E402

T = list(ct.LABELS)


def load():
    d = pd.read_csv(ROOT / "processed" / "dataset.csv", low_memory=False)
    d = d[(d.board == 101) & (d.channel != 2)]
    d = d[d["label.ok"].astype(str).str.lower().eq("true") & (d["trace.n_usable"] > 0)].copy()
    plan = {p["packet_id"]: str(p.get("session")) for p in __import__("json").loads(ct.PLAN.read_text())["packets"]}
    d = d[d.packet_id.map(plan) != "C"].reset_index(drop=True)
    tiers = tl.load_tiers(ct.TIERS)
    feats = [f for f, ts in tiers.items() if ts & {"P", "A1", "A2"} and f"trace.{f}" in d.columns]
    F = d[[f"trace.{f}" for f in feats]].to_numpy(float)
    pos = np.nanmin(F, 0) > 0
    F = np.where(pos, np.log(np.where(pos, F, 1)), F)                  # log for positive features
    Th = d[[c for c in d.columns if c.startswith("theta.")]].to_numpy(float)
    Y = np.log(d[[ct.LABELS[t] for t in T]].to_numpy(float))
    return d, F, Th, Y


def pairs(d, max_hours=None):
    """Every pair of packets of one amplifier; with max_hours, only pairs taken within that many hours."""
    t = pd.to_datetime(d["date_start"]).astype("int64").to_numpy() / 3.6e12
    out = []
    for ch, g in d.groupby("channel"):
        out += [(i, j) for i, j in combinations(g.index.to_numpy(), 2)
                if max_hours is None or abs(t[j] - t[i]) <= max_hours]
    return np.array(out)


def predict(kind, X, Y, tr, te):
    xtr, xte, _ = tl.prep_columns(X[tr], X[te])
    if kind == "no change":
        return np.zeros((len(te), Y.shape[1]))
    if xtr.shape[1] == 0:
        return np.zeros((len(te), Y.shape[1]))
    m = {"trees": lambda: models.GBT(), "gp": lambda: models.GP(max_points=600)}[kind]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.fit(xtr, Y[tr]).predict(xte)[0]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-hours", type=float, help="only pairs of packets taken within this many hours")
    parser.add_argument("--models", nargs="*", default=["no change", "settings change: trees", "settings change: gp",
                                                        "trace change: trees", "trace change: gp"])
    args = parser.parse_args()
    d, F, Th, Y = load()
    P = pairs(d, args.max_hours)
    i, j = P[:, 0], P[:, 1]
    Xf, Xs, Yp = F[j] - F[i], Th[j] - Th[i], Y[j] - Y[i]
    amp, pk = d.channel.to_numpy(), d.packet_id.to_numpy()
    upk = np.unique(pk)
    fold_of = {p: k for k, f in enumerate(grouped_folds(upk, 5, seed=0)) for p in upk[f]}
    fi, fj = np.array([fold_of[p] for p in pk[i]]), np.array([fold_of[p] for p in pk[j]])
    pa = amp[i]
    print(f"{len(d)} rows, {len(P)} pairs ({len(P)//len(np.unique(amp))} per amplifier); "
          f"true change RMS: " + ", ".join(f"{t} {100*np.sqrt(np.mean(Yp[:, k]**2)):.1f} %" for k, t in enumerate(T)))
    clear = {"sigma_e": 0.05, "lambda": 0.15, "gain": 0.012}
    for name, kind, X in (("no change", "no change", Xs), ("settings change: trees", "trees", Xs),
                          ("settings change: gp", "gp", Xs), ("trace change: trees", "trees", Xf),
                          ("trace change: gp", "gp", Xf)):
        if name not in args.models:
            continue
        pred = np.full(Yp.shape, np.nan)
        for a in np.unique(amp):
            for k in range(5):
                te = np.flatnonzero((pa == a) & (fi == k) & (fj == k))
                tr = np.flatnonzero((pa != a) & (fi != k) & (fj != k))
                if len(te) and len(tr):
                    pred[te] = predict(kind, X, Yp, tr, te)
        ok = np.isfinite(pred[:, 0])
        cells = []
        for k, t in enumerate(T):
            y, p = Yp[ok, k], pred[ok, k]
            c = np.abs(y) > clear[t]
            direction = np.mean(np.sign(p[c]) == np.sign(y[c])) if c.any() else np.nan
            cells.append(f"{t}: {100*np.sqrt(np.mean((p-y)**2)):5.1f} % (median {100*np.median(np.abs(p-y)):4.1f}, "
                         f"direction {direction:.2f} of {c.sum()})")
        print(f"{name:24s} " + "  ".join(cells) + f"   [{ok.sum()} test pairs]", flush=True)


if __name__ == "__main__":
    main()

"""
feature_importance.py -- let trained models say which trace features matter (a learned alternative to the evidence
tiers of lvl1/feature_study.py, which need no labels).

Three independent methods, each on the own-trace features (X_trace) plus the settings (theta), for every target
(log sigma_e, log lambda, log gain), with packet-grouped resampling so a feature is not judged on the rows it was fitted
to:

* stability selection   elastic net (l1_ratio 0.5, alpha from ElasticNetCV on all rows) refitted on packet-bootstrap
                        resamples; importance = how often the feature's coefficient is non-zero (Meinshausen & Buehlmann
                        2010).
* permutation           gradient-boosted trees (sklearn HistGradientBoosting) fitted on 4/5 of the packets; importance =
                        the increase of the held-out squared error when the feature is shuffled (Breiman 2001), averaged
                        over folds and repeats.
* SAAS relevance        (--saas) the median inverse length scale of a SAAS GP (src/ml/bogp.py) fitted to all rows;
                        slow, so off by default.

A feature only counts as important if it beats SHADOW features: a row-shuffled copy of every column is added and the
feature's importance must exceed the 95th percentile of the shadows' in most of --rounds rounds with fresh shadows
(Boruta, Kursa & Rudnicki 2010).  Ranking all
features against each other does NOT work at this data size: on synthetic 41-packet data, beyond the one or two features
that identify the amplifier, real and pure-noise features had the same importances (about 1e-4).  Learned tiers:
L1 = both methods confirm the feature, each by more than 1.5 x the shadows' 95th percentile (for at least one target);
L2 = a candidate (one method, or both narrowly); L3 = not shown to matter.
--add-noise k appends k pure-noise columns ("null i") as a check: they should almost all land in L3.

    .venv/bin/python lvl1/feature_importance.py --inputs <model_inputs> [--out DIR] [--add-noise 20] [--saas]
                                                [--boot 50] [--repeats 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ml import real_inputs  # noqa: E402
from ml.evaluate import grouped_folds  # noqa: E402

TARGETS = ("sigma_e", "lambda", "gain")
MARGIN = 1.5        # L1 needs both methods to beat the shadows' 95th percentile by this factor


def prepare(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float).copy()
    med = np.nanmedian(X, axis=0)
    med[~np.isfinite(med)] = 0
    X = np.where(np.isfinite(X), X, med)
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1)


def stability(X, y, packets, n_boot, rng) -> np.ndarray:
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # alpha by packet-grouped CV (rows of one packet share its noise; row-wise folds would favour a small alpha)
        folds = grouped_folds(packets, min(5, len(np.unique(packets))), seed=0)
        cv = [(np.setdiff1d(np.arange(len(y)), te), te) for te in folds]
        alpha = ElasticNetCV(l1_ratio=0.5, cv=cv, alphas=30, max_iter=5000).fit(X, y).alpha_
        ups = np.unique(packets)
        hits = np.zeros(X.shape[1])
        for _ in range(n_boot):
            pick = rng.choice(ups, len(ups), replace=True)
            idx = np.concatenate([np.flatnonzero(packets == p) for p in pick])
            hits += ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=5000).fit(X[idx], y[idx]).coef_ != 0
    return hits / n_boot


def permutation(X, y, packets, repeats, rng) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    folds = grouped_folds(packets, min(5, len(np.unique(packets))), seed=0)
    imp = np.zeros(X.shape[1])
    for test in folds:
        train = np.setdiff1d(np.arange(len(y)), test)
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15).fit(X[train], y[train])
        base = np.mean((m.predict(X[test]) - y[test]) ** 2)
        for j in range(X.shape[1]):
            Xp = X[test].copy()
            for _ in range(repeats):
                Xp[:, j] = rng.permutation(Xp[:, j])
                imp[j] += np.mean((m.predict(Xp) - y[test]) ** 2) - base
    return imp / (len(folds) * repeats)


def saas_relevance(X, y) -> np.ndarray:
    from ml.bogp import BoTorchGP
    gp = BoTorchGP("saas", warmup=96, num_samples=48, thinning=8).fit(X, y)
    return gp.relevance(0)


def with_shadows(X: np.ndarray, rng) -> np.ndarray:
    """X plus a "shadow" copy of every column with its rows shuffled independently (Boruta, Kursa & Rudnicki 2010):
    the importances of the shadows show what a feature WITHOUT information scores."""
    S = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    return np.hstack([X, S])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out")
    parser.add_argument("--add-noise", type=int, default=0, help="append this many pure-noise columns as a check")
    parser.add_argument("--saas", action="store_true", help="also SAAS GP relevance (slow)")
    parser.add_argument("--boot", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5, help="rounds of fresh shadow features; a feature must win most of them")
    parser.add_argument("--board", type=int, default=101)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    d = real_inputs.load(Path(args.inputs) / "features.npz")
    rows = (d["board"] == args.board) & np.isfinite(d["Y"]).all(1) & d["label_ok"] & ~d["fake"]
    names = [str(n) for n in d["feature_names"]]
    X = d["X_trace"][rows]
    if args.add_noise:
        X = np.hstack([X, rng.normal(0, 1, (len(X), args.add_noise))])
        names += [f"null {i}" for i in range(args.add_noise)]
    theta = d["theta"][rows] if np.asarray(d["theta"]).size else np.zeros((len(X), 0))
    Xs = prepare(np.hstack([X, theta]))
    nf = len(names)
    packets = d["packet_id"][rows].astype(str)
    Y = d["Y"][rows]
    table = pd.DataFrame(index=names)
    n_base = Xs.shape[1]                                 # features + settings
    hits_any = {"stability": np.zeros(nf, bool), "permutation": np.zeros(nf, bool)}
    win_margin = {"stability": np.zeros(nf), "permutation": np.zeros(nf)}   # best margin among the targets it wins
    for t, tn in enumerate(TARGETS):
        y = Y[:, t]
        wins = {"stability": np.zeros(nf), "permutation": np.zeros(nf)}
        vals = {"stability": np.zeros(nf), "permutation": np.zeros(nf)}
        margin = {"stability": np.zeros(nf), "permutation": np.zeros(nf)}
        for _ in range(args.rounds):                     # fresh shadows every round (Boruta)
            XS = with_shadows(Xs, rng)
            for meth, v in (("stability", stability(XS, y, packets, args.boot, rng)),
                            ("permutation", permutation(XS, y, packets, args.repeats, rng))):
                real, shadow = v[:nf], v[n_base:n_base + nf]
                thr = np.quantile(shadow, 0.95)
                wins[meth] += real > thr
                vals[meth] += real / args.rounds
                margin[meth] += (real / thr if thr > 0 else np.where(real > 0, 10.0, 0.0)) / args.rounds
        for meth in wins:
            table[f"{meth} {tn}"] = vals[meth]
            table[f"{meth} {tn} / shadow 95 %"] = margin[meth]
            table[f"{meth} {tn} wins"] = wins[meth] / args.rounds
            won = wins[meth] / args.rounds > 0.5                     # beats the shadows in most rounds
            hits_any[meth] |= won
            win_margin[meth] = np.maximum(win_margin[meth], np.where(won, margin[meth], 0))
        if args.saas:
            table[f"SAAS {tn}"] = saas_relevance(Xs, y)[:nf]
        print(f"{tn}: {int((wins['stability'] / args.rounds > 0.5).sum())} features beat the shadows in most rounds by stability, "
              f"{int((wins['permutation'] / args.rounds > 0.5).sum())} by permutation", flush=True)
    n_hits = hits_any["stability"].astype(int) + hits_any["permutation"].astype(int)
    table["methods confirming"] = n_hits
    margins = table[[c for c in table.columns if c.endswith("/ shadow 95 %")]].replace(np.inf, 1e6)
    table["best margin over shadows"] = margins.max(axis=1)
    # learned tiers: L1 = both methods confirm, each by more than MARGIN x the shadows' 95th percentile; L2 = a
    # candidate (one method, or both but narrowly); L3 = not shown to matter.  On synthetic data (3 datasets, 20 noise
    # columns) no noise column reached L1, but 5-6 of the 20 reached L2: L2 is a list to look at, not a feature set.
    strong = np.minimum(win_margin["stability"], win_margin["permutation"]) > MARGIN
    table["learned tier"] = np.select([strong, n_hits >= 1], ["L1", "L2"], "L3")
    tiers_file = ROOT / "lvl1" / "feature_study" / "evidence_tiers.json"
    if tiers_file.exists():
        ev = {n: k for k, v in json.loads(tiers_file.read_text()).items() for n in v}
        table["evidence tier"] = [ev.get(n, "–") for n in table.index]
    table = table.sort_values(["methods confirming", "best margin over shadows"], ascending=False)
    out = Path(args.out) if args.out else Path(args.inputs).parent / "feature_importance"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "feature_importance.csv")
    print(table[["methods confirming", "best margin over shadows", "learned tier"] + (["evidence tier"] if "evidence tier" in table else [])].round(2).to_string())
    print("\nlearned tiers:", table["learned tier"].value_counts().to_dict())
    if args.add_noise:
        nulls = table.index.str.startswith("null ")
        print(f"check: of the {args.add_noise} pure-noise columns, {int((table.loc[nulls, 'learned tier'] != 'L3').sum())} reach L1/L2 "
              f"(false positives; L1: {int((table.loc[nulls, 'learned tier'] == 'L1').sum())})")
    print(f"wrote {out / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()

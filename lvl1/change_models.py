"""
change_models.py -- models built for predicting the CHANGE of sigma_e, lambda and gain between two traces of the same
amplifier (nominal reference -> new setting), in size and direction.

Models
* shared encoder ("Siamese"): one network g scores a single trace; the predicted change is g(B) - g(A), so swapping
  the traces flips the sign by construction.  Inputs to g: the 81 trace features (log for positive ones), or the
  16-number embedding of the trace curves by the CNN encoder pre-trained on unlabelled board-101 traces.
  Losses: "size" (Huber on the change), "direction" (pairwise logistic on its sign) or "both"; in the direction term a
  change of the size of the label noise is down-weighted, w = 1 - exp(-d^2 / (2 s^2)).
* subtracted waveforms: the 16 curves of the new trace minus those of the reference (standardized per point), reduced
  to their 10 main patterns (PCA, fitted on the training rows); trees, GP and an intercept-free ridge (antisymmetric) on
  them.  Also a map of where along each difference curve the change of sigma_e and lambda is correlated.
* references: no change; trees on the feature differences (change_test.py's best); GP on the settings change.

Training rows: the within-packet changes of the training amplifiers and packets, plus (option --cross-hours) pairs of
packets of one amplifier taken within that many hours.  Test: a new amplifier (trained on the other amplifiers and the
other packet folds) and session C (trained on A+B).  Board 101 without ch2.

    .venv/bin/python lvl1/change_models.py [--cross-hours 1] [--out processed/lvl1_results/2026-11-04_change_models]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

import change_test as ct  # noqa: E402
import train_lvl1 as tl  # noqa: E402
from ml import models, real_inputs  # noqa: E402
from ml.evaluate import grouped_folds  # noqa: E402

T = list(ct.LABELS)
NOISE = np.array([0.019, 0.12, 0.008])          # label noise of a change (change_test2.py), log units
CLEAR = np.array([ct.CLEAR[t] for t in T])


# --------------------------------------------------------------------------- data
def build():
    d = ct.load()                                                    # within-packet changes, no nominal packets
    tiers = tl.load_tiers(ct.TIERS)
    feats = [f for f, ts in tiers.items() if ts & {"P", "A1", "A2"} and f"trace.{f}" in d and f"ref.{f}" in d]
    A, B = d[[f"ref.{f}" for f in feats]].to_numpy(float), d[[f"trace.{f}" for f in feats]].to_numpy(float)
    pos = (np.nanmin(A, 0) > 0) & (np.nanmin(B, 0) > 0)
    lg = lambda X: np.where(pos, np.log(np.where(pos, X, 1)), X)
    c = real_inputs.load(ROOT / "processed" / "model_inputs" / "curves.npz")
    key = {(p, int(b), int(ch)): i for i, (p, b, ch) in enumerate(zip(c["packet_id"], c["board"], c["channel"]))}
    idx = np.array([key[(p, 101, int(ch))] for p, ch in zip(d.packet_id, d.channel)])
    Y = d[[f"d_{t}" for t in T]].to_numpy(float)
    logy = d[[f"log_{t}" for t in T]].to_numpy(float)
    ts = pd.to_datetime(d["date_start"])
    hours = ((ts - ts.min()).dt.total_seconds() / 3600).to_numpy()
    return dict(d=d, FA=lg(A), FB=lg(B), CA=np.nan_to_num(c["curves_ref"][idx]), CB=np.nan_to_num(c["curves_theta"][idx]),
                set=d[[col for col in d.columns if col.startswith("delta.")]].to_numpy(float), Y=Y, logy=logy,
                hours=hours, amp=d.channel.to_numpy(), packet=d.packet_id.to_numpy(), session=d.session.to_numpy())


def cross_pairs(D, rows, max_hours):
    """Pairs (i, j) of training rows of one amplifier within max_hours: own trace i -> own trace j."""
    out = []
    for a in np.unique(D["amp"][rows]):
        r = rows[D["amp"][rows] == a]
        out += [(i, j) for i, j in combinations(r, 2) if abs(D["hours"][j] - D["hours"][i]) <= max_hours]
    return np.array(out, int).reshape(-1, 2)


# --------------------------------------------------------------------------- shared encoder
class Siamese:
    """g: input -> 3 scores (one per target); change = g(B) - g(A).  Ensemble of small networks."""

    def __init__(self, loss="both", hidden=32, epochs=500, lr=3e-3, wd=3e-3, n_models=5, kappa=3.0):
        self.loss, self.hidden, self.epochs, self.lr, self.wd, self.n_models, self.kappa = loss, hidden, epochs, lr, wd, n_models, kappa

    def fit(self, XA, XB, Y):
        import torch
        allx = np.vstack([XA, XB])
        self.med = np.nanmedian(allx, 0)
        self.mu, self.sd = np.nanmean(allx, 0), np.nanstd(allx, 0) + 1e-9
        prep = lambda X: torch.tensor(np.clip((np.where(np.isnan(X), self.med, X) - self.mu) / self.sd, -8, 8), dtype=torch.float32)
        a, b = prep(XA), prep(XB)
        self.ys = np.sqrt(np.mean(Y ** 2, 0)) + 1e-9
        y = torch.tensor(Y / self.ys, dtype=torch.float32)
        noise = torch.tensor(NOISE / self.ys, dtype=torch.float32)
        w = 1 - torch.exp(-y ** 2 / (2 * noise ** 2))
        self.nets = []
        for seed in range(self.n_models):
            torch.manual_seed(seed)
            net = torch.nn.Sequential(torch.nn.Linear(a.shape[1], self.hidden), torch.nn.SiLU(),
                                      torch.nn.Linear(self.hidden, self.hidden), torch.nn.SiLU(), torch.nn.Linear(self.hidden, 3))
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
            for _ in range(self.epochs):
                opt.zero_grad()
                p = net(b) - net(a)
                loss = 0.0
                if self.loss in ("size", "both"):
                    loss = loss + torch.nn.functional.huber_loss(p, y, delta=1.0)
                if self.loss in ("direction", "both"):
                    loss = loss + (w * torch.nn.functional.softplus(-torch.sign(y) * self.kappa * p)).mean()
                loss.backward()
                opt.step()
            net.eval()
            self.nets.append(net)
        self.prep = prep
        return self

    def predict(self, XA, XB):
        import torch
        a, b = self.prep(XA), self.prep(XB)
        with torch.no_grad():
            return np.mean([(n(b) - n(a)).numpy() for n in self.nets], 0) * self.ys


# --------------------------------------------------------------------------- difference-curve features
class DiffPCA:
    """The difference curves (B - A, 16 x 256), standardized per point, projected on their first k principal
    components (fitted on the training differences)."""

    def __init__(self, k=10):
        self.k = k

    def fit(self, D):
        X = D.reshape(len(D), -1)
        self.sd = X.std(0) + 1e-6
        Z = X / self.sd
        _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        self.v = vt[: self.k]
        return self

    def transform(self, D):
        return (D.reshape(len(D), -1) / self.sd) @ self.v.T


def plain(kind, Xtr, Ytr, Xte):
    xtr, xte, _ = tl.prep_columns(Xtr, Xte)
    if xtr.shape[1] == 0:
        return np.zeros((len(Xte), 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "ridge0":
            from sklearn.linear_model import RidgeCV
            m = RidgeCV(alphas=np.logspace(-2, 3, 20), fit_intercept=False).fit(xtr / xtr.std(0), Ytr)
            return m.predict(xte / xtr.std(0))
        m = {"trees": models.GBT, "gp": models.GP}[kind]()
        return m.fit(xtr, Ytr).predict(xte)[0]


# --------------------------------------------------------------------------- evaluation
def score(Y, P):
    out = {}
    for k, t in enumerate(T):
        y, p = Y[:, k], P[:, k]
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        clear = np.abs(y) > CLEAR[k]
        out[t] = {"error": float(100 * np.sqrt(np.mean((p - y) ** 2))), "median": float(100 * np.median(np.abs(p - y))),
                  "direction": float(np.mean(np.sign(p[clear]) == np.sign(y[clear]))) if clear.any() else None,
                  "n_clear": int(clear.sum()),
                  "spearman": float(spearmanr(y, p)[0]) if np.std(p) > 0 else None}
    return out


def run_model(name, D, tr, te, cross_hours, pre):
    """Predicted within-packet changes of rows te from a model trained on rows tr (and their cross pairs)."""
    Y = D["Y"]
    pairs = cross_pairs(D, tr, cross_hours) if cross_hours else np.zeros((0, 2), int)
    if name == "no change":
        return np.zeros((len(te), 3))
    if name == "settings change: gp":
        return plain("gp", D["set"][tr], Y[tr], D["set"][te])
    if name == "feature difference: trees":
        return plain("trees", D["FB"][tr] - D["FA"][tr], Y[tr], D["FB"][te] - D["FA"][te])
    if name.startswith("siamese"):
        loss = name.split("(")[1].rstrip(")")
        if "curves" in name:
            EA, EB = pre.encode(D["CA"]), pre.encode(D["CB"])
        else:
            EA, EB = D["FA"], D["FB"]
        XA, XB, YY = EA[tr], EB[tr], Y[tr]
        if len(pairs):
            XA = np.vstack([XA, EB[pairs[:, 0]]])
            XB = np.vstack([XB, EB[pairs[:, 1]]])
            YY = np.vstack([YY, D["logy"][pairs[:, 1]] - D["logy"][pairs[:, 0]]])
        return Siamese(loss=loss).fit(XA, XB, YY).predict(EA[te], EB[te])
    if name.startswith("difference curves"):
        pca = DiffPCA().fit(D["CB"][tr] - D["CA"][tr])
        Ztr, Zte = pca.transform(D["CB"][tr] - D["CA"][tr]), pca.transform(D["CB"][te] - D["CA"][te])
        return plain(name.split(": ")[1], Ztr, Y[tr], Zte)
    raise ValueError(name)


def diff_curve_map(D, rows, out: Path):
    """Correlation of the difference curves (point by point) with the change of sigma_e and lambda."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [str(n) for n in real_inputs.load(ROOT / "processed" / "model_inputs" / "curves.npz")["curve_names"]]
    Dc = D["CB"][rows] - D["CA"][rows]
    summary = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, k in zip(axes, (0, 1)):
        y = D["Y"][rows, k]
        yz = (y - y.mean()) / y.std()
        X = (Dc - Dc.mean(0)) / (Dc.std(0) + 1e-9)
        R = np.einsum("n,nij->ij", yz, X) / len(y)
        im = ax.imshow(R, aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
        ax.set_title(f"change of {T[k]}: correlation with the difference curve", fontsize=11)
        ax.set_xlabel("point along the curve (0-255)")
        best = np.argsort(-np.max(np.abs(R), 1))[:5]
        summary[T[k]] = [{"curve": names[c], "max_abs_corr": float(np.max(np.abs(R[c]))),
                          "at_point": int(np.argmax(np.abs(R[c])))} for c in best]
    axes[0].set_yticks(range(len(names)), names, fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="correlation")
    fig.savefig(out / "difference_curve_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cross-hours", type=float, default=1.0, help="add cross-packet pairs within this many hours (0: none)")
    parser.add_argument("--out", default=str(ROOT / "processed" / "lvl1_results" / "2026-11-04_change_models"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    D = build()
    ab, c = np.flatnonzero(D["session"] != "C"), np.flatnonzero(D["session"] == "C")
    folds = [ab[f] for f in grouped_folds(D["packet"][ab], 5, seed=0)]
    from ml import deep
    p = real_inputs.load(ROOT / "processed" / "model_inputs" / "pretrain.npz")
    pre = deep.Pretrainer(epochs=40).fit(np.nan_to_num(p["curves"][p["board"] == 101]), None)
    results = {"_info": {"rows_AB": int(len(ab)), "rows_C": int(len(c)), "cross_hours": args.cross_hours},
               "_difference_curve_map": diff_curve_map(D, ab, out)}
    names = ["no change", "settings change: gp", "feature difference: trees",
             "siamese features (size)", "siamese features (direction)", "siamese features (both)",
             "siamese curves (size)", "siamese curves (direction)", "siamese curves (both)",
             "difference curves: trees", "difference curves: gp", "difference curves: ridge0"]
    for name in names:
        pl = np.full(D["Y"].shape, np.nan)
        for a in np.unique(D["amp"][ab]):
            for te in folds:
                tst = te[D["amp"][te] == a]
                trn = np.setdiff1d(ab[D["amp"][ab] != a], te)
                if len(tst) and len(trn):
                    pl[tst] = run_model(name, D, trn, tst, args.cross_hours, pre)
        pc = run_model(name, D, ab, c, args.cross_hours, pre)
        results[name] = {"new_amp": score(D["Y"][ab], pl[ab]), "c": score(D["Y"][c], pc)}
        r = results[name]["new_amp"]
        print(f"{name:30s} " + "  ".join(f"{t}: {r[t]['error']:5.1f} (med {r[t]['median']:4.1f}, dir "
                                        f"{r[t]['direction'] if r[t]['direction'] is not None else float('nan'):.2f})" for t in T), flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=1))
    print("difference-curve map:", json.dumps(results["_difference_curve_map"], indent=1))


if __name__ == "__main__":
    main()

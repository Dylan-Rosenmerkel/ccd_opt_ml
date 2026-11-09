"""
scorecard.py -- one scorecard for the level-1 models, in four cells (board 101 without ch2, all 43 packets):

                         absolute (one trace -> value)     relative (two traces -> change)
    trained amplifiers   (1)                               (3)
    new amplifier        (2)                               (4)

Folds: the packets in time order, cut into 5 contiguous blocks; a block's packets are never in training.  Trained
amplifiers: every amplifier in training, the test is the block.  New amplifier: one amplifier left out, crossed with the
blocks (trained on the OTHER amplifiers of the OTHER blocks).

Absolute: the train_lvl1.py models.  Trace models see the traces only, never the settings; their feature sets are fixed
in advance (train_lvl1.MODEL_TIERS, P+A+B otherwise), so nothing is chosen on the test.  Against: the mean, the
amplifier's mean (trained cells), settings-only GP / trees, and settings + the amplifier's identity (trained cells).

Relative: pairs of traces of one amplifier, by type: within a packet (reference trace at nominal -> own trace; the
nominal label interpolated in time between the nominal packets), neighbouring packets (n -> n+1) and n -> n+2 (both labels
measured).  A test pair has both packets in the test block; training pairs (all types pooled, both orders) have neither.
Models: no change; settings change (GP, trees; + the amplifier's identity in the trained cells); trace change (feature
differences: trees, GP; the shared-encoder network of change_models.py; hand-picked difference-curve bands + GP,
change_handpicked.py); and every absolute model's two predictions differenced (packet pairs only).

Ranking (5): the settings ranked from the traces under the joint objective w dlog sigma_e + (1-w) dlog lambda for
every --weights w, with two shortlist rules (lvl1/shortlist.py): is the truly best setting on a 5- or 10-setting
shortlist, and how much worse is the best shortlisted one (regret)?  The screening step of level 2 (session F).

Training pairs are used in both orders, so no model can learn "a change away from nominal goes up" (96 % of the clear
within-packet sigma_e changes do: nominal is near the best setting); the report gives that base rate per pair type.
Neighbouring-packet pairs are balanced (about half go up) and are the fair test of direction.

Scores: absolute -- typical error (RMS of the log error, %); relative -- direction right on clear changes
(change_test.CLEAR), size error (RMS, %), Spearman.  Each model is compared with its cell's settings-only reference
(settings GP; settings + amplifier GP in the trained cells) on the same rows; a packet bootstrap gives the 95 % interval
of the difference ("better" / "worse" when it excludes 0).

    .venv/bin/python lvl1/scorecard.py [--out processed/lvl1_results/2026-11-04_scorecard] [--quick]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "processed"   # processed data folder; set with --data
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

import change_handpicked as hp  # noqa: E402
import change_models as cm  # noqa: E402
import change_test as ct  # noqa: E402
import shortlist as sl  # noqa: E402
import train_lvl1 as tl  # noqa: E402
from ml import models, real_inputs  # noqa: E402

T = list(tl.TARGETS)
CLEAR = np.array([ct.CLEAR[t] for t in T])
CELLS = ("trained", "new")
CELL_NAME = {"trained": "trained amplifiers", "new": "new amplifier"}
PAIR_TYPES = ("within packet", "n -> n+1", "n -> n+2")
N_BLOCKS = 5
#: the settings-only reference of each cell (fixed in advance), absolute and relative
REF = {"trained": "settings + amplifier: gp", "new": "settings: gp"}
REL_REF = {"trained": "settings + amplifier: gp", "new": "settings change: gp"}
#: absolute models: name -> (train_lvl1 method, feature set, sees the settings, cells)
ABS_MODELS = {
    "mean": ("mean", "none", False, CELLS),
    "amplifier mean": ("amp_mean", "none", False, ("trained",)),
    "settings: gp": ("settings", "none", True, CELLS),
    "settings: trees": ("settings_trees", "none", True, CELLS),
    "settings + amplifier: gp": ("settings_amp", "none", True, ("trained",)),
    "settings + amplifier: trees": ("settings_amp_trees", "none", True, ("trained",)),
    "trace: physics": ("physics_white", tl.DEFAULT_MODEL_TIERS, False, CELLS),
    "trace: gp": ("gp", tl.MODEL_TIERS["gp"], False, CELLS),
    "trace: trees": ("trees", tl.MODEL_TIERS["trees"], False, CELLS),
    "trace: nn": ("nn", tl.MODEL_TIERS["nn"], False, CELLS),
    "trace: cnn": ("cnn", tl.DEFAULT_MODEL_TIERS, False, CELLS),
    "trace: pretrained cnn": ("pretrained", tl.DEFAULT_MODEL_TIERS, False, CELLS),
    "trace: deep kernel": ("deep_kernel", tl.DEFAULT_MODEL_TIERS, False, CELLS),
}
REL_MODELS = ("no change", "settings change: gp", "settings change: trees", "settings + amplifier: gp",
              "settings + amplifier: trees", "trace change: trees", "trace change: gp", "trace change: shared-encoder net",
              "trace change: hand-picked bands + gp")
NOT_DIFFERENCED = ("mean", "amplifier mean")                 # their differenced prediction is 0 = "no change"


class Runner(tl.Runner):
    """train_lvl1's models plus two baselines."""

    def amp_mean(self, train, test):
        """Baseline: the training mean of the test row's amplifier."""
        m = np.array([self.Y[train][self.amp_id[train] == a].mean(0) for a in self.amp_id[test]])
        return m, np.tile(self.Y[train].std(0, ddof=1), (len(test), 1))

    def settings_trees(self, train, test):
        """Baseline: gradient-boosted trees on the settings alone (no trace, no amplifier)."""
        th = self.f("features")["theta"]
        xtr, xte, _ = tl.prep_columns(th[train], th[test])
        return models.GBT(max_iter=100 if self.quick else 300).fit(xtr, self.Y[train]).predict(xte)


# --------------------------------------------------------------------------- data
def load(channels=(0, 1, 3)):
    files = {p.stem: real_inputs.load(p) for p in sorted((DATA / "model_inputs").glob("*.npz"))}
    b = files["physics"]
    ok = ((b["board"] == 101) & np.isin(b["channel"], channels) & b["label_ok"] & ~b["fake"] & b["trace_ok"].astype(bool)
          & np.isfinite(b["Y"]).all(1))
    rows = np.flatnonzero(ok)
    d = pd.read_csv(DATA / "dataset.csv", low_memory=False).iloc[rows].reset_index(drop=True).copy()
    assert (d.packet_id.to_numpy() == b["packet_id"][rows]).all() and (d.channel.to_numpy() == b["channel"][rows]).all()
    ts = pd.to_datetime(d["date_start"])
    d["hours"] = (ts - ts.min()).dt.total_seconds() / 3600
    first = d.groupby("packet_id").hours.min().sort_values()
    block = {p: min(N_BLOCKS - 1, r * N_BLOCKS // len(first)) for r, p in enumerate(first.index)}
    d["block"] = d.packet_id.map(block)
    return files, rows, d


def trace_table(files, rows, d):
    """Every trace twice per row: own (index r) and reference (index n + r).  Log trace features (P+A tiers), log white
    noise, settings, curves."""
    n = len(d)
    tiers = tl.load_tiers(ct.TIERS)
    feats = [f for f, ts in tiers.items() if ts & {"P", "A1", "A2"} and f"trace.{f}" in d and f"ref.{f}" in d]
    own, ref = d[[f"trace.{f}" for f in feats]].to_numpy(float), d[[f"ref.{f}" for f in feats]].to_numpy(float)
    pos = (np.nanmin(own, 0) > 0) & (np.nanmin(ref, 0) > 0)
    lg = lambda X: np.where(pos, np.log(np.where(pos, X, 1)), X)
    th_names = [c for c in d.columns if c.startswith("theta.")]
    th_own = d[th_names].to_numpy(float)
    th_ref = th_own - d[["delta." + c[len("theta."):] for c in th_names]].to_numpy(float)
    c = files["curves"]
    white = np.log(np.concatenate([d[f"trace.{hp.WHITE}"], d[f"ref.{hp.WHITE}"]]).astype(float))
    return dict(F=np.vstack([lg(own), lg(ref)]), TH=np.vstack([th_own, th_ref]), white=white,
                C=np.nan_to_num(np.concatenate([c["curves_theta"][rows], c["curves_ref"][rows]])), n=n)


def make_pairs(d, Y):
    """Pairs (trace A, trace B) with their target log y(B) - log y(A), type, and the two rows (packets) they come from."""
    n = len(d)
    nom = np.full(Y.shape, np.nan)                       # the nominal label interpolated in time, per amplifier
    for ch, g in d.groupby("channel"):
        r = g.index.to_numpy()
        m = r[g.packet_id.isin(ct.NOMINAL).to_numpy()]
        m = m[np.argsort(d.hours[m].to_numpy())]
        for k in range(Y.shape[1]):
            nom[r, k] = np.interp(d.hours[r], d.hours[m], Y[m, k])
    A, B, typ, ra, rb = [], [], [], [], []
    for r in range(n):
        if d.packet_id[r] not in ct.NOMINAL:
            A.append(n + r), B.append(r), typ.append(0), ra.append(r), rb.append(r)
    for ch, g in d.groupby("channel"):
        r = g.index.to_numpy()[np.argsort(g.hours.to_numpy())]
        for step in (1, 2):
            for i, j in zip(r[:-step], r[step:]):
                A.append(i), B.append(j), typ.append(step), ra.append(i), rb.append(j)
    A, B, typ, ra, rb = map(np.array, (A, B, typ, ra, rb))
    Yfull = np.vstack([Y, nom])                          # the label of trace n + r is the interpolated nominal
    return dict(A=A, B=B, type=typ, ra=ra, rb=rb, Y=Yfull[B] - Yfull[A])


# --------------------------------------------------------------------------- absolute
def run_absolute(files, rows, d, quick, only=None):
    runner = Runner(files, rows, quick, tl.load_tiers(ct.TIERS))
    amp, blk = d.channel.to_numpy(), d.block.to_numpy()
    splits = {"trained": [(np.flatnonzero(blk != k), np.flatnonzero(blk == k)) for k in range(N_BLOCKS)],
              "new": [(np.flatnonzero((blk != k) & (amp != a)), np.flatnonzero((blk == k) & (amp == a)))
                      for a in np.unique(amp) for k in range(N_BLOCKS)]}
    preds = {cell: {} for cell in CELLS}
    # settings models first: the pre-trained encoder is built on the first trace model's call, which must not see theta
    for name, (method, spec, sees_settings, cells) in sorted(ABS_MODELS.items(), key=lambda kv: not kv[1][2]):
        if only and name not in only:
            continue
        runner.no_settings = not sees_settings
        for cell in cells:
            t0 = time.time()
            preds[cell][name] = tl.run_model(runner, method, splits[cell], spec)[0]
            print(f"absolute {cell:8s} {name:28s} " + "  ".join(
                f"{t} {100 * rms(runner.Y[:, i] - preds[cell][name][:, i]):5.1f}" for i, t in enumerate(T))
                + f"   {time.time() - t0:4.0f} s", flush=True)
    return runner.Y, runner.Y_err, preds


# --------------------------------------------------------------------------- relative
def fit_predict(kind, X, y, Xte, groups=None):
    xtr, xte, _ = tl.prep_columns(X, Xte)
    if xtr.shape[1] == 0:
        return np.zeros((len(Xte), y.shape[1]))
    m = {"trees": lambda: models.GBT(), "gp": lambda: models.GP(max_points=800)}[kind]()
    return m.fit(xtr, y).predict(xte)[0]


def rel_predict(name, TR, P, tr, te, onehot):
    """One relative model: train on pairs tr (both orders where the model is not antisymmetric), predict pairs te."""
    A, B, Y = P["A"], P["B"], P["Y"]
    if name == "no change":
        return np.zeros((len(te), 3))
    if name.startswith("settings"):
        X = TR["TH"][B] - TR["TH"][A]
        if "amplifier" in name:
            X2 = np.hstack([X, onehot]), np.hstack([-X, onehot])
        else:
            X2 = X, -X
        Xtr = np.vstack([X2[0][tr], X2[1][tr]])
        return fit_predict(name.split(": ")[1], Xtr, np.vstack([Y[tr], -Y[tr]]), X2[0][te])
    if name in ("trace change: trees", "trace change: gp"):
        X = TR["F"][B] - TR["F"][A]
        return fit_predict(name.split(": ")[1], np.vstack([X[tr], -X[tr]]), np.vstack([Y[tr], -Y[tr]]), X[te])
    if name == "trace change: shared-encoder net":
        net = cm.Siamese(loss="both").fit(TR["F"][A[tr]], TR["F"][B[tr]], Y[tr])
        return net.predict(TR["F"][A[te]], TR["F"][B[te]])
    if name == "trace change: hand-picked bands + gp":
        Dc = TR["C"][B] - TR["C"][A]
        w = TR["white"][B] - TR["white"][A]
        out = np.zeros((len(te), 3))
        for i in range(3):
            h = hp.Handpicked(6).fit(Dc[tr], Y[tr, i])
            Xtr, Xte = np.column_stack([h.transform(Dc[tr]), w[tr]]), np.column_stack([h.transform(Dc[te]), w[te]])
            out[:, i] = fit_predict("gp", np.vstack([Xtr, -Xtr]), np.concatenate([Y[tr, i], -Y[tr, i]])[:, None], Xte)[:, 0]
        return out
    raise ValueError(name)


def run_relative(TR, P, d, abs_preds, only=None):
    amp, blk = d.channel.to_numpy(), d.block.to_numpy()
    ba, bb, pamp = blk[P["ra"]], blk[P["rb"]], amp[P["ra"]]
    onehot = np.eye(len(np.unique(amp)))[np.unique(amp, return_inverse=True)[1][P["ra"]]]
    preds = {cell: {} for cell in CELLS}
    for cell in CELLS:
        splits = []
        for k in range(N_BLOCKS):
            for a in (np.unique(amp) if cell == "new" else [None]):
                te = np.flatnonzero((ba == k) & (bb == k) & ((pamp == a) if a is not None else True))
                tr = np.flatnonzero((ba != k) & (bb != k) & ((pamp != a) if a is not None else True))
                if len(te) and len(tr):
                    splits.append((tr, te))
        for name in REL_MODELS:
            if only and name not in only:
                continue
            if "amplifier" in name and cell == "new":
                continue
            t0 = time.time()
            pr = np.full(P["Y"].shape, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for tr, te in splits:
                    pr[te] = rel_predict(name, TR, P, tr, te, onehot)
            preds[cell][name] = pr
            print(f"relative {cell:8s} {name:36s} {time.time() - t0:4.0f} s", flush=True)
        # the absolute models, differenced: both packets were predicted by the same fit (same block, same amplifier)
        packet_pair = P["type"] > 0
        for name, m in abs_preds[cell].items():
            if name in NOT_DIFFERENCED:
                continue
            pr = np.full(P["Y"].shape, np.nan)
            pr[packet_pair] = m[P["rb"][packet_pair]] - m[P["ra"][packet_pair]]
            preds[cell][f"absolute {name}, differenced"] = pr
    return preds


# --------------------------------------------------------------------------- scores
def rms(e, w=None):
    ok = np.isfinite(e)
    w = np.ones(len(e)) if w is None else w
    return float(np.sqrt(np.sum(w[ok] * e[ok] ** 2) / max(np.sum(w[ok]), 1e-12)))


def direction(y, p, clear, w=None):
    c = (np.abs(y) > clear) & np.isfinite(p)
    w = np.ones(len(y)) if w is None else w
    s = np.sum(w[c])
    right = (np.sign(p[c]) == np.sign(y[c])) + 0.5 * (p[c] == 0)      # a prediction of exactly 0 is a coin flip
    return float(np.sum(w[c] * right) / s) if s > 0 else np.nan


def boot_weights(packets_a, packets_b, n_boot, seed):
    """Packet bootstrap: each packet drawn a Poisson(1) number of times; a pair's weight is the product over its two
    packets (one factor when both traces come from the same packet)."""
    ups = np.unique(np.concatenate([packets_a, packets_b]))
    ia, ib = np.searchsorted(ups, packets_a), np.searchsorted(ups, packets_b)
    cnt = np.random.default_rng(seed).poisson(1.0, (n_boot, len(ups))).astype(float)
    return np.where(ia == ib, cnt[:, ia], cnt[:, ia] * cnt[:, ib])


def interval(v):
    v = np.asarray(v)[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]


def verdict(iv, lower_is_better=True):
    if not np.all(np.isfinite(iv)):
        return "-"
    if iv[1] < 0:
        return "better" if lower_is_better else "worse"
    if iv[0] > 0:
        return "worse" if lower_is_better else "better"
    return "same"


def score_absolute(Y, preds, packets, n_boot):
    W = boot_weights(packets, packets, n_boot, seed=0)
    out = {}
    for cell in CELLS:
        ref = preds[cell][REF[cell]]
        out[cell] = {}
        for name, p in preds[cell].items():
            r = {}
            for i, t in enumerate(T):
                e, er = p[:, i] - Y[:, i], ref[:, i] - Y[:, i]
                diff = [rms(e, w) - rms(er, w) for w in W]
                iv = interval(diff)
                r[t] = {"error": 100 * rms(e), "median": float(100 * np.nanmedian(np.abs(e))),
                        "vs_settings_pp": 100 * (rms(e) - rms(er)), "vs_settings_95": [100 * x for x in iv],
                        "vs_settings": verdict(iv) if name != REF[cell] else "reference"}
            out[cell][name] = r
    return out


def score_relative(P, d, preds, n_boot):
    pk = d.packet_id.to_numpy()
    W = boot_weights(pk[P["ra"]], pk[P["rb"]], n_boot, seed=1)
    out = {}
    for cell in CELLS:
        ref = preds[cell][REL_REF[cell]]
        out[cell] = {}
        for name, p in preds[cell].items():
            out[cell][name] = {}
            for ty, tname in enumerate(PAIR_TYPES):
                sel = (P["type"] == ty) & np.isfinite(p[:, 0]) & np.isfinite(ref[:, 0])
                if not sel.any():
                    continue
                r = {}
                for i, t in enumerate(T):
                    y, q, qr = P["Y"][sel, i], p[sel, i], ref[sel, i]
                    Ws = W[:, sel]
                    dd = [direction(y, q, CLEAR[i], w) - direction(y, qr, CLEAR[i], w) for w in Ws]
                    de = [rms(q - y, w) - rms(qr - y, w) for w in Ws]
                    ivd, ive = interval(dd), interval(de)
                    r[t] = {"n": int(sel.sum()), "n_clear": int((np.abs(y) > CLEAR[i]).sum()),
                            "clear_up": float(np.mean(y[np.abs(y) > CLEAR[i]] > 0)),
                            "direction": direction(y, q, CLEAR[i]), "size_error": 100 * rms(q - y),
                            "spearman": float(spearmanr(y, q)[0]) if np.std(q) > 0 else np.nan,
                            "true_change_rms": 100 * rms(y),
                            "direction_vs_settings": verdict(ivd, lower_is_better=False) if name != REL_REF[cell] else "reference",
                            "direction_vs_settings_95": [100 * x for x in ivd],
                            "size_vs_settings": verdict(ive) if name != REL_REF[cell] else "reference",
                            "size_vs_settings_95": [100 * x for x in ive]}
                out[cell][name][tname] = r
    return out


# --------------------------------------------------------------------------- ranking
RANK_K = (5, 10)


def rank_metrics(true, pred_s, pred_l, w, rule, rng, n_draws=1):
    """Shortlist the k settings predicted best under J(w) with a shortlist rule (lvl1/shortlist.py); ties broken at
    random (averaged over n_draws).  The truth is J(w) of the measured changes.  hits: how many of the k truly best are
    on the shortlist; regret: how much worse (%, J in log units x 100) the best shortlisted setting truly is than the
    best of all; caught: the single best setting is on the shortlist."""
    pred = sl.objective(pred_s, pred_l, w)
    out = {"spearman": float(spearmanr(true, pred)[0]) if np.std(pred) > 0 else 0.0}
    for k in RANK_K:
        h, r, c = [], [], []
        for _ in range(n_draws):
            top = sl.order(pred_s, pred_l, w, rule, rng)[:k]
            h.append(len(np.intersect1d(top, np.argsort(true)[:k])))
            r.append(100 * (true[top].min() - true.min()))
            c.append(float(np.argmin(true) in top))
        out[f"hits@{k}"], out[f"regret@{k}"], out[f"caught@{k}"] = float(np.mean(h)), float(np.mean(r)), float(np.mean(c))
    return out


def score_ranking(P, d, rel_preds, abs_preds, weights):
    """Rank the settings (the non-nominal packets) from the traces and check the shortlist, for every sigma_e weight w
    and both shortlist rules.  The truth is each setting's effect: log y(packet) - log y(nominal, interpolated in
    time), i.e. the within-packet change, so the night's drift does not decide the ranking.  Relative models rank by
    their predicted within-packet change, absolute models by their predicted value; "random" averages 2000 random
    orders.  Two levels: "board" -- a setting applies to every amplifier of the board, so its objective is the mean
    over the amplifiers (predicted and true; the funnel's choice); "amplifier" -- each amplifier ranked on its own,
    then averaged."""
    sel = np.flatnonzero(P["type"] == 0)
    amp = d.channel.to_numpy()[P["ra"][sel]]
    pk = d.packet_id.to_numpy()[P["ra"][sel]]
    amps = np.unique(amp)
    common = sorted(set.intersection(*[set(pk[amp == a]) for a in amps]))
    rng = np.random.default_rng(0)
    Ytrue = P["Y"][sel]
    out = {}
    for cell in CELLS:
        sources = {"random": None}
        sources.update({name: p[sel] for name, p in rel_preds[cell].items() if not name.startswith("absolute")})
        sources.update({f"absolute {name}": p[P["rb"][sel]] for name, p in abs_preds[cell].items() if name not in NOT_DIFFERENCED})
        # hybrid: sigma_e from the absolute trace trees, lambda from the trace-change trees (the absolute level is a
        # per-amplifier constant, which does not change a ranking within one amplifier)
        if "absolute trace: trees" in sources and "trace change: trees" in sources:
            hyb = sources["absolute trace: trees"].copy()
            hyb[:, 1] = sources["trace change: trees"][:, 1]
            sources["hybrid: sigma_e absolute trees, lambda change trees"] = hyb
        out[cell] = {}
        for name, p in sources.items():
            out[cell][name] = {}
            for w in weights:
                for rule in sl.RULES:
                    key = f"{sl.weight_label(w)} {rule}"
                    groups = {"board": [np.array([np.flatnonzero((pk == q) & (amp == a))[0] for q in common]) for a in amps]}
                    res = {}
                    # board: mean over the amplifiers of each setting's changes
                    idx = groups["board"]
                    true = np.mean([sl.objective(Ytrue[i, 0], Ytrue[i, 1], w) for i in idx], 0)
                    if p is None:
                        z = np.zeros(len(common))
                        res["board"] = rank_metrics(true, z, z, w, rule, rng, n_draws=2000)
                    else:
                        ps, pl = np.mean([p[i, 0] for i in idx], 0), np.mean([p[i, 1] for i in idx], 0)
                        res["board"] = rank_metrics(true, ps, pl, w, rule, rng, n_draws=50)
                    per_amp = {}
                    for a in amps:
                        m = np.flatnonzero(amp == a)
                        t = sl.objective(Ytrue[m, 0], Ytrue[m, 1], w)
                        if p is None:
                            z = np.zeros(len(m))
                            per_amp[str(a)] = rank_metrics(t, z, z, w, rule, rng, n_draws=2000)
                        else:
                            per_amp[str(a)] = rank_metrics(t, p[m, 0], p[m, 1], w, rule, rng, n_draws=50)
                    res["amplifier"] = {k: float(np.mean([v[k] for v in per_amp.values()])) for k in next(iter(per_amp.values()))}
                    res["per amplifier"] = per_amp
                    res["n_settings"] = len(common)
                    out[cell][name][key] = res
    return out


def ranking_report(rank_sc, weights) -> list[str]:
    L = ["## (5) Ranking: shortlist the settings from the traces, joint objective J(w) = w dlog sigma_e + (1-w) dlog lambda",
         "",
         "Each setting (non-nominal packet) applies to the whole board, so its objective is the mean over the 3 amplifiers "
         "(predicted and true; truth = the setting's effect relative to nominal, drift removed). Weight w:(1-w) of "
         "sigma_e:lambda. Shortlist rules (lvl1/shortlist.py): weighted = by J(w); pareto = the predicted Pareto front "
         "first. Cells: top-5 regret (%) = how much worse the best of the 5 shortlisted settings truly is than the truly "
         "best setting / whether the truly best is on the 10-setting shortlist (1 = yes). The 3 amplifiers share their "
         "settings, so each column is ONE trial: read the pattern, not single numbers.", ""]
    keys = [f"{sl.weight_label(w)} {rule}" for rule in sl.RULES for w in weights]
    for cell in CELLS:
        n = next(iter(rank_sc[cell].values()))[keys[0]]["n_settings"]
        L += [f"### {CELL_NAME[cell]} ({n} settings)", "",
              "| model | " + " | ".join(f"{k}" for k in keys) + " |", "|---|" + "---|" * len(keys)]
        for name, r in rank_sc[cell].items():
            L.append(f"| {name} | " + " | ".join(f"{r[k]['board']['regret@5']:.1f} / {r[k]['board']['caught@10']:.0f}"
                                                 if name != "random" else
                                                 f"{r[k]['board']['regret@5']:.1f} / {r[k]['board']['caught@10']:.2f}" for k in keys) + " |")
        L.append("")
    return L


# --------------------------------------------------------------------------- report
def report(abs_sc, rel_sc, info) -> str:
    L = ["# Level-1 scorecard", "",
         f"Board 101 ch{'/'.join(str(c) for c in info.get('channels', [0, 1, 3]))}, {info['n_rows']} rows from {info['n_packets']} packets, {N_BLOCKS} contiguous time blocks. "
         "Trace models never see the settings. vs settings: the cell's settings-only reference "
         f"(absolute: `{REF['trained']}` / `{REF['new']}`; relative: `{REL_REF['trained']}` / `{REL_REF['new']}`), packet bootstrap 95 %: better / same / worse.", "",
         f"Label noise (floor): absolute {', '.join(f'{t} {v:.1f} %' for t, v in info['label_noise_abs'].items())}; "
         f"of a change {', '.join(f'{t} {100 * v:.1f} %' for t, v in zip(T, cm.NOISE))}.", ""]
    for k, cell in enumerate(CELLS):
        L += [f"## ({k + 1}) Absolute, {CELL_NAME[cell]}: typical error (%)", "",
              "| model | " + " | ".join(T) + " |", "|---|" + "---|" * len(T)]
        for name, r in abs_sc[cell].items():
            L.append(f"| {name} | " + " | ".join(f"{r[t]['error']:.1f} ({r[t]['vs_settings']})" for t in T) + " |")
        L.append("")
    for k, cell in enumerate(CELLS):
        for tname in PAIR_TYPES:
            L += [f"## ({k + 3}) Relative, {CELL_NAME[cell]}, {tname}: direction right on clear changes / size error (%)", ""]
            any_row = next((r[tname] for r in rel_sc[cell].values() if tname in r), None)
            if any_row is None:
                continue
            L += ["clear changes: " + ", ".join(f"{t} {any_row[t]['n_clear']} of {any_row[t]['n']}" for t in T)
                  + "; true change RMS: " + ", ".join(f"{t} {any_row[t]['true_change_rms']:.1f} %" for t in T)
                  + "; clear changes that go UP (= the score of always saying up): "
                  + ", ".join(f"{t} {100 * any_row[t]['clear_up']:.0f} %" for t in T), "",
                  "| model | " + " | ".join(f"{t} direction | {t} size" for t in T) + " |", "|---|" + "---|---|" * len(T)]
            for name, r in rel_sc[cell].items():
                if tname not in r:
                    continue
                L.append(f"| {name} | " + " | ".join(
                    f"{100 * r[tname][t]['direction']:.0f} % ({r[tname][t]['direction_vs_settings']}) | "
                    f"{r[tname][t]['size_error']:.1f} ({r[tname][t]['size_vs_settings']})" for t in T) + " |")
            L.append("")
    return "\n".join(L)


def group_colour(name):
    if name.startswith(("trace", "absolute trace")):
        return "#2a6fb0"
    if "settings" in name:
        return "#d9822b"
    return "#8c8c8c"


def plot(abs_sc, rel_sc, info, out: Path):
    """Two figures: absolute error per model (filled = trained amplifiers, open = new amplifier), and the relative
    direction score per model and pair type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(abs_sc["trained"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.32 * len(names) + 1.6), sharey=True)
    for ax, t in zip(axes, T):
        for k, name in enumerate(names):
            c = group_colour(name)
            for cell, face in (("trained", c), ("new", "white")):
                if name in abs_sc[cell]:
                    ax.plot(abs_sc[cell][name][t]["error"], k, "o", ms=7, mfc=face, mec=c, mew=1.6)
        ax.axvline(info["label_noise_abs"][t], color="k", ls=":", lw=1)
        ax.set_title(f"{t}: typical error (%)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].invert_yaxis()
    axes[0].plot([], [], "o", mfc="k", mec="k", label="trained amplifiers")
    axes[0].plot([], [], "o", mfc="white", mec="k", label="new amplifier")
    axes[0].plot([], [], "k:", label="label noise")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "scorecard_absolute.png", dpi=150)
    plt.close(fig)
    names = [n for n in rel_sc["trained"] if not n.startswith("absolute")] + \
            [n for n in rel_sc["trained"] if n.startswith("absolute") and "trace" in n or n == "absolute settings: gp, differenced"]
    types = PAIR_TYPES[:2]
    fig, axes = plt.subplots(len(types), 3, figsize=(13, 2 * (0.3 * len(names) + 1)), sharey=True, sharex=True)
    for row, ty in enumerate(types):
        for ax, t in zip(axes[row], T):
            for k, name in enumerate(names):
                c = group_colour(name)
                for cell, face in (("trained", c), ("new", "white")):
                    r = rel_sc[cell].get(name, {}).get(ty)
                    if r and np.isfinite(r[t]["direction"]):
                        ax.plot(100 * r[t]["direction"], k, "o", ms=7, mfc=face, mec=c, mew=1.6)
            ax.axvline(50, color="k", ls=":", lw=1)
            any_r = next(r[ty] for r in rel_sc["trained"].values() if ty in r)
            ax.set_title(f"{ty}, {t}: direction right (%)\n{any_r[t]['n_clear']} clear changes", fontsize=9)
            ax.grid(axis="x", alpha=0.3)
            ax.set_xlim(20, 100)
        axes[row][0].set_yticks(range(len(names)), names, fontsize=8)
    axes[0][0].invert_yaxis()
    axes[0][0].plot([], [], "o", mfc="k", mec="k", label="trained amplifiers")
    axes[0][0].plot([], [], "o", mfc="white", mec="k", label="new amplifier")
    axes[0][0].plot([], [], "k:", label="coin flip")
    axes[0][0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "scorecard_relative.png", dpi=150)
    plt.close(fig)


def main():
    global DATA
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=str(DATA), help="processed data folder (dataset.csv, model_inputs/); default %(default)s")
    parser.add_argument("--out", help="output folder (default <data>/lvl1_results/2026-11-04_scorecard)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 1, 3], help="board-101 amplifiers (default 0 1 3: no ch2)")
    parser.add_argument("--abs-models", nargs="+", help="only these absolute models (names as in ABS_MODELS; keep the references)")
    parser.add_argument("--rel-models", nargs="+", help="only these relative models (names as in REL_MODELS; keep the references)")
    parser.add_argument("--rescore", action="store_true", help="score the predictions saved in --out (no training)")
    parser.add_argument("--weights", type=float, nargs="+", default=list(sl.DEFAULT_WEIGHTS),
                        help="sigma_e weights w of the ranking objective w dlog sigma_e + (1-w) dlog lambda (default 1 .75 .5 .25 0)")
    args = parser.parse_args()
    cm.set_data(args.data)
    DATA = cm.DATA
    out = Path(args.out) if args.out else DATA / "lvl1_results" / "2026-11-04_scorecard"
    out.mkdir(parents=True, exist_ok=True)
    files, rows, d = load(tuple(args.channels))
    cache = out / "predictions.pkl"
    if args.rescore:
        Y, Y_err, abs_preds, P, rel_preds = pickle.loads(cache.read_bytes())
    else:
        Y, Y_err, abs_preds = run_absolute(files, rows, d, args.quick, args.abs_models)
        TR = trace_table(files, rows, d)
        P = make_pairs(d, Y)
        rel_preds = run_relative(TR, P, d, abs_preds, args.rel_models)
        cache.write_bytes(pickle.dumps((Y, Y_err, abs_preds, P, rel_preds)))
    abs_sc = score_absolute(Y, abs_preds, d.packet_id.to_numpy(), args.n_boot)
    rel_sc = score_relative(P, d, rel_preds, args.n_boot)
    rank_sc = score_ranking(P, d, rel_preds, abs_preds, args.weights)
    info = {"n_rows": len(d), "n_packets": int(d.packet_id.nunique()), "channels": args.channels,
            "blocks": d.groupby("block").packet_id.unique().apply(lambda v: [str(v.min()), str(v.max())]).to_dict(),
            "label_noise_abs": {t: float(100 * np.sqrt(np.mean(Y_err[:, i] ** 2))) for i, t in enumerate(T)},
            "n_pairs": {t: int((P["type"] == k).sum()) for k, t in enumerate(PAIR_TYPES)}, "ref": REF}
    (out / "scorecard.json").write_text(json.dumps({"info": info, "absolute": abs_sc, "relative": rel_sc, "ranking": rank_sc},
                                                       indent=1, default=float))
    (out / "scorecard.md").write_text(report(abs_sc, rel_sc, info) + "\n" + "\n".join(ranking_report(rank_sc, args.weights)))
    plot(abs_sc, rel_sc, info, out)
    cols = {"packet_id": d.packet_id, "channel": d.channel, "block": d.block}
    for cell in CELLS:
        for name, p in abs_preds[cell].items():
            for i, t in enumerate(T):
                cols[f"{cell} | {name} | log {t}"] = p[:, i]
    pd.DataFrame(cols).to_csv(out / "absolute_predictions.csv", index=False)
    pcols = {"type": [PAIR_TYPES[k] for k in P["type"]], "packet A": d.packet_id.to_numpy()[P["ra"]],
             "packet B": d.packet_id.to_numpy()[P["rb"]], "channel": d.channel.to_numpy()[P["ra"]]}
    for i, t in enumerate(T):
        pcols[f"true change {t}"] = P["Y"][:, i]
    for cell in CELLS:
        for name, p in rel_preds[cell].items():
            for i, t in enumerate(T):
                pcols[f"{cell} | {name} | {t}"] = p[:, i]
    pd.DataFrame(pcols).to_csv(out / "relative_predictions.csv", index=False)
    print(f"wrote {out / 'scorecard.md'}")


if __name__ == "__main__":
    main()

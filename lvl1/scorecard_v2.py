"""
scorecard_v2.py -- the four-way scorecard (lvl1/scorecard.py) for the v2 data, with everything the v2 training plan
adds to task B (docs/overleaf_v2_training_plan_2026-11-04, "Task B", "The four-way scorecard", "Session F"):

* nominal packets, sessions and retakes from the v2 plan (L2_02b counts as L2_02); board 101 ch0/1/3 (ch2 left out:
  its image fit chi2 is ~25 against ~5); rows with fractional (N / N+1 sample) CDS windows left out by default, their
  image labels are biased (src/decode_trace.fractional_window);
* change labels from three sources, each its own pair type:
    - "within packet (image)": new trace vs reference trace; label = the packet's image against the reference's label:
      nominal interpolated in time between the nominal packets, or -- for the 21 packets whose reference is the
      previous packet's settings -- the latest earlier packet imaged at those settings;
    - "within packet (strip)": the same two traces; label = new strip vs reference strip (minutes apart, no drift, no
      nominal image needed): gain, lambda (CIC + serial register) and sigma_e of the strips;
    - "n -> n+1", "n -> n+2": neighbouring packets of one amplifier, both labels measured;
    - "random pairs": any two packets of one amplifier, labels drift-corrected (each packet's setting effect = log y -
      nominal interpolated in time), the funnel's training pairs;
* image-label models are trained on the within-packet (image), n -> n+1, n -> n+2 and random pairs (both orders); strip-label models on the
  strip pairs (both orders);
* relative models: no change; settings change (GP, trees; + amplifier in the trained cells); feature differences
  (trees, GP); combined trees on [A, B, B - A]; the direction classifier on [A, B, B - A] (the funnel default:
  P(B > A), changes of the size of the label noise down-weighted; direction only); the shared-encoder network;
  difference curves reduced to main patterns (PCA) + trees and to hand-picked bands + GP; the physics difference for
  sigma_e (d log w - predicted d log g, plus a learned correction); every absolute model differenced;
* scores per pair type: direction right on clear changes with the "always up" rate next to it, size error (RMS),
  median absolute error, Spearman; trained amplifiers and a new amplifier; vs the settings-only reference (bootstrap);
* --holdout-session E: two more cells, "trained, E" and "new amplifier, E": trained once on all session-D rows / pairs
  (the new-amplifier cell without the test amplifier), scored on the session-E rows and pairs only.  Run it ONCE,
  after the choice is frozen.  The level-1 pass mark: direction of clear changes >= 85 % on amplifiers not in
  training, on session E.

    .venv/bin/python lvl1/scorecard_v2.py --data processed_v2_trainD --out <dir>                 # session D only
    .venv/bin/python lvl1/scorecard_v2.py --data processed_v2 --holdout-session E --out <dir>    # after the freeze
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

import change_handpicked as hp  # noqa: E402
import change_models as cm  # noqa: E402
import change_test as ct  # noqa: E402
import funnel_shortlist as fs  # noqa: E402
import scorecard as sc  # noqa: E402
import shortlist as sl  # noqa: E402
import train_lvl1 as tl  # noqa: E402
from ml import models, real_inputs  # noqa: E402

T = sc.T
CLEAR = sc.CLEAR
PLAN = ROOT / "lvl1" / "packet_plan_v2" / "plan.json"
PAIR_TYPES = ("within packet (image)", "within packet (strip)", "n -> n+1", "n -> n+2", "random pairs")
IMAGE_TRAIN = (0, 2, 3, 4)    # pair types the image-label models train on: within packet, n -> n+1, n -> n+2, random
STRIP_TRAIN = (1,)            # ... and the strip-label models
STRIP = {"sigma_e": "l1.l1: sigma_e (e)", "lambda": "l1.l1: lambda (e/pixel)", "gain": "l1.l1: gain (ADU/e)"}
N_BLOCKS = 5
MAX_RANDOM_TRAIN = 800        # random training pairs kept per split (fixed-seed subsample; all other pairs kept)
REL_MODELS = ("no change", "settings change: gp", "settings change: trees", "settings + amplifier: gp",
              "settings + amplifier: trees", "trace change: trees", "trace change: gp", "trace: combined trees [A, B, B-A]",
              "trace: direction classifier [A, B, B-A]", "trace change: shared-encoder net",
              "trace change: difference curves PCA + trees", "trace change: hand-picked bands + gp",
              "trace: physics difference (sigma_e)", "trace: physics difference + correction (sigma_e)")
DIRECTION_ONLY = ("trace: direction classifier [A, B, B-A]",)
SIGMA_ONLY = ("trace: physics difference (sigma_e)", "trace: physics difference + correction (sigma_e)")
CELLS = ["trained", "new"]
CELL_NAME = {"trained": "trained amplifiers", "new": "new amplifier", "trained E": "trained amplifiers, session E",
             "new E": "new amplifier, session E"}
REF = {"trained": "settings + amplifier: gp", "new": "settings: gp", "trained E": "settings + amplifier: gp",
       "new E": "settings: gp"}
REL_REF = {"trained": "settings + amplifier: gp", "new": "settings change: gp", "trained E": "settings + amplifier: gp",
           "new E": "settings change: gp"}


# --------------------------------------------------------------------------- data
def load(data: Path, channels, holdout: str | None, keep_fractional: bool):
    files = {p.stem: real_inputs.load(p) for p in sorted((data / "model_inputs").glob("*.npz"))}
    b = files["physics"]
    ok = ((b["board"] == 101) & np.isin(b["channel"], channels) & b["label_ok"] & ~b["fake"] & b["trace_ok"].astype(bool)
          & np.isfinite(b["Y"]).all(1))
    if not keep_fractional:
        ok &= ~b["fractional_window"].astype(bool)
    d = pd.read_csv(data / "dataset.csv", low_memory=False)
    plan = json.loads(PLAN.read_text())["packets"]
    session = {p["packet_id"]: str(p["session"]) for p in plan}
    planned = [ct.planned_id(p, session) for p in b["packet_id"]]
    ses = np.array([session.get(p, "?") if p else "?" for p in planned])
    ok &= np.isin(ses, ["D", "E"]) if holdout else (ses == "D")
    rows = np.flatnonzero(ok)
    d = d.iloc[rows].reset_index(drop=True).copy()
    assert (d.packet_id.to_numpy() == b["packet_id"][rows]).all() and (d.channel.to_numpy() == b["channel"][rows]).all()
    d["planned_id"] = np.array(planned, dtype=object)[rows]
    d["session"] = ses[rows]
    nominal = [p["packet_id"] for p in plan if str(p.get("purpose", "")).startswith("nominal")]
    d["nominal"] = d.planned_id.isin(nominal)
    ts = pd.to_datetime(d["date_start"])
    d["hours"] = (ts - ts.min()).dt.total_seconds() / 3600
    first = d[d.session == "D"].groupby("packet_id").hours.min().sort_values()
    block = {p: min(N_BLOCKS - 1, r * N_BLOCKS // len(first)) for r, p in enumerate(first.index)}
    d["block"] = d.packet_id.map(block).fillna(-1).astype(int)          # session-E rows: block -1 (never in the CV)
    return files, rows, d


def labels(d: pd.DataFrame, Y: np.ndarray):
    """Per row: nominal label interpolated in time (log), the reference's label (log), the strip changes (log)."""
    nom = np.full(Y.shape, np.nan)
    for ch, g in d.groupby("channel"):
        r = g.index.to_numpy()
        m = r[d.nominal.to_numpy()[r]]
        m = m[np.argsort(d.hours.to_numpy()[m])]
        for k in range(3):
            nom[r, k] = np.interp(d.hours.to_numpy()[r], d.hours.to_numpy()[m], Y[m, k])
    nom_theta = d.loc[d.nominal, "theta_id"].mode().iloc[0]
    ref = np.full(Y.shape, np.nan)
    for i in range(len(d)):
        if d.nominal[i]:
            continue
        if d.reference_id[i] == nom_theta:
            ref[i] = nom[i]
        else:                                    # reference at the previous packet's settings: that packet's own image
            same = np.flatnonzero((d.channel.to_numpy() == d.channel[i]) & (d.theta_id.to_numpy() == d.reference_id[i])
                                  & (d.hours.to_numpy() < d.hours[i]))
            if len(same):
                ref[i] = Y[same[np.argmax(d.hours.to_numpy()[same])]]
    strip = np.full(Y.shape, np.nan)
    usable = d["l1.usable"].astype(str).str.lower().eq("true") & d["l1ref.usable"].astype(str).str.lower().eq("true")
    for k, t in enumerate(T):
        new, old = d[STRIP[t]].to_numpy(float), d[STRIP[t].replace("l1.", "l1ref.", 1)].to_numpy(float)
        with np.errstate(all="ignore"):
            strip[:, k] = np.where(usable & (new > 0) & (old > 0), np.log(new) - np.log(old), np.nan)
    return nom, ref, strip


def make_pairs(d: pd.DataFrame, Y: np.ndarray):
    """Pairs (trace A, trace B): target log y(B) - log y(A), pair type, the two rows.  Trace r = own trace of row r,
    trace n + r = its reference trace."""
    n = len(d)
    nom, ref, strip = labels(d, Y)
    eff = Y - nom
    A, B, typ, ra, rb, Yp = [], [], [], [], [], []

    def add(a, b_, t, i, j, y):
        A.append(a), B.append(b_), typ.append(t), ra.append(i), rb.append(j), Yp.append(y)

    for r in range(n):
        if not d.nominal[r] and np.isfinite(ref[r]).all():
            add(n + r, r, 0, r, r, Y[r] - ref[r])
        if not d.nominal[r] and np.isfinite(strip[r]).any():
            add(n + r, r, 1, r, r, strip[r])
    sess = d.session.to_numpy()
    for ch, g in d.groupby("channel"):
        for s in np.unique(sess[g.index]):                  # pairs never cross sessions (E stays apart)
            r = g.index.to_numpy()[sess[g.index] == s]
            r = r[np.argsort(d.hours.to_numpy()[r])]
            for step in (1, 2):
                for i, j in zip(r[:-step], r[step:]):
                    add(i, j, 1 + step, i, j, Y[j] - Y[i])
            for i, j in combinations(r, 2):
                if np.isfinite(eff[i]).all() and np.isfinite(eff[j]).all():
                    add(i, j, 4, i, j, eff[j] - eff[i])
    A, B, typ, ra, rb = map(np.array, (A, B, typ, ra, rb))
    return dict(A=A, B=B, type=typ, ra=ra, rb=rb, Y=np.array(Yp), eff=eff)


# --------------------------------------------------------------------------- relative models
def _both(X, Y):
    return np.vstack([X, -X]), np.vstack([Y, -Y])


def rel_predict(name, TR, P, tr, te, onehot):
    """One relative model: train on pairs tr (both orders), predict pairs te.  NaN where a model has no prediction."""
    A, B, Y = P["A"], P["B"], P["Y"]
    F = TR["F"]
    out = np.full((len(te), 3), np.nan)
    if name == "no change":
        return np.zeros((len(te), 3))
    if name.startswith("settings"):
        return sc.rel_predict(name, TR, P, tr, te, onehot)
    if name in ("trace change: trees", "trace change: gp"):
        return sc.rel_predict(name, TR, P, tr, te, onehot)
    if name == "trace change: shared-encoder net":
        return sc.rel_predict(name, TR, P, tr, te, onehot)
    if name == "trace change: hand-picked bands + gp":
        return sc.rel_predict(name, TR, P, tr, te, onehot)
    XA, XB = F[A], F[B]
    if name in ("trace: combined trees [A, B, B-A]", "trace: direction classifier [A, B, B-A]"):
        X = np.hstack([XA, XB, XB - XA])
        Xs = np.hstack([XB, XA, XA - XB])
        XX, YY = np.vstack([X[tr], Xs[tr]]), np.vstack([Y[tr], -Y[tr]])
        if "classifier" in name:
            f = fs.direction_model(XX, YY, noise=tuple(cm.NOISE))
        else:
            f = fs.gbt(XX, YY)
        # antisymmetrise the prediction: average f(A, B) and -f(B, A)
        return 0.5 * (f(X[te]) - f(Xs[te]))
    if name == "trace change: difference curves PCA + trees":
        from sklearn.decomposition import PCA
        Dc = TR["C"][B] - TR["C"][A]
        Z = Dc.reshape(len(Dc), -1)
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        Z = (Z - mu) / sd
        pca = PCA(10, random_state=0).fit(np.vstack([Z[tr], -Z[tr]]))
        Xp = pca.transform(Z)
        Xtr, Ytr = _both(Xp[tr], Y[tr])
        return sc.fit_predict("trees", Xtr, Ytr, Xp[te])
    if name.startswith("trace: physics difference"):
        dw = TR["white"][B] - TR["white"][A]                           # d log w (the trace's white noise per skip)
        dX = F[B] - F[A]
        Xtr, Ytr = _both(dX[tr], Y[tr, 2:3])
        dg = sc.fit_predict("trees", Xtr, Ytr, dX)[:, 0]               # predicted d log g, every pair
        base = dw - dg
        if "correction" in name:
            Xc = np.column_stack([base, dX])
            Xtr, Rtr = _both(Xc[tr], (Y[tr, 0] - base[tr])[:, None])
            base = base + sc.fit_predict("trees", Xtr, Rtr, Xc)[:, 0]
        out[:, 0] = base[te]
        return out
    raise ValueError(name)


def run_relative(TR, P, d, abs_preds, cells, only=None):
    amp, blk, ses = d.channel.to_numpy(), d.block.to_numpy(), d.session.to_numpy()
    ba, bb, pamp, typ = blk[P["ra"]], blk[P["rb"]], amp[P["ra"]], P["type"]
    pses = ses[P["ra"]]
    onehot = np.eye(len(np.unique(amp)))[np.unique(amp, return_inverse=True)[1][P["ra"]]]
    families = {"image": (np.isin(typ, IMAGE_TRAIN), ~np.isin(typ, STRIP_TRAIN)),
                "strip": (np.isin(typ, STRIP_TRAIN), np.isin(typ, STRIP_TRAIN))}     # (train pool, scored types)
    preds = {cell: {} for cell in cells}
    for cell in cells:
        holdout = cell.endswith("E")
        base = "new" if cell.startswith("new") else "trained"
        splits = []
        if holdout:
            for a in (np.unique(amp) if base == "new" else [None]):
                te = np.flatnonzero((pses == "E") & ((pamp == a) if a is not None else True))
                tr = np.flatnonzero((pses == "D") & ((pamp != a) if a is not None else True))
                if len(te) and len(tr):
                    splits.append((tr, te))
        else:
            for k in range(N_BLOCKS):
                for a in (np.unique(amp) if base == "new" else [None]):
                    te = np.flatnonzero((ba == k) & (bb == k) & ((pamp == a) if a is not None else True))
                    tr = np.flatnonzero((ba != k) & (bb != k) & (ba >= 0) & (bb >= 0)
                                        & ((pamp != a) if a is not None else True))
                    if len(te) and len(tr):
                        splits.append((tr, te))
        for name in REL_MODELS:
            if only and name not in only and name not in (REL_REF[cell], "no change"):
                continue
            if "amplifier" in name and base == "new":
                continue
            t0 = time.time()
            pr = np.full(P["Y"].shape, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for fam, (pool, scored) in families.items():
                    if fam == "strip" and name in SIGMA_ONLY:
                        continue
                    for tr, te in splits:
                        trf, tef = tr[pool[tr]], te[scored[te]]
                        rnd = trf[typ[trf] == 4]
                        if MAX_RANDOM_TRAIN and len(rnd) > MAX_RANDOM_TRAIN:
                            keep = np.random.default_rng(len(trf)).choice(rnd, MAX_RANDOM_TRAIN, replace=False)
                            trf = np.concatenate([trf[typ[trf] != 4], np.sort(keep)])
                        if len(trf) < 10 or not len(tef):
                            continue
                        pr[tef] = rel_predict(name, TR, P, trf, tef, onehot)
            preds[cell][name] = pr
            print(f"relative {cell:10s} {name:48s} {time.time() - t0:5.0f} s", flush=True)
        packet_pair = np.isin(P["type"], (2, 3, 4))
        for name, m in abs_preds[cell].items():
            if name in sc.NOT_DIFFERENCED:
                continue
            pr = np.full(P["Y"].shape, np.nan)
            pr[packet_pair] = m[P["rb"][packet_pair]] - m[P["ra"][packet_pair]]
            preds[cell][f"absolute {name}, differenced"] = pr
    return preds


def run_absolute(files, rows, d, cells, quick, only=None):
    runner = sc.Runner(files, rows, quick, tl.load_tiers(ct.TIERS))
    amp, blk, ses = d.channel.to_numpy(), d.block.to_numpy(), d.session.to_numpy()
    D = ses == "D"
    splits = {"trained": [(np.flatnonzero(D & (blk != k)), np.flatnonzero(blk == k)) for k in range(N_BLOCKS)],
              "new": [(np.flatnonzero(D & (blk != k) & (amp != a)), np.flatnonzero((blk == k) & (amp == a)))
                      for a in np.unique(amp) for k in range(N_BLOCKS)],
              "trained E": [(np.flatnonzero(D), np.flatnonzero(ses == "E"))],
              "new E": [(np.flatnonzero(D & (amp != a)), np.flatnonzero((ses == "E") & (amp == a))) for a in np.unique(amp)]}
    preds = {cell: {} for cell in cells}
    for name, (method, spec, sees_settings, mcells) in sorted(sc.ABS_MODELS.items(), key=lambda kv: not kv[1][2]):
        if only and name not in only and name not in REF.values() and name != "mean":
            continue
        runner.no_settings = not sees_settings
        for cell in cells:
            if cell.split()[0] not in mcells:
                continue
            t0 = time.time()
            preds[cell][name] = tl.run_model(runner, method, [s for s in splits[cell] if len(s[1])], spec)[0]
            print(f"absolute {cell:10s} {name:28s} " + "  ".join(
                f"{t} {100 * sc.rms(runner.Y[:, i] - preds[cell][name][:, i]):5.1f}" for i, t in enumerate(T))
                + f"   {time.time() - t0:4.0f} s", flush=True)
    return runner.Y, runner.Y_err, preds


# --------------------------------------------------------------------------- scores
def score_absolute(Y, preds, packets, n_boot, cells):
    W = sc.boot_weights(packets, packets, n_boot, seed=0)
    out = {}
    for cell in cells:
        ref = preds[cell][REF[cell]]
        out[cell] = {}
        for name, p in preds[cell].items():
            r = {}
            for i, t in enumerate(T):
                e, er = p[:, i] - Y[:, i], ref[:, i] - Y[:, i]
                iv = sc.interval([sc.rms(e, w) - sc.rms(er, w) for w in W])
                r[t] = {"error": 100 * sc.rms(e), "median": float(100 * np.nanmedian(np.abs(e))), "n": int(np.isfinite(e).sum()),
                        "vs_settings_95": [100 * x for x in iv],
                        "vs_settings": sc.verdict(iv) if name != REF[cell] else "reference"}
            out[cell][name] = r
    return out


def score_relative(P, d, preds, n_boot, cells):
    pk = d.packet_id.to_numpy()
    W = sc.boot_weights(pk[P["ra"]], pk[P["rb"]], n_boot, seed=1)
    out = {}
    for cell in cells:
        ref = preds[cell][REL_REF[cell]]
        out[cell] = {}
        for name, p in preds[cell].items():
            out[cell][name] = {}
            for ty, tname in enumerate(PAIR_TYPES):
                r = {}
                for i, t in enumerate(T):
                    sel = (P["type"] == ty) & np.isfinite(p[:, i]) & np.isfinite(ref[:, i]) & np.isfinite(P["Y"][:, i])
                    if not sel.any():
                        continue
                    y, q, qr = P["Y"][sel, i], p[sel, i], ref[sel, i]
                    Ws = W[:, sel]
                    clear = np.abs(y) > CLEAR[i]
                    ivd = sc.interval([sc.direction(y, q, CLEAR[i], w) - sc.direction(y, qr, CLEAR[i], w) for w in Ws])
                    size = name not in DIRECTION_ONLY
                    ive = sc.interval([sc.rms(q - y, w) - sc.rms(qr - y, w) for w in Ws]) if size else [np.nan, np.nan]
                    r[t] = {"n": int(sel.sum()), "n_clear": int(clear.sum()),
                            "clear_up": float(np.mean(y[clear] > 0)) if clear.any() else np.nan,
                            "direction": sc.direction(y, q, CLEAR[i]),
                            "size_error": 100 * sc.rms(q - y) if size else np.nan,
                            "median_error": float(100 * np.median(np.abs(q - y))) if size else np.nan,
                            "spearman": float(spearmanr(y, q)[0]) if np.std(q) > 0 else np.nan,
                            "true_change_rms": 100 * sc.rms(y),
                            "direction_vs_settings": sc.verdict(ivd, lower_is_better=False) if name != REL_REF[cell] else "reference",
                            "direction_vs_settings_95": [100 * x for x in ivd],
                            "size_vs_settings": (sc.verdict(ive) if name != REL_REF[cell] else "reference") if size else "-",
                            "size_vs_settings_95": [100 * x for x in ive]}
                if r:
                    out[cell][name][tname] = r
    return out


def score_ranking(P, d, rel_preds, abs_preds, weights, cells):
    """scorecard.score_ranking on the within-packet (image) pairs of packets with a NOMINAL reference: their target is the
    setting's effect against nominal, drift removed, so the settings can be ranked against each other."""
    nomref = np.zeros(len(P["type"]), bool)
    nom_theta = d.loc[d.nominal, "theta_id"].mode().iloc[0]
    sel0 = P["type"] == 0
    nomref[sel0] = d.reference_id.to_numpy()[P["ra"][sel0]] == nom_theta
    Q = {k: (v[nomref] if isinstance(v, np.ndarray) and len(v) == len(P["type"]) else v) for k, v in P.items()}
    Q["type"] = np.zeros(nomref.sum(), int)
    out = {}
    for cell in cells:
        rp = {name: p[nomref] for name, p in rel_preds[cell].items() if not name.startswith("absolute")}
        rp = {k: np.nan_to_num(v) for k, v in rp.items() if np.isfinite(v[:, :2]).any()}
        if not len(Q["type"]):
            continue
        ses = d.session.to_numpy()[Q["ra"]]
        want = "E" if cell.endswith("E") else "D"
        keep = ses == want
        Qc = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(Q["type"]) else v) for k, v in Q.items()}
        rpc = {k: v[keep] for k, v in rp.items()}
        abc = {k: v for k, v in abs_preds[cell].items()}
        sc_cells, sc.CELLS = sc.CELLS, (cell,)
        try:
            out[cell] = sc.score_ranking(Qc, d, {cell: rpc}, {cell: abc}, weights)[cell]
        finally:
            sc.CELLS = sc_cells
    return out


# --------------------------------------------------------------------------- report
def fmt(v, pct=True):
    return "-" if v is None or not np.isfinite(v) else (f"{100 * v:.0f} %" if pct else f"{v:.1f}")


def report(abs_sc, rel_sc, info, cells) -> str:
    L = ["# Level-1 scorecard, v2", "",
         f"Board 101 ch{'/'.join(map(str, info['channels']))}; {info['n_rows']} rows from {info['n_packets']} packets "
         f"({info['sessions']}); {N_BLOCKS} contiguous time blocks of session D; fractional-window rows "
         f"{'kept' if info['keep_fractional'] else 'left out'}. Trace models never see the settings. "
         f"Pairs: {info['n_pairs']}.", "",
         "Image-label change models train on the within-packet (image), n -> n+1, n -> n+2 and random pairs (random capped per split); strip-label models on the strip "
         "pairs; both orders. A test pair never shares a time block with a training pair.", "",
         f"Label noise of a change: {', '.join(f'{t} {100 * v:.1f} %' for t, v in zip(T, cm.NOISE))}; clear = "
         f"{', '.join(f'{t} > {100 * c:.1f} %' for t, c in zip(T, CLEAR))}.", ""]
    for cell in cells:
        L += [f"## Absolute, {CELL_NAME[cell]}: typical error % (vs settings-only `{REF[cell]}`)", "",
              "| model | " + " | ".join(T) + " |", "|---|" + "---|" * len(T)]
        for name, r in abs_sc[cell].items():
            L.append(f"| {name} | " + " | ".join(f"{r[t]['error']:.1f} ({r[t]['vs_settings']})" for t in T) + " |")
        L.append("")
    for cell in cells:
        for tname in PAIR_TYPES:
            rows = {n: r[tname] for n, r in rel_sc[cell].items() if tname in r}
            if not rows:
                continue
            ref = rows.get(REL_REF[cell]) or next(iter(rows.values()))
            L += [f"## Relative, {CELL_NAME[cell]}, {tname}", "",
                  "Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | "
                  "median error % | Spearman. " + "; ".join(
                      f"{t}: {ref[t]['n_clear']} clear of {ref[t]['n']}, true change RMS {ref[t]['true_change_rms']:.1f} %"
                      for t in T if t in ref), "",
                  "| model | " + " | ".join(f"{t} dir | up | size | med | rho" for t in T) + " |",
                  "|---|" + "---|---|---|---|---|" * len(T)]
            for name, r in rows.items():
                cells_ = []
                for t in T:
                    if t not in r:
                        cells_.append("- | - | - | - | -")
                        continue
                    x = r[t]
                    rho = "-" if not np.isfinite(x["spearman"]) else f"{x['spearman']:.2f}"
                    cells_.append(f"{fmt(x['direction'])} ({x['direction_vs_settings']}) | {fmt(x['clear_up'])} | "
                                  f"{fmt(x['size_error'], False)} | {fmt(x['median_error'], False)} | {rho}")
                L.append(f"| {name} | " + " | ".join(cells_) + " |")
            L.append("")
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="processed data folder (dataset.csv, model_inputs/)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout-session", choices=("E",), help="also score session E (trained on D): ONCE, after the freeze")
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 1, 3])
    parser.add_argument("--keep-fractional", action="store_true", help="keep the rows with fractional CDS windows")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--abs-models", nargs="+")
    parser.add_argument("--rel-models", nargs="+")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--weights", type=float, nargs="+", default=list(sl.DEFAULT_WEIGHTS))
    parser.add_argument("--cells", nargs="+", help="only these cells (e.g. 'trained E' 'new E' for the session-E test)")
    parser.add_argument("--max-random-train", type=int, default=MAX_RANDOM_TRAIN,
                        help="random pairs kept per training split (0 = all)")
    args = parser.parse_args()
    globals()["MAX_RANDOM_TRAIN"] = args.max_random_train
    data = Path(args.data).resolve()
    cm.set_data(data)
    sc.DATA = data
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = CELLS + (["trained E", "new E"] if args.holdout_session else [])
    if args.cells:
        cells = [c for c in cells if c in args.cells]
    files, rows, d = load(data, tuple(args.channels), args.holdout_session, args.keep_fractional)
    print(f"{len(d)} rows, {d.packet_id.nunique()} packets, sessions {d.session.value_counts().to_dict()}", flush=True)
    cache = out / "predictions.pkl"
    if args.rescore:
        Y, Y_err, abs_preds, P, rel_preds = pickle.loads(cache.read_bytes())
    else:
        Y, Y_err, abs_preds = run_absolute(files, rows, d, cells, args.quick, args.abs_models)
        TR = sc.trace_table(files, rows, d)
        P = make_pairs(d, Y)
        print("pairs:", {t: int((P["type"] == k).sum()) for k, t in enumerate(PAIR_TYPES)}, flush=True)
        rel_preds = run_relative(TR, P, d, abs_preds, cells, args.rel_models)
        cache.write_bytes(pickle.dumps((Y, Y_err, abs_preds, P, rel_preds)))
    abs_sc = score_absolute(Y, abs_preds, d.packet_id.to_numpy(), args.n_boot, cells)
    rel_sc = score_relative(P, d, rel_preds, args.n_boot, cells)
    rank_sc = score_ranking(P, d, rel_preds, abs_preds, args.weights, cells)
    info = {"n_rows": len(d), "n_packets": int(d.packet_id.nunique()), "channels": args.channels,
            "sessions": d.session.value_counts().to_dict(), "keep_fractional": args.keep_fractional,
            "n_pairs": {t: int((P["type"] == k).sum()) for k, t in enumerate(PAIR_TYPES)},
            "max_random_train": MAX_RANDOM_TRAIN}
    (out / "scorecard.json").write_text(json.dumps({"info": info, "absolute": abs_sc, "relative": rel_sc, "ranking": rank_sc},
                                                   indent=1, default=float))
    sc_cells, sc.CELLS = sc.CELLS, tuple(c for c in cells if c in rank_sc)
    try:
        rank_md = sc.ranking_report(rank_sc, args.weights)
    finally:
        sc.CELLS = sc_cells
    (out / "scorecard.md").write_text(report(abs_sc, rel_sc, info, cells) + "\n" + "\n".join(rank_md))
    pcols = {"type": [PAIR_TYPES[k] for k in P["type"]], "packet A": d.packet_id.to_numpy()[P["ra"]],
             "packet B": d.packet_id.to_numpy()[P["rb"]], "channel": d.channel.to_numpy()[P["ra"]],
             "session": d.session.to_numpy()[P["ra"]]}
    for i, t in enumerate(T):
        pcols[f"true change {t}"] = P["Y"][:, i]
    for cell in cells:
        for name, p in rel_preds[cell].items():
            for i, t in enumerate(T):
                pcols[f"{cell} | {name} | {t}"] = p[:, i]
    pd.DataFrame(pcols).to_csv(out / "relative_predictions.csv", index=False)
    print(f"wrote {out / 'scorecard.md'}")


if __name__ == "__main__":
    main()

"""
fleet_tests.py -- the tests of the ML plan of 2026-11-10 (docs/overleaf_ml_plan_2026-11-10/ml_plan.tex): can machine
learning cut the number of IMAGES needed to set up a new CCD, by learning across amplifiers (later: across CCDs)?
Every test runs on data already taken (v2 sessions D, E; v1 for a smoke test) and treats one amplifier at a time as
the "new CCD", the others as the fleet it learns from.

  fewshot   T1  How many images of a new amplifier does the model need?  Leave one amplifier out, give the model k of
                its packets (k = 0, 1, 2, 4, 8; random draws), score the rest: error of sigma_e / lambda / gain, and the
                REGRET of the setting the model would pick (how much worse than the best of the remaining settings).
                Models: the fleet mean; a shared settings GP (amplifier ignored; k > 0: + the mean offset of the k
                packets); a MULTI-TASK GP (shared response x amplifier similarity rho + an amplifier offset term, rho and
                the offset size learned from the fleet, the new amplifier's k packets enter as data); trace trees (own
                trace features, no settings; + offset).  The board-102 "one calibration image" result is k = 1 with an
                offset only.
  bo        T2  Bayesian optimisation replayed on the measured packets.  For the left-out amplifier the pool of its
                measured settings is the candidate set; a strategy picks the next setting to "image", sees its label,
                updates, and so on.  Score: best-so-far regret against the number of images, images needed to come
                within 2 % / 5 % of the best in the pool.  Strategies: random order (many draws); the fleet model's
                ranking without updates; BO with the fleet as prior (multi-task GP, expected improvement); BO with no
                fleet data; BO with the fleet + the trace features of every candidate (traces are cheap).  Objective
                J = w log sigma_e + (1 - w) log lambda for w in --weights.  Starts from the nominal packet (one
                calibration image) unless --start none.
  strips    T3  Can the level-1 STRIP (2 rows, under a minute) stand in for the image (32 rows, 9.5 min)?  Per target:
                strip vs image on the same packet (correlation, error after an amplifier offset, error of a calibration
                model log image ~ log strip + amplifier, cross-validated by packet), repeatability at nominal, and the
                direction of within-packet changes (strip new vs strip reference) against the image change.  v2 only
                (needs l1.* columns).
  residual  T4  What is "obvious" and what is left for ML.  Per target: variance explained by the timing knobs alone
                (the known physics), what the voltages add on top (on trained amplifiers and on a new one), what is left
                against the label noise, and per voltage knob the effect all amplifiers share against the
                amplifier-specific part (the part a new CCD must measure).

Common options: --data (processed folder with dataset.csv), --plan (plan.json files, for sessions and nominal packets),
--sessions, --board, --channels, --drift (subtract each amplifier's nominal packets interpolated in time, keeping its
mean level; uses the left-out amplifier's nominal packets), --out.

    python lvl1/fleet_tests.py fewshot  --data processed_v2 --plan lvl1/packet_plan_v2/plan.json --sessions D E --out <dir>
    python lvl1/fleet_tests.py bo       ... ; python lvl1/fleet_tests.py strips ... ; python lvl1/fleet_tests.py residual ...
    python lvl1/fleet_tests.py all      ...      # the four in a row
(smoke test on v1: --data processed --plan lvl1/packet_plan/plan.json --sessions A B C --channels 0 1 2 3)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lvl1"))
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore")

LAB = {"sigma_e": "label.sigma_e", "lambda": "label.lambda_e_per_bin", "gain": "label.gain_adu_per_e"}
STRIP = {"sigma_e": "l1.l1: sigma_e (e)", "lambda": "l1.l1: lambda (e/pixel)", "gain": "l1.l1: gain (ADU/e)"}
CLEAR = {"sigma_e": 0.05, "lambda": 0.15, "gain": 0.012}     # clear change (log), as lvl1/change_test.py
TIMING_KNOBS = ("delayInt", "delayRGH", "delay_Ha", "delay_Hb", "delay_Hc", "delayRGL", "delaySWH", "delaySWL",
                "delayOGL", "delayOGH", "delayDGL", "delayDGH", "delay_Va", "delay_Vb", "NPSD", "NVWAIT")
PCT = 100.0          # log differences are reported as % (100 x log ratio)


# =========================================================================== data
def planned_id(pid, planned):
    if pid in planned:
        return pid
    for q in planned:
        if pid.startswith(q) and not pid[len(q)].isdigit():
            return q
    return None


def load(args):
    d = pd.read_csv(Path(args.data) / "dataset.csv", low_memory=False)
    packets = [p for pl in args.plan for p in json.loads(Path(pl).read_text())["packets"]]
    session = {p["packet_id"]: str(p.get("session", "?")) for p in packets}
    purpose = {p["packet_id"]: str(p.get("purpose", "")) for p in packets}
    pid = [planned_id(str(p), session) for p in d.packet_id]
    d["planned_id"] = pid
    d["session"] = [session.get(p, "?") if p else "?" for p in pid]
    d["purpose"] = [purpose.get(p, "") if p else "" for p in pid]
    istrue = lambda c: d[c].astype(str).str.lower().eq("true") if c in d else pd.Series(False, index=d.index)
    ok = (d.board == args.board) & d.channel.isin(args.channels) & d.session.isin(args.sessions) & istrue("label.ok")
    ok &= ~istrue("label.fractional_window")
    if "fake" in d:
        ok &= ~istrue("fake")
    d = d[ok].reset_index(drop=True)
    for t, c in LAB.items():
        x = pd.to_numeric(d[c], errors="coerce").astype(float)
        d[f"log_{t}"] = np.log(x.where(x > 0))
    d = d[np.isfinite(d[[f"log_{t}" for t in LAB]]).all(1)].reset_index(drop=True)
    d["amp"] = d.board.astype(str) + ":" + d.channel.astype(str)
    ts = pd.to_datetime(d.date_start)
    d["hours"] = (ts - ts.min()).dt.total_seconds() / 3600
    d["nominal"] = d.purpose.str.startswith("nominal")
    if not d.nominal.any():                                  # no purposes: the most common setting is nominal
        d["nominal"] = d.theta_id.eq(d.theta_id.mode().iloc[0])
    if args.drift:
        for t in LAB:
            d[f"log_{t}"] = drift_corrected(d, f"log_{t}")
    if len(d) == 0:
        raise SystemExit("no rows: check --data, --plan, --sessions, --board, --channels")
    return d


def drift_corrected(d, col):
    """log y - (the amplifier's nominal packets interpolated in time) + their mean: the drift removed, the level kept.
    A nominal packet is interpolated from the OTHER nominal packets."""
    out = d[col].to_numpy(float).copy()
    for a, g in d.groupby("amp"):
        m = g[g.nominal].sort_values("hours")
        if len(m) < 2:
            continue
        h, y = m.hours.to_numpy(), m[col].to_numpy()
        for i in g.index:
            keep = m.index != i
            out[i] = d.at[i, col] - np.interp(d.at[i, "hours"], h[keep], y[keep]) + y.mean()
    return out


def knobs(d):
    """The varied settings (theta.*), standardised; settings that always moved together become one knob; timing on
    a log2 scale.  Returns (Z array, names, is_timing list)."""
    th = [c for c in d.columns if c.startswith("theta.") and pd.to_numeric(d[c], errors="coerce").nunique() > 1]
    groups = []
    for c in th:
        x = pd.to_numeric(d[c], errors="coerce").to_numpy(float)
        for g in groups:
            z = pd.to_numeric(d[g[0]], errors="coerce").to_numpy(float)
            if np.allclose(x, z, equal_nan=True) or abs(np.corrcoef(x, z)[0, 1]) > 0.999:
                g.append(c)
                break
        else:
            groups.append([c])
    cols, names, timing = [], [], []
    for g in groups:
        short = g[0][len("theta."):]
        x = pd.to_numeric(d[g[0]], errors="coerce").to_numpy(float)
        is_t = short in TIMING_KNOBS
        x = np.log2(np.clip(x, 1e-9, None)) if is_t else x
        sd = x.std()
        cols.append((x - x.mean()) / (sd if sd > 0 else 1.0))
        names.append(short + (f" (+{len(g) - 1})" if len(g) > 1 else ""))
        timing.append(is_t)
    return np.column_stack(cols), names, timing


def trace_matrix(d, n_pc=10):
    """Own-trace features (trace.*), standardised, NaN -> 0, reduced to n_pc principal components (unsupervised, so
    it may use every row: traces of every candidate setting are cheap to take).  None if there are none."""
    cols = [c for c in d.columns if c.startswith("trace.") and pd.api.types.is_numeric_dtype(d[c])]
    if not cols:
        return None, None
    X = d[cols].to_numpy(float)
    keep = np.isfinite(X).mean(0) > 0.8
    X = X[:, keep]
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    X = np.nan_to_num((X - mu) / np.where(sd > 0, sd, 1.0))
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    pcs = U[:, :n_pc] * S[:n_pc]
    return X, pcs / pcs.std(0)


def packet_blocks(d, n):
    """Each row's block: the packets in time order cut into n contiguous blocks.  A left-out amplifier is always scored
    on one block while the fleet trains on the OTHER blocks: the amplifiers of one packet share its moment (drift, a
    stalled queue), which a new CCD measured another day would not share."""
    first = d.groupby("packet_id").hours.min().sort_values()
    blk = {p: min(n - 1, r * n // len(first)) for r, p in enumerate(first.index)}
    return d.packet_id.map(blk).to_numpy()


def label_noise(d, t):
    """Repeatability of the label at nominal (log, per amplifier: RMS of successive differences / sqrt 2), median."""
    out = []
    for a, g in d[d.nominal].sort_values("hours").groupby("amp"):
        y = g[f"log_{t}"].to_numpy()
        if len(y) >= 3:
            out.append(np.sqrt(np.mean(np.diff(y) ** 2) / 2))
    return float(np.median(out)) if out else np.nan


# =========================================================================== GP pieces
def fit_shared_gp(X, y):
    """Shared GP (sklearn): constant x Matern-5/2 (ARD) + white.  Returns (signal var, length scales, noise var)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
    k = C(np.var(y) + 1e-6, (1e-5, 1e2)) * Matern(np.ones(X.shape[1]) * 2.0, (0.05, 1e3), nu=2.5) \
        + WhiteKernel(0.1 * np.var(y) + 1e-6, (1e-6, 1e1))
    gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=1, random_state=0)
    gp.fit(X, y - y.mean())
    kk = gp.kernel_
    return float(kk.k1.k1.constant_value), np.atleast_1d(kk.k1.k2.length_scale), float(kk.k2.noise_level)


def matern(A, B, ls):
    from sklearn.gaussian_process.kernels import Matern
    return Matern(length_scale=ls, nu=2.5)(A, B)


class MultiTaskGP:
    """K = s2 M(z, z') (rho + (1 - rho) [same amp]) + c2 [same amp] + n2 [same row], constant mean.
    rho = 1, c2 > 0: one shared response + an offset per amplifier; rho < 1: amplifiers differ in shape too.
    Hyperparameters: s2, length scales, n2 from a shared GP on the fleet; rho and c2 by maximum marginal likelihood
    on the fleet (grid).  The new amplifier's packets are data with their own amp label."""

    RHOS = (0.0, 0.5, 0.75, 0.9, 0.97, 1.0)
    C2S = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)

    def __init__(self, X, y, amp, hyper=None):
        self.mu = float(y.mean())
        self.s2, self.ls, self.n2 = hyper if hyper else fit_shared_gp(X, y)
        best = -np.inf
        for rho in self.RHOS:
            for c in self.C2S:
                ll = self._loglik(X, y, amp, rho, c * self.s2)
                if ll > best:
                    best, self.rho, self.c2 = ll, rho, c * self.s2
        self.ll = best

    def K(self, A, aa, B, ab):
        same = (np.asarray(aa)[:, None] == np.asarray(ab)[None, :]).astype(float)
        return self.s2 * matern(A, B, self.ls) * (self.rho + (1 - self.rho) * same) + self.c2 * same

    def _loglik(self, X, y, amp, rho, c2):
        same = (amp[:, None] == amp[None, :]).astype(float)
        K = self.s2 * matern(X, X, self.ls) * (rho + (1 - rho) * same) + c2 * same + self.n2 * np.eye(len(y))
        try:
            L = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return -np.inf
        r = y - self.mu
        return float(-0.5 * r @ cho_solve(L, r) - np.log(np.diag(L[0])).sum())

    def predict(self, Xtr, ytr, atr, Xte, ate):
        K = self.K(Xtr, atr, Xtr, atr) + self.n2 * np.eye(len(ytr))
        L = cho_factor(K + 1e-9 * np.eye(len(ytr)), lower=True)
        Ks = self.K(Xte, ate, Xtr, atr)
        m = self.mu + Ks @ cho_solve(L, ytr - self.mu)
        v = self.s2 + self.c2 - np.einsum("ij,ji->i", Ks, cho_solve(L, Ks.T))
        return m, np.sqrt(np.clip(v, 1e-12, None))


def shared_predict(hyper, Xtr, ytr, Xte):
    """A plain GP with fixed hyperparameters (no amplifier information)."""
    s2, ls, n2 = hyper
    mu = ytr.mean()
    L = cho_factor(s2 * matern(Xtr, Xtr, ls) + (n2 + 1e-9) * np.eye(len(ytr)), lower=True)
    Ks = s2 * matern(Xte, Xtr, ls)
    m = mu + Ks @ cho_solve(L, ytr - mu)
    v = s2 - np.einsum("ij,ji->i", Ks, cho_solve(L, Ks.T))
    return m, np.sqrt(np.clip(v, 1e-12, None))


def trees():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8,
                                         random_state=0)


# =========================================================================== T1 few-shot
def test_fewshot(d, args, out):
    Z, names, _ = knobs(d)
    T, _ = trace_matrix(d)
    amps = sorted(d.amp.unique())
    rng = np.random.default_rng(args.seed)
    blk = packet_blocks(d, args.blocks)
    res = {"amps": amps, "ks": args.ks, "blocks": args.blocks, "targets": {}, "label_noise": {t: label_noise(d, t) for t in LAB}}
    preds_all = {}
    for t in LAB:
        y = d[f"log_{t}"].to_numpy()
        rows = {}
        for a, b in [(a, b) for a in amps for b in range(args.blocks)]:
            amp_arr = d.amp.to_numpy()
            test = np.flatnonzero((amp_arr == a) & (blk == b))              # scored: this amplifier, this block
            pool_k = np.flatnonzero((amp_arr == a) & (blk != b))            # its k images come from other times
            tr = np.flatnonzero((amp_arr != a) & (blk != b))                # the fleet, other times
            if len(test) < 3:
                continue
            hyper = fit_shared_gp(Z[tr], y[tr])
            mt = MultiTaskGP(Z[tr], y[tr], d.amp.to_numpy()[tr], hyper)
            tree = None
            if T is not None:
                tree = trees().fit(T[tr], y[tr])
                tree_te = tree.predict(T)
            base_shared, _ = shared_predict(hyper, Z[tr], y[tr], Z)
            for k in args.ks:
                if k > len(pool_k):
                    continue
                for r in range(args.draws if k > 0 else 1):
                    known = rng.choice(pool_k, size=k, replace=False) if k else np.array([], int)
                    P = {"fleet mean": np.full(len(test), y[tr].mean())}
                    if k:
                        P["mean of the k images"] = np.full(len(test), y[known].mean())
                    off = (y[known] - base_shared[known]).mean() if k else 0.0
                    P["shared GP (+ offset)"] = base_shared[test] + off
                    Xtr = np.vstack([Z[tr], Z[known]])
                    ytr = np.concatenate([y[tr], y[known]])
                    atr = np.concatenate([d.amp.to_numpy()[tr], [a] * k])
                    P["multi-task GP"], _ = mt.predict(Xtr, ytr, atr, Z[test], [a] * len(test))
                    if tree is not None:
                        toff = (y[known] - tree_te[known]).mean() if k else 0.0
                        P["trace trees (+ offset)"] = tree_te[test] + toff
                    for m, p in P.items():
                        e = p - y[test]
                        best = y[test].min()
                        rows.setdefault((m, k), []).append({"block": b,
                            "amp": a, "rms": float(np.sqrt(np.mean(e ** 2)) * PCT),
                            "regret": float((y[test][np.argmin(p)] - best) * PCT) if m not in ("fleet mean", "mean of the k images") else np.nan,
                            "spearman": float(spearmanr(p, y[test]).correlation) if np.std(p) > 0 else np.nan})
                        preds_all.setdefault((t, m, k, a, b, r), (test, p))
            print(f"  fewshot {t} {a} block {b}: rho {mt.rho:.2f}, offset sd {np.sqrt(mt.c2):.3f}, shape sd {np.sqrt(mt.s2):.3f}",
                  flush=True)
            res.setdefault("multitask_hyper", {}).setdefault(t, {})[f"{a} block {b}"] = {"rho": mt.rho, "offset_sd": float(np.sqrt(mt.c2)),
                                                                           "signal_sd": float(np.sqrt(mt.s2)),
                                                                           "noise_sd": float(np.sqrt(mt.n2))}
        summ = {}
        for (m, k), v in rows.items():
            summ.setdefault(m, {})[str(k)] = {s: float(np.nanmean([x[s] for x in v])) for s in ("rms", "regret", "spearman")}
            summ[m][str(k)]["per_amp_rms"] = {a: float(np.mean([x["rms"] for x in v if x["amp"] == a])) for a in amps
                                              if any(x["amp"] == a for x in v)}
        res["targets"][t] = summ
    # joint objective regret (w = 0.5) from the two models' predictions on the same test rows
    joint = {}
    for (t, m, k, a, b, r), (test, p) in preds_all.items():
        if t != "sigma_e" or ("lambda", m, k, a, b, r) not in preds_all or m in ("fleet mean", "mean of the k images"):
            continue
        test2, p2 = preds_all[("lambda", m, k, a, b, r)]
        J = 0.5 * d.log_sigma_e.to_numpy()[test] + 0.5 * d.log_lambda.to_numpy()[test]
        pj = 0.5 * p + 0.5 * p2
        joint.setdefault(m, {}).setdefault(str(k), []).append((J[np.argmin(pj)] - J.min()) * PCT)
    res["joint_regret_w0.5"] = {m: {k: float(np.mean(v)) for k, v in kk.items()} for m, kk in joint.items()}
    write(out, "fewshot", res, fewshot_md(res))
    fewshot_fig(res, out)


def fewshot_md(res):
    L = ["# T1: images of a new amplifier needed (leave one amplifier out)", "",
         f"Amplifiers: {', '.join(res['amps'])}; packets in {res['blocks']} time blocks. The left-out amplifier is scored on one "
         "block; the fleet (the other amplifiers) trains on the other blocks; its k images come from the other blocks. "
         "Error = RMS of the log error x 100 (%), mean over draws, blocks and amplifiers. Regret = how much worse (%) the "
         "setting the model ranks best in the block is than the best in the block.", "",
         "Label noise at nominal (%): " + ", ".join(f"{t} {v * PCT:.1f}" for t, v in res["label_noise"].items()), ""]
    for t, summ in res["targets"].items():
        ks = res["ks"]
        L += [f"## {t}", "", "| model | " + " | ".join(f"k={k} error / regret" for k in ks) + " |",
              "|---|" + "---|" * len(ks)]
        for m, v in summ.items():
            cells = [f"{v[str(k)]['rms']:.1f} / {v[str(k)]['regret']:.1f}" if str(k) in v else "–" for k in ks]
            L.append(f"| {m} | " + " | ".join(c.replace("/ nan", "/ –") for c in cells) + " |")
        L.append("")
    L += ["## joint objective (w = 0.5), regret %", "", "| model | " + " | ".join(f"k={k}" for k in res["ks"]) + " |",
          "|---|" + "---|" * len(res["ks"])]
    for m, v in res.get("joint_regret_w0.5", {}).items():
        L.append(f"| {m} | " + " | ".join(f"{v[str(k)]:.1f}" if str(k) in v else "–" for k in res["ks"]) + " |")
    L += ["", "Multi-task GP, learned on the fleet (per left-out amplifier and block): rho = how alike the amplifiers' responses "
          "are (1 = the same shape), offset sd = how far apart their levels are (log).", ""]
    for t, h in res.get("multitask_hyper", {}).items():
        L.append(f"* {t}: " + "; ".join(f"{a} rho {v['rho']:.2f}, offset sd {v['offset_sd']:.2f}" for a, v in h.items()))
    return "\n".join(L) + "\n"


def fewshot_fig(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    for j, (t, summ) in enumerate(res["targets"].items()):
        for m, v in summ.items():
            ks = [k for k in res["ks"] if str(k) in v]
            axs[0, j].plot(ks, [v[str(k)]["rms"] for k in ks], "o-", label=m)
            if not all(np.isnan(v[str(k)]["regret"]) for k in ks):
                axs[1, j].plot(ks, [v[str(k)]["regret"] for k in ks], "o-", label=m)
        axs[0, j].axhline(res["label_noise"][t] * PCT, color="k", ls=":", lw=1, label="label noise")
        axs[0, j].set(title=t, ylabel="error on the new amplifier (%)", xscale="symlog", xlabel="k images of it")
        axs[1, j].set(ylabel="regret of the chosen setting (%)", xscale="symlog", xlabel="k images of it")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fewshot.png", dpi=130)
    plt.close(fig)


# =========================================================================== T2 BO replay
def ei(m, s, best):
    z = (best - m) / s
    return (best - m) * norm.cdf(z) + s * norm.pdf(z)


def test_bo(d, args, out):
    Z, names, _ = knobs(d)
    _, PCS = trace_matrix(d)
    amps = sorted(d.amp.unique())
    rng = np.random.default_rng(args.seed)
    blk = packet_blocks(d, args.bo_blocks)
    res = {"amps": amps, "weights": args.weights, "start": args.start, "blocks": args.bo_blocks, "curves": {}, "images_to": {}}
    for w in args.weights:
        J = w * d.log_sigma_e.to_numpy() + (1 - w) * d.log_lambda.to_numpy()
        curves = {}
        for a, bb in [(a, bb) for a in amps for bb in range(args.bo_blocks)]:
            pool = np.flatnonzero((d.amp.to_numpy() == a) & (blk == bb))   # candidates: this amplifier, one block
            tr = np.flatnonzero((d.amp.to_numpy() != a) & (blk != bb))     # the fleet: other amplifiers, other times
            if len(pool) < 5:
                continue
            Jp = J[pool]
            best = Jp.min()
            n = len(pool)
            start = pool[d.nominal.to_numpy()[pool]]
            start = [int(start[0])] if (args.start == "nominal" and len(start)) else []

            def regret_curve(order):
                return [(J[order[:i + 1]].min() - best) * PCT for i in range(len(order))]

            # random order
            rc = [regret_curve(list(start) + list(rng.permutation(np.setdiff1d(pool, start)))) for _ in range(200)]
            curves.setdefault("random order", []).append(np.mean(rc, 0))
            amp_arr = d.amp.to_numpy()
            variants = {"BO, fleet as prior (multi-task GP)": Z}
            if PCS is not None:
                variants["BO, fleet + traces of every candidate"] = np.hstack([Z, PCS])
            for name, X in variants.items():
                mt = MultiTaskGP(X[tr], J[tr], amp_arr[tr])
                if name.startswith("BO, fleet as prior"):
                    m0, _ = mt.predict(X[tr], J[tr], amp_arr[tr], X[pool], [a] * n)
                    order = list(start) + [p for p in pool[np.argsort(m0)] if p not in start]
                    curves.setdefault("fleet model ranking, no updates", []).append(regret_curve(order))
                revealed = list(start)
                while len(revealed) < n:
                    rem = np.setdiff1d(pool, revealed)
                    Xtr = np.vstack([X[tr], X[revealed]]) if revealed else X[tr]
                    ytr = np.concatenate([J[tr], J[revealed]])
                    atr = np.concatenate([amp_arr[tr], [a] * len(revealed)])
                    m, s = mt.predict(Xtr, ytr, atr, X[rem], [a] * len(rem))
                    b = J[revealed].min() if revealed else m.min()
                    revealed.append(int(rem[np.argmax(ei(m, s, b))]))
                curves.setdefault(name, []).append(regret_curve(revealed))
            # BO without fleet data (the same kernel shape, only this amplifier's images)
            hyper = fit_shared_gp(Z[tr], J[tr])        # the kernel shape from the fleet, none of its data
            revealed = list(start) + [int(x) for x in rng.choice(np.setdiff1d(pool, start), size=2, replace=False)]
            while len(revealed) < n:
                rem = np.setdiff1d(pool, revealed)
                m, s = shared_predict(hyper, Z[revealed], J[revealed], Z[rem])
                revealed.append(int(rem[np.argmax(ei(m, s, J[revealed].min()))]))
            curves.setdefault("BO, no fleet data", []).append(regret_curve(revealed))
            print(f"  bo w={w} {a} block {bb}: pool {n}", flush=True)
        nmin = min(len(c) for v in curves.values() for c in v)
        res["curves"][str(w)] = {k: np.mean([c[:nmin] for c in v], 0).tolist() for k, v in curves.items()}
        res["images_to"][str(w)] = {k: {f"{x}%": float(np.mean([next((i + 1 for i, r in enumerate(c) if r <= x), len(c))
                                                                for c in v])) for x in (2, 5)} for k, v in curves.items()}
    write(out, "bo", res, bo_md(res))
    bo_fig(res, out)


def bo_md(res):
    L = ["# T2: Bayesian optimisation replayed on the measured packets", "",
         f"Each amplifier in turn is the new CCD; its measured settings in one of {res['blocks']} time blocks are the "
         f"candidates, the fleet is the other amplifiers in the other blocks. Start: {res['start']}. "
         "Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool "
         "(J = w log sigma_e + (1 - w) log lambda).", ""]
    for w, v in res["images_to"].items():
        L += [f"## w = {w}", "", "| strategy | images to 2 % | images to 5 % |", "|---|---|---|"]
        for k, x in v.items():
            L.append(f"| {k} | {x['2%']:.1f} | {x['5%']:.1f} |")
        L.append("")
    return "\n".join(L)


def bo_fig(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ws = list(res["curves"])
    fig, axs = plt.subplots(1, len(ws), figsize=(5 * len(ws), 4), squeeze=False)
    for j, w in enumerate(ws):
        for k, c in res["curves"][w].items():
            axs[0, j].plot(np.arange(1, len(c) + 1), c, label=k)
        axs[0, j].set(title=f"w = {w}", xlabel="images taken", ylabel="best so far vs best in pool (%)", xlim=(1, 25))
        axs[0, j].axhline(2, color="k", ls=":", lw=1)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "bo.png", dpi=130)
    plt.close(fig)


# =========================================================================== T3 strips
def test_strips(d, args, out):
    if not all(c in d for c in STRIP.values()):
        print("  strips: no level-1 strip columns (l1.*) in this dataset: skipped (v2 has them)")
        return
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    istrue = lambda c: d[c].astype(str).str.lower().eq("true") if c in d else pd.Series(True, index=d.index)
    res = {"targets": {}}
    nom_theta = d.loc[d.nominal, "theta_id"].mode().iloc[0]
    for t, c in STRIP.items():
        s = pd.to_numeric(d[c], errors="coerce").to_numpy(float)
        sref = pd.to_numeric(d[c.replace("l1.", "l1ref.", 1)], errors="coerce").to_numpy(float) if c.replace("l1.", "l1ref.", 1) in d else np.full(len(d), np.nan)
        ok = istrue("l1.usable").to_numpy() & (s > 0)
        y = d[f"log_{t}"].to_numpy()
        ls = np.log(np.where(s > 0, s, np.nan))
        g = d[ok]
        x, yy = ls[ok], y[ok]
        r = {"n": int(ok.sum()), "pearson_log": float(np.corrcoef(x, yy)[0, 1])}
        # error after one offset per amplifier (the strip as a relative measure)
        e = yy - x
        e = e - pd.Series(e).groupby(g.amp.to_numpy()).transform("mean").to_numpy()
        r["rms_after_amp_offset"] = float(np.sqrt(np.mean(e ** 2)) * PCT)
        # calibration model: log image ~ log strip + amplifier, CV by packet
        A = pd.get_dummies(g.amp).to_numpy(float)
        X = np.column_stack([x, A])
        pred = np.full(len(yy), np.nan)
        for trn, tst in GroupKFold(n_splits=min(5, g.packet_id.nunique())).split(X, yy, g.packet_id):
            pred[tst] = RidgeCV(alphas=np.logspace(-3, 2, 11)).fit(X[trn], yy[trn]).predict(X[tst])
        r["rms_calibrated_cv"] = float(np.sqrt(np.mean((pred - yy) ** 2)) * PCT)
        r["image_spread"] = float(np.std(yy) * PCT)
        # repeatability at nominal: strip vs image
        nom = g.nominal.to_numpy()
        def rep(v):
            out_ = []
            for a, gg in pd.DataFrame({"a": g.amp[nom].to_numpy(), "h": g.hours[nom].to_numpy(), "v": v[nom]}).sort_values("h").groupby("a"):
                if len(gg) >= 3:
                    out_.append(np.sqrt(np.mean(np.diff(gg.v.to_numpy()) ** 2) / 2))
            return float(np.median(out_) * PCT) if out_ else np.nan
        r["repeat_nominal_strip"], r["repeat_nominal_image"] = rep(x), rep(yy)
        # within-packet changes: strip (new vs reference strip) against the image change (vs nominal in time)
        nom_int = np.full(len(d), np.nan)
        for a, gg in d.groupby("amp"):
            m = gg[gg.nominal].sort_values("hours")
            if len(m) >= 2:
                nom_int[gg.index] = np.interp(gg.hours, m.hours, m[f"log_{t}"])
        okc = ok & istrue("l1ref.usable").to_numpy() & (sref > 0) & (d.reference_id.to_numpy() == nom_theta) & ~d.nominal.to_numpy()
        ds = np.log(s[okc]) - np.log(sref[okc])
        di = y[okc] - nom_int[okc]
        clear = np.abs(di) > CLEAR[t]
        r["changes"] = {"n": int(okc.sum()), "n_clear": int(clear.sum()),
                        "direction_on_clear": float(np.mean(np.sign(ds[clear]) == np.sign(di[clear]))) if clear.any() else np.nan,
                        "pearson": float(np.corrcoef(ds, di)[0, 1]) if okc.sum() > 2 else np.nan}
        res["targets"][t] = r
        print(f"  strips {t}: r {r['pearson_log']:.2f}, after offset {r['rms_after_amp_offset']:.1f} %, "
              f"calibrated {r['rms_calibrated_cv']:.1f} %, changes dir {r['changes']['direction_on_clear']}", flush=True)
    L = ["# T3: can the level-1 strip stand in for the image?", "",
         "| target | rows | r (log) | error after amp offset % | calibrated (CV) % | image spread % | repeat at nominal strip / image % | change direction (clear) | n clear |",
         "|---|---|---|---|---|---|---|---|---|"]
    for t, r in res["targets"].items():
        L.append(f"| {t} | {r['n']} | {r['pearson_log']:.2f} | {r['rms_after_amp_offset']:.1f} | {r['rms_calibrated_cv']:.1f} | "
                 f"{r['image_spread']:.1f} | {r['repeat_nominal_strip']:.1f} / {r['repeat_nominal_image']:.1f} | "
                 f"{r['changes']['direction_on_clear']:.2f} | {r['changes']['n_clear']} |")
    L += ["", "Pass (plan): calibrated error at or below the image's own repeatability, and change direction >= 90 %."]
    write(out, "strips", res, "\n".join(L) + "\n")


# =========================================================================== T4 residual
def test_residual(d, args, out):
    from sklearn.model_selection import GroupKFold
    Z, names, timing = knobs(d)
    timing = np.array(timing)
    amps = d.amp.to_numpy()
    res = {"knobs": names, "timing_knobs": [n for n, t in zip(names, timing) if t], "targets": {}}
    A = pd.get_dummies(d.amp).to_numpy(float)
    folds = list(GroupKFold(n_splits=min(5, d.packet_id.nunique())).split(Z, groups=d.packet_id))

    def cv_pred(X, y, splits):
        p = np.full(len(y), np.nan)
        for tr, te in splits:
            hy = fit_shared_gp(X[tr], y[tr])
            p[te], _ = shared_predict(hy, X[tr], y[tr], X[te])
        return p

    blk = packet_blocks(d, args.blocks)
    loao = [(np.flatnonzero((amps != a) & (blk != b)), np.flatnonzero((amps == a) & (blk == b)))
            for a in sorted(set(amps)) for b in range(args.blocks) if ((amps == a) & (blk == b)).sum() >= 3]
    for t in LAB:
        y = d[f"log_{t}"].to_numpy()
        # amplifier offsets removed with the amp's mean (a calibration image), so the comparison is about the response
        ya = y - pd.Series(y).groupby(amps).transform("mean").to_numpy()
        v = np.var(ya)
        r = {"variance_pct2": float(v * PCT ** 2), "label_noise_pct": label_noise(d, t) * PCT}
        if timing.any():
            pt = cv_pred(Z[:, timing], ya, folds)
            r["R2_timing_only"] = float(1 - np.var(ya - pt) / v)
        else:
            pt = np.zeros_like(ya)
            r["R2_timing_only"] = 0.0
        pa = cv_pred(Z, ya, folds)
        r["R2_all_settings"] = float(1 - np.var(ya - pa) / v)
        resid = ya - pt
        if (~timing).any():
            pv = cv_pred(Z[:, ~timing], resid, folds)
            r["voltages_on_residual_R2_trained"] = float(1 - np.var(resid - pv) / np.var(resid))
            pvn = cv_pred(Z[:, ~timing], resid, loao)
            r["voltages_on_residual_R2_new_amp"] = float(1 - np.var(resid - pvn) / np.var(resid))
        r["rms_left_after_all_pct"] = float(np.std(ya - pa) * PCT)
        r["rms_residual_after_timing_pct"] = float(np.std(resid) * PCT)
        # per voltage knob: shared slope vs amplifier-specific spread (linear on the residual, per amplifier)
        slopes = {}
        V = Z[:, ~timing]
        vnames = [n for n, tt in zip(names, timing) if not tt]
        per = {}
        for a in sorted(set(amps)):
            m = amps == a
            X1 = np.column_stack([np.ones(m.sum()), V[m]])
            b, *_ = np.linalg.lstsq(X1, resid[m], rcond=None)
            s2 = np.sum((resid[m] - X1 @ b) ** 2) / max(1, m.sum() - X1.shape[1])
            cov = s2 * np.linalg.pinv(X1.T @ X1)
            per[a] = (b[1:], np.sqrt(np.clip(np.diag(cov)[1:], 0, None)))
        B = np.array([per[a][0] for a in per])
        E = np.array([per[a][1] for a in per])
        for j, n in enumerate(vnames):
            shared = float(np.mean(B[:, j]))
            spread = float(np.std(B[:, j], ddof=1)) if len(B) > 1 else np.nan
            noise = float(np.sqrt(np.mean(E[:, j] ** 2)))
            slopes[n] = {"shared_pct_per_sd": shared * PCT, "amp_spread_pct_per_sd": spread * PCT,
                         "slope_error_pct_per_sd": noise * PCT,
                         "amp_specific": bool(spread > 2 * noise) if np.isfinite(spread) else False}
        r["voltage_slopes"] = slopes
        res["targets"][t] = r
        print(f"  residual {t}: R2 timing {r['R2_timing_only']:.2f}, all {r['R2_all_settings']:.2f}, "
              f"left {r['rms_left_after_all_pct']:.1f} % (label noise {r['label_noise_pct']:.1f} %)", flush=True)
    L = ["# T4: the obvious part (timing) and what is left for ML", "",
         "Amplifier offsets removed first (the amplifier's mean; one calibration image), GP models, cross-validated by "
         "packet; 'new amp' = the amplifier left out AND the fleet trained on other time blocks. R2 = share of the "
         "remaining variance explained.", "",
         "| target | spread % | R2 timing only | R2 all settings | voltages on the timing residual: R2 trained / new amp | left after all % | label noise % |",
         "|---|---|---|---|---|---|---|"]
    for t, r in res["targets"].items():
        L.append(f"| {t} | {np.sqrt(r['variance_pct2']):.1f} | {r['R2_timing_only']:.2f} | {r['R2_all_settings']:.2f} | "
                 f"{r.get('voltages_on_residual_R2_trained', np.nan):.2f} / {r.get('voltages_on_residual_R2_new_amp', np.nan):.2f} | "
                 f"{r['rms_left_after_all_pct']:.1f} | {r['label_noise_pct']:.1f} |")
    for t, r in res["targets"].items():
        L += ["", f"## {t}: voltage effects on the timing residual (% per SD of the knob)", "",
              "| knob | shared | amplifier-to-amplifier spread | slope error | amplifier-specific? |", "|---|---|---|---|---|"]
        for n, s in sorted(r["voltage_slopes"].items(), key=lambda kv: -abs(kv[1]["shared_pct_per_sd"])):
            L.append(f"| {n} | {s['shared_pct_per_sd']:+.1f} | {s['amp_spread_pct_per_sd']:.1f} | {s['slope_error_pct_per_sd']:.1f} | "
                     f"{'yes' if s['amp_specific'] else ''} |")
    write(out, "residual", res, "\n".join(L) + "\n")


# =========================================================================== main
def write(out, name, res, md):
    (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o)))
    (out / f"{name}.md").write_text(md)
    print(f"  -> {out / name}.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("test", choices=["fewshot", "bo", "strips", "residual", "all"])
    ap.add_argument("--data", default=str(ROOT / "processed_v2"))
    ap.add_argument("--plan", nargs="+", default=[str(ROOT / "lvl1" / "packet_plan_v2" / "plan.json")])
    ap.add_argument("--sessions", nargs="+", default=["D", "E"])
    ap.add_argument("--board", type=int, default=101)
    ap.add_argument("--channels", type=int, nargs="+", default=[0, 1, 3])
    ap.add_argument("--drift", action="store_true", help="remove each amplifier's drift with its nominal packets")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=5, help="time blocks of packets (fewshot, residual)")
    ap.add_argument("--bo-blocks", type=int, default=2, help="time blocks of packets (bo): the candidate pool is one block")
    ap.add_argument("--weights", type=float, nargs="+", default=[1.0, 0.5, 0.0])
    ap.add_argument("--start", choices=["nominal", "none"], default="nominal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(args)
    print(f"{len(d)} rows, {d.packet_id.nunique()} packets, amplifiers {sorted(d.amp.unique())}, "
          f"{int(d.nominal.sum())} nominal rows, drift correction {'on' if args.drift else 'off'}")
    (out / "run_args.json").write_text(json.dumps(vars(args), indent=1))
    tests = ["fewshot", "bo", "strips", "residual"] if args.test == "all" else [args.test]
    for t in tests:
        t0 = time.time()
        print(f"{t} ...", flush=True)
        {"fewshot": test_fewshot, "bo": test_bo, "strips": test_strips, "residual": test_residual}[t](d, args, out)
        print(f"{t} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

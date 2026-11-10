"""
parameter_influence.py -- how much each operating parameter moves the noise (sigma_e), the dark charge (lambda) and
the gain: the second goal of the project (2026-11-10), next to the trace models.  Works on any processed data set
(v2 now, v3 when it is taken); every number comes with an uncertainty.

Per target (log units, so every effect is a % change), on board 101, label.ok, no fractional-window rows:

 A  EFFECTS      linear model per amplifier (log y ~ offset + sum_k beta_k z_k, timing knobs on a log scale,
                 z = the knob in units of its standard deviation over the data) and pooled (shared slopes, one offset
                 per amplifier); % per SD and % per unit (V, or per doubling for the timing), 68 % bootstrap intervals
                 (resampling packets).  -> effects_<target>.png, the tables
 B  IMPORTANCE   boosted trees on the knobs and the amplifier: cross-validated R^2 (grouped by packet), permutation
                 importance per knob (spread over the folds), partial dependence of the top knobs per amplifier.
                 -> importance_<target>.png, partial_<target>.png
 C  SENSITIVITY  variance-based (Sobol) indices over the design box, from the tree surrogate: the share of the
                 target's variance over the box due to each knob alone (first order) and with its interactions
                 (total order); total - first = interactions.  -> sobol_<target>.png
 D  ONE AT A TIME  model-free: packets that differ from nominal in ONE knob (v3's sweeps; found automatically) against
                 the nominal packets interpolated in time on the same amplifier; error from the nominal scatter.
                 -> oat_<target>.png
 E  AMPLIFIERS   per knob, the effect all amplifiers share against the amplifier-specific part (spread of the slopes
                 beyond their errors): what transfers to a new amplifier.  -> amplifiers_<target>.png
 F  DARK RATE vs CLOCK-INDUCED CHARGE (lambda only): lambda = t^alpha * exp(r_amp + sum r_k z_k) + exp(c_amp + sum
                 c_k z_k), t = the skip cycle time (src timing), over the voltage knobs: a voltage that scales with the
                 readout time changes the dark-current RATE, one that adds the same at every timing changes the
                 clock-induced charge.  Needs voltage points at several timings (v3: delayInt 500 / 1000 / 2500).
                 -> dc_vs_cic.png
 G  LEVEL vs SWING  when a clock's high AND low were varied (v3): the same effects with (low, swing = high - low).

Outputs in --out: influence.md (the ranked tables), influence.json, the figures.

    .venv/bin/python lvl1/parameter_influence.py --data processed_v2 --plan lvl1/packet_plan_v2/plan.json \\
        --sessions D E --out processed_v2/lvl1_results/parameter_influence
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
sys.path.insert(0, str(ROOT / "lvl1"))
sys.path.insert(0, str(ROOT / "src"))
import change_test as ct  # noqa: E402
from make_packet_plan import TIMING, seconds_per_skip  # noqa: E402

LAB = {"sigma_e": "label.sigma_e", "lambda": "label.lambda_e_per_bin", "gain": "label.gain_adu_per_e"}
#: --noise-components: the noise of the trace split into its parts (every trace measures them, with less noise than
#: one image's sigma_e): what each setting does to the white noise, the slow (1/f) noise and the bands
NOISE = {"white noise": "trace.fit: white noise per skip (ADU)",
         "1/f corner": "trace.noise: 1/f corner frequency (kHz)",
         "Allan dev n=50": "trace.noise: Allan deviation n=50 (ADU)",
         "power 0.1-1 MHz": "trace.skip: noise power 0.1-1 MHz",
         "power 1-3 MHz": "trace.skip: noise power 1-3 MHz",
         "power 3-7.5 MHz": "trace.skip: noise power 3-7.5 MHz",
         "slow/fast CDS power": "trace.noise: CDS power <1 kHz / >10 kHz"}
TIMING_KNOBS = ("delayInt", "delayRGH", "delay_Ha", "delay_Hb", "delay_Hc", "delayRGL", "delaySWH", "delaySWL",
                "delayOGL", "delayOGH", "delayDGL", "delayDGH", "delay_Va", "delay_Vb")
COL = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- data
def load(data: Path, plans: list[Path], sessions, channels, extra=None):
    d = pd.read_csv(data / "dataset.csv", low_memory=False)
    packets = [p for pl in plans for p in json.loads(Path(pl).read_text())["packets"]]
    session = {p["packet_id"]: str(p["session"]) for p in packets}
    purpose = {p["packet_id"]: str(p.get("purpose", "")) for p in packets}
    pid = [ct.planned_id(p, session) for p in d.packet_id]
    d["session"] = [session.get(p, "?") if p else "?" for p in pid]
    d["purpose"] = [purpose.get(p, "") if p else "" for p in pid]
    ok = (d.board == 101) & d.session.isin(sessions) & d.channel.isin(channels)
    ok &= d["label.ok"].astype(str).str.lower().eq("true")
    ok &= ~d["label.fractional_window"].astype(str).str.lower().eq("true")
    d = d[ok].reset_index(drop=True)
    cols = {**LAB, **(extra or {})}
    for t, c in cols.items():
        x = pd.to_numeric(d[c], errors="coerce").astype(float)
        d[f"log_{t}"] = np.log(x.where(x > 0))
    d = d[np.isfinite(d[[f"log_{t}" for t in cols]]).all(1)].reset_index(drop=True)
    ts = pd.to_datetime(d.date_start)
    d["hours"] = (ts - ts.min()).dt.total_seconds() / 3600
    d["nominal"] = d.purpose.str.startswith("nominal")
    return d


def knobs(d):
    """The varied settings; columns that always moved together (equal, or in lockstep: correlation > 0.999) become one
    knob (named after the first, +n): their effects cannot be told apart."""
    th = [c for c in d.columns if c.startswith("theta.") and d[c].nunique() > 1]
    groups = []
    for c in th:
        for g in groups:
            x, z = d[c].to_numpy(float), d[g[0]].to_numpy(float)
            if np.allclose(x, z, equal_nan=True) or abs(np.corrcoef(x, z)[0, 1]) > 0.999:   # moved in lockstep
                g.append(c)
                break
        else:
            groups.append([c])
    K, members, timing = {}, {}, {}
    for g in groups:
        short = [c[len("theta."):] for c in g]
        name = short[0] if len(short) == 1 else f"{short[0]} (+{len(short) - 1})"
        x = d[g[0]].to_numpy(float)
        is_t = short[0] in TIMING_KNOBS
        K[name] = np.log2(x) if is_t else x          # timing: per doubling
        members[name], timing[name] = short, is_t
    return pd.DataFrame(K), members, timing


def boot_packets(pk, n, seed=0):
    u = np.unique(pk)
    rng = np.random.default_rng(seed)
    idx = {p: np.flatnonzero(pk == p) for p in u}
    for _ in range(n):
        yield np.concatenate([idx[p] for p in rng.choice(u, len(u))])


def gbt():
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    return H(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8, random_state=0)


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b


# --------------------------------------------------------------------------- analyses
def effects(d, K, t, n_boot):
    """A: per-amplifier and pooled slopes, % per SD, with bootstrap intervals."""
    y = d[f"log_{t}"].to_numpy()
    Z = ((K - K.mean()) / K.std()).to_numpy()
    ch = d.channel.to_numpy()
    amps = np.unique(ch)
    onehot = (ch[:, None] == amps[None, :]).astype(float)
    pk = d.packet_id.to_numpy()

    def fit(idx):
        pooled = np.linalg.lstsq(np.column_stack([onehot[idx], Z[idx]]), y[idx], rcond=None)[0][len(amps):]
        per = np.full((len(amps), Z.shape[1]), np.nan)
        for i, a in enumerate(amps):
            m = idx[ch[idx] == a]
            if len(m) > Z.shape[1] + 3:
                per[i] = ols(Z[m], y[m])[1:]
        return pooled, per

    pooled, per = fit(np.arange(len(y)))
    bs = [fit(i) for i in boot_packets(pk, n_boot)]
    bp = np.array([b[0] for b in bs])
    ba = np.array([b[1] for b in bs])
    sd = K.std().to_numpy()
    return {"knobs": list(K.columns), "pooled": 100 * pooled, "pooled_lo": 100 * np.percentile(bp, 16, 0),
            "pooled_hi": 100 * np.percentile(bp, 84, 0), "per_amp": 100 * per, "per_amp_se": 100 * np.nanstd(ba, 0),
            "amps": amps, "sd": sd}


def importance(d, K, t):
    """B: trees, CV R^2 and permutation importance (grouped by packet)."""
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import GroupKFold
    y = d[f"log_{t}"].to_numpy()
    ch = d.channel.to_numpy()
    amps = np.unique(ch)
    X = np.hstack([K.to_numpy(), (ch[:, None] == amps[None, :]).astype(float)])
    pk = d.packet_id.to_numpy()
    vals, r2 = [], []
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(min(5, len(np.unique(pk)))).split(X, y, pk):
        m = gbt().fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
        r2.append(m.score(X[te], y[te]))
        vals.append(permutation_importance(m, X[te], y[te], n_repeats=5, random_state=0).importances_mean[:K.shape[1]])
    full = gbt().fit(X, y)
    return {"r2_cv": float(1 - np.mean((y - p) ** 2) / np.var(y)), "imp": np.mean(vals, 0), "imp_sd": np.std(vals, 0),
            "model": full, "amps": amps}


def sobol(K, imp, n=4096, seed=0):
    """C: Sobol first- and total-order indices of the tree surrogate over the box (each knob uniform over its
    observed range, independent), averaged over the amplifiers (Saltelli sampling, Jansen estimators)."""
    rng = np.random.default_rng(seed)
    lo, hi = K.min().to_numpy(), K.max().to_numpy()
    k = len(lo)
    A = lo + rng.random((n, k)) * (hi - lo)
    B = lo + rng.random((n, k)) * (hi - lo)
    S1, ST = np.zeros(k), np.zeros(k)
    amps = imp["amps"]
    for i, a in enumerate(amps):
        oh = np.zeros((n, len(amps)))
        oh[:, i] = 1
        f = lambda Z: imp["model"].predict(np.hstack([Z, oh]))
        fA, fB = f(A), f(B)
        var = np.var(np.concatenate([fA, fB]))
        for j in range(k):
            ABj = A.copy()
            ABj[:, j] = B[:, j]
            fABj = f(ABj)
            S1[j] += np.mean(fB * (fABj - fA)) / var / len(amps)
            ST[j] += 0.5 * np.mean((fA - fABj) ** 2) / var / len(amps)
    ST = np.clip(ST, 0, 1)
    return {"S1": np.minimum(np.clip(S1, 0, 1), ST), "ST": ST}     # first order <= total (estimator noise)


def one_at_a_time(d, K, t, tol_sd=0.1):
    """D: packets that differ from the nominal setting in exactly one knob, against the nominal packets interpolated in
    time (same amplifier).  Error: the RMS scatter of the nominal packets about that interpolation."""
    nom = d[d.nominal]
    if nom.empty:
        return pd.DataFrame()
    Kn = K.loc[nom.index].median()
    dev = (K - Kn).abs() / K.std()
    moved = dev > tol_sd
    one = moved.sum(1) == 1
    y = d[f"log_{t}"].to_numpy()
    base = np.full(len(d), np.nan)
    scat = {}
    for ch, g in d.groupby("channel"):
        n = g[g.nominal].sort_values("hours")
        if len(n) < 2:
            continue
        base[g.index] = np.interp(g.hours, n.hours, n[f"log_{t}"])
        loo = [n[f"log_{t}"].iloc[i] - np.interp(n.hours.iloc[i], n.hours.drop(n.index[i]), n[f"log_{t}"].drop(n.index[i]))
               for i in range(1, len(n) - 1)]
        scat[ch] = float(np.sqrt(np.mean(np.square(loo)))) if loo else np.nan
    rows = []
    for r in np.flatnonzero(one & ~d.nominal.to_numpy() & np.isfinite(base)):
        kn = moved.columns[moved.iloc[r].to_numpy()][0]
        rows.append({"knob": kn, "delta": float(K.iloc[r][kn] - Kn[kn]), "channel": int(d.channel.iloc[r]),
                     "packet": d.packet_id.iloc[r], "effect_pct": 100 * (y[r] - base[r]),
                     "err_pct": 100 * scat.get(d.channel.iloc[r], np.nan)})
    return pd.DataFrame(rows)


def dc_vs_cic(d, K, timing, n_boot=50):
    """F: lambda = t^alpha exp(r_a + sum r_k z_k) + exp(c_a + sum c_k z_k) over the voltage knobs; t = skip cycle."""
    from scipy.optimize import least_squares
    th = {k: d.get(f"theta.{k}", pd.Series(v, index=d.index)).fillna(v).to_numpy(float) for k, v in TIMING.items()}
    tcyc = np.array([seconds_per_skip({k: th[k][i] for k in TIMING}) for i in range(len(d))])
    tt = np.log(tcyc / np.median(tcyc))
    if np.std(tt) < 0.05:
        return None
    volts = [k for k in K.columns if not timing[k]]
    Z = ((K[volts] - K[volts].mean()) / K[volts].std()).to_numpy()
    lam = d["label.lambda_e_per_bin"].to_numpy(float)
    ch = d.channel.to_numpy()
    amps = np.unique(ch)
    ai = np.searchsorted(amps, ch)
    na, nv = len(amps), len(volts)

    def model(p, idx):
        alpha, r_a, c_a = p[0], p[1:1 + na], p[1 + na:1 + 2 * na]
        r_k, c_k = p[1 + 2 * na:1 + 2 * na + nv], p[1 + 2 * na + nv:]
        return (np.exp(alpha * tt[idx] + r_a[ai[idx]] + Z[idx] @ r_k) + np.exp(c_a[ai[idx]] + Z[idx] @ c_k))

    def fit(idx, p0):
        res = least_squares(lambda p: np.log(model(p, idx)) - np.log(lam[idx]) , p0, loss="soft_l1", f_scale=0.05,
                            max_nfev=4000)
        return res.x

    base = np.log(np.median(lam))
    p0 = np.concatenate([[0.5], np.full(na, base - 0.2), np.full(na, base - 2.0), np.zeros(2 * nv)])
    pk = d.packet_id.to_numpy()
    p = fit(np.arange(len(d)), p0)
    bs = np.array([fit(i, p) for i in boot_packets(pk, n_boot, seed=3)])
    r_k, c_k = p[1 + 2 * na:1 + 2 * na + nv], p[1 + 2 * na + nv:]
    # share of lambda that is clock-induced at nominal timing (tt = 0), z = 0, per amplifier
    cic_share = np.exp(p[1 + na:1 + 2 * na]) / (np.exp(p[1:1 + na]) + np.exp(p[1 + na:1 + 2 * na]))
    rs = bs[:, 1 + 2 * na:1 + 2 * na + nv]
    cs = bs[:, 1 + 2 * na + nv:]
    return {"voltages": volts, "alpha": float(p[0]), "alpha_se": float(np.std(bs[:, 0])),
            "rate_pct_per_sd": 100 * r_k, "rate_se": 100 * np.std(rs, 0), "cic_pct_per_sd": 100 * c_k,
            "cic_se": 100 * np.std(cs, 0), "cic_share_nominal": dict(zip(map(int, amps), cic_share.tolist())),
            "timing_spread": float(np.std(tt))}


def level_swing(d, K, members):
    """G: when the clock highs were varied (v3), re-express the V and H clocks as (low levels, swing): the swing of a
    family = mean over its pins of (high - low); the knobs made of highs are replaced by the swings.  None when no
    swing varies (v1, v2: the highs were fixed)."""
    swings = {}
    for fam in ("V", "H"):
        pins = [c[len("theta."):] for c in d.columns
                if c.startswith(f"theta.{fam}") and c.endswith("_H") and len(c) > 8 and c[7] in "123"
                and f"theta.{c[len('theta.'):-2]}_L" in d]
        if pins:
            sw = np.mean([d[f"theta.{p}"].to_numpy(float) - d[f"theta.{p[:-2]}_L"].to_numpy(float) for p in pins], 0)
            if np.std(sw) > 1e-6:
                swings[f"{fam} swing"] = sw
    if not swings:
        return None
    highs = [k for k, ps in members.items() if all(p[:1] in "VH" and p[1:2] in "123" and p.endswith("_H") for p in ps)]
    K2 = K.drop(columns=highs)
    for k, v in swings.items():
        K2[k] = v
    return K2


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(ROOT / "processed_v2"))
    ap.add_argument("--plan", nargs="+", default=[str(ROOT / "lvl1" / "packet_plan_v2" / "plan.json")])
    ap.add_argument("--sessions", nargs="+", default=["D", "E"])
    ap.add_argument("--channels", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--targets", nargs="+", default=list(LAB))
    ap.add_argument("--drop", nargs="*", default=[], help="packets left out")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--noise-components", action="store_true",
                    help="also the parts of the trace noise (NOISE: white, correlated floor, 1/f corner, bands)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import partial_dependence

    extra = NOISE if args.noise_components else None
    d = load(Path(args.data), args.plan, args.sessions, args.channels, extra)
    if extra:
        args.targets = list(args.targets) + [t for t in NOISE if t not in args.targets]
    d = d[~d.packet_id.isin(args.drop)].reset_index(drop=True)
    K, members, timing = knobs(d)
    names = list(K.columns)
    unit = {k: ("per doubling" if timing[k] else "per V") for k in names}
    res = {"n_rows": len(d), "n_packets": int(d.packet_id.nunique()), "sessions": args.sessions,
           "channels": args.channels, "knobs": members, "targets": {}}
    md = [f"# Parameter influence: {', '.join(args.targets)}", "",
          f"{len(d)} rows, {d.packet_id.nunique()} packets (sessions {'+'.join(args.sessions)}, board 101, channels "
          f"{args.channels}, label.ok, no fractional windows{', without ' + ', '.join(args.drop) if args.drop else ''}). "
          f"{len(names)} knobs (columns that always moved together are one knob). Timing knobs per doubling, voltages "
          "per volt; 'per SD' = per standard deviation of the knob over these data. Effects on log y, i.e. % changes. "
          "Method: lvl1/parameter_influence.py (docstring).", ""]
    print(md[2])
    for t in args.targets:
        print(f"--- {t}", flush=True)
        E = effects(d, K, t, args.n_boot)
        I = importance(d, K, t)
        S = sobol(K, I)
        O = one_at_a_time(d, K, t)
        rt = {"r2_cv_trees": I["r2_cv"], "knobs": {}}
        order = np.argsort(-S["ST"])
        for j in order:
            k = names[j]
            per = E["per_amp"][:, j]
            se = E["per_amp_se"][:, j]
            spread = float(np.sqrt(max(np.nanvar(per, ddof=1) - np.nanmean(se ** 2), 0))) if np.isfinite(per).sum() > 1 else np.nan
            rt["knobs"][k] = {"pct_per_sd": float(E["pooled"][j]), "pct_per_sd_68": [float(E["pooled_lo"][j]), float(E["pooled_hi"][j])],
                              "pct_per_unit": float(E["pooled"][j] / E["sd"][j]), "unit": unit[k],
                              "per_amp_pct_per_sd": dict(zip(map(int, E["amps"]), per.tolist())),
                              "amp_specific_pct_per_sd": spread, "perm_importance": float(I["imp"][j]),
                              "sobol_first": float(S["S1"][j]), "sobol_total": float(S["ST"][j]),
                              "range": [float(K[k].min()), float(K[k].max())]}
        # ---- tables
        md += [f"## {t}", "", f"Trees on the knobs and the amplifier: cross-validated R^2 = {I['r2_cv']:.2f}.", "",
               "| knob | % per SD (68 %) | % per unit | Sobol first / total | interactions | permutation imp. | "
               "amp-specific % per SD | per amplifier % per SD |", "|---|---|---|---|---|---|---|---|"]
        for k, v in rt["knobs"].items():
            pa = ", ".join(f"ch{a} {x:+.1f}" for a, x in v["per_amp_pct_per_sd"].items() if np.isfinite(x))
            md.append(f"| {k} | {v['pct_per_sd']:+.1f} ({v['pct_per_sd_68'][0]:+.1f}, {v['pct_per_sd_68'][1]:+.1f}) | "
                      f"{v['pct_per_unit']:+.2f} {v['unit']} | {v['sobol_first']:.2f} / {v['sobol_total']:.2f} | "
                      f"{max(v['sobol_total'] - v['sobol_first'], 0):.2f} | {v['perm_importance']:.3f} | "
                      f"{v['amp_specific_pct_per_sd']:.1f} | {pa} |")
        md.append("")
        if len(O):
            O["dir"] = np.sign(O.delta)
            g = O.groupby(["knob", "dir"]).agg(n=("effect_pct", "size"), delta=("delta", "mean"),
                                                          effect=("effect_pct", "mean"), err=("err_pct", "mean")).reset_index()
            rt["one_at_a_time"] = g.to_dict("records")
            md += ["One-at-a-time packets (one knob away from nominal; effect against the nominal packets interpolated "
                   "in time, same amplifier; error = nominal scatter / sqrt(n)):", "",
                   "| knob | direction | rows | mean delta | effect % | +- |", "|---|---|---|---|---|---|"]
            for _, r in g.iterrows():
                md.append(f"| {r.knob} | {'+' if r.dir > 0 else '-'} | {r.n} | {r.delta:+.3g} | {r.effect:+.1f} | "
                          f"{r.err / np.sqrt(r.n):.1f} |")
            md.append("")
        res["targets"][t] = rt

        # ---- figures
        top = [names[j] for j in order[:min(12, len(names))]]
        jj = [names.index(k) for k in top]
        fig, axes = plt.subplots(1, 3, figsize=(18, 0.45 * len(top) + 2.5), sharey=True)
        yy = np.arange(len(top))[::-1]
        axes[0].errorbar(E["pooled"][jj], yy, xerr=[E["pooled"][jj] - E["pooled_lo"][jj], E["pooled_hi"][jj] - E["pooled"][jj]],
                         fmt="ko", capsize=2, label="pooled (68 %)")
        for i, a in enumerate(E["amps"]):
            axes[0].plot(E["per_amp"][i, jj], yy + 0.12 * (i - 1.5) / 1.5, "o", ms=4, color=COL.get(int(a)), alpha=0.8, label=f"ch{a}")
        axes[0].axvline(0, color="k", lw=0.8)
        axes[0].set_xlabel(f"% change of {t} per 1 SD of the knob")
        axes[0].set_yticks(yy, top, fontsize=8)
        axes[0].legend(fontsize=7)
        axes[0].set_title(f"{t}: effect of each knob (linear)")
        axes[1].barh(yy - 0.2, S["S1"][jj], 0.4, label="first order (alone)", color="#4c72b0")
        axes[1].barh(yy + 0.2, S["ST"][jj], 0.4, label="total (with interactions)", color="#c44e52")
        axes[1].set_xlabel("share of the variance over the box (Sobol)")
        axes[1].legend(fontsize=7)
        axes[1].set_title(f"variance share (tree surrogate, CV R$^2$ {I['r2_cv']:.2f})")
        axes[2].barh(yy, I["imp"][jj], xerr=I["imp_sd"][jj], color="#55a868")
        axes[2].set_xlabel("permutation importance (drop in R$^2$)")
        axes[2].set_title("what the tree model relies on")
        for ax in axes:
            ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(out / f"influence_{t.replace('/', '_').replace(' ', '_')}.png", dpi=130)
        plt.close(fig)
        # partial dependence of the top 4, per amplifier
        top4 = top[:4]
        fig, axes = plt.subplots(1, len(top4), figsize=(4.2 * len(top4), 3.6))
        X = np.hstack([K.to_numpy(), (d.channel.to_numpy()[:, None] == I["amps"][None, :]).astype(float)])
        for ax, k in zip(np.atleast_1d(axes), top4):
            j = names.index(k)
            for i, a in enumerate(I["amps"]):
                Xa = X[d.channel.to_numpy() == a]
                pdp = partial_dependence(I["model"], Xa, [j], grid_resolution=25, kind="average")
                gx, gy = pdp["grid_values"][0], pdp["average"][0]
                ax.plot(2 ** gx if timing[k] else gx, 100 * (gy - gy.mean()), "-", color=COL.get(int(a)), label=f"ch{a}")
            if timing[k]:
                ax.set_xscale("log", base=2)
            ax.set_xlabel(k)
            ax.set_ylabel(f"{t}, % (partial dependence)")
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"partial_{t.replace('/', '_').replace(' ', '_')}.png", dpi=130)
        plt.close(fig)
        if len(O):
            fig, ax = plt.subplots(figsize=(10, 0.35 * O.knob.nunique() * 2 + 2))
            for i, (k, g) in enumerate(O.groupby("knob")):
                for _, r in g.iterrows():
                    ax.errorbar(r.effect_pct, i + (0.15 if r.delta > 0 else -0.15), xerr=r.err_pct, fmt="o",
                                color=COL.get(r.channel), ms=4, alpha=0.7)
            ax.set_yticks(range(O.knob.nunique()), sorted(O.knob.unique()), fontsize=8)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel(f"{t} against the nominal packets, % (upper: knob raised, lower: lowered)")
            ax.set_title(f"{t}: one-at-a-time packets")
            ax.grid(alpha=0.3, axis="x")
            fig.tight_layout()
            fig.savefig(out / f"oat_{t.replace('/', '_').replace(' ', '_')}.png", dpi=130)
            plt.close(fig)

    # ---- F: dark rate vs clock-induced charge
    if "lambda" in args.targets:
        F = dc_vs_cic(d, K, timing)
        if F:
            res["dc_vs_cic"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in F.items()}
            md += ["## lambda: dark-current rate against clock-induced charge", "",
                   f"lambda = t^alpha exp(rate terms) + exp(CIC terms), t = the skip cycle time; alpha = "
                   f"{F['alpha']:.2f} +- {F['alpha_se']:.2f}. Share of lambda that is clock-induced at nominal: "
                   + ", ".join(f"ch{a} {100 * s:.0f} %" for a, s in F["cic_share_nominal"].items()) + ". "
                   "Only meaningful when the voltages were varied at several timings (v3); with v2 the two terms are "
                   "poorly separated.", "", "| voltage | dark rate % per SD | CIC % per SD |", "|---|---|---|"]
            o = np.argsort(-np.abs(F["rate_pct_per_sd"]) - np.abs(F["cic_pct_per_sd"]))
            for j in o:
                md.append(f"| {F['voltages'][j]} | {F['rate_pct_per_sd'][j]:+.1f} +- {F['rate_se'][j]:.1f} | "
                          f"{F['cic_pct_per_sd'][j]:+.1f} +- {F['cic_se'][j]:.1f} |")
            md.append("")
            fig, ax = plt.subplots(figsize=(12, 4.5))
            x = np.arange(len(F["voltages"]))
            ax.bar(x - 0.2, F["rate_pct_per_sd"], 0.4, yerr=F["rate_se"], label="dark-current rate (scales with time)", color="#4c72b0")
            ax.bar(x + 0.2, F["cic_pct_per_sd"], 0.4, yerr=F["cic_se"], label="clock-induced charge (per transfer)", color="#c44e52")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(x, F["voltages"], rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("% change of the term per 1 SD")
            ax.set_title(f"lambda split into rate and clock-induced charge (alpha = {F['alpha']:.2f})")
            ax.legend()
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(out / "dc_vs_cic.png", dpi=130)
            plt.close(fig)

    # ---- G: level vs swing
    K2 = level_swing(d, K, members)
    if K2 is not None:
        md += ["## Level against swing (clocks with both rails varied)", ""]
        for t in args.targets:
            E2 = effects(d, K2, t, max(50, args.n_boot // 4))
            md += [f"{t}: " + ", ".join(f"{k} {v:+.1f} %/SD" for k, v in zip(E2["knobs"], E2["pooled"])
                                        if "swing" in k or "_L" in k), ""]

    # ---- the ranking across targets
    md += ["## Ranking (Sobol total-order share of the variance over the box)", "",
           "| knob | " + " | ".join(args.targets) + " |", "|---|" + "---|" * len(args.targets)]
    for k in sorted(names, key=lambda k: -max(res["targets"][t]["knobs"][k]["sobol_total"] for t in args.targets)):
        md.append(f"| {k} | " + " | ".join(f"{res['targets'][t]['knobs'][k]['sobol_total']:.2f}" for t in args.targets) + " |")
    (out / "influence.md").write_text("\n".join(md) + "\n")
    (out / "influence.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("\n".join(md[-(len(names) + 3):]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

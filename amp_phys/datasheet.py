"""
datasheet.py -- phase 1 of amp_phys/TEST_PLAN.md: an electrical "datasheet" per amplifier from processed data.

Per amplifier (board, channel), every quantity with a bootstrap error (resampling packets):
  A  NOMINAL values (the packets at nominal: board 101's nominal packets, every board-102 packet): gain, C_sn relative to
     the board-101 mean (~1/gain), sigma_e, lambda (dark / CIC / overscan / active), noise in e- (white, null-CDS,
     Allan, 1/f), RTS / discreteness, clock coupling in e- per volt (feedthrough / gain), bandwidth (rise time,
     recovery, ringing), charge transfer, mains lines.
  B  STABILITY over the nominal packets: scatter (%) and drift (%/h) of gain, white noise, lambda, sigma_e.
  C  RESPONSES (board 101, the v3 voltage scan): per amplifier, every output regressed on all voltages at once (in
     volts) -> % per volt of the physically meaningful ones (gain vs VDD/VR: source-follower operating point; white
     noise vs VDD; feedthrough vs its own clock: a check that the coupling is capacitive).
  D  KNEES: hinge fits (y = a + b x + c max(0, x - k) + the other voltages linearly) of lambda vs VSUB (depletion onset)
     and sigma_e vs VR (reset-drain headroom), with the knee voltage k and its error.
  E  FUNCTIONAL check: a simple automatic rule that flags channels with no amplifier signal (no charged pixels in the
     trace, lambda at the fit floor).
  F  OUTLIER score per quantity: (x - mean of the other working amplifiers) / sqrt(err^2 + spread of the others^2).

    .venv/bin/python amp_phys/datasheet.py --data processed_v3n --plan lvl1/packet_plan_v3n/plan.json \\
        --out amp_phys/results/v3n
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lvl1"))
import change_test as ct  # noqa: E402

warnings.filterwarnings("ignore")
N_BOOT = 300
RNG = np.random.default_rng(0)

# (name, column, scale) ; scale: None = as is; "gain" = divide by the gain (ADU -> e-)
NOMINAL = {
    "gain": [("gain (ADU/e)", "label.gain_adu_per_e", None),
             ("C_sn relative to board-101 mean", "label.gain_adu_per_e", "csn")],
    "image": [("sigma_e (e)", "label.sigma_e", None),
              ("lambda (e/pix)", "label.lambda_e_per_bin", None),
              ("lambda dark", "label.lambda_dark", None),
              ("lambda CIC", "label.lambda_cic", None),
              ("lambda overscan", "label.lambda_overscan", None),
              ("lambda active", "label.lambda_active", None)],
    "noise": [("white noise per skip (ADU)", "trace.fit: white noise per skip (ADU)", None),
              ("null-CDS noise (ADU)", "trace.noise: null-CDS noise (ADU)", None),
              ("white noise per skip (e)", "trace.fit: white noise per skip (ADU)", "gain"),
              ("null-CDS noise (e)", "trace.noise: null-CDS noise (ADU)", "gain"),
              ("jump-robust CDS sigma (e)", "trace.noise: jump-robust CDS sigma (ADU)", "gain"),
              ("correlated noise floor (e)", "trace.fit: correlated noise floor (ADU)", "gain"),
              ("Allan dev n=50 (e)", "trace.noise: Allan deviation n=50 (ADU)", "gain"),
              ("noise of 50-skip average (e)", "trace.skip: noise of 50-skip average", "gain"),
              ("1/f corner (kHz)", "trace.noise: 1/f corner frequency (kHz)", None),
              ("1/f exponent", "trace.noise: 1/f exponent", None),
              ("Lorentzian corner (kHz)", "trace.noise: Lorentzian corner (kHz)", None),
              ("slow/fast CDS power", "trace.noise: CDS power <1 kHz / >10 kHz", None)],
    "RTS / traps (amplifier)": [("CDS excess kurtosis", "trace.discrete: CDS excess kurtosis", None),
                                ("whitened CDS kurtosis", "trace.discrete: whitened CDS kurtosis", None),
                                ("in-window step rate (%)", "trace.discrete: in-window step rate (%)", None),
                                ("in-window step amplitude (e)", "trace.discrete: in-window step amplitude (ADU)", "gain"),
                                ("jump share of CDS variance", "trace.discrete: jump share of CDS variance", None),
                                ("largest CDS step (t)", "trace.discrete: largest CDS step (t)", None)],
    "coupling (e/V)": [("SW -> sense node (e/V)", "trace.edge: dip per volt SW (ADU/V)", "gain"),
                       ("RG -> sense node (e/V)", "trace.edge: pulse per volt RG (ADU/V)", "gain"),
                       ("H -> first skip (e/V)", "trace.start: first-skip excess per volt H (ADU/V)", "gain"),
                       ("pulse/dip ratio (RG/SW)", "trace.edge: pulse/dip ratio", None),
                       ("H-transfer peak-to-peak (e)", "trace.slow: H-transfer peak-to-peak (ADU)", "gain"),
                       ("pre-pulse dip / pulse", "trace.edge: pre-pulse dip / pulse height", None),
                       ("baseline coupling (1e-3)", "trace.noise: baseline coupling (1e-3)", None)],
    "bandwidth": [("rise time (ns)", "trace.edge: rise time (ns)", None),
                  ("recovery tau1 (us)", "trace.edge: recovery tau1 (us)", None),
                  ("recovery tau2 (us)", "trace.edge: recovery tau2 (us)", None),
                  ("ringing frequency (MHz)", "trace.edge: ringing frequency (MHz)", None),
                  ("ringing rms (e)", "trace.edge: ringing rms (ADU)", "gain"),
                  ("settling time after pulse (us)", "trace.fit: settling time after pulse (us)", None),
                  ("dip recovery time (us)", "trace.fit: dip recovery time (us)", None),
                  ("edge jitter (ns)", "trace.edge: jitter (ns)", None)],
    "charge transfer": [("charge-loss slope (%)", "trace.level: charge-loss slope (%)", None),
                        ("charged-pixel skewness", "trace.level: charged-pixel skewness", None)],
    "lines / grounding": [("line count", "trace.discrete: line count (F-test, p<0.001)", None),
                          ("strongest line (kHz)", "trace.discrete: strongest line frequency (kHz)", None),
                          ("strongest line amplitude (e)", "trace.discrete: strongest line amplitude (ADU)", "gain"),
                          ("line phase coherence, pixel clock", "trace.discrete: line phase coherence, pixel clock", None)],
}
STABILITY = {"gain": "label.gain_adu_per_e", "sigma_e": "label.sigma_e", "lambda": "label.lambda_e_per_bin",
             "white noise": "trace.fit: white noise per skip (ADU)"}
# responses: (name, output column, knob, in_log)
RESPONSES = [("gain vs VDD (%/V)", "label.gain_adu_per_e", "theta.VDD1", True),
             ("gain vs VR (%/V)", "label.gain_adu_per_e", "theta.VR1", True),
             ("white noise (ADU) vs VDD (%/V)", "trace.fit: white noise per skip (ADU)", "theta.VDD1", True),
             ("rise time vs VDD (%/V)", "trace.edge: rise time (ns)", "theta.VDD1", True),
             ("recovery tau1 vs VDD (%/V)", "trace.edge: recovery tau1 (us)", "theta.VDD1", True),
             ("SW feedthrough vs SW low (%/V)", "trace.edge: dip per volt SW (ADU/V)", "theta.SW_1_L", True),
             ("RG feedthrough vs RG high (%/V)", "trace.edge: pulse per volt RG (ADU/V)", "theta.RG_1_H", True),
             ("lambda vs VSUB (%/V)", "label.lambda_e_per_bin", "theta.VSUB", True),
             ("lambda active vs VSUB (%/V)", "label.lambda_active", "theta.VSUB", True),
             ("lambda overscan vs VSUB (%/V)", "label.lambda_overscan", "theta.VSUB", True),
             ("lambda dark vs VSUB (%/V)", "label.lambda_dark", "theta.VSUB", True),
             ("lambda CIC vs VSUB (%/V)", "label.lambda_cic", "theta.VSUB", True),
             ("sigma_e vs VSUB (%/V)", "label.sigma_e", "theta.VSUB", True),
             ("white noise (ADU) vs VSUB (%/V)", "trace.fit: white noise per skip (ADU)", "theta.VSUB", True),
             ("CIC vs H swing (%/V)", "label.lambda_cic", "Hswing", True),
             ("CIC vs V swing (%/V)", "label.lambda_cic", "Vswing", True),
             ("1/f corner vs H swing (%/V)", "trace.noise: 1/f corner frequency (kHz)", "Hswing", True)]
KNEES = [("lambda knee in VSUB (V)", "label.lambda_e_per_bin", "theta.VSUB", np.arange(42, 68.1, 0.5)),
         ("sigma_e knee in VR (V)", "label.sigma_e", "theta.VR1", np.arange(-3.6, -2.25, 0.05))]


def num(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def load(data: Path, plan: Path):
    d = pd.read_csv(data / "dataset.csv", low_memory=False)
    packets = json.loads(plan.read_text())["packets"]
    session = {p["packet_id"]: str(p["session"]) for p in packets}
    purpose = {p["packet_id"]: str(p.get("purpose", "")) for p in packets}
    pid = [ct.planned_id(p, session) for p in d.packet_id]
    d["session"] = [session.get(p, "?") if p else "?" for p in pid]
    d["purpose"] = [purpose.get(p, "") if p else "" for p in pid]
    d["ok"] = d["label.ok"].astype(str).str.lower().eq("true")
    ts = pd.to_datetime(d.date_start)
    d["hours"] = (ts - ts.min()).dt.total_seconds() / 3600
    d["amp"] = d.board.astype(str) + " ch" + d.channel.astype(str)
    hs = [c for c in d.columns if c.startswith("theta.H") and c.endswith(("_H", "_L"))]
    vs = [c for c in d.columns if c.startswith("theta.V") and c[7:8].isdigit() and c.endswith(("_H", "_L"))]
    def swing(cols):
        his = [c for c in cols if c.endswith("_H")]
        return np.mean([num(d[c]) - num(d[c[:-2] + "_L"]) for c in his if c[:-2] + "_L" in d], axis=0)
    d["Hswing"], d["Vswing"] = swing(hs), swing(vs)
    return d


def functional(d: pd.DataFrame) -> pd.DataFrame:
    """automatic rule for "is there an amplifier on this channel": found from the data (v3n, 2026-11-16) by comparing the
    three dead board-102 channels with the working ones: the signal-window settling (the output's response to the
    summing-well / output-gate edge, which needs a live sense node) is ~82-84 ADU on every dead channel against 150-202
    on every working one.  Rule: works if its median is above 0.7 x the median over all channels of board 101 (the
    reference board).  Also listed: the row-start / inner first-skip ratio (0.62-0.64 on board 101) and the label error."""
    col = "trace.noise: signal-window settling (ADU)"
    x = num(d[col])
    ref = np.nanmedian(x[d.board.eq(101)])
    rows = []
    for amp, g in d.groupby("amp"):
        m = float(np.nanmedian(num(g[col])))
        rows.append({"amp": amp, "rows": len(g), "label ok": int(g.ok.sum()),
                     "signal-window settling (ADU)": round(m, 1), "ratio to board-101 median": round(m / ref, 2),
                     "row-start/inner first-skip ratio": round(float(np.nanmedian(num(g["trace.start: row-start/inner first-skip ratio"]))), 2),
                     "median lambda": round(float(np.nanmedian(num(g["label.lambda_e_per_bin"]))), 4),
                     "working (rule)": bool(m > 0.7 * ref)})
    return pd.DataFrame(rows).set_index("amp")


def boot(vals: np.ndarray, pk: np.ndarray, fn, n=N_BOOT):
    u = np.unique(pk)
    est = fn(vals)
    bs = []
    idx = {p: np.flatnonzero(pk == p) for p in u}
    for _ in range(n):
        s = np.concatenate([idx[p] for p in RNG.choice(u, len(u))])
        bs.append(fn(vals[s]))
    return est, float(np.nanstd(bs))


def nominal_values(d, amps, nominal_mask):
    gain_ref = np.nanmedian(num(d.loc[nominal_mask & d.board.eq(101) & d.channel.isin([0, 1, 3]), "label.gain_adu_per_e"]))
    rows = []
    for group, items in NOMINAL.items():
        for name, col, scale in items:
            if col not in d:
                continue
            r = {"group": group, "quantity": name}
            for amp in amps:
                g = d[nominal_mask & d.amp.eq(amp)]
                x = num(g[col]).to_numpy()
                gn = num(g["label.gain_adu_per_e"]).to_numpy()
                if scale == "gain":
                    x = x / gn
                elif scale == "csn":
                    x = gain_ref / x
                ok = np.isfinite(x)
                if ok.sum() < 3:
                    r[amp], r[amp + " err"] = np.nan, np.nan
                    continue
                v, e = boot(x[ok], g.packet_id.to_numpy()[ok], np.nanmedian)
                r[amp], r[amp + " err"] = v, e
            rows.append(r)
    return pd.DataFrame(rows)


def stability(d, amps, nominal_mask):
    rows = []
    for name, col in STABILITY.items():
        for kind in ("scatter (%)", "drift (%/h)"):
            r = {"group": "stability", "quantity": f"{name} {kind}"}
            for amp in amps:
                g = d[nominal_mask & d.amp.eq(amp)]
                y = np.log(num(g[col]).to_numpy())
                h = g.hours.to_numpy()
                ok = np.isfinite(y) & np.isfinite(h)
                if ok.sum() < 5:
                    r[amp], r[amp + " err"] = np.nan, np.nan
                    continue
                y, h, pk = y[ok], h[ok], g.packet_id.to_numpy()[ok]
                if kind.startswith("scatter"):
                    # scatter after removing a linear drift
                    fn = lambda idx, y=y, h=h: 100 * np.std(y[idx] - np.polyval(np.polyfit(h[idx], y[idx], 1), h[idx]), ddof=2)
                else:
                    fn = lambda idx, y=y, h=h: 100 * np.polyfit(h[idx], y[idx], 1)[0]
                v, e = boot(np.arange(len(y)), pk, fn)
                r[amp], r[amp + " err"] = v, e
            rows.append(r)
    return pd.DataFrame(rows)


def knob_matrix(g):
    th = [c for c in g.columns if c.startswith("theta.") and num(g[c]).nunique() > 1]
    keep, seen = [], []
    for c in th:                                        # one column per group of lockstep knobs
        x = num(g[c]).to_numpy()
        if any(np.allclose(x, s) for s in seen):
            continue
        seen.append(x)
        keep.append(c)
    return keep


def responses(d, amps_scan, scan_mask):
    rows = []
    for name, col, knob, in_log in RESPONSES:
        r = {"group": "responses (v3 scan)", "quantity": name}
        for amp in amps_scan:
            g = d[scan_mask & d.amp.eq(amp)]
            ks = knob_matrix(g)
            if knob not in ks and knob not in ("Hswing", "Vswing"):
                r[amp], r[amp + " err"] = np.nan, np.nan
                continue
            cols = ks + ([knob] if knob in ("Hswing", "Vswing") else [])
            X = np.column_stack([num(g[c]) for c in cols] + [np.ones(len(g))])
            y = num(g[col]).to_numpy()
            y = np.log(np.where(y > 0, y, np.nan)) if in_log else y
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            j = cols.index(knob)
            Xo, yo, pk = X[ok], y[ok], g.packet_id.to_numpy()[ok]
            if len(yo) < len(cols) + 10:
                r[amp], r[amp + " err"] = np.nan, np.nan
                continue
            fn = lambda idx, Xo=Xo, yo=yo, j=j: 100 * np.linalg.lstsq(Xo[idx], yo[idx], rcond=None)[0][j]
            v, e = boot(np.arange(len(yo)), pk, fn, n=200)
            r[amp], r[amp + " err"] = v, e
        rows.append(r)
    return pd.DataFrame(rows)


def hinge_fit(X, x, y, grid):
    best = (np.inf, np.nan)
    for k in grid:
        A = np.column_stack([X, np.maximum(0, x - k)])
        res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        s = float(res @ res)
        if s < best[0]:
            best = (s, k)
    return best[1]


def knees(d, amps_scan, scan_mask, out: Path):
    rows = []
    fig, axs = plt.subplots(1, len(KNEES), figsize=(6 * len(KNEES), 4))
    cols_amp = {"101 ch0": "#2a78d6", "101 ch1": "#eb6834", "101 ch2": "#8a8984", "101 ch3": "#1baf7a"}
    for ax, (name, col, knob, grid) in zip(np.atleast_1d(axs), KNEES):
        r = {"group": "knees (v3 scan)", "quantity": name}
        for amp in amps_scan:
            g = d[scan_mask & d.amp.eq(amp)]
            ks = [c for c in knob_matrix(g) if c != knob]
            y = np.log(num(g[col]).to_numpy())
            x = num(g[knob]).to_numpy()
            X = np.column_stack([num(g[c]) for c in ks] + [x, np.ones(len(g))])
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            Xo, xo, yo, pk = X[ok], x[ok], y[ok], g.packet_id.to_numpy()[ok]
            if len(yo) < X.shape[1] + 10:
                r[amp], r[amp + " err"] = np.nan, np.nan
                continue
            fn = lambda idx: hinge_fit(Xo[idx], xo[idx], yo[idx], grid)
            v, e = boot(np.arange(len(yo)), pk, fn, n=100)
            r[amp], r[amp + " err"] = v, e
            # partial residual plot: y minus the other voltages' linear part, against the knob
            beta = np.linalg.lstsq(Xo, yo, rcond=None)[0]
            part = yo - Xo[:, :-2] @ beta[:-2]
            ax.scatter(xo, part - np.median(part), s=10, color=cols_amp.get(amp, "#6250d6"), label=f"{amp}: knee {v:.2f} ± {e:.2f}")
            ax.axvline(v, color=cols_amp.get(amp, "#6250d6"), lw=1, ls="--")
        ax.set_xlabel(knob.replace("theta.", "") + " (V)")
        ax.set_ylabel(f"log {col.split('.')[-1]}, other voltages removed")
        ax.set_title(name, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        rows.append(r)
    fig.tight_layout()
    fig.savefig(out / "knees.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


def outliers(T: pd.DataFrame, working: list[str]) -> pd.DataFrame:
    Z = pd.DataFrame(index=T.index, columns=working, dtype=float)
    for i, r in T.iterrows():
        for a in working:
            others = [b for b in working if b != a and np.isfinite(r.get(b, np.nan))]
            if not np.isfinite(r.get(a, np.nan)) or len(others) < 2:
                continue
            xo = np.array([r[b] for b in others])
            eo = np.array([r[b + " err"] for b in others])
            spread = np.sqrt(max(np.var(xo, ddof=1) - np.nanmean(eo ** 2), 0) + np.nanmean(eo ** 2) / len(others))
            Z.loc[i, a] = (r[a] - xo.mean()) / np.sqrt(r[a + " err"] ** 2 + spread ** 2)
    return Z


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--scan-session", default="G", help="the voltage-scan session (responses, knees)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(Path(args.data), Path(args.plan))
    F = functional(d)
    print(F.to_string())
    amps = sorted(d.amp.unique())
    working = [a for a in amps if F.loc[a, "working (rule)"]]
    # nominal rows: board 101 nominal packets; board 102 (never moved in v3): every packet
    nominal = (d.board.eq(101) & d.purpose.str.startswith("nominal")) | d.board.eq(102)
    nominal &= d.ok | d.board.eq(102)
    scan = d.board.eq(101) & d.session.eq(args.scan_session) & d.ok & (num(d["label.lambda_e_per_bin"]) > 0.005)
    scan_amps = [a for a in amps if a.startswith("101")]
    T = pd.concat([nominal_values(d, amps, nominal), stability(d, amps, nominal),
                   responses(d, scan_amps, scan), knees(d, scan_amps, scan, out)], ignore_index=True)
    T.index = T.group + " | " + T.quantity
    T.drop(columns=["group", "quantity"]).round(5).to_csv(out / "datasheet.csv")
    Z = outliers(T, [a for a in working])
    Z.round(2).to_csv(out / "outlier_z.csv")
    F.to_csv(out / "functional.csv")

    # markdown
    L = ["# Amplifier datasheet (phase 1)", "", f"Data: {args.data}; nominal rows = board-101 nominal packets + every "
         "board-102 packet; responses and knees from the board-101 voltage scan (session "
         f"{args.scan_session}). value ± bootstrap error (packets).", "", "## Which amplifiers work (automatic rule)", "",
         F.to_markdown(), "", "## Datasheet", ""]
    for grp in T.index.str.split(" \\| ").str[0].unique():
        sub = T[T.index.str.startswith(grp + " |")]
        L += [f"### {grp}", "", "| quantity | " + " | ".join(amps) + " |", "|---|" + "---|" * len(amps)]
        for i, r in sub.iterrows():
            cells = []
            for a in amps:
                v, e = r.get(a, np.nan), r.get(a + " err", np.nan)
                if not np.isfinite(v):
                    cells.append("–")
                    continue
                z = Z.loc[i, a] if a in Z.columns else np.nan
                mark = " **⚠**" if np.isfinite(z) and abs(z) > 3 else ""
                fmt = ".3g"
                cells.append(f"{v:{fmt}} ± {e:.2g}{mark}" + ("" if a in working else " (dead)"))
            L.append(f"| {i.split(' | ', 1)[1]} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Largest outliers per working amplifier (|z| > 3: the amplifier differs from the others by more than "
          "3 combined errors)", ""]
    for a in Z.columns:
        z = Z[a].dropna()
        top = z.reindex(z.abs().sort_values(ascending=False).index)[:8]
        L.append(f"- **{a}**: " + "; ".join(f"{k.split(' | ', 1)[1]} z={v:+.1f}" for k, v in top.items() if abs(v) > 3)
                 or f"- **{a}**: none above 3")
    (out / "datasheet.md").write_text("\n".join(L) + "\n")

    # outlier heatmap
    Zp = Z.copy()
    Zp.index = [i.split(" | ", 1)[1] for i in Zp.index]
    fig, ax = plt.subplots(figsize=(1.2 * len(Zp.columns) + 4, 0.22 * len(Zp) + 1.5))
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("d", ["#1c5cab", "#2a78d6", "#f0efec", "#e34948", "#b02a2a"])
    ax.imshow(np.clip(Zp.to_numpy(float), -6, 6), cmap=cm, vmin=-6, vmax=6, aspect="auto")
    for i in range(Zp.shape[0]):
        for j in range(Zp.shape[1]):
            v = Zp.iat[i, j]
            if np.isfinite(v) and abs(v) >= 2:
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 4 else "#0b0b0b")
    ax.set_xticks(range(Zp.shape[1]), Zp.columns)
    ax.set_yticks(range(Zp.shape[0]), Zp.index, fontsize=6.5)
    ax.set_title("How far each working amplifier sits from the others (z, combined errors; |z|>3 = outlier)", loc="left", fontsize=9)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out / "outliers.png", dpi=150)
    plt.close(fig)
    print("\n".join(L[-len(Z.columns) - 2:]))


if __name__ == "__main__":
    main()

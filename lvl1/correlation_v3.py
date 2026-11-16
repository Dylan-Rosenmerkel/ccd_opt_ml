"""
correlation_v3.py -- the full correlation picture of a voltage-only data set (v3n): every voltage knob, the three image
labels (sigma_e, lambda, gain) and the noise components of the trace, against each other.

Figures (in --out):
  corr_full.png        all quantities x all quantities, Pearson, within amplifier (each quantity minus its amplifier
                       mean, so the amplifier-to-amplifier offsets do not count); outputs in log units.
  corr_full_spearman.png  the same, rank correlation (robust to the steep VSUB knee and to outliers).
  corr_knobs_outputs.png  knobs (rows) x outputs (columns): plain correlation | partial correlation (each knob with the
                       other knobs and the amplifier held fixed) side by side; the partial one is the "this knob alone".
  corr_per_amp.png     knobs x outputs, one panel per amplifier: does every amplifier respond the same way?
  corr_design.png      knobs x knobs: did the plan vary them independently? (should be near 0 off the diagonal)
Tables: corr_full.csv, corr_partial.csv, corr_per_amp.csv, correlations.md (top correlations per output).

Significance: the rows of one packet share a setting, so the effective sample size is the number of packets
(n_eff); a cell is drawn faded when |r| is below the 95 % threshold for n_eff (|r| < 1.96 / sqrt(n_eff)).

    .venv/bin/python lvl1/correlation_v3.py --data processed_v3n --plan lvl1/packet_plan_v3n/plan.json \\
        --sessions G --out processed_v3n/lvl1_results/correlations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parameter_influence as pi  # noqa: E402

# diverging: blue (negative) - neutral gray - red (positive)
CMAP = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#2a78d6", "#f0efec", "#e34948", "#b02a2a"])
INK, MUTED = "#0b0b0b", "#8a8984"
PRETTY = {"SW_1_L": "SW low", "RG_1_H": "RG high", "RG_1_L": "RG low", "V1_B_H": "V1 high", "V1_B_L": "V1 low",
          "V2_C_H": "V2 high", "V2_C_L": "V2 low", "V3_B_H": "V3 high", "V3_B_L": "V3 low", "H1_B_H": "H1 high",
          "H1_B_L": "H1 low", "H2_C_H": "H2 high", "H2_C_L": "H2 low", "H3_B_H": "H3 high", "H3_B_L": "H3 low",
          "VSUB": "VSUB", "VDD1": "VDD", "VR1": "VR"}
OUT_NICE = {"sigma_e": "σe (noise)", "lambda": "λ (dark charge)", "gain": "gain"}


def nice_knob(k: str) -> str:
    return PRETTY.get(k.split(" (")[0], k)


def partial_corr(X: np.ndarray, y: np.ndarray, C: np.ndarray) -> np.ndarray:
    """partial correlation of each column of X with y, holding the other columns of X and C fixed"""
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        Z = np.column_stack([np.delete(X, j, 1), C, np.ones(len(y))])
        rx = X[:, j] - Z @ np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
        ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        out[j] = np.corrcoef(rx, ry)[0, 1]
    return out


def heat(ax, M: pd.DataFrame, thr: float, title: str, fs=7, annotate=True, seps=()):
    v = M.to_numpy(float)
    ax.imshow(v, cmap=CMAP, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=60, ha="right", fontsize=fs + 1)
    ax.set_yticks(range(M.shape[0]), M.index, fontsize=fs + 1)
    ax.set_title(title, fontsize=fs + 3, loc="left", color=INK)
    if annotate:
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if np.isfinite(v[i, j]):
                    sig = abs(v[i, j]) >= thr
                    ax.text(j, i, f"{v[i, j]:.2f}".replace("0.", ".").replace("-", "−"), ha="center", va="center",
                            fontsize=fs, color=("white" if abs(v[i, j]) > 0.6 else INK) if sig else MUTED,
                            fontweight="bold" if sig else "normal")
    for s in seps:
        ax.axhline(s - 0.5, color=INK, lw=1.2)
        ax.axvline(s - 0.5, color=INK, lw=1.2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--plan", nargs="+", required=True)
    ap.add_argument("--sessions", nargs="+", default=["G"])
    ap.add_argument("--channels", type=int, nargs="+", default=[0, 1, 3])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pi.load(Path(args.data), [Path(p) for p in args.plan], args.sessions, args.channels, pi.NOISE)
    K, members, _ = pi.knobs(d)
    K.columns = [nice_knob(c) for c in K.columns]
    outs = {OUT_NICE[t]: d[f"log_{t}"] for t in pi.LAB} | {f"trace: {n}": d[f"log_{n}"] for n in pi.NOISE}
    O = pd.DataFrame(outs)
    amp = d.channel.to_numpy()
    n_eff = d.packet_id.nunique()
    thr = 1.96 / np.sqrt(n_eff)
    A = pd.concat([K, O], axis=1)
    W = A - A.groupby(amp).transform("mean")                  # within amplifier
    nk = K.shape[1]

    # 1 full matrices
    for meth, fn in (("pearson", "corr_full.png"), ("spearman", "corr_full_spearman.png")):
        C = W.corr(method=meth)
        if meth == "pearson":
            C.round(3).to_csv(out / "corr_full.csv")
        fig, ax = plt.subplots(figsize=(17, 15))
        heat(ax, C, thr, f"All voltages, image labels and trace noise components: {meth.title()} correlation within "
                         f"amplifier (session {'+'.join(args.sessions)}, {len(d)} rows, {n_eff} packets; faded = not "
                         f"significant, |r| < {thr:.2f})", fs=6.5, seps=(nk, nk + 3))
        fig.colorbar(ax.images[0], ax=ax, fraction=0.025, pad=0.01, label="correlation")
        fig.tight_layout()
        fig.savefig(out / fn, dpi=150)
        plt.close(fig)

    # 2 knobs x outputs: plain vs partial
    Cko = W.corr().loc[K.columns, O.columns]
    onehot = np.eye(len(np.unique(amp)))[np.unique(amp, return_inverse=True)[1]][:, 1:]
    P = pd.DataFrame({c: partial_corr(K.to_numpy(float), O[c].to_numpy(float), onehot) for c in O.columns},
                     index=K.columns)
    P.round(3).to_csv(out / "corr_partial.csv")
    fig, axs = plt.subplots(1, 2, figsize=(17, 8.5))
    heat(axs[0], Cko, thr, "Correlation: voltage vs output (within amplifier)", fs=7, seps=(3,))
    heat(axs[1], P, thr, "Partial correlation: this voltage alone (other voltages + amplifier held fixed)", fs=7, seps=(3,))
    fig.colorbar(axs[1].images[0], ax=axs, fraction=0.02, pad=0.01, label="correlation")
    fig.savefig(out / "corr_knobs_outputs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3 per amplifier
    amps = np.unique(amp)
    fig, axs = plt.subplots(1, len(amps), figsize=(7.2 * len(amps), 8.5))
    rows = []
    for ax, a in zip(np.atleast_1d(axs), amps):
        m = amp == a
        ne = d.packet_id[m].nunique()
        Ca = A[m].corr().loc[K.columns, O.columns]
        heat(ax, Ca, 1.96 / np.sqrt(ne), f"amplifier ch{a} ({m.sum()} rows)", fs=6.5, seps=(3,))
        rows.append(Ca.assign(amplifier=a))
    pd.concat(rows).round(3).to_csv(out / "corr_per_amp.csv")
    fig.suptitle("Voltage vs output correlation, per amplifier (faded = not significant)", x=0.01, ha="left", fontsize=12)
    fig.savefig(out / "corr_per_amp.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4 design
    fig, ax = plt.subplots(figsize=(10, 9))
    heat(ax, K.groupby(d.packet_id).first().corr(), thr,
         "Design check: correlation between the voltages over the packets (0 = varied independently)", fs=7)
    fig.tight_layout()
    fig.savefig(out / "corr_design.png", dpi=150)
    plt.close(fig)

    # 5 table
    L = [f"# Correlations, v3 voltage-only data (session {'+'.join(args.sessions)})", "",
         f"{len(d)} rows, {n_eff} packets, amplifiers {list(amps)}; within amplifier; outputs in log units. Significant "
         f"(95 %, n_eff = packets) when |r| >= {thr:.2f}. Partial = with the other voltages and the amplifier held fixed.",
         "", "Knob groups that moved together: " + "; ".join(f"{nice_knob(k)} = {'+'.join(v)}" for k, v in members.items()
                                                            if len(v) > 1), ""]
    L += ["| output | 1st | 2nd | 3rd | 4th | 5th |", "|---|---|---|---|---|---|"]
    for c in O.columns:
        top = P[c].abs().sort_values(ascending=False).index[:5]
        L.append(f"| {c} | " + " | ".join(f"{k} {P.loc[k, c]:+.2f}{'' if abs(P.loc[k, c]) >= thr else ' (n.s.)'}"
                                         for k in top) + " |")
    L += ["", "Largest correlations between outputs (within amplifier):", ""]
    Co = W[O.columns].corr()
    pairs = [(abs(Co.iloc[i, j]), Co.index[i], Co.columns[j], Co.iloc[i, j])
             for i in range(len(Co)) for j in range(i + 1, len(Co))]
    for _, a_, b_, r in sorted(pairs, reverse=True)[:10]:
        L.append(f"- {a_} ~ {b_}: {r:+.2f}")
    D = K.groupby(d.packet_id).first().corr().to_numpy()
    off = np.abs(D[~np.eye(len(D), dtype=bool)])
    L += ["", f"Design: largest |correlation| between two voltages over the packets {off.max():.2f}, median {np.median(off):.2f}."]
    (out / "correlations.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

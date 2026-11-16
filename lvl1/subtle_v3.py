"""
subtle_v3.py -- can the models see the SMALL voltage effects?  The v3n packets are a Sobol design (every packet moves
every voltage), so the change between two packets is dominated by the strongest knob: VSUB for sigma_e and lambda,
VDD and VR for the gain.  Here the pairs are split by how much that dominant knob moved; in the pairs where it barely
moved (|change| < --hold SD), what is left is the sum of the small effects of all the other voltages.  Per subset and
model (on the pairs the scorecard scores): Spearman rank correlation of predicted vs true change, size error, and the true change RMS next to the label
noise of a change (what no model can beat).

    .venv/bin/python lvl1/subtle_v3.py --scorecard processed_v3n/lvl1_results/scorecard_G --data processed_v3n \\
        --out processed_v3n/lvl1_results/subtle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

T = ("sigma_e", "lambda", "gain")
TN = {"sigma_e": "noise σe", "lambda": "dark charge λ", "gain": "gain"}
DOMINANT = {"sigma_e": ("theta.VSUB",), "lambda": ("theta.VSUB",), "gain": ("theta.VDD1", "theta.VR1")}
DOMN = {"sigma_e": "VSUB", "lambda": "VSUB", "gain": "VDD and VR"}
MODELS = {"settings change: trees": ("settings only", "#9fc0ea"),
          "trace change: trees": ("traces only", "#eb6834"),
          "ensemble: trace models": ("traces only (ensemble)", "#f2a07e"),
          "settings + trace: combined trees": ("settings + traces", "#2a78d6"),
          "ensemble: settings + trace": ("settings + traces (ensemble)", "#1c5cab")}
PAIRS = ("random pairs", "within packet (image)", "n -> n+1", "n -> n+2")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scorecard", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--hold", type=float, default=0.25, help="dominant knob 'held': |change| below this many SD")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = pd.read_csv(Path(args.scorecard) / "relative_predictions.csv", low_memory=False)
    R = R[R["type"].isin(PAIRS)]
    sc = json.loads((Path(args.scorecard) / "scorecard.json").read_text())
    d = pd.read_csv(Path(args.data) / "dataset.csv", low_memory=False)
    cols = sorted({c for v in DOMINANT.values() for c in v})
    th = d.groupby("packet_id")[cols].first()
    sd = th.std()
    res, md = [], ["# Subtle changes: pairs where the dominant knob barely moved", "",
                   f"'held' = the dominant knob changed by less than {args.hold} SD between the two packets. Pairs: "
                   f"{', '.join(PAIRS)}. rho = Spearman of predicted vs true change; size = RMS error of the change (%).", ""]
    for cell in ("trained", "new"):
        for t in T:
            dz = np.zeros(len(R))
            for c in DOMINANT[t]:
                dz = np.maximum(dz, np.abs((th.loc[R["packet B"], c].to_numpy() - th.loc[R["packet A"], c].to_numpy()) / sd[c]))
            # the pairs the scorecard scores: those with a prediction of the reference (settings change: gp)
            scored = np.isfinite(R[f"{cell} | settings change: gp | {t}"].to_numpy())
            for subset, m in (("all pairs", scored), (f"{DOMN[t]} held", scored & (dz < args.hold))):
                y = 100 * R.loc[m, f"true change {t}"].to_numpy()
                row = {"cell": cell, "target": t, "subset": subset, "n": int(m.sum()), "true_rms": float(np.sqrt(np.mean(y ** 2)))}
                for name in MODELS:
                    p = 100 * R.loc[m, f"{cell} | {name} | {t}"].to_numpy()
                    ok = np.isfinite(p)
                    row[f"{name} rho"] = float(spearmanr(p[ok], y[ok])[0]) if ok.sum() > 5 else np.nan
                    row[f"{name} size"] = float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.sum() > 5 else np.nan
                res.append(row)
    df = pd.DataFrame(res)
    df.round(3).to_csv(out / "subtle.csv", index=False)
    noise = {"sigma_e": 1.9, "lambda": 12.0, "gain": 0.8}
    try:
        txt = (Path(args.scorecard) / "scorecard.md").read_text()
        for t in T:
            noise[t] = float(txt.split("Label noise of a change:")[1].split(f"{t} ")[1].split(" %")[0])
    except Exception:
        pass
    for cell in ("trained", "new"):
        md += [f"## {'trained amplifiers' if cell == 'trained' else 'new amplifier'}", "",
               "| target | subset | n | true change RMS % (label noise) | " + " | ".join(v[0] for v in MODELS.values()) + " |",
               "|---|---|---|---|" + "---|" * len(MODELS)]
        for _, r in df[df.cell == cell].iterrows():
            md.append(f"| {TN[r.target]} | {r.subset} | {r.n} | {r.true_rms:.1f} ({noise[r.target]:.1f}) | "
                      + " | ".join(f"ρ {r[f'{k} rho']:.2f}, {r[f'{k} size']:.1f}" for k in MODELS) + " |")
        md.append("")
    (out / "subtle.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    for r_, cell in enumerate(("trained", "new")):
        for c_, t in enumerate(T):
            ax = axs[r_, c_]
            sub = df[(df.cell == cell) & (df.target == t)]
            x = np.arange(len(sub))
            w = 0.8 / len(MODELS)
            for k, (name, (lab, col)) in enumerate(MODELS.items()):
                ax.bar(x + (k - (len(MODELS) - 1) / 2) * w, sub[f"{name} rho"], w * 0.9, color=col,
                       label=lab if r_ == c_ == 0 else None)
            ax.set_xticks(x, [f"{s}\n(n={n}, true RMS {v:.1f} %)" for s, n, v in zip(sub.subset, sub.n, sub.true_rms)], fontsize=8)
            ax.axhline(0, color="#8a8984", lw=0.8)
            ax.set_ylim(-0.1, 1)
            ax.set_title(f"{TN[t]} — {'trained amps' if cell == 'trained' else 'new amp'}", loc="left", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].set_ylabel("ρ predicted vs true change")
    axs[1, 0].set_ylabel("ρ predicted vs true change")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper right", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Subtle changes: all pairs vs pairs where the dominant knob moved < {args.hold} SD "
                 "(what is left = the small effects of all the other voltages)", x=0.01, ha="left", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "subtle.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

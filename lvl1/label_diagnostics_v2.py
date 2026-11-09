"""
label_diagnostics_v2.py -- task D of the v2 training plan, the label side (docs/overleaf_v2_training_plan_2026-11-04,
"Task D: diagnostics"):

* label noise at every setting: the repeat scatter of log sigma_e, log lambda and log gain between packets at identical
  settings (the 3 session-D replicates and their design points; the nominal packets, raw and after removing a smooth
  drift in time), per amplifier;
* drift: the nominal packets' sigma_e, lambda and gain against time (figure + linear slope per hour);
* strip vs image: the new strip's sigma_e, gain and lambda against the image's, per amplifier (ratio and correlation).

Board 101 (all four amplifiers, ch2 reported separately: its image fit chi2 is ~5x the others), rows with label.ok;
fractional-window rows (biased image labels) are listed separately.  Session D only unless --sessions says otherwise.

    .venv/bin/python lvl1/label_diagnostics_v2.py --data processed_v2_trainD --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lvl1"))
import change_test as ct  # noqa: E402

LAB = {"sigma_e": "label.sigma_e", "lambda": "label.lambda_e_per_bin", "gain": "label.gain_adu_per_e"}
STRIP = {"sigma_e": "l1.l1: sigma_e (e)", "lambda": "l1.l1: lambda (e/pixel)", "gain": "l1.l1: gain (ADU/e)"}
PLAN = ROOT / "lvl1" / "packet_plan_v2" / "plan.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", nargs="+", default=["D"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(Path(args.data) / "dataset.csv", low_memory=False)
    plan = json.loads(PLAN.read_text())["packets"]
    session = {p["packet_id"]: p["session"] for p in plan}
    purpose = {p["packet_id"]: p["purpose"] for p in plan}
    d["planned_id"] = [ct.planned_id(p, session) for p in d.packet_id]
    d["session"] = d.planned_id.map(session)
    d["purpose"] = d.planned_id.map(purpose)
    d = d[(d.board == 101) & d.session.isin(args.sessions) & d["label.ok"].astype(str).str.lower().eq("true")].copy()
    d["frac"] = d["label.fractional_window"].astype(str).str.lower().eq("true")
    d["t"] = pd.to_datetime(d.date_start)
    d["hours"] = (d.t - d.t.min()).dt.total_seconds() / 3600
    for t, c in LAB.items():
        d[f"log_{t}"] = np.log(d[c])
    L = ["# Label diagnostics (task D), board 101, sessions " + "+".join(args.sessions), ""]
    res = {}

    # ---- repeat scatter at identical settings
    L += ["## Repeat scatter at identical settings", "",
          "Scatter of a single label = RMS over groups of the deviation from the group mean, with the degrees of freedom "
          "counted (sqrt(sum (x - mean)^2 / (n - groups))), log units, %. Nominal: raw, and after removing a straight line "
          "in time per amplifier (the drift). fractional = groups at a delayInt with N / N+1 sample windows.", "",
          "| amplifier | group | packets | sigma_e | lambda | gain |", "|---|---|---|---|---|---|"]
    res["repeat"] = {}
    for ch, g in d.groupby("channel"):
        nom = g[g.purpose.str.startswith("nominal")]
        rep_groups = g[g.theta_id.isin(g[g.purpose.str.startswith("REPLICATE")].theta_id)]
        for name, sub, detrend in (("nominal, raw", nom, False), ("nominal, drift removed", nom, True),
                                   ("replicates (3 design points)", rep_groups, False)):
            if len(sub) < 2:
                continue
            row = []
            for t in LAB:
                x = sub[f"log_{t}"].to_numpy()
                if detrend:
                    x = x - np.polyval(np.polyfit(sub.hours, x, 1), sub.hours)
                    dof = len(x) - 2
                    s = np.sqrt(np.sum(x ** 2) / max(dof, 1))
                else:
                    grp = sub.theta_id.to_numpy()
                    dev = x - pd.Series(x).groupby(grp).transform("mean").to_numpy()
                    dof = len(x) - len(np.unique(grp))
                    s = np.sqrt(np.sum(dev ** 2) / max(dof, 1))
                row.append(100 * s)
            res["repeat"][f"ch{ch} {name}"] = dict(zip(LAB, row), n=len(sub))
            L.append(f"| ch{ch}{' (fit chi2 high)' if ch == 2 else ''} | {name} | {len(sub)} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
    L.append("")

    # ---- drift of the nominal packets
    L += ["## Drift: nominal packets against time", "", "Slope of a straight line in time, % per hour (log units).", "",
          "| amplifier | nominal packets | hours spanned | sigma_e %/h | lambda %/h | gain %/h |", "|---|---|---|---|---|---|"]
    res["drift"] = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ch, g in d.groupby("channel"):
        nom = g[g.purpose.str.startswith("nominal")].sort_values("hours")
        sl = {t: 100 * np.polyfit(nom.hours, nom[f"log_{t}"], 1)[0] for t in LAB}
        res["drift"][f"ch{ch}"] = sl
        L.append(f"| ch{ch} | {len(nom)} | {nom.hours.max() - nom.hours.min():.1f} | " + " | ".join(f"{sl[t]:+.2f}" for t in LAB) + " |")
        for ax, t in zip(axes, LAB):
            ax.plot(nom.hours, nom[LAB[t]], "o-", ms=4, label=f"ch{ch}")
    for ax, t in zip(axes, LAB):
        ax.set_xlabel("hours since the first packet")
        ax.set_title(f"nominal packets: {t}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "drift_nominal.png", dpi=140)
    plt.close(fig)
    L += ["", "Figure: drift_nominal.png", ""]

    # ---- strip vs image
    L += ["## Strip vs image (new strip at the packet's settings against the packet's image)", "",
          "Median ratio strip / image, RMS of log(strip / image) after removing the median, Pearson r of the logs; rows "
          "with both usable. Fractional-window rows left out (their image labels are biased).", "",
          "| amplifier | n | sigma_e ratio / scatter % / r | lambda ratio / scatter % / r | gain ratio / scatter % / r |",
          "|---|---|---|---|---|"]
    res["strip_vs_image"] = {}
    use = d[d["l1.usable"].astype(str).str.lower().eq("true") & ~d.frac]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ch, g in use.groupby("channel"):
        cells, r_ = [], {}
        for ax, t in zip(axes, LAB):
            s, im = g[STRIP[t]].to_numpy(float), g[LAB[t]].to_numpy(float)
            ok = (s > 0) & (im > 0)
            lr = np.log(s[ok] / im[ok])
            ratio, scat = float(np.exp(np.median(lr))), float(100 * np.sqrt(np.mean((lr - np.median(lr)) ** 2)))
            r = float(np.corrcoef(np.log(s[ok]), np.log(im[ok]))[0, 1]) if ok.sum() > 2 else np.nan
            r_[t] = {"ratio": ratio, "scatter_pct": scat, "r": r, "n": int(ok.sum())}
            cells.append(f"{ratio:.2f} / {scat:.0f} / {r:.2f}")
            ax.loglog(im[ok], s[ok], "o", ms=3, label=f"ch{ch}")
        res["strip_vs_image"][f"ch{ch}"] = r_
        L.append(f"| ch{ch} | {len(g)} | " + " | ".join(cells) + " |")
    for ax, t in zip(axes, LAB):
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], "k:", lw=1)
        ax.set_xlabel(f"image {t}")
        ax.set_ylabel(f"strip {t}")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "strip_vs_image.png", dpi=140)
    plt.close(fig)
    L += ["", "Figure: strip_vs_image.png", ""]

    # ---- fractional-window bias (what the flag removes)
    L += ["## Fractional-window rows", "",
          f"{int(d.frac.sum())} of {len(d)} rows ({d[d.frac].packet_id.nunique()} packets) have a delayInt that is not a "
          "whole number of ADC samples; their traces and strips are fine, their image labels are biased (on-board CDS "
          "sums N and N+1 samples). Left out of training by --exclude-fractional.", ""]
    (out / "label_diagnostics.md").write_text("\n".join(L))
    (out / "label_diagnostics.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""
amplifier_identity.py -- how much of each label is simply "which amplifier is this?", whether a trace gives the
amplifier away, and what that means for predicting an amplifier the model has never seen.  Plain-language figures for
people who do not know the models.

Figures (in --out):
 amp_levels.png         each amplifier's dark charge and noise at the SAME (nominal) settings: the levels differ
 amp_share.png          how much of each label's spread is explained by the amplifier alone, the settings alone, both
 amp_from_trace.png     can a model tell the amplifier from one trace? (cross-validated accuracy)
 amp_known_vs_new.png   absolute models: error on amplifiers seen in training against a new amplifier; and the new
                        amplifier after one calibration image (its level measured once)
 amp_calibration_image.png  a new amplifier's error after 0, 1, 2, 4, 8 images of it (fleet test T1)
 amp_changes.png        change models: direction on seen and on new amplifiers (a change cancels the level)
Plus amplifier_identity.md.

    .venv/bin/python lvl1/amplifier_identity.py --out processed_v2_trainD/lvl1_results/diagnostics
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
sys.path[:0] = [str(ROOT / "lvl1"), str(ROOT / "src")]
warnings.filterwarnings("ignore")
LAB = {"sigma_e": "label.sigma_e", "lambda": "label.lambda_e_per_bin", "gain": "label.gain_adu_per_e"}
NICE = {"sigma_e": "noise (sigma_e)", "lambda": "dark charge (lambda)", "gain": "gain"}
COL = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(ROOT / "processed_v2"))
    ap.add_argument("--scorecard", default=str(ROOT / "processed_v2_trainD" / "lvl1_results" / "taskB_scorecard_D_newmodels" / "scorecard.json"))
    ap.add_argument("--fewshot", default=str(ROOT / "processed_v2" / "lvl1_results" / "fleet_tests_2026-11-10" / "main" / "fewshot.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import lambda_settings_figures as L
    from sklearn.ensemble import HistGradientBoostingClassifier as HC
    from sklearn.model_selection import GroupKFold, cross_val_predict

    d = L.load(Path(args.data))                       # sessions D+E, board 101, label.ok, no fractional windows
    K, _ = L.knobs(d)
    ch = d.channel.to_numpy().astype(int)
    pk = d.packet_id.to_numpy()
    md = ["# Which amplifier is it? (v2 sessions D+E, board 101, all four amplifiers)", ""]

    # ---- 1. levels at nominal
    nom = d[d.purpose.str.startswith("nominal")] if "purpose" in d else None
    if nom is None or nom.empty:
        import json as _j
        plan = {p["packet_id"]: p["purpose"] for p in _j.loads(L.PLAN.read_text())["packets"]}
        import change_test as ct
        d["purpose"] = [plan.get(ct.planned_id(p, plan), "") for p in d.packet_id]
        nom = d[d.purpose.str.startswith("nominal")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    md += ["## 1. Same settings, different amplifiers", "",
           "At the nominal settings (every amplifier read out the same way, many repeats):", "",
           "| amplifier | " + " | ".join(NICE[t] for t in LAB) + " |", "|---|---|---|---|"]
    for ax, t in zip(axes, LAB):
        vals = [nom[nom.channel == a][LAB[t]].astype(float) for a in range(4)]
        ax.bar(range(4), [v.mean() for v in vals], yerr=[v.std() for v in vals], color=[COL[a] for a in range(4)], capsize=4)
        ax.set_xticks(range(4), [f"ch{a}" for a in range(4)])
        ax.set_title(NICE[t])
        ax.set_ylabel("mean at nominal (bar: spread of the repeats)")
        ax.grid(alpha=0.3, axis="y")
    for a in range(4):
        md.append(f"| ch{a} | " + " | ".join(f"{nom[nom.channel == a][LAB[t]].astype(float).mean():.3g}" for t in LAB) + " |")
    md.append("")
    fig.suptitle("Four amplifiers of the same CCD at the same settings: the dark charge differs by about 2x between them")
    fig.tight_layout()
    fig.savefig(out / "amp_levels.png", dpi=130)
    plt.close(fig)

    # ---- 2. share of the spread: amplifier alone, settings alone, both
    onehot = np.eye(4)[ch]
    shares = {}
    for t in LAB:
        y = d[f"log_{t}"].to_numpy()
        shares[t] = {"amplifier alone": L.cv_r2(onehot, y, pk)[0], "settings alone": L.cv_r2(K.to_numpy(), y, pk)[0],
                     "settings + amplifier": L.cv_r2(np.hstack([K.to_numpy(), onehot]), y, pk)[0]}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(3)
    for j, (k, c) in enumerate(zip(["amplifier alone", "settings alone", "settings + amplifier"], ["#c44e52", "#4c72b0", "#55a868"])):
        ax.bar(x + (j - 1) * 0.27, [max(0, shares[t][k]) for t in LAB], 0.27, label=k, color=c)
    ax.set_xticks(x, [NICE[t] for t in LAB])
    ax.set_ylabel("share of the spread explained (cross-validated R²)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("What sets each label: the amplifier, the settings, or both")
    fig.tight_layout()
    fig.savefig(out / "amp_share.png", dpi=130)
    plt.close(fig)
    md += ["## 2. What sets each label", "", "Share of the spread explained (cross-validated by packet, 1 = all of it):", "",
           "| label | amplifier alone | settings alone | settings + amplifier |", "|---|---|---|---|"]
    for t in LAB:
        md.append(f"| {NICE[t]} | " + " | ".join(f"{shares[t][k]:.2f}" for k in ["amplifier alone", "settings alone", "settings + amplifier"]) + " |")
    md.append("")

    # ---- 3. can a trace tell the amplifier?
    excl = set(Path(L.FP).read_text().split()) if L.FP.exists() else set()
    allc = [c for c in d.columns if c.startswith("trace.") and pd.api.types.is_numeric_dtype(d[c]) and d[c].notna().mean() > 0.9]
    sets = {"all trace features": allc,
            "without the amplifier-fingerprint features (what the models use)": [c for c in allc if c[len("trace."):] not in excl]}
    acc = {}
    for name, cols in sets.items():
        X = d[cols].to_numpy(float)
        p = cross_val_predict(HC(max_iter=200, random_state=0), X, ch, groups=pk, cv=GroupKFold(5))
        acc[name] = float(np.mean(p == ch))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(range(len(acc)), [100 * v for v in acc.values()], color=["#c44e52", "#dd8452"])
    ax.axvline(25, color="k", ls=":", lw=1)
    ax.text(26, len(acc) - 0.6, "guessing (1 in 4)", fontsize=8)
    ax.set_yticks(range(len(acc)), list(acc), fontsize=8)
    ax.set_xlabel("amplifier identified correctly from one trace, % (new packets)")
    ax.set_xlim(0, 100)
    ax.set_title("A trace gives away which amplifier it came from")
    fig.tight_layout()
    fig.savefig(out / "amp_from_trace.png", dpi=130)
    plt.close(fig)
    md += ["## 3. Does a trace give the amplifier away?", "",
           "A classifier told only one trace's features names its amplifier (tested on packets it never saw):", ""]
    md += [f"* {k}: {100 * v:.0f} % right (guessing: 25 %)" for k, v in acc.items()]
    md.append("")

    # ---- 4. absolute models: known amplifier vs new amplifier (+ one calibration image)
    sc_ = json.loads(Path(args.scorecard).read_text())["absolute"]
    models = [m for m in ["settings: gp", "trace: physics", "trace: trees", "trace: nn", "trace: cnn", "trace: pretrained cnn"]
              if m in sc_["trained"] and m in sc_["new"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    md += ["## 4. Absolute models: amplifier seen in training against a new amplifier", "",
           "Typical error % (session D cross-validation). 'settings + amplifier' is told which amplifier it is.", "",
           "| model | label | seen amplifier | new amplifier |", "|---|---|---|---|"]
    for ax, t in zip(axes, LAB):
        known = [sc_["trained"][m][t]["error"] for m in models]
        new = [sc_["new"][m][t]["error"] for m in models]
        xx = np.arange(len(models))
        ax.bar(xx - 0.2, known, 0.4, label="amplifier seen in training", color="#55a868")
        ax.bar(xx + 0.2, new, 0.4, label="new amplifier", color="#c44e52")
        ref = sc_["trained"].get("settings + amplifier: gp", {}).get(t, {}).get("error")
        if ref:
            ax.axhline(ref, color="k", ls="--", lw=1, label=f"settings + told the amplifier: {ref:.0f} %")
        ax.set_xticks(xx, [m.replace("trace: ", "trace ").replace("settings: ", "settings ") for m in models], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("typical error %")
        ax.set_title(NICE[t])
        ax.grid(alpha=0.3, axis="y")
        for m, a, b in zip(models, known, new):
            md.append(f"| {m} | {NICE[t]} | {a:.1f} | {b:.1f} |")
    axes[0].legend(fontsize=7)
    md.append("")
    fig.suptitle("Absolute models: good on amplifiers they know, poor on a new one (the dark charge most)")
    fig.tight_layout()
    fig.savefig(out / "amp_known_vs_new.png", dpi=130)
    plt.close(fig)
    if Path(args.fewshot).exists():
        fsr = json.loads(Path(args.fewshot).read_text())["targets"]
        ks = sorted(fsr["lambda"]["shared GP (+ offset)"], key=int)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        md += ["## 4b. A new amplifier after k calibration images (its own level measured; fleet test T1, settings GP "
               "trained on the other amplifiers + the mean offset of the k images), typical error %", "",
               "| label | amplifier | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|---|" + "---|" * len(ks)]
        for ax, t in zip(axes, LAB):
            r = fsr[t]["shared GP (+ offset)"]
            amps = list(r[ks[0]]["per_amp_rms"])
            for a in amps:
                v = [r[k]["per_amp_rms"][a] for k in ks]
                ax.plot([int(k) for k in ks], v, "o-", color=COL[int(a.split(":")[1])], label=f"ch{a.split(':')[1]}")
                md.append(f"| {NICE[t]} | ch{a.split(':')[1]} | " + " | ".join(f"{x:.1f}" for x in v) + " |")
            ax.plot([int(k) for k in ks], [r[k]["rms"] for k in ks], "k--", label="all")
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xticks([int(k) for k in ks], ks)
            ax.set_xlabel("images of the new amplifier (k)")
            ax.set_ylabel("typical error %")
            ax.set_title(NICE[t])
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        md.append("")
        fig.suptitle("A new amplifier: one image of it (its level) removes most of the error; more images add little")
        fig.tight_layout()
        fig.savefig(out / "amp_calibration_image.png", dpi=130)
        plt.close(fig)

    # ---- 5. change models: seen vs new amplifier
    rel = json.loads(Path(args.scorecard).read_text())["relative"]
    cm = [m for m in ["settings change: gp", "trace change: trees", "trace: combined trees [A, B, B-A]",
                      "trace: direction classifier [A, B, B-A]"] if m in rel["trained"] and m in rel["new"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    md += ["## 5. Change models: seen against new amplifier (random pairs of packets, direction right on clear changes %)", "",
           "| model | label | seen amplifier | new amplifier |", "|---|---|---|---|"]
    for ax, t in zip(axes, LAB):
        a_ = [100 * rel["trained"][m]["random pairs"][t]["direction"] for m in cm]
        b_ = [100 * rel["new"][m]["random pairs"][t]["direction"] for m in cm]
        xx = np.arange(len(cm))
        ax.bar(xx - 0.2, a_, 0.4, label="amplifier seen in training", color="#55a868")
        ax.bar(xx + 0.2, b_, 0.4, label="new amplifier", color="#c44e52")
        ax.set_ylim(40, 100)
        ax.set_xticks(xx, [m.replace(" [A, B, B-A]", "").replace("trace: ", "trace ").replace("trace change: ", "trace Δ ")
                           .replace("settings change: ", "settings Δ ") for m in cm], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("direction right %")
        ax.set_title(NICE[t])
        ax.grid(alpha=0.3, axis="y")
        for m, a, b in zip(cm, a_, b_):
            md.append(f"| {m} | {NICE[t]} | {a:.0f} | {b:.0f} |")
    axes[0].legend(fontsize=7)
    md.append("")
    fig.suptitle("Change models: about as good on a new amplifier as on a known one (a change cancels the amplifier's level)")
    fig.tight_layout()
    fig.savefig(out / "amp_changes.png", dpi=130)
    plt.close(fig)
    (out / "amplifier_identity.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

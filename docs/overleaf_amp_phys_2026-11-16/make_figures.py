"""
make_figures.py -- the explanatory (pedagogical) figures of the amp_phys document: diagrams of the readout and the
device physics, and annotated data figures built from amp_phys/results.

    cd ccd_opt_ml && .venv/bin/python docs/overleaf_amp_phys_2026-11-16/make_figures.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, RED, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#e34948", "#0b0b0b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def box(ax, x, y, w, h, text, fc="#eef3fb", ec=BLUE, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, color=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.2))


def readout_chain():
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.add_patch(Rectangle((0.1, 1.2), 4.3, 3.1, fc="#f6f6f4", ec=GRAY, ls="--"))
    ax.text(0.2, 4.15, "inside the cryostat (CCD, 130 K)", fontsize=8, color=GRAY)
    ax.add_patch(Rectangle((4.6, 1.2), 5.2, 3.1, fc="#f6f6f4", ec=GRAY, ls="--"))
    ax.text(4.7, 4.15, "frontend board FE_ADA4817 (room temperature)", fontsize=8, color=GRAY)
    ax.add_patch(Rectangle((10.0, 1.2), 2.9, 3.1, fc="#f6f6f4", ec=GRAY, ls="--"))
    ax.text(10.1, 4.15, "ACM", fontsize=8, color=GRAY)
    box(ax, 0.3, 2.3, 1.6, 1.2, "sense node\n$C_{sn}\\approx$ 80–90 fF\n1 e$^-$ → 1.8–2.0 µV")
    box(ax, 2.4, 2.3, 1.8, 1.2, "source follower\n(output FET, W/L≈10)\ngain $A\\approx$0.8–0.9")
    box(ax, 4.8, 2.3, 1.6, 1.2, "load 20 kΩ to GND\n‖ 100 kΩ after\n100 nF coupling")
    box(ax, 6.7, 2.3, 1.4, 1.2, "ADA4817\n×13.64")
    box(ax, 8.3, 2.3, 1.3, 1.2, "ADA4932\n×2 (diff.)")
    box(ax, 10.2, 2.3, 1.2, 1.2, "ADC 18 bit\n31.25 µV")
    box(ax, 11.6, 2.3, 1.1, 1.2, "ADU")
    for x0, x1 in ((1.9, 2.4), (4.2, 4.8), (6.4, 6.7), (8.1, 8.3), (9.6, 10.2), (11.4, 11.6)):
        arrow(ax, x0, 2.9, x1, 2.9)
    ax.text(3.3, 1.75, "video line = 'AIN'", ha="center", fontsize=8)
    notes = [(1.1, 0.95, "measured: 1.51–1.62 µV/e$^-$ at AIN\n(chain calibration)", BLUE),
             (3.3, 0.55, "FET thermal noise ≤ 4.9 nV/√Hz at AIN\n(physics bound, 130 K)", RED),
             (5.6, 0.95, "Johnson noise of 16.7 kΩ:\n16.6 nV/√Hz (open input)", RED),
             (7.4, 0.55, "preamp 4 nV/√Hz", RED),
             (11.0, 0.95, "chain gain 0.853 ADU/µV measured\n0.875 ADU/µV by design", BLUE)]
    for x, y, t, c in notes:
        ax.text(x, y, t, ha="center", fontsize=7.5, color=c)
    ax.set_title("The readout chain of one amplifier, with the numbers measured in this study (blue) and the noise "
                 "sources (red)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "readout_chain.png", dpi=170)
    plt.close(fig)


def skip_cycle():
    f = sorted(glob.glob(str(ROOT / "processed_v3n" / "N3_0*" / "traces" / "ref1_acm101_ch1.npz")))[0]
    z = np.load(f, allow_pickle=True)
    w = z["curve: skip: mean shape"].astype(float)
    t = np.arange(len(w)) / 15.0                                  # us
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(t, w, color=INK, lw=1.2)
    spans = [(0, 150, "pedestal window\n(10 µs average)", "#dbe8f8"), (150, 207, "SW → sense node\n(charge transfer)", "#fde3d6"),
             (207, 357, "signal window\n(10 µs average)", "#d8f2e7"), (357, 489, "OG, reset (RG)\nback to the pedestal", "#efefec")]
    for a, b, lab, c in spans:
        ax.axvspan(a / 15, b / 15, color=c, zorder=0)
        ax.text((a + b) / 30, ax.get_ylim()[1] * 0.85 if a != 357 else ax.get_ylim()[1] * 0.55, lab, ha="center", fontsize=8)
    ax.set_xlabel("time within one skip cycle (µs); one cycle = 489 samples = 32.6 µs")
    ax.set_ylabel("video signal (ADU, pedestal level = 0)")
    ax.set_title("One skip cycle (mean of 490 skips of a real trace, 101 ch1). CDS = mean(signal window) − "
                 "mean(pedestal window).\nA clock that sits at the same level in both windows cancels in the CDS; the "
                 "large reset/OG swings outside the windows do not enter it.", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "skip_cycle.png", dpi=170)
    plt.close(fig)


def dead_channel_calibration():
    T = pd.read_csv(ROOT / "amp_phys/results/v3n/datasheet.csv", index_col=0)
    r = T.loc[[i for i in T.index if i.endswith("| white noise per skip (ADU)")][0]]
    k, Tb, R = 1.380649e-23, 300, 1 / (1 / 20e3 + 1 / 100e3)
    e = np.sqrt(4 * k * Tb * R + (4e-9) ** 2)
    sig = e / np.sqrt(10e-6) * 1e6
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    ax = axs[0]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    box(ax, 0.3, 3.6, 2.4, 1.4, "dead channel:\nno amplifier,\nonly the load", fc="#fdeee8", ec=ORANGE)
    box(ax, 3.3, 3.6, 2.6, 1.4, "Johnson noise\n√(4kTR), R = 16.7 kΩ, 300 K\n+ preamp → 17.1 nV/√Hz", fc="#fdeee8", ec=ORANGE)
    box(ax, 6.5, 3.6, 3.2, 1.4, "CDS of 10 µs windows:\nσ = e$_n$/√τ = 5.40 µV per skip", fc="#fdeee8", ec=ORANGE)
    arrow(ax, 2.7, 4.3, 3.3, 4.3)
    arrow(ax, 5.9, 4.3, 6.5, 4.3)
    box(ax, 1.5, 0.8, 7.0, 1.6, "measured on the dead channels: 4.50–4.78 ADU per skip\n→ chain gain = 4.6 ADU / 5.40 µV = "
        "0.853 ± 0.027 ADU/µV\nschematic: 13.64 × 2.00 / 31.25 µV = 0.875 ADU/µV  (agree within 2.5 %)", fc="#eef3fb")
    arrow(ax, 8.1, 3.6, 6.5, 2.4)
    ax.set_title("A resistor as a built-in noise standard", loc="left", fontsize=10)
    ax = axs[1]
    names = ["102 ch0", "102 ch1", "102 ch2"]
    vals = [r[n] for n in names]
    ax.bar(names, vals, color=ORANGE, width=0.5, label="measured (dead)")
    ax.axhline(sig * 0.875, color=BLUE, ls="--", label="predicted with the design gain")
    for n, v in zip(names, vals):
        ax.text(n, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("white noise per skip (ADU)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Dead channels: measured vs Johnson-noise prediction", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "dead_channel_calibration.png", dpi=170)
    plt.close(fig)


def noise_budget():
    k = 1.380649e-23
    R = 1 / (1 / 20e3 + 1 / 100e3)
    meas = 20.4e-9
    fig, ax = plt.subplots(figsize=(10, 4))
    cases = []
    for gm, lab in ((300e-6, "if g$_m$ ≈ 300 µS\n(layout + device physics)"), (20e-6, "if g$_m$ ≈ 20 µS\n(low current)")):
        A = gm * 20e3 / (1 + gm * 20e3)
        Reff = 1 / (1 / R + gm)
        load = np.sqrt(4 * k * 300 / R) * Reff
        pre = 4e-9
        fet = A * np.sqrt(8 / 3 * k * 130 / gm)
        rest = np.sqrt(max(meas ** 2 - load ** 2 - pre ** 2 - fet ** 2, 0))
        cases.append((lab, [load ** 2, pre ** 2, fet ** 2, rest ** 2]))
    parts = ["load resistor (setup)", "preamp (setup)", "output FET thermal (intrinsic)", "unexplained"]
    cols = [ORANGE, "#f2a07e", BLUE, GRAY]
    for i, (lab, v) in enumerate(cases):
        left = 0
        for p, c, x in zip(parts, cols, v):
            ax.barh(i, x * 1e18, left=left, color=c, label=p if i == 0 else None, height=0.5)
            left += x * 1e18
        ax.text(left + 5, i, f"total {np.sqrt(left):.1f} nV/√Hz", va="center", fontsize=8)
    ax.set_yticks([0, 1], [c[0] for c in cases])
    ax.set_xlabel("noise power at the video input (nV²/Hz); the bars add in quadrature")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    ax.set_title("White-noise budget of board 101 (measured 20.4 nV/√Hz). Which branch is true is decided by one DC "
                 "reading of the video line.", loc="left", fontsize=9, pad=40)
    fig.tight_layout()
    fig.savefig(OUT / "noise_budget.png", dpi=170)
    plt.close(fig)


def buried_channel():
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axs[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.6, 1.05)
    ax.axis("off")
    ax.add_patch(Rectangle((2, 0.55), 6, 0.35, fc="#d9a6a6", ec=INK))
    ax.text(5, 0.72, "poly-Si gate (RG, H, V …)", ha="center", fontsize=8)
    ax.add_patch(Rectangle((2, 0.42), 6, 0.13, fc="#cfe3f7", ec=INK))
    ax.text(8.15, 0.48, "Si$_3$N$_4$ 50 nm", fontsize=7.5, va="center")
    ax.add_patch(Rectangle((2, 0.30), 6, 0.12, fc="#e9f2fb", ec=INK))
    ax.text(8.15, 0.36, "SiO$_2$ 50 nm", fontsize=7.5, va="center")
    ax.add_patch(Rectangle((0.5, -1.5), 9, 1.8, fc="#f4f1ea", ec=INK))
    ax.text(0.6, -1.4, "n-type high-resistivity silicon (≳ 25 kΩ·cm), 615–655 µm thick, depleted by VSUB", fontsize=7.5)
    yy = np.linspace(0.3, -1.4, 200)
    x_ch = 0.6
    depth = (0.3 - yy) / 1.7 * 2.0                                 # µm, 0 at the oxide
    pot = np.exp(-((depth - x_ch) / 0.25) ** 2)
    ax.plot(5 + 2.2 * pot, yy, color=BLUE, lw=2)
    yc = 0.3 - x_ch / 2.0 * 1.7
    ax.axhline(yc, xmin=0.2, xmax=0.8, color=BLUE, ls=":", lw=1)
    ax.text(7.4, yc, "channel potential maximum:\nthe holes sit here\n(x$_{ch}$ ≈ 0.6 µm below the oxide)", fontsize=7.5,
            color=BLUE, va="center")
    ax.annotate("", xy=(3.1, yc), xytext=(3.1, 0.3), arrowprops=dict(arrowstyle="<->", color=INK))
    ax.text(2.2, (yc + 0.3) / 2, "x$_{ch}$", fontsize=9)
    ax.text(5, -1.05, "boron implant 1.3×10$^{13}$ cm$^{-2}$ at 150 keV (buried p-channel)", ha="center", fontsize=7.5)
    ax.set_title("Buried channel: the charge sits inside the silicon, not at the oxide", loc="left", fontsize=10)
    ax = axs[1]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    box(ax, 3.5, 8.3, 3, 1.1, "gate", fc="#f6dede", ec=RED)
    box(ax, 3.5, 5.9, 3, 1.3, "C$_{ox}$ = ε$_{ox}$/t$_{eq}$\n0.454 fF/µm² (76 nm eq.)")
    box(ax, 3.5, 3.4, 3, 1.3, "C$_{Si}$ = ε$_{Si}$/x$_{ch}$\n(silicon above the channel)")
    box(ax, 3.5, 1.2, 3, 1.1, "channel charge", fc="#e2f4ec", ec=AQUA)
    for y0, y1 in ((8.3, 7.2), (5.9, 4.7), (3.4, 2.3)):
        arrow(ax, 5, y0, 5, y1)
    ax.text(0.2, 0.2, "in series:  1/C$_{eff}$ = t$_{eq}$/ε$_{ox}$ + x$_{ch}$/ε$_{Si}$\n"
            "measured (reset injection): C$_{eff}$ ≈ 0.125 fF/µm²\n→ x$_{ch}$ ≈ 0.6 µm if t$_{eq}$ = 76 nm (recipe)",
            fontsize=8.5)
    ax.set_title("The gate sees the channel through two capacitors", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "buried_channel.png", dpi=170)
    plt.close(fig)


def charge_injection():
    J = pd.read_csv(ROOT / "amp_phys/results/v3n/clock_edges_per_trace.csv")
    d = pd.read_csv(ROOT / "processed_v3n/dataset.csv", low_memory=False, usecols=["packet_id", "board", "channel",
                    "theta.RG_1_H", "theta.VR1", "label.gain_adu_per_e"])
    d = d[d.board == 101]
    th = d.groupby("packet_id")[["theta.RG_1_H", "theta.VR1"]].first()
    g = d.groupby("channel")["label.gain_adu_per_e"].median()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axs[0]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.add_patch(Rectangle((1, 1.8), 2.2, 1.0, fc="#e2f4ec", ec=INK))
    ax.text(2.1, 2.3, "sense node\n(floating)", ha="center", fontsize=8)
    ax.add_patch(Rectangle((6.8, 1.8), 2.2, 1.0, fc="#efefec", ec=INK))
    ax.text(7.9, 2.3, "VR\n(reset drain)", ha="center", fontsize=8)
    ax.add_patch(Rectangle((3.2, 1.8), 3.6, 1.0, fc="#dbe8f8", ec=INK))
    ax.text(5, 2.3, "channel charge Q = C$_{eff}$ W L (overdrive)", ha="center", fontsize=7.5)
    ax.add_patch(Rectangle((3.2, 3.0), 3.6, 0.7, fc="#d9a6a6", ec=INK))
    ax.text(5, 3.35, "reset gate RG (6 × 6 µm)", ha="center", fontsize=8)
    arrow(ax, 4.4, 1.7, 2.6, 1.0, color=BLUE)
    arrow(ax, 5.6, 1.7, 7.4, 1.0, color=GRAY)
    ax.text(1.0, 0.5, "≈ ½ Q onto the sense node", fontsize=8, color=BLUE)
    ax.text(6.4, 0.5, "≈ ½ Q into VR", fontsize=8, color=GRAY)
    ax.text(0.2, 4.6, "When RG switches the reset transistor off, the charge in its channel leaves through both ends.\n"
            "The part that lands on the sense node shifts the output by ½ C$_{eff}$·W·L per volt of overdrive,\n"
            "and the overdrive is set by RG$_{high}$ − VR.", fontsize=8)
    ax.set_title("Reset charge injection", loc="left", fontsize=10)
    ax = axs[1]
    cols = {0: BLUE, 1: ORANGE, 2: GRAY, 3: AQUA}
    for c in range(4):
        jj = J[(J.board == 101) & (J.channel == c)].join(th, on="packet_id").dropna(subset=["edge0 jump", "theta.RG_1_H"])
        x = jj["theta.RG_1_H"] - jj["theta.VR1"]
        y = jj["edge0 jump"] / g[c] * 1.602e-19 * 1e15                # fF-equivalent (charge / 1 V)
        ax.scatter(x, y, s=10, color=cols[c], alpha=0.6, label=f"101 ch{c}")
        p = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 10)
        ax.plot(xx, np.polyval(p, xx), color=cols[c], lw=1)
    ax.set_xlabel("RG$_{high}$ − VR (V), as set in the 80 v3 packets (other voltages also varied)")
    ax.set_ylabel("reset jump, as charge on the sense node (fC = fF × V)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Measured: the jump grows by 2.1–2.4 fF per volt (slope), the same on all four CCDs", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "charge_injection.png", dpi=170)
    plt.close(fig)


def bootstrap():
    q = 1.602e-19
    cgs = 105
    cfix = np.linspace(10, 100, 200)
    cg = 1.56e-6
    A = cg * (cfix * 1e-15 + cgs * 1e-15) / (q + cg * cgs * 1e-15)
    csn = cfix + cgs * (1 - A)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axs[0]
    ax.plot(cfix, A, color=BLUE, lw=2, label="follower gain A required by the measured\nconversion gain (1.56 µV/e$^-$ at AIN)")
    ax.axhspan(0.77, 0.94, color=ORANGE, alpha=0.2, label="A from device physics\n(g$_m$ 170–750 µS, 20 kΩ load)")
    ok = (A > 0.77) & (A < 0.94)
    ax.axvspan(cfix[ok].min(), cfix[ok].max(), color=GRAY, alpha=0.15)
    ax.set_xlabel("C$_{fixed}$ (fF): sense-node diffusion + overlaps + ~100 µm of sense line")
    ax.set_ylabel("follower gain A")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Two routes cross: A ≈ 0.8–0.9, C$_{fixed}$ ≈ 55–80 fF", loc="left", fontsize=9)
    ax = axs[1]
    ax.plot(cfix, csn, color=BLUE, lw=2)
    ax.axvspan(cfix[ok].min(), cfix[ok].max(), color=GRAY, alpha=0.15)
    ax.set_xlabel("C$_{fixed}$ (fF)")
    ax.set_ylabel("total sense-node capacitance C$_{sn}$ (fF)")
    ax.text(15, csn.max() * 0.95, "C$_{sn}$ = C$_{fixed}$ + C$_{gs}$(1 − A)\nC$_{gs}$ ≈ ⅔ C$_{ox}$WL ≈ 105 fF (layout)\n"
            "the follower 'bootstraps' its own gate:\nonly (1 − A) of C$_{gs}$ loads the node", fontsize=8, va="top")
    ax.set_title("→ C$_{sn}$ ≈ 80–90 fF, sense-node gain ≈ 1.8–2.0 µV/e$^-$", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "bootstrap.png", dpi=170)
    plt.close(fig)


def crosstalk_window():
    fr = []
    for g, tau in (("short", 441), ("nominal", 1000), ("long", 2046)):
        r = pd.read_csv(ROOT / f"amp_phys/results/v2/crosstalk_101_{g}.csv")
        r["tau_us"] = tau * 0.01
        fr.append(r)
    D = pd.concat(fr)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, pairs, title in ((axs[0], ["ch0->ch1", "ch2->ch3", "ch3->ch2", "ch1->ch0"], "positive pairs: fall as 1/τ → capacitance between video lines"),
                             (axs[1], ["ch0->ch2", "ch1->ch3", "ch3->ch1", "ch2->ch0"], "negative pairs: a τ-independent part → shared return impedance")):
        for pr, c in zip(pairs, (BLUE, ORANGE, AQUA, GRAY)):
            s, t = pr.split("->")
            g = D[(D.source == s) & (D.target == t)].sort_values("tau_us")
            ax.errorbar(1 / g.tau_us, g["crosstalk (1e-4)"], yerr=[g["crosstalk (1e-4)"] - g.lo, g.hi - g["crosstalk (1e-4)"]],
                        fmt="o-", color=c, label=pr, capsize=3)
        ax.axhline(0, color=GRAY, lw=0.6)
        ax.set_xlabel("1 / CDS window (1/µs)")
        ax.set_ylabel("crosstalk (target e$^-$ per 10$^4$ source e$^-$)")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(title, loc="left", fontsize=9)
    fig.suptitle("Separating the two crosstalk mechanisms by the CDS window length (v2 images, three window groups)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "crosstalk_window.png", dpi=170)
    plt.close(fig)


def coupling_bars():
    lines = [("RG", 0.15, 0.37, "measured"), ("SW", 0.08, 0.13, "measured"), ("H1–H3", 0.011, 0.016, "measured"),
             ("V1–V3", 0.0005, 0.003, "upper limit"), ("OG", 0.2, 0.6, "layout estimate"),
             ("VDD / drain", 2.0, 4.0, "layout estimate"), ("VSUB", 0.008, 0.033, "estimate")]
    fig, ax = plt.subplots(figsize=(10, 4))
    col = {"measured": BLUE, "upper limit": GRAY, "layout estimate": ORANGE, "estimate": ORANGE}
    for i, (n, lo, hi, kind) in enumerate(lines):
        ax.plot([lo, hi], [i, i], color=col[kind], lw=8, solid_capstyle="butt", label=kind)
        if kind == "upper limit":
            ax.annotate("", xy=(lo * 0.5, i), xytext=(hi, i), arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.set_xscale("log")
    ax.set_yticks(range(len(lines)), [l[0] for l in lines])
    ax.set_xlabel("mutual capacitance to the sense node (fF), log scale")
    h, l = ax.get_legend_handles_labels()
    u = dict(zip(l, h))
    ax.legend(u.values(), u.keys(), frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Every line's coupling to the sense node (C$_{sn}$ ≈ 85 fF for scale). The line needing the least noise "
                 "to inject 22 nV/√Hz: VDD (0.6 µV/√Hz)", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "coupling_bars.png", dpi=170)
    plt.close(fig)


def depletion():
    q, eps = 1.602e-19, 11.7 * 8.854e-14
    V = np.linspace(0, 100, 300)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for rho, c in ((5, GRAY), (10, ORANGE), (25, BLUE), (35, AQUA)):
        N = 1 / (q * 1350 * rho * 1e3)
        ax.plot(V, np.sqrt(2 * eps * V / (q * N)) * 1e4, color=c, lw=1.8, label=f"{rho} kΩ·cm (N$_d$ = {N:.1e} cm$^{{-3}}$)")
    ax.axhspan(615, 655, color=INK, alpha=0.12, label="thickness from muon tracks (615–655 µm)")
    ax.axvline(45.1, color=INK, ls=":", lw=1)
    ax.text(46, 120, "nominal VSUB 45 V:\nalready fully depleted\n(muon diffusion profile)", fontsize=8)
    ax.axvspan(54, 57.5, color=RED, alpha=0.12)
    ax.text(58, 820, "λ knee 54–57 V: NOT depletion\n(serial-register effect)", fontsize=8, color=RED)
    ax.set_ylim(0, 1000)
    ax.set_xlabel("VSUB (V)")
    ax.set_ylabel("depletion depth (µm)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("Depletion depth vs VSUB: full depletion at ≤ 45–55 V of 615–655 µm needs ρ ≳ 25–35 kΩ·cm", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "depletion.png", dpi=170)
    plt.close(fig)


def not_simultaneous():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for c in range(4):
        ax.add_patch(Rectangle((c * 1.0, 3 - c), 0.2, 0.6, fc=BLUE, ec=INK))
        ax.text(c * 1.0 + 0.3, 3.3 - c, f"trace of ch{c}: its own acquisition (18:56:0{4 + c})", fontsize=8, va="center")
    ax.add_patch(Rectangle((6.2, 0), 0.5, 3.6, fc=AQUA, ec=INK, alpha=0.6))
    ax.text(6.8, 1.8, "image: ch0–ch3 read in ONE acquisition,\npixel (r, c) of every amplifier digitised\n"
            "at the same instant → valid for common-mode", fontsize=8, va="center")
    ax.set_xlim(-0.2, 11)
    ax.set_ylim(-0.3, 3.9)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title("Why the trace-based 'common-mode' analysis had to be withdrawn: the four channels' traces are taken one "
                 "after the other", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "not_simultaneous.png", dpi=170)
    plt.close(fig)


def knees_concept():
    x = np.linspace(4.5, 8, 200)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k, c, lab in ((6.3, BLUE, "CCD_D, CCD_C: knee ≈ 6.2–6.45 V"), (5.45, AQUA, "CCD_B: knee ≈ 5.35–5.55 V")):
        ax.plot(x, 0.06 * (x - 5) + 0.6 * np.maximum(0, x - k), color=c, lw=2, label=lab)
        ax.axvline(k, color=c, ls=":")
    ax.annotate("", xy=(5.45, 1.25), xytext=(6.3, 1.25), arrowprops=dict(arrowstyle="<->"))
    ax.text(5.55, 1.3, "0.85 V ≈ 0.6 % of the implant dose\n(ΔV ≈ q ΔN$_s$ / C$_{eff}$)", fontsize=8)
    ax.set_xlabel("H3 low rail (V)")
    ax.set_ylabel("trace white noise (log, arb.)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("A 'knee': where a barrier or well potential is crossed (schematic of the hinge fits)", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "knees_concept.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    for f in (readout_chain, skip_cycle, dead_channel_calibration, noise_budget, buried_channel, charge_injection,
              bootstrap, crosstalk_window, coupling_bars, depletion, not_simultaneous, knees_concept):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, e)


def cds_transfer():
    """Why the transistor's 1/f noise survives the CDS and then averages down like white noise."""
    f = np.logspace(1, np.log10(7.5e6), 4000)
    tau, dt, T = 10e-6, 13.8e-6, 32.6e-6
    x = np.pi * f * tau
    cds = 4 * np.sin(np.pi * f * dt) ** 2 * (np.sin(x) / x) ** 2
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3))
    ax = axs[0]
    ax.loglog(f, cds, color=INK, lw=1.2, label="CDS filter |H$_{CDS}$|$^2$")
    ax.loglog(f, cds / f * 1e4, color=ORANGE, lw=1.5, label="1/f noise after the CDS (arb.)")
    ax.loglog(f, cds * 0.5, color=BLUE, lw=1.0, ls="--", label="white noise after the CDS (arb.)")
    ax.axvspan(1e4, 4e4, color=ORANGE, alpha=0.12)
    ax.text(1.1e4, 3e-4, "what is left of the 1/f\nsits at 10–40 kHz,\nnear the skip rate", fontsize=8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylim(1e-6, 10)
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    ax.set_title("The CDS suppresses low frequencies as f$^2$: 1/f noise survives only near the skip rate", loc="left", fontsize=9)
    ax = axs[1]
    Ns = np.array([1, 2, 5, 10, 20, 40, 80, 160, 240])
    df = np.gradient(f)
    def sig(N, w):
        y = np.pi * f * T
        avg = np.nan_to_num((np.sin(N * y) / (N * np.sin(y))) ** 2, nan=1.0)
        return np.sqrt(np.sum(cds * avg * w * df))
    for w, lab, c in ((np.ones_like(f), "white", BLUE), (1 / f, "transistor 1/f", ORANGE)):
        s = np.array([sig(N, w) for N in Ns])
        ax.loglog(Ns, s / s[0], "o-", color=c, label=lab, ms=4)
    ax.loglog(Ns, 1 / np.sqrt(Ns), "k:", lw=1, label="1/√N")
    ax.set_xlabel("skips averaged N")
    ax.set_ylabel("noise of the N-skip average (relative)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("…so averaging skips reduces it like white noise:\nits level cannot be read from the noise-vs-N curve",
                 loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "cds_transfer.png", dpi=170)
    plt.close(fig)


def window_optimum():
    import sys
    sys.path.insert(0, str(ROOT / "amp_phys"))
    from oxide_traps_tau import integrals
    R = pd.read_csv(ROOT / "amp_phys/results/v2/oxide_traps_tau/oxide_traps_tau.csv")
    R = R[R.kind == "own"]
    taus = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30])
    I = {t: integrals(t * 1e-6) for t in taus}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    cols = {0: BLUE, 1: ORANGE, 2: GRAY, 3: AQUA}
    for _, r in R.iterrows():
        c = int(r.channel)
        w = np.array([r["S_w (ADU^2/Hz)"] * I[t][0] for t in taus])
        k = np.array([r["K (ADU^2)"] * I[t][1] for t in taus])
        axs[0].plot(taus, np.sqrt(w), ls="--", color=cols[c])
        axs[0].plot(taus, np.sqrt(k), ls=":", color=cols[c])
        axs[0].plot(taus, np.sqrt(w + k), color=cols[c], lw=1.8, label=f"ch{c}")
        T = 32.6 + 2 * (taus - 10)
        ft = (w + k) * T
        axs[1].plot(taus, np.sqrt(ft / ft[list(taus).index(10)]), "o-", color=cols[c], ms=3, label=f"ch{c}")
    axs[0].set_xlabel("CDS window τ (µs)")
    axs[0].set_ylabel("own single-skip noise (ADU)")
    axs[0].set_title("total (solid) = white (dashed, ∝ 1/√τ) + 1/f (dotted, ≈ constant)", loc="left", fontsize=9)
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].axvline(10, color=INK, ls=":", lw=1)
    axs[1].text(10.4, 1.27, "now: 10 µs", fontsize=8)
    axs[1].set_xlabel("CDS window τ (µs)")
    axs[1].set_ylabel("noise at FIXED readout time (relative to 10 µs)")
    axs[1].set_title("optimum 5–8 µs; longer windows cost 9–32 %", loc="left", fontsize=9)
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "window_optimum.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    for f_ in (cds_transfer, window_optimum):
        try:
            f_()
            print("ok", f_.__name__)
        except Exception as e:
            print("FAILED", f_.__name__, e)

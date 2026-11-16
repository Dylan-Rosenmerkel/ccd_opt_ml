"""
make_tables.py -- colour-graded LaTeX tables for channel_scorecard.tex, generated from the amp_phys result CSVs:
  amp_phys/results/per_channel/physics_sheet.csv      (device physics per channel)
  amp_phys/results/per_channel/channel_scorecard.csv  (trace setup metrics, dead channel, static records)
  amp_phys/results/anatomy/trace_anatomy.csv          (PA03 101 ch1 October / Nov v1 history)

Every graded quantity has explicit thresholds (GRADES): green = good, amber = fair, red = poor; the thresholds are
printed in the document next to each table.  Run from the repository root:

    .venv/bin/python docs/overleaf_channel_scorecard_2026-11-16/make_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "amp_phys" / "results"
OUT = Path(__file__).resolve().parent / "tables"

# quantity -> (direction, good limit, fair limit); direction "low" = smaller is better, "high" = larger is better,
# "abs" = |value| smaller is better
# Limits: "good" = what a well-working skipper amplifier/setup reaches, "fair" = works but worth attention,
# beyond = a real problem.  Quantities that depend on temperature/exposure or include pickup are not graded.
GRADES = {
    "csn": ("low", 95, 105),                      # fF: the design value +- 10 %
    "cds": ("low", 19.0, 22.0),                   # nV/rtHz: open-input frontend floor 17.1 + amplifier
    "skip_e": ("low", 4.0, 5.0),                  # e per skip: typical skipper single-sample noise 3-4 e
    "img": ("low", 0.20, 0.30),                   # e after 500 skips: 0.3 e still resolves single electrons
    "img_ratio": ("low", 1.25, 2.0),              # images vs white-noise averaging
    "white": ("low", 10, 15),
    "corner": ("low", 100, 200),                  # kHz (method-dependent: shown, not used in the card)
    "onef": ("low", 0.10, 0.15),                  # e: transistor 1/f noise left in a 500-skip pixel
    "nt": ("low", 3.0, 10.0),                     # 1e16 /eV/cm3: good thermal oxide 1e16-1e17
    "kurt": ("abs", 0.5, 1.5), "kurt90": ("abs", 0.15, 0.5),  # telegraph noise (p90: episodes over many traces)
    "resid": ("low", 0.05, 0.05),
    "settle": ("low", 1.2, 2.0),                  # us, inside the 3.5 us SW->signal wait
    "slope": ("abs", 6.0, 15.0),                  # e/us inside the signal window
    "offset": ("abs", 100, 500),                  # e: unsettled SW/OG transient
    "slow": ("low", 5.0, 15.0),                   # uV/rtHz 0.2-1 kHz
    "corr": ("low", 0.3, 0.6), "burst": ("low", 0.05, 0.15), "static": ("low", 2.0, 4.0),
    "setup500": ("low", 0.04, 0.08),               # e: setup (100 Hz-1 kHz) left in a 500-skip pixel, static record
    "charge_rel": ("low", 1.3, 2.0),              # blank-image charge / median of the same board
}
CAT = {  # summary card: category -> graded keys
    "Sense node": ["csn"],
    "Read noise": ["cds", "skip_e", "img", "img_ratio"],
    "Transistor (1/f, traps)": ["onef", "nt", "kurt", "kurt90"],
    "Timing": ["settle", "slope", "offset"],
    "Setup": ["setup500", "slow", "corr", "burst", "static"],
    "Charge (vs same board)": ["charge_rel"],
}
RANK = {"good": 0, "fair": 1, "poor": 2}


def grade(key, v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    d, g, f = GRADES[key]
    x = abs(v) if d == "abs" else v
    if d == "high":
        return "good" if x >= g else ("fair" if x >= f else "poor")
    return "good" if x <= g else ("fair" if x <= f else "poor")


def fmt(v, nd):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if nd == "k":
        return f"{v / 1e3:.1f}"
    if nd == "e":
        return f"{v:.1e}".replace("e+0", "e").replace("e-0", "e-")
    return f"{v:.{nd}f}"


def cell(key, v, nd):
    g = grade(key, v) if key else None
    s = fmt(v, nd)
    if s.startswith("-") and s != "--":
        s = "$-$" + s[1:]
    return f"\\cellcolor{{{g}}}{s}" if g else s


def load():
    ps = pd.read_csv(RES / "per_channel/physics_sheet.csv").set_index("channel")
    sc = pd.read_csv(RES / "per_channel/channel_scorecard.csv").set_index("label")
    rows = []
    tr = pd.read_csv(RES / "trap_from_trace_v3/trap_from_trace.csv").groupby("label").median(numeric_only=True)
    order = ["PA03 101 ch0", "PA03 101 ch1", "PA03 101 ch2", "PA03 101 ch3", "PA03 102 ch3", "PA03 102 ch0 (dead)",
             "PA03 102 ch1 (no charge)", "PA03 102 ch2 (no charge)"] + \
            [f"ACM{b} ch{c}" for b in (106, 109, 110) for c in range(4)]
    for lab in order:
        p = ps.loc[lab] if lab in ps.index else pd.Series(dtype=float)
        s = sc.loc[lab] if lab in sc.index else pd.Series(dtype=float)
        g = lambda ser, k: float(ser[k]) if k in ser.index and pd.notna(ser[k]) else np.nan  # noqa: E731
        fit_ok = g(p, "trap-fit residual") <= 0.05 if np.isfinite(g(p, "trap-fit residual")) else False
        rows.append({
            "label": lab, "setup": "PA03, 130 K" if lab.startswith("PA03") else "LSM, 180 K",
            "gain": g(p, "gain (ADU/e)"), "cg": g(p, "conversion gain at input (uV/e)"), "csn": g(p, "C_sn, A=0.85 (fF)"),
            "cds": g(s, "CDS noise (nV/rtHz)"), "skip_e": g(p, "single-skip noise (e)"),
            "img": g(p, "image read noise, 500 skips (e)"), "img_ratio": g(p, "image / white-noise expectation"),
            "white": g(s, "white part (nV/rtHz)") if fit_ok else np.nan,
            "corner": g(s, "1/f corner (kHz)") if fit_ok else np.nan, "share": g(s, "1/f share") if fit_ok else np.nan,
            "nt": g(s, "N_t (1e16 /eV/cm3)") if fit_ok else np.nan, "resid": g(p, "trap-fit residual"),
            "onef": g(p, "transistor 1/f at 500 skips (e)") if fit_ok else np.nan,
            "kurt": g(s, "kurtosis"), "kurt90": g(s, "kurtosis p90 (all traces)"),
            "settle": g(s, "reset settling (us)"), "slope": g(p, "signal-window slope (e/us)"),
            "offset": g(p, "CDS offset (signal - pedestal, e)"), "rg": g(p, "RG feedthrough per volt (e/V)") / 1e3,
            "hft": g(p, "H-transfer waveform per volt (e/V)"), "vft": g(p, "V-transfer waveform per volt (e/V)"),
            "slow": g(s, "slow noise 0.2-1 kHz (uV/rtHz)"), "mid": g(s, "noise 1-10 kHz (uV/rtHz)"),
            "corr": g(s, "skip-to-skip corr"), "burst": g(s, "burst share"), "static": g(s, "static 0.2-1 kHz (uV/rtHz)"),
            "charge": g(p, "blank-image charge (e/pix)"), "traces": g(s, "traces"),
        })
        # board-102 ch0-2: a transistor, but no charge clocked to it -> trap numbers from trap_from_trace
        tl = lab.replace("(dead)", "(no charge)")
        if "no charge" in tl and tl in tr.index:
            t = tr.loc[tl]
            rows[-1].update({"label": tl, "resid": t["fit rms rel. residual"], "white": t["white ASD (nV/rtHz)"],
                             "corner": t["1/f corner (kHz)"], "share": t["1/f share at 10 us"],
                             "nt": t["N_t A=0.85 C_ox (eV^-1 cm^-3)"] / 1e16})
    D = pd.DataFrame(rows)
    sn = RES / "per_channel/setup_noise_N/setup_noise_N.csv"
    if sn.exists():
        S = pd.read_csv(sn).groupby("label").median(numeric_only=True)
        D["setup500"] = D["label"].map(S["setup 100-1000 Hz at 500 skips (e)"])
        D["setupshare"] = D["label"].map(S["setup share of 500-skip variance"])
    D["board"] = D["label"].str.extract(r"(1\d\d)")[0]
    D["charge_rel"] = D["charge"] / D.groupby("board")["charge"].transform("median")
    return D


def name(lab):
    return lab.replace("PA03 ", "").replace("ACM", "ACM\\,").replace("(no charge)", "\\textit{(no charge)}")


def table(D, cols, caption, label, note):
    """cols: list of (header, key, digits, grade?)"""
    spec = "ll" + "r" * len(cols)
    L = ["\\begin{table}[H]\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}",
         "\\begin{adjustbox}{max width=\\linewidth}", f"\\begin{{tabular}}{{{spec}}}\\toprule",
         "setup & channel & " + " & ".join(h for h, *_ in cols) + "\\\\\\midrule"]
    prev = None
    for _, r in D.iterrows():
        if prev is not None and r.setup != prev:
            L.append("\\midrule")
        st = r.setup if r.setup != prev else ""
        prev = r.setup
        L.append(f"{st} & {name(r.label)} & " + " & ".join(cell(k if gr else None, r[k], nd) for h, k, nd, gr in cols)
                 + "\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{adjustbox}", f"\\par\\smallskip\\footnotesize {note}", "\\end{table}"]
    return "\n".join(L) + "\n"


def thresholds(keys, unit):
    parts = []
    for k in keys:
        d, g, f = GRADES[k]
        op = "\\ge" if d == "high" else "\\le"
        bar = "|x|" if d == "abs" else "x"
        parts.append(f"{unit[k]}: \\colorbox{{good}}{{${bar}{op}{g:g}$}} \\colorbox{{fair}}{{${bar}{op}{f:g}$}}")
    return "Grades --- " + "; ".join(parts) + "; otherwise \\colorbox{poor}{poor}."


def summary(D):
    L = ["\\begin{table}[H]\\centering\\small",
         "\\caption{Report card: the typical grade of each category per channel (detail in Tables~\\ref{tab:sn}--\\ref{tab:setup}).}"
         "\\label{tab:card}",
         "\\begin{adjustbox}{max width=\\linewidth}", "\\begin{tabular}{ll" + "c" * len(CAT) + "}\\toprule",
         "setup & channel & " + " & ".join(CAT) + "\\\\\\midrule"]
    prev = None
    for _, r in D.iterrows():
        if prev is not None and r.setup != prev:
            L.append("\\midrule")
        st = r.setup if r.setup != prev else ""
        prev = r.setup
        cells = []
        for cat, keys in CAT.items():
            gs = [grade(k, r[k]) for k in keys]
            gs = [x for x in gs if x]
            if not gs:
                cells.append("--")
                continue
            srt = sorted(gs, key=RANK.get)
            w = srt[(len(srt) - 1) // 2 + (len(srt) % 2 == 0)]          # median grade (the worse one on a tie)
            cnt = "/".join(str(sum(x == k for x in gs)) for k in ("good", "fair", "poor"))
            cells.append(f"\\cellcolor{{{w}}}{w} \\scriptsize{{{cnt}}}")
        L.append(f"{st} & {name(r.label)} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{adjustbox}",
          "\\par\\smallskip\\footnotesize Colour = the typical (median) grade of the category; the small numbers "
          "count its graded quantities as good/fair/poor, so a single problem stays visible without colouring the whole "
          "category; ``--'': not measurable for that channel.",
          "\\end{table}"]
    return "\n".join(L) + "\n"


def history():
    a = pd.read_csv(RES / "anatomy/trace_anatomy.csv")
    a = a[a["duration (ms)"] < 300]                 # equal 0.2 s lengths (9 of 12 October traces are 1.1 s long)
    a = a.groupby("label").median(numeric_only=True)
    m = {"Oct 101ch1": "101 ch1, October (before cabling; 0.2\\,s traces)", "Nov v1 101ch1": "101 ch1, Nov 2--3",
         "Nov v3 101ch1": "101 ch1, Nov 10--12", "Nov v3 dead 102ch0": "102 ch0, no charge (same-board reference)"}
    L = ["\\begin{table}[H]\\centering\\small",
         "\\caption{Setup history of PA03 at equal trace length (0.2\\,s): the cabling change seen from the trace alone.}\\label{tab:hist}",
         "\\begin{tabular}{lrrrrr}\\toprule",
         "trace set & CDS noise & slow noise 0.2--1\\,kHz & skip-to-skip corr. & bursts & 1--7.5\\,MHz\\\\",
         " & (nV/$\\sqrt{\\text{Hz}}$) & (\\textmu V/$\\sqrt{\\text{Hz}}$) & & & (nV/$\\sqrt{\\text{Hz}}$)\\\\\\midrule"]
    for k, lab in m.items():
        r = a.loc[k]
        L.append(f"{lab} & {cell('cds', r['CDS white (nV/rtHz)'], 1)} & "
                 f"{cell('slow', r['ASD 200-1000 Hz (nV/rtHz)'] / 1e3, 1)} & {cell('corr', r['CDS lag-1 autocorr'], 2)} & "
                 f"{cell('burst', r['burst fraction (1 ms blocks > 1.5x median rms)'], 2)} & "
                 f"{fmt(r['ASD 1e+06-7.5e+06 Hz (nV/rtHz)'], 1)}\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(exist_ok=True)
    D = load()
    U = {"cg": "conv.\\ gain", "csn": "$C_\\text{sn}$", "cds": "CDS noise", "skip_e": "per skip", "img": "image",
         "img_ratio": "image/expectation", "white": "white", "corner": "1/f corner", "onef": "1/f at 500 skips", "nt": "$N_t$", "kurt": "kurtosis",
         "kurt90": "kurtosis p90", "settle": "settling", "slope": "slope", "offset": "offset", "rg": "RG",
         "setup500": "setup at 500 skips", "slow": "slow noise", "corr": "corr.", "burst": "bursts", "static": "static", "charge_rel": "charge/median"}
    (OUT / "card.tex").write_text(summary(D))
    (OUT / "sense_node.tex").write_text(table(D, [
        ("gain (ADU/e)", "gain", 2, False), ("conv.\\ gain (\\textmu V/e)", "cg", 2, False),
        ("$C_\\text{sn}$ (fF)", "csn", 0, True)],
        "Sense node: conversion gain and capacitance ($A=0.85$).", "tab:sn",
        thresholds(["csn"], U)))
    (OUT / "noise.tex").write_text(table(D, [
        ("CDS (nV/$\\sqrt{\\text{Hz}}$)", "cds", 1, True), ("per skip (e)", "skip_e", 2, True),
        ("image, 500 skips (e)", "img", 3, True), ("image / expectation", "img_ratio", 2, True)],
        "Read noise: per skip at the input and in electrons, and what the images reach after 500 skips.", "tab:noise",
        thresholds(["cds", "skip_e", "img", "img_ratio"], U)))
    (OUT / "transistor.tex").write_text(table(D, [
        ("1/f at 500 skips (e)", "onef", 3, True),
        ("white (nV/$\\sqrt{\\text{Hz}}$)", "white", 1, False), ("1/f corner (kHz)", "corner", 0, False),
        ("1/f share", "share", 2, False), ("$N_t$ ($10^{16}$/eV/cm$^3$)", "nt", 2, True),
        ("kurtosis", "kurt", 2, True), ("kurtosis p90", "kurt90", 2, True), ("fit resid.", "resid", 3, True)],
        "Output transistor: white and 1/f noise, oxide trap density, telegraph noise.", "tab:tr",
        thresholds(["onef", "nt", "kurt", "kurt90"], U)
        + " Fit residual above 5\\,\\% $\\Rightarrow$ the trap numbers are withheld."))
    (OUT / "timing.tex").write_text(table(D, [
        ("reset settling (\\textmu s)", "settle", 2, True), ("window slope (e/\\textmu s)", "slope", 1, True),
        ("CDS offset (e)", "offset", 0, True), ("RG (ke/V)", "rg", 1, False),
        ("H (ke/V)", "hft", "k", False), ("V (ke/V)", "vft", "k", False)],
        "Timing and clocks: settling, in-window slope, the CDS offset, and clock feedthrough per volt of swing.",
        "tab:tim", thresholds(["settle", "slope", "offset"], U)
        + " RG, H and V: relative only, not graded (they include board pickup and the whole transfer waveform)."))
    (OUT / "setup.tex").write_text(table(D, [
        ("setup at 500 skips (e)", "setup500", 3, True), ("its share", "setupshare", 2, False),
        ("slow 0.2--1\\,kHz (\\textmu V/$\\sqrt{\\text{Hz}}$)", "slow", 1, True),
        ("1--10\\,kHz (\\textmu V/$\\sqrt{\\text{Hz}}$)", "mid", 2, False),
        ("skip corr.", "corr", 2, True), ("bursts", "burst", 2, True),
        ("static 0.2--1\\,kHz", "static", 2, True), ("blank-image charge (e/pix)", "charge", 2, False),
        ("charge / board median", "charge_rel", 2, True)],
        "The setup (and the charge in blank images).", "tab:setup",
        thresholds(["setup500", "slow", "corr", "burst", "static", "charge_rel"], U)
        + " Setup at 500 skips: the static-record spectrum at 100\\,Hz--1\\,kHz integrated through the exact CDS + "
        "500-skip filter, in electrons (LSM only: PA03 has no static records)."
        + " Blank-image charge depends on temperature and exposure, so it is graded only against the other channels of the same board."))
    (OUT / "history.tex").write_text(history())
    D.to_csv(OUT / "graded_values.csv", index=False)
    print("tables written to", OUT)


if __name__ == "__main__":
    main()

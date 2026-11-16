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
GRADES = {
    "cg": ("high", 1.55, 1.45), "csn": ("low", 88, 95),
    "cds": ("low", 17.5, 20.0), "skip_e": ("low", 3.8, 4.2), "img": ("low", 0.16, 0.20), "img_ratio": ("low", 1.0, 1.3),
    "white": ("low", 10, 15), "corner": ("low", 50, 120), "nt": ("low", 2.0, 5.0), "kurt": ("abs", 0.3, 1.0),
    "kurt90": ("abs", 0.15, 0.5), "resid": ("low", 0.05, 0.05),
    "settle": ("low", 0.6, 1.2), "slope": ("abs", 2.0, 8.0), "offset": ("abs", 50, 300), "rg": ("low", 7.0, 10.0),
    "slow": ("low", 5.0, 15.0), "corr": ("low", 0.2, 0.5), "burst": ("low", 0.02, 0.08), "static": ("low", 1.0, 3.0),
    "charge": ("low", 0.3, 0.6),
}
CAT = {  # summary card: category -> graded keys
    "Sense node": ["cg", "csn"],
    "Read noise": ["cds", "skip_e", "img", "img_ratio"],
    "Transistor (1/f, traps)": ["corner", "nt", "kurt", "kurt90"],
    "Timing / clocks": ["settle", "slope", "offset", "rg"],
    "Setup": ["slow", "corr", "burst", "static"],
    "Charge in images": ["charge"],
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
    order = ["PA03 101 ch0", "PA03 101 ch1", "PA03 101 ch2", "PA03 101 ch3", "PA03 102 ch3", "PA03 102 ch0 (dead)"] + \
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
            "kurt": g(s, "kurtosis"), "kurt90": g(s, "kurtosis p90 (all traces)"),
            "settle": g(s, "reset settling (us)"), "slope": g(p, "signal-window slope (e/us)"),
            "offset": g(p, "CDS offset (signal - pedestal, e)"), "rg": g(p, "RG feedthrough per volt (e/V)") / 1e3,
            "hft": g(p, "H-transfer waveform per volt (e/V)"), "vft": g(p, "V-transfer waveform per volt (e/V)"),
            "slow": g(s, "slow noise 0.2-1 kHz (uV/rtHz)"), "mid": g(s, "noise 1-10 kHz (uV/rtHz)"),
            "corr": g(s, "skip-to-skip corr"), "burst": g(s, "burst share"), "static": g(s, "static 0.2-1 kHz (uV/rtHz)"),
            "charge": g(p, "blank-image charge (e/pix)"), "traces": g(s, "traces"),
        })
        if "dead" in lab:                         # no amplifier: only the electronics-noise and setup numbers apply
            for k in ("settle", "slope", "offset", "rg", "hft", "vft", "white", "corner", "share", "nt", "resid"):
                rows[-1][k] = np.nan
    return pd.DataFrame(rows)


def name(lab):
    return lab.replace("PA03 ", "").replace("ACM", "ACM\\,").replace("(dead)", "\\textit{(dead)}")


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
         "\\caption{Report card: the worst grade of each category per channel (detail in Tables~\\ref{tab:sn}--\\ref{tab:setup}).}"
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
            w = max(gs, key=RANK.get)
            n = sum(x == w for x in gs)
            cells.append(f"\\cellcolor{{{w}}}{w}" + (f" ({n}/{len(gs)})" if w != "good" else ""))
        L.append(f"{st} & {name(r.label)} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{adjustbox}",
          "\\par\\smallskip\\footnotesize ``fair (2/4)'': the worst grade is fair, reached by 2 of the 4 graded "
          "quantities of that category; ``--'': not measurable for that channel.",
          "\\end{table}"]
    return "\n".join(L) + "\n"


def history():
    a = pd.read_csv(RES / "anatomy/trace_anatomy.csv").groupby("label").median(numeric_only=True)
    m = {"Oct 101ch1": "101 ch1, October (before cabling)", "Nov v1 101ch1": "101 ch1, Nov 2--3",
         "Nov v3 101ch1": "101 ch1, Nov 10--12", "Nov v3 dead 102ch0": "102 ch0, dead (electronics only)"}
    L = ["\\begin{table}[H]\\centering\\small",
         "\\caption{Setup history of PA03 (12 traces each): the cabling change seen from the trace alone.}\\label{tab:hist}",
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
         "img_ratio": "image/expectation", "white": "white", "corner": "1/f corner", "nt": "$N_t$", "kurt": "kurtosis",
         "kurt90": "kurtosis p90", "settle": "settling", "slope": "slope", "offset": "offset", "rg": "RG",
         "slow": "slow noise", "corr": "corr.", "burst": "bursts", "static": "static", "charge": "charge"}
    (OUT / "card.tex").write_text(summary(D))
    (OUT / "sense_node.tex").write_text(table(D, [
        ("gain (ADU/e)", "gain", 2, False), ("conv.\\ gain (\\textmu V/e)", "cg", 2, True),
        ("$C_\\text{sn}$ (fF)", "csn", 0, True)],
        "Sense node: conversion gain and capacitance ($A=0.85$).", "tab:sn",
        thresholds(["cg", "csn"], U)))
    (OUT / "noise.tex").write_text(table(D, [
        ("CDS (nV/$\\sqrt{\\text{Hz}}$)", "cds", 1, True), ("per skip (e)", "skip_e", 2, True),
        ("image, 500 skips (e)", "img", 3, True), ("image / expectation", "img_ratio", 2, True)],
        "Read noise: per skip at the input and in electrons, and what the images reach after 500 skips.", "tab:noise",
        thresholds(["cds", "skip_e", "img", "img_ratio"], U)))
    (OUT / "transistor.tex").write_text(table(D, [
        ("white (nV/$\\sqrt{\\text{Hz}}$)", "white", 1, True), ("1/f corner (kHz)", "corner", 0, True),
        ("1/f share", "share", 2, False), ("$N_t$ ($10^{16}$/eV/cm$^3$)", "nt", 2, True),
        ("kurtosis", "kurt", 2, True), ("kurtosis p90", "kurt90", 2, True), ("fit resid.", "resid", 3, True)],
        "Output transistor: white and 1/f noise, oxide trap density, telegraph noise.", "tab:tr",
        thresholds(["white", "corner", "nt", "kurt", "kurt90"], U)
        + " Fit residual above 5\\,\\% $\\Rightarrow$ the trap numbers are withheld."))
    (OUT / "timing.tex").write_text(table(D, [
        ("reset settling (\\textmu s)", "settle", 2, True), ("window slope (e/\\textmu s)", "slope", 1, True),
        ("CDS offset (e)", "offset", 0, True), ("RG (ke/V)", "rg", 1, True),
        ("H (ke/V)", "hft", "k", False), ("V (ke/V)", "vft", "k", False)],
        "Timing and clocks: settling, in-window slope, the CDS offset, and clock feedthrough per volt of swing.",
        "tab:tim", thresholds(["settle", "slope", "offset", "rg"], U)
        + " H and V: relative only, not graded (they include board pickup and the whole transfer waveform)."))
    (OUT / "setup.tex").write_text(table(D, [
        ("slow 0.2--1\\,kHz (\\textmu V/$\\sqrt{\\text{Hz}}$)", "slow", 1, True),
        ("1--10\\,kHz (\\textmu V/$\\sqrt{\\text{Hz}}$)", "mid", 2, False),
        ("skip corr.", "corr", 2, True), ("bursts", "burst", 2, True),
        ("static 0.2--1\\,kHz", "static", 2, True), ("blank-image charge (e/pix)", "charge", 2, True)],
        "The setup (and the charge in blank images).", "tab:setup",
        thresholds(["slow", "corr", "burst", "static", "charge"], U)))
    (OUT / "history.tex").write_text(history())
    D.to_csv(OUT / "graded_values.csv", index=False)
    print("tables written to", OUT)


if __name__ == "__main__":
    main()

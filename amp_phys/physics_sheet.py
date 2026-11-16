"""
physics_sheet.py -- the device physics of every amplifier, PA03 (130 K) and LSM DM07/09/10 (180 K), side by side.

Per channel: conversion gain and sense-node capacitance, read noise (per skip, after N skips), noise physics (white,
1/f corner, oxide trap density, telegraph), clock -> sense-node couplings in e/V and fF, bandwidth/settling, and the
setup.  Sources: v3n/datasheet.csv (PA03 image gain, sigma_e), per_channel/ (trace anatomy, traps), lsm_hightemp/
(trace anatomy, traps, waders image fits: MEGAIN/MESIG/MEDC; EXTn = channel n-1).

  conversion gain at the video input CG = gain / G (G = 0.875 ADU/uV design; 0.853 board 102)
  C_sn = A q / CG (A = 0.85, follower gain from the layout)          [model]
  feedthrough per volt (e/V) = waveform size (ADU) / gain / swing (V): RELATIVE only -- it includes board pickup and the
  whole transfer waveform; the per-line mutual capacitances (fF) need a voltage scan (clock_edges.py, PA03 only)

    .venv/bin/python amp_phys/physics_sheet.py --out amp_phys/results/per_channel
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "results"
Q = 1.602176634e-19
A_SF = 0.85
SW_PA03 = {"RG": 8.88 - 3.0, "SW": 1.0 + 7.1, "H": 9.1 - 6.1, "V": 9.85 - 8.0}
SW_LSM = {"RG": 9.0 - 3.0, "SW": 1.0 + 6.0, "H": 9.5 - 6.5, "V": 9.9 - 7.7}


_F = np.logspace(-1, np.log10(7.5e6), 400000)
_DF = np.gradient(_F)


def onef_integral(N, tau=10e-6, dt=13.8e-6, T=32.6e-6):
    """int |H|^2/f df for the CDS (windows tau, centres dt apart) averaged over N skips of period T."""
    x = np.pi * _F * tau
    H = 4 * np.sin(np.pi * _F * dt) ** 2 * (np.sin(x) / x) ** 2
    y = np.pi * _F * T
    with np.errstate(all="ignore"):
        A = np.nan_to_num((np.sin(N * y) / (N * np.sin(y))) ** 2, nan=1.0)
    return np.sum(H * A / _F * _DF)


def anat(path, labels=None):
    d = pd.read_csv(path)
    return d.groupby("label").median(numeric_only=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # ---- PA03
    ds = pd.read_csv(R / "v3n/datasheet.csv").set_index("Unnamed: 0")
    an = pd.concat([pd.read_csv(f) for f in (R / "per_channel").glob("anat_*/trace_anatomy.csv")]).groupby("label").median(numeric_only=True)
    tr = pd.read_csv(R / "trap_from_trace_v3/trap_from_trace.csv").groupby("label").median(numeric_only=True)
    sa = pd.read_csv(R / "per_channel/skip_averaging/skip_averaging.csv")
    rows = []
    for amp in ("101 ch0", "101 ch1", "101 ch2", "101 ch3", "102 ch3"):
        lab = f"PA03 {amp}"
        G = 0.853 if amp.startswith("102") else 0.875
        rows.append(dict(setup="PA03 130 K", channel=lab, G=G, gain=ds.loc["gain | gain (ADU/e)", amp],
                         sig_img=ds.loc["image | sigma_e (e)", amp], img_skips=500,
                         charge=ds.loc["image | lambda (e/pix)", amp], swings=SW_PA03,
                         kurt_p90=sa[sa.label == lab]["kurtosis"].quantile(0.9), **an.loc[lab].to_dict(),
                         **{f"trap {k}": v for k, v in tr.loc[lab].to_dict().items()}))
    # ---- LSM
    la = pd.read_csv(R / "lsm_hightemp/anatomy/trace_anatomy.csv").groupby("label").median(numeric_only=True)
    lt = pd.read_csv(R / "lsm_hightemp/traps/trap_from_trace.csv").groupby("label").median(numeric_only=True)
    wf = pd.read_csv(R / "lsm_hightemp/waders_image_fits.csv")
    w = wf[wf.skips == 1000].groupby(["board", "channel"])[["gain", "dc"]].median()
    w["sig"] = wf[wf.skips == 500].groupby(["board", "channel"])["sig"].median()
    for b in (106, 109, 110):
        for c in range(4):
            lab = f"ACM{b} ch{c}"
            rows.append(dict(setup="LSM 180 K", channel=lab, G=0.875, gain=w.loc[(b, c), "gain"],
                             sig_img=w.loc[(b, c), "sig"], img_skips=1000, charge=w.loc[(b, c), "dc"], swings=SW_LSM,
                             kurt_p90=np.nan, **la.loc[lab].to_dict(),
                             **{f"trap {k}": v for k, v in lt.loc[lab].to_dict().items()}))
    I500 = onef_integral(500)
    out = []
    for r in rows:
        g, sw = r["gain"], r["swings"]
        cg = g / r["G"]
        o = {"setup": r["setup"], "channel": r["channel"],
             # sense node
             "gain (ADU/e)": g, "conversion gain at input (uV/e)": cg,
             "C_sn, A=0.85 (fF)": A_SF * Q / (cg * 1e-6) * 1e15,
             # read noise
             "single-skip noise (e)": r["CDS white per skip (ADU)"] / g,
             "image read noise, 500 skips (e)": r["sig_img"],
             "image / white-noise expectation": r["sig_img"] / (r["CDS white per skip (ADU)"] / g / np.sqrt(500)),
             "CDS white noise (nV/rtHz)": r["CDS white (nV/rtHz)"],
             # noise physics
             "white part (nV/rtHz)": r["trap white ASD (nV/rtHz)"], "1/f corner (kHz)": r["trap 1/f corner (kHz)"],
             "N_t (1e16 /eV/cm3)": r["trap N_t A=0.85 C_ox (eV^-1 cm^-3)"] / 1e16,
             "trap-fit residual": r["trap fit rms rel. residual"],
             # amplifier health: the transistor's own 1/f noise in the 500-skip pixel (upper limit; robust: the 1/f
             # level K agrees with the static-record spectrum to 1.01 +- 0.13 on the LSM channels)
             "transistor 1/f at 500 skips (e)": np.sqrt(max(r["trap 1/f K (ADU^2)"], 0) * I500) / g,
             "kurtosis (median trace)": r["excess kurtosis of the CDS"], "kurtosis p90 (all traces)": r["kurt_p90"],
             # couplings
             "RG feedthrough per volt (e/V)": r["reset pulse height (ADU)"] / g / sw["RG"],
             "H-transfer waveform per volt (e/V)": r["H-transfer waveform peak-to-peak (ADU)"] / g / sw["H"],
             "V-transfer waveform per volt (e/V)": r["V-transfer waveform peak-to-peak (ADU)"] / g / sw["V"],
             "CDS offset (signal - pedestal, e)": r["SW feedthrough (signal - pedestal level of the template, ADU)"] / g,
             # bandwidth
             "reset settling (us)": r["settling tau after reset (us)"],
             "signal-window slope (e/us)": r["signal-window slope (ADU/us)"] / g,
             # setup
             "slow noise 0.2-1 kHz (uV/rtHz)": r["ASD 200-1000 Hz (nV/rtHz)"] / 1e3,
             "burst share": r["burst fraction (1 ms blocks > 1.5x median rms)"],
             "blank-image charge (e/pix)": r["charge"]}
        out.append(o)
    S = pd.DataFrame(out).set_index(["setup", "channel"])
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    S.to_csv(o / "physics_sheet.csv")
    (o / "physics_sheet.md").write_text(S.T.round(3).to_markdown() + "\n")
    pd.set_option("display.width", 400)
    pd.set_option("display.max_columns", 40)
    print(S.T.to_string(float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()

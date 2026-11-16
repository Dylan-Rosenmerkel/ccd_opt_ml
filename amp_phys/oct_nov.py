"""
oct_nov.py -- October vs November (the cabling change), with data that is valid for it.

IMAGES (the four amplifiers of a board are read SIMULTANEOUSLY into one file): board 101, 500 skips, unbinned, at the
October nominal voltages (VSUB, VR1, VDD1, H1_A_H, V1_A_H, DELAY_HA equal to October's).  Per image and amplifier,
on the pixels where every amplifier is in its 0-electron peak (|x| < 2.5 sigma):
  sigma_0e        robust width of the 0-e peak (ADU, and e- with the period's gain)
  corr            mean simultaneous correlation with the other three amplifiers (row baseline removed / not removed)
  CMS reduction   % lower sigma after subtracting beta x (mean of the other amplifiers), beta fitted on half the pixels
TRACES (each channel's trace is its own acquisition, so only per-channel quantities): white noise per skip
(cubic-detrended CDS per pixel), converted to nV/rtHz at the video input with the chain gain (0.875 ADU/uV) and the
10 us window, and the slow part (pixel-mean scatter between the 10 pixels, after the per-position pattern).

    .venv/bin/python amp_phys/oct_nov.py --out amp_phys/results/oct_nov
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RAW = Path("/pool/storage/DAQ1_Data/run/PA_PQ_testing")
GAIN = {"Oct": [1.324, 1.304, 1.248, 1.256], "Nov": [1.398, 1.384, 1.332, 1.336]}   # image agent / v3 labels
NOM = {"VSUB": 45.095, "VR1": None, "VDD1": None, "H1_A_H": None, "V1_A_H": None, "DELAY_HA": 1800}


def period(date):
    if date < "20261101":
        return "Oct"
    return {"20261102": "Nov 2-3 (v1)", "20261103": "Nov 2-3 (v1)"}.get(date, "Nov 5-7 (v2)" if date < "20261110" else "Nov 10-12 (v3)")


def image_stats(f, gains):
    h = fits.open(f)
    X0 = np.array([h[i].data.astype(float) for i in range(1, 5)])[:, :, 10:]
    out = []
    for base in ("row", "none"):
        X = X0 - np.median(X0, axis=(1, 2), keepdims=True)
        if base == "row":
            X = X - np.median(X, axis=2, keepdims=True)
        s = 1.4826 * np.median(np.abs(X), axis=(1, 2))
        zero = np.all(np.abs(X) < 2.5 * s[:, None, None], 0).ravel()
        V = X.reshape(4, -1)[:, zero]
        if V.shape[1] < 500:
            return []
        C = np.corrcoef(V)
        half = np.zeros(V.shape[1], bool)
        half[::2] = True
        for i in range(4):
            x, ref = V[i], np.delete(V, i, 0).mean(0)
            beta = (x[half] @ ref[half]) / (ref[half] @ ref[half])
            sb = 1.4826 * np.median(np.abs(x[~half] - np.median(x[~half])))
            y = x[~half] - beta * ref[~half]
            sa = 1.4826 * np.median(np.abs(y - np.median(y)))
            out.append({"baseline": base, "amp": i, "sigma_0e (ADU)": sb, "sigma_0e (e)": sb / gains[i],
                        "corr": (C[i].sum() - 1) / 3, "CMS reduction %": 100 * (1 - sa / sb), "n_pix": int(zero.sum())})
    return out


def trace_stats_raw(f):
    from decode_trace import decode_trace
    t = decode_trace(f)
    s = t.skips()
    npx = len(s["pixel_ids"])
    c = s["cds"].reshape(npx, -1)[:, 20:]
    return c


def trace_numbers(c, G=0.875, tau=10e-6):
    tt = np.linspace(-1, 1, c.shape[1])
    V = np.vander(tt, 4)
    det = c - (c @ np.linalg.pinv(V).T) @ V.T
    return det.std() / G * 1e-6 * np.sqrt(tau) * 1e9, c.std(1).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-nov", type=int, default=25, help="images per November period")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    I = pd.read_csv(ROOT / "amp_phys/results/images/inventory.csv")
    I = I[I.file.str.startswith("avg_img") & I.ACMID.astype(str).str.contains("101") & (I.NDCM == 500) & (I.NPBIN == 1)].copy()
    I["date"] = I.file.str.extract(r"_(2026\d{4})_")[0]
    I = I[I.date >= "20261001"]
    octo = I[I.date < "20261101"]
    nomv = {k: octo[k].median() for k in ("VSUB", "VR1", "VDD1", "H1_A_H", "V1_A_H", "DELAY_HA")}
    sel = np.ones(len(I), bool)
    for k, v in nomv.items():
        sel &= np.isclose(pd.to_numeric(I[k], errors="coerce"), v, atol=0.06)
    I = I[sel]
    I["period"] = I.date.map(period)
    rows = []
    for per, g in I.groupby("period"):
        g = g.sort_values("file")
        if not per.startswith("Oct"):
            g = g.iloc[np.linspace(0, len(g) - 1, min(args.max_nov, len(g))).astype(int)]
        for _, r in g.iterrows():
            f = RAW / r.run / r.file
            for s in image_stats(f, GAIN["Oct" if per == "Oct" else "Nov"]):
                rows.append({"period": per, "file": r.file, **s})
    R = pd.DataFrame(rows)
    R.to_csv(out / "images.csv", index=False)
    T = R.groupby(["period", "baseline", "amp"])[["sigma_0e (e)", "corr", "CMS reduction %"]].median().unstack("amp")
    n = R.groupby("period").file.nunique()
    # traces: October raw (Oct 8, Oct 12) vs November processed ref traces
    trows = []
    for d in ("2026-10-08-PA03_PQ04_traces", "2026-10-12-PA03_PQ04_Packet_Tests"):
        for ch in range(4):
            for fm in sorted(glob.glob(str(RAW / d / f"trace_2x5x500_*_ch{ch}_101_*.RJ45ADCDATA{ch}")))[:40]:
                try:
                    c = trace_stats_raw(fm)
                except Exception:
                    continue
                ew, _ = trace_numbers(c)
                trows.append({"period": "Oct " + d[8:10], "channel": ch, "e_w (nV/rtHz)": ew})
    for run, per in (("v1", "Nov 2-3 (v1)"), ("v2", "Nov 5-7 (v2)"), ("v3n", "Nov 10-12 (v3)")):
        for ch in range(4):
            for f in sorted(glob.glob(str(ROOT / f"processed_{run}" / "*" / "traces" / f"ref*_acm101_ch{ch}.npz")))[:40]:
                z = np.load(f, allow_pickle=True)
                if "cds" not in z.files:
                    continue
                ew, _ = trace_numbers(z["cds"].astype(float)[:, 20:])
                trows.append({"period": per, "channel": ch, "e_w (nV/rtHz)": ew})
    TR = pd.DataFrame(trows)
    TR.to_csv(out / "traces.csv", index=False)
    TT = TR.groupby(["period", "channel"])["e_w (nV/rtHz)"].median().unstack("channel")
    L = ["# October vs November (board 101)", "", f"Images per period: {n.to_dict()}; nominal voltages {nomv}.", "",
         "## Images (simultaneous amplifiers), medians", "", T.round(3).to_markdown(), "",
         "## Traces: white input noise density per channel (nV/rtHz; per acquisition)", "", TT.round(2).to_markdown()]
    (out / "oct_nov.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

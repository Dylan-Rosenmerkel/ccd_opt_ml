"""
trace_anatomy.py -- the WHOLE trace, section by section and as a whole: what each part says about the amplifier's
physics, its performance and the setup.

A trace (one channel, 15 MS/s, ~188 ms) is segmented with the sequencer flags (bit 7 V transfer, bit 6 H transfer,
bit 3 the static gaps between H transfers, bit 0/1 pedestal/signal windows, bit 4 low during the reset pulse):

  SECTION              WHAT IT MEASURES
  V transfers          vertical-clock edges at the output (coupling; slow V edges through 2.35 us RC)
  H transfers          H-clock edges: jumps (coupling), and the reset events inside them
  H gaps (static)      the output with NO clock switching: settling time constant after the last edge (output
                       node R x C: bandwidth), then the noise of the quiet system (setup floor, own spectrum)
  reset (each skip)    reset pulse height (RG feedthrough), the level jump at reset (charge injection + kTC +
                       electronics), its skip-to-skip scatter
  pedestal / signal    CDS noise (white), window settling (slope inside the window), the in-window spectrum
  WHOLE TRACE          template-removed noise record: low-frequency wander index (setup), line inventory (lines
                       present also on a dead channel = electronics/environment), bursts (1 ms blocks with excess
                       rms), drift over 188 ms, stationarity, and the noise during clocking vs with clocks static
                       (clocking-induced noise)

    .venv/bin/python amp_phys/trace_anatomy.py --traces <npz | RJ45ADCDATA> ... --label <name> --out <dir> [--plot]
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
from scipy.optimize import curve_fit  # noqa: E402
from scipy.signal import welch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
FS = 15e6
G = 0.875                                   # ADU per uV at the video input (chain_calibration.py)


def load(path):
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return z["values"].astype(float), z["flags"].astype(np.int64)
    from decode_trace import decode_trace
    t = decode_trace(path)
    fl = np.zeros(len(t.values), np.int64)
    fl |= np.asarray(t.rd).astype(np.int64)
    fl |= np.asarray(t.ru).astype(np.int64) << 1
    if hasattr(t, "flags"):
        fl = np.asarray(t.flags).astype(np.int64)
    return t.values.astype(float), fl


def rising(fl, bit):
    x = (fl >> bit) & 1
    return np.flatnonzero(np.diff(x) == 1) + 1


def runs(mask):
    d = np.diff(mask.astype(int))
    s = np.flatnonzero(d == 1) + 1
    e = np.flatnonzero(d == -1) + 1
    if mask[0]:
        s = np.r_[0, s]
    if mask[-1]:
        e = np.r_[e, len(mask)]
    return list(zip(s, e))


def exp_tau(seg):
    t = np.arange(len(seg)) / FS * 1e6
    try:
        p, _ = curve_fit(lambda t, a, b, c: a + b * np.exp(-t / c), t, seg, p0=[seg[-1], seg[0] - seg[-1], 5],
                         bounds=([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 500]), maxfev=5000)
        return p[2]
    except Exception:
        return np.nan


def anatomy(v, fl):
    out = {"duration (ms)": len(v) / FS * 1e3}
    # ---------------- skip cycles
    e = rising(fl, 0)
    gaps = np.diff(e)
    P = int(np.median(gaps))
    out["skip period (us)"] = P / FS * 1e6
    good = np.r_[np.abs(gaps - P) <= 3, False]
    # the window offsets inside a cycle, then keep only real skip cycles: the last pedestal of every pixel has NO
    # signal window after it (it would put the next region into the "signal" window)
    sig0 = int(np.median([s for k in e[good][:300] for s in rising(fl[k:k + P], 1)[:1]]))
    has_sig = np.array([k + sig0 + 2 < len(fl) and ((fl[k + sig0 + 2] >> 1) & 1) == 1 for k in e])
    cand = e[good & has_sig]
    # drop the first 20 skips of EACH pixel (the first-skip transient), not of the trace
    newpix = np.r_[True, np.abs(np.diff(cand) - P) > 3]
    pix_id = np.cumsum(newpix)
    pos_in_pix = np.arange(len(cand)) - np.flatnonzero(newpix)[pix_id - 1]
    cyc_starts = cand[pos_in_pix >= 20]
    C = np.array([v[k:k + P] for k in cyc_starts if k + P <= len(v)])
    tmpl = np.median(C, 0)
    R = C - tmpl
    ped_len = int(np.median([b - a for a, b in runs(((fl >> 0) & 1).astype(bool))][:200]))
    sig_start = int(np.median([s for k in cyc_starts[:200] for s in rising(fl[k:k + P], 1)[:1]]))
    ped = C[:, 5:ped_len - 5].mean(1)
    sig = C[:, sig_start + 5:sig_start + ped_len - 5].mean(1)
    cds = sig - ped
    # every per-skip statistic WITHIN a pixel: consecutive cycles one period apart (pixel charge and the per-pixel
    # transient do not enter)
    same = np.abs(np.diff(cyc_starts[:len(cds)]) - P) <= 3
    pid = np.r_[0, np.cumsum(~same)]
    cds_c = cds - pd.Series(cds).groupby(pid).transform("mean").to_numpy()
    out["CDS white per skip (ADU)"] = np.std(np.diff(cds)[same]) / np.sqrt(2)
    out["CDS white (nV/rtHz)"] = out["CDS white per skip (ADU)"] / G * 1e3 * np.sqrt(ped_len / FS)
    out["CDS lag-1 autocorr"] = np.corrcoef(cds_c[:-1][same], cds_c[1:][same])[0, 1]
    lvl = (ped + sig) / 2
    out["per-skip level scatter (ADU)"] = np.std(np.diff(lvl)[same]) / np.sqrt(2)    # reset: kTC + electronics
    # reset pulse: where bit 4 is low inside the cycle
    b4 = (fl[cyc_starts[0]:cyc_starts[0] + P] >> 4) & 1
    rs = np.flatnonzero(b4 == 0)
    if len(rs):
        r0, r1 = rs[0], rs[-1]
        out["reset pulse height (ADU)"] = tmpl[r0:r1 + 30].max() - tmpl[max(r0 - 20, 0):r0].mean()
        out["settling tau after reset (us)"] = exp_tau(tmpl[r1 + 3:min(r1 + 3 + 400, P)]) if r1 + 50 < P else np.nan
    # in-window settling: slope of the mean pedestal window (fA-level leakage or settling tail)
    w = tmpl[5:ped_len - 5]
    out["pedestal-window slope (ADU/us)"] = np.polyfit(np.arange(len(w)) / FS * 1e6, w, 1)[0]
    ws = tmpl[sig_start + 5:sig_start + ped_len - 5]
    out["signal-window slope (ADU/us)"] = np.polyfit(np.arange(len(ws)) / FS * 1e6, ws, 1)[0]
    out["SW feedthrough (signal - pedestal level of the template, ADU)"] = ws.mean() - w.mean()
    # in-window random noise spectrum
    Wn = R[:, 5:ped_len - 5]
    f_w = np.fft.rfftfreq(Wn.shape[1], 1 / FS)
    P_w = (np.abs(np.fft.rfft(Wn * np.hanning(Wn.shape[1]), axis=1)) ** 2).mean(0)
    hi = f_w > 3e6
    out["in-window power above 3 MHz (share)"] = P_w[hi].sum() / P_w[1:].sum()
    # ---------------- H transfers and the static gaps after them
    eh = rising(fl, 6)
    hruns = runs(((fl >> 6) & 1).astype(bool))
    hlen = int(np.median([b - a for a, b in hruns]))
    gap_noise, gap_tau, H = [], [], []
    for (a, b) in hruns:
        nxt = [k for k in e if k > b]
        end = min(nxt[0] if nxt else len(v), b + 1200)
        seg = v[b:end]
        if len(seg) > 700:
            gap_tau.append(exp_tau(seg[:400] - seg[-50:].mean()))
            q = seg[-600:]
            gap_noise.append(q - q.mean())
        if a > 300 and a + hlen + 1200 < len(v):
            H.append(v[a - 300:a + hlen + 900])
    if gap_noise:
        Gn = np.array([g for g in gap_noise if len(g) == 600])
        out["static-gap noise rms (ADU)"] = np.median(Gn.std(1))
        f_g = np.fft.rfftfreq(600, 1 / FS)
        P_g = (np.abs(np.fft.rfft(Gn * np.hanning(600), axis=1)) ** 2).mean(0)
        out["settling tau after H transfer (us)"] = np.nanmedian(gap_tau)
        # the same band from the windows (clocking on) vs the gaps (clocks static): clocking-induced noise
        f_b = (f_w > 0.2e6) & (f_w < 3e6)
        g_b = (f_g > 0.2e6) & (f_g < 3e6)
        out["windows / static-gap noise density 0.2-3 MHz"] = np.sqrt((P_w[f_b].mean() / Wn.shape[1]) /
                                                                        (P_g[g_b].mean() / 600))
    if H:
        Hm = np.median(np.array(H), 0)
        out["H-transfer waveform peak-to-peak (ADU)"] = np.ptp(Hm)
    # ---------------- V transfers
    vr = runs(((fl >> 7) & 1).astype(bool))
    if vr:
        a, b = vr[0]
        seg = v[max(a - 200, 0):min(b + 3000, len(v))]
        out["V-transfer waveform peak-to-peak (ADU)"] = np.ptp(seg - np.median(seg[:150]))
        out["V transfers in trace"] = len(vr)
    # ---------------- whole trace
    x = R.ravel()                                                          # skip region with the waveform removed
    f, Pr = welch(x, fs=FS, nperseg=2 ** 15, window="hann")
    asd = np.sqrt(Pr) / G * 1e3
    for lo, hi_ in ((200, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 7.5e6)):
        m = (f >= lo) & (f < hi_)
        out[f"ASD {lo:g}-{hi_:g} Hz (nV/rtHz)"] = np.median(asd[m])
    med = pd.Series(Pr).rolling(31, center=True, min_periods=1).median().to_numpy()
    ex = Pr / med
    pk = [i for i in np.argsort(-ex) if f[i] > 2e4][:6]
    out["lines (kHz)"] = ", ".join(f"{f[i] / 1e3:.0f}" for i in sorted(pk))
    # bursts and stationarity: rms of the residual in ~1 ms blocks
    blk = int(1e-3 * FS) // P * P
    nb = len(x) // blk
    br = x[:nb * blk].reshape(nb, blk).std(1)
    out["burst fraction (1 ms blocks > 1.5x median rms)"] = np.mean(br > 1.5 * np.median(br))
    out["stationarity (CV of 1 ms block rms)"] = br.std() / br.mean()
    out["drift of the per-skip level over the trace (ADU/ms)"] = np.polyfit(np.arange(len(lvl)) * P / FS * 1e3, lvl, 1)[0]
    # CDS kurtosis from consecutive-skip differences WITHIN a pixel (pixel boundaries = cycle starts not one period apart)
    starts = np.array([k for k in cyc_starts if k + P <= len(v)])
    within = np.abs(np.diff(starts) - P) <= 3
    dc = np.diff(cds)[within]
    out["excess kurtosis of the CDS"] = float(pd.Series(dc).kurt())
    return out, (tmpl, P, ped_len, sig_start, f, asd, (Hm if H else None))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, keep = [], None
    for t in a.traces:
        try:
            v, fl = load(t)
            r, extra = anatomy(v, fl)
        except Exception as ex:
            print("skip", Path(t).name, ex)
            continue
        r.update({"label": a.label, "file": Path(t).name})
        rows.append(r)
        keep = keep or extra
    R = pd.DataFrame(rows)
    p = out / "trace_anatomy.csv"
    R.to_csv(p, mode="a", header=not p.exists(), index=False)
    med = R.drop(columns=["file", "label", "lines (kHz)"]).median(numeric_only=True)
    print(f"== {a.label} ({len(R)} traces)")
    print(med.round(3).to_string())
    print("lines (kHz), first trace:", R["lines (kHz)"].iloc[0])
    if a.plot and keep:
        tmpl, P, ped_len, sig_start, f, asd, Hm = keep
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        ax = axs[0, 0]
        t = np.arange(P) / FS * 1e6
        ax.plot(t, tmpl, color="#0b0b0b", lw=1)
        ax.axvspan(0, ped_len / FS * 1e6, color="#dbe8f8")
        ax.axvspan(sig_start / FS * 1e6, (sig_start + ped_len) / FS * 1e6, color="#d8f2e7")
        ax.set_title("skip cycle (median): pedestal (blue), signal (green), reset/SW outside", loc="left", fontsize=9)
        ax.set_xlabel("µs")
        if Hm is not None:
            ax = axs[0, 1]
            ax.plot(np.arange(len(Hm)) / FS * 1e6 - 20, Hm - np.median(Hm[:250]), color="#2a78d6", lw=1)
            ax.set_title("horizontal transfer + static gap (median)", loc="left", fontsize=9)
            ax.set_xlabel("µs after the H flag")
        ax = axs[1, 0]
        ax.loglog(f[1:], asd[1:], color="#eb6834", lw=0.8)
        ax.set_title("whole-trace noise spectrum, waveform removed (nV/√Hz at the input)", loc="left", fontsize=9)
        ax.set_xlabel("Hz")
        ax = axs[1, 1]
        ax.axis("off")
        txt = "\n".join(f"{k}: {v:.3g}" for k, v in med.items())
        ax.text(0, 1, txt, va="top", fontsize=7, family="monospace")
        fig.suptitle(f"Trace anatomy: {a.label}", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / f"anatomy_{a.label.replace(' ', '_')}.png", dpi=140)


if __name__ == "__main__":
    main()

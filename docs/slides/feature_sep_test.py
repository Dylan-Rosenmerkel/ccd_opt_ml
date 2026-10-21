"""Do the 'rejected' cross-field CDS statistics separate SETTINGS (not only amplifiers)?  All 59 2026-10-12 traces.

Per trace: median over usable pixels of each statistic of the detrended CDS (skips >= 10).  Then, per amplifier
(ch0, ch1, ch3), a one-way ANOVA F over the four changed settings groups (all NCOLPRE 185, so the reference's NCOLPRE 0
cannot fake a separation), and an amplifier F over the four amplifiers within each group.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/dylanrosenmerkel/Documents/Software/ccd_opt_ml")
sys.path.insert(0, str(ROOT / "src"))
from compare_trace import usable_pixels  # noqa: E402
from trace_features import detrend  # noqa: E402
from trace_regions import load_regions  # noqa: E402

DATA = ROOT / "data/2026-10-12-PA03_PQ04_Packet_Tests"
OUT = Path(__file__).parent / "data" / "feature_sep_test.json"


def perm_entropy(x, m=3):
    pats = np.array([np.argsort(x[i:i + m]) for i in range(len(x) - m + 1)])
    codes = (pats * (m ** np.arange(m))).sum(1)
    _, c = np.unique(codes, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(m)))


def dfa_alpha(x, scales=(4, 6, 8, 12, 16, 24, 32, 48)):
    y = np.cumsum(x - x.mean())
    F = []
    for s in scales:
        m = len(y) // s
        seg = y[:m * s].reshape(m, s)
        t = np.arange(s)
        res = [seg[i] - np.polyval(np.polyfit(t, seg[i], 1), t) for i in range(m)]
        F.append(np.sqrt(np.mean(np.square(res))))
    return float(np.polyfit(np.log(scales), np.log(F), 1)[0])


def psd(x):
    S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    return S[1:]


def flatness(x):
    S = psd(x)
    return float(np.exp(np.mean(np.log(S))) / np.mean(S))


def spectral_slope(x):
    S = psd(x)
    f = np.arange(1, len(S) + 1)
    return float(np.polyfit(np.log(f), np.log(S), 1)[0])


def spectral_centroid(x):
    S = psd(x)
    f = np.arange(1, len(S) + 1) / (2 * len(S))
    return float((f * S).sum() / S.sum())


def sta_lta(x, ns=5, nl=50):
    e = x ** 2
    c = np.r_[0, np.cumsum(e)]
    k = np.arange(nl, len(x) - ns)
    sta = (c[k + ns] - c[k]) / ns
    lta = (c[k] - c[k - nl]) / nl
    return float(np.max(sta / lta))


def ljung_box_sq(x, h=10):
    e = x ** 2 - np.mean(x ** 2)
    n = len(e)
    d = np.dot(e, e)
    r = [np.dot(e[:-k], e[k:]) / d for k in range(1, h + 1)]
    return float(n * (n + 2) * sum(rk ** 2 / (n - k) for k, rk in enumerate(r, 1)))


def sample_entropy(x, m=2, r=0.2):
    x = (x - x.mean()) / x.std()
    n = len(x)

    def count(mm):
        emb = np.array([x[i:i + mm] for i in range(n - mm)])
        d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
        return (np.sum(d <= r) - len(emb)) / 2
    B, A = count(m), count(m + 1)
    return float(-np.log(A / B)) if A > 0 and B > 0 else np.nan


def jump_share(x):
    """1 - bipower variation / realized variance (of successive differences): the jump part of the variance."""
    rv = np.sum(x ** 2)
    bv = (math.pi / 2) * np.sum(np.abs(x[1:]) * np.abs(x[:-1]))
    return float(max(0.0, 1 - bv / rv))


def red_noise_beta(x, n=50):
    """Astronomy 'beta': rms of n-skip bin means divided by the white-noise expectation sigma/sqrt(n)."""
    m = len(x) // n
    b = x[:m * n].reshape(m, n).mean(1)
    return float(b.std(ddof=1) / (x.std(ddof=1) / math.sqrt(n)))


STATS = {
    "permutation entropy (m=3)": perm_entropy,
    "permutation entropy (m=4)": lambda e: perm_entropy(e, 4),
    "DFA exponent": dfa_alpha,
    "spectral flatness": flatness,
    "spectral slope (log-log)": spectral_slope,
    "spectral centroid": spectral_centroid,
    "crest factor": lambda e: float(np.abs(e).max() / e.std()),
    "RMSSD / SD (Poincaré)": lambda e: float(np.sqrt(np.mean(np.diff(e) ** 2)) / e.std()),
    "zero-crossing rate": lambda e: float(np.mean(np.signbit(e[1:]) != np.signbit(e[:-1]))),
    "STA/LTA max": sta_lta,
    "Ljung-Box of squared CDS (volatility clustering)": ljung_box_sq,
    "sample entropy": sample_entropy,
    "jump share (bipower)": jump_share,
    "red-noise beta (50 skips)": red_noise_beta,
}

rows = list(csv.DictReader(open(ROOT / "docs/slides/data/oct12_trace_features.csv")))
EXISTING = ["skip: CDS rms (detrended)", "noise: Allan deviation n=50 (ADU)", "noise: CDS autocorrelation at lag 1",
            "discrete: CDS excess kurtosis", "horizontal: minimum", "vertical: minimum"]
table = []
for row in rows:
    f = DATA / (row["file"] + f".RJ45ADCDATA{row['channel']}")
    r = load_regions(str(f))
    vals = {k: [] for k in STATS}
    for p in usable_pixels(r):
        e = detrend(np.asarray(r.cds[p][10:], float))
        for k, fn in STATS.items():
            vals[k].append(fn(e))
    rec = {"group": row["group"], "channel": int(row["channel"]), **{k: float(np.nanmedian(v)) for k, v in vals.items()}}
    rec.update({k: float(row[k]) for k in EXISTING})
    table.append(rec)
    print(row["file"][-18:], row["channel"], flush=True)


def anova_F(groups):
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2]
    allv = np.concatenate(groups)
    k, n = len(groups), len(allv)
    msb = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n - k)
    return msb / msw if msw > 0 else np.inf


CHANGED = ["VDD −20.5 V", "V low rails + TG", "all H low rails", "H3_A_L only"]
GROUPS = ["reference"] + CHANGED
result = {}
for k in list(STATS) + EXISTING:
    Fset = [anova_F([[t[k] for t in table if t["channel"] == ch and t["group"] == g] for g in CHANGED]) for ch in (0, 1, 3)]
    Famp = [anova_F([[t[k] for t in table if t["channel"] == ch and t["group"] == g] for ch in (0, 1, 2, 3)]) for g in GROUPS]
    result[k] = {"F_settings_min": float(np.min(Fset)), "F_settings_median": float(np.median(Fset)),
                 "F_amps_median": float(np.median(Famp)), "existing": k in EXISTING}
    print(f"{k:50s} settings F min {np.min(Fset):9.1f} median {np.median(Fset):9.1f}   amplifiers F median {np.median(Famp):9.1f}")
OUT.write_text(json.dumps({"per_trace": table, "separation": result}, indent=1))
print("wrote", OUT)

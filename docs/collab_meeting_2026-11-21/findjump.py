import glob, math, numpy as np
def med_rv(e):
    a = np.abs(e); m = np.median(np.vstack([a[:-2], a[1:-1], a[2:]]), 0); n = len(a)
    return math.pi / (6 - 4 * math.sqrt(3) + math.pi) * n / (n - 2) * np.sum(m ** 2)
best = []
fs = sorted(glob.glob("/Users/dylanrosenmerkel/Documents/Software/ccd_opt_ml/processed/L1_*/traces/*acm101_ch[013].npz"))
for f in fs:
    try: C = np.load(f, allow_pickle=True)["cds"]
    except Exception: continue
    for i, row in enumerate(C):
        c = row[10:]; c = c[np.isfinite(c)]
        if len(c) < 400: continue
        n = np.arange(len(c)); e = c - np.polyval(np.polyfit(n, c, 4), n)
        rms = e.std(); rob = math.sqrt(med_rv(e) / len(e))
        if rob > 2: best.append((rms / rob, rms, rob, f, i))
best.sort(reverse=True)
print(len(fs), "files")
for b in best[:8]: print(round(b[0], 2), round(b[1], 1), round(b[2], 1), b[3].split("processed/")[1], b[4])
import statistics; print("median ratio", statistics.median([b[0] for b in best]))

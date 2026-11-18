"""
trace_read_noise.py -- the image's read noise measured DIRECTLY in the traces: each trace has 10 pixels x 500 skips, the
same 500 skips as the image, so the scatter of its pixel means (robust: MAD, so the rare charged pixel does not count)
is a mini image-noise measurement with no Allan extrapolation.  Per packet and amplifier, from the traces at the packet's
settings (theta1-3), after the per-skip common mode and the per-pixel-position pattern are removed.

    .venv/bin/python amp_phys/trace_read_noise.py --data processed_v3n --out amp_phys/results/v3n/trace_read_noise.csv
"""
import argparse, glob, re
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--skip0", type=int, default=20)
a = ap.parse_args()
rows = []
for board in (101, 102):
    for ch in range(4):
        recs = []
        for f in sorted(glob.glob(f"{a.data}/N3_*/traces/theta*_acm{board}_ch{ch}.npz")):
            z = np.load(f, allow_pickle=True)
            if "cds" not in z.files:
                continue
            c = z["cds"].astype(float)[:, a.skip0:]
            c = c - np.median(c, 0, keepdims=True)
            recs.append((Path(f).parts[-3], c.mean(1), c.shape[1]))
        if not recs:
            continue
        M = np.array([r[1] for r in recs])
        df = pd.DataFrame({"packet_id": [r[0] for r in recs]})
        for pid, idx in df.groupby("packet_id").groups.items():
            Mi = M[list(idx)]                                   # the traces of one packet: same settings
            if len(Mi) < 2:
                continue
            # same pixel position, different traces: the settings-dependent position pattern cancels
            dv = (Mi - Mi.mean(0, keepdims=True)).ravel() * np.sqrt(len(Mi) / (len(Mi) - 1))
            mad = 1.4826 * np.median(np.abs(dv))
            rows.append({"packet_id": pid, "board": board, "channel": ch, "n_pix": dv.size,
                         "trace read noise 500 skips (ADU)": mad / np.sqrt(0.9), "skips used": recs[0][2]})
pd.DataFrame(rows).to_csv(a.out, index=False)
print(pd.DataFrame(rows).groupby(["board", "channel"])["trace read noise 500 skips (ADU)"].median())

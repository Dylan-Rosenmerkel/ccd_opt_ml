"""
plot_trace_csv.py -- plain plot of a decoded trace CSV (columns sample,val,RD,RU).

Like ``commant l0plot``: the ADC value against time (sample / 15 MHz), with the pedestal
(RD) and signal (RU) samples coloured.  Make the CSV with
``python decode_trace.py TRACE_FILE --csv out.csv``.

Usage:  python plot_trace_csv.py out.csv plot.png [--tmin 0] [--tmax 200]   (times in us)
Windows longer than ~100k samples (~6.7 ms) are drawn as a thin line without markers.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_RATE_MHZ = 15

parser = argparse.ArgumentParser()
parser.add_argument("csv")
parser.add_argument("out")
parser.add_argument("--tmin", type=float, default=0, help="start time in us")
parser.add_argument("--tmax", type=float, default=200, help="end time in us (full trace: 202000)")
args = parser.parse_args()

data = np.loadtxt(args.csv, delimiter=",", skiprows=1, dtype=np.int64)
sample, val, rd, ru = data.T
t_us = sample / SAMPLE_RATE_MHZ
keep = (t_us >= args.tmin) & (t_us <= args.tmax)

fig, ax = plt.subplots(figsize=(14, 4.5))
n_points = int(keep.sum())
many = n_points > 100_000                       # long stretch: thin lines, no markers
ax.plot(t_us[keep], val[keep], color="0.4", lw=0.3 if many else 0.8, label="trace")
if not many:
    ax.plot(t_us[keep & (rd == 1)], val[keep & (rd == 1)], ".", ms=3, color="tab:red", label="pedestal (RD)")
    ax.plot(t_us[keep & (ru == 1)], val[keep & (ru == 1)], ".", ms=3, color="tab:green", label="signal (RU)")
ax.set(xlabel="time (us)", ylabel="ADC value (ADU)", title=args.csv.split("/")[-1])
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(args.out, dpi=110)
print(f"wrote {args.out}")

"""
common_mode.py -- how much of the read noise is COMMON to the channels (setup: grounding, supplies, substrate, clocks)
and how much is each amplifier's own.  The ACM digitises its 4 channels simultaneously, so the per-skip CDS values of
ch0-3 in one trace are synchronous; noise injected through a shared path appears as a correlation between channels,
an amplifier's own noise does not.  Board 102 adds three dead channels (no CCD amplifier) and a second ACM.

Per trace (packet x trace kind), per channel: the 10 x 500 CDS values, skips >= --skip0, each pixel's mean removed
(charge and pixel pattern out) -> residual noise per skip.  Then:
  * the 8 x 8 correlation matrix of the residuals (same trace kind, same packet; boards 101 and 102 recorded in the same
    packet step but by two ACMs: their cross-correlation tests a setup-wide source);
  * the common-mode part: per board, the residual's projection on the mean of the OTHER channels of that board
    (variance share explained), and the cross-spectral coherence vs frequency (which frequencies are shared);
  * NOISE THERMOMETRY: the dead channels read the Johnson noise of 16.7 k at the frontend board's temperature; with
    the chain gain fixed, relative changes of their noise variance = relative changes of the board temperature
    (dT/T = d(sigma^2)/sigma^2 x (Johnson share)), tracked over the 49 h of the run.

    .venv/bin/python amp_phys/common_mode.py --data processed_v3n --out amp_phys/results/v3n
"""

from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CH = [(101, 0), (101, 1), (101, 2), (101, 3), (102, 0), (102, 1), (102, 2), (102, 3)]
NAMES = [f"{b} ch{c}" for b, c in CH]


DETREND = 0


def residual(f, skip0):
    z = np.load(f, allow_pickle=True)
    if "cds" not in z.files:
        return None
    c = z["cds"].astype(float)[:, skip0:]
    if DETREND > 0:                                           # per-pixel polynomial: removes the AC-coupling droop
        t = np.linspace(-1, 1, c.shape[1])
        V = np.vander(t, DETREND + 1)
        c = c - (c @ np.linalg.pinv(V).T) @ V.T
        return c.ravel()
    return (c - c.mean(1, keepdims=True)).ravel()            # per-pixel mean removed, skip order kept


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--skip0", type=int, default=20)
    ap.add_argument("--detrend", type=int, default=0, help="per-pixel polynomial order removed (0 = mean only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    globals()["DETREND"] = args.detrend
    out = Path(args.out)
    d = pd.read_csv(Path(args.data) / "dataset.csv", low_memory=False, usecols=["packet_id", "board", "channel",
                    "date_start", "theta.VSUB", "theta.VDD1", "theta.VR1"])
    when = d.groupby("packet_id").date_start.first()
    groups = defaultdict(dict)
    for f in glob.glob(str(Path(args.data) / "*" / "traces" / "*_acm*_ch*.npz")):
        p = Path(f)
        m = re.match(r"(\w+?)_acm(\d+)_ch(\d)\.npz", p.name)
        groups[(p.parts[-3], m.group(1))][(int(m.group(2)), int(m.group(3)))] = f
    C_sum, n_c = np.zeros((8, 8)), 0
    share_rows, therm, coh_acc = [], [], defaultdict(list)
    for (pk, kind), files in sorted(groups.items()):
        if not all(k in files for k in CH):
            continue
        R = [residual(files[k], args.skip0) for k in CH]
        if any(r is None for r in R) or len({len(r) for r in R}) != 1:
            continue
        R = np.array(R)
        R = R[:, np.all(np.isfinite(R), 0)]
        C = np.corrcoef(R)
        C_sum += C
        n_c += 1
        row = {"packet_id": pk, "kind": kind, "time": when.get(pk)}
        for i, (b, c) in enumerate(CH):
            others = [j for j, (bb, _) in enumerate(CH) if bb == b and j != i]
            ref = R[others].mean(0)
            beta = (R[i] @ ref) / (ref @ ref)
            row[f"{b} ch{c} common share"] = 1 - np.var(R[i] - beta * ref) / np.var(R[i])
            row[f"{b} ch{c} rms"] = R[i].std()
            # cross-board: share explained by the other board's mean
            ob = [j for j, (bb, _) in enumerate(CH) if bb != b]
            ref2 = R[ob].mean(0)
            beta2 = (R[i] @ ref2) / (ref2 @ ref2)
            row[f"{b} ch{c} other-board share"] = 1 - np.var(R[i] - beta2 * ref2) / np.var(R[i])
        share_rows.append(row)
        # coherence spectrum (per-skip sample rate) of each channel with its board's other channels' mean
        for i, (b, c) in enumerate(CH):
            others = [j for j, (bb, _) in enumerate(CH) if bb == b and j != i]
            X, Y = np.fft.rfft(R[i]), np.fft.rfft(R[others].mean(0))
            coh_acc[i].append((X * np.conj(Y), np.abs(X) ** 2, np.abs(Y) ** 2))
    C = C_sum / n_c
    S = pd.DataFrame(share_rows)
    S.to_csv(out / "common_mode_per_trace.csv", index=False)
    pd.DataFrame(C, index=NAMES, columns=NAMES).round(3).to_csv(out / "common_mode_corr.csv")
    # thermometry from the dead channels
    S["time"] = pd.to_datetime(S.time)
    S["hours"] = (S.time - S.time.min()).dt.total_seconds() / 3600
    dead = ["102 ch0", "102 ch1", "102 ch2"]
    var = (S[[f"{a} rms" for a in dead]] ** 2).mean(1)
    S["board-102 temperature change (K, rel. 300 K)"] = 300 * (var / var.median() - 1) / 0.945   # Johnson share 94.5 %
    # coherence vs frequency
    skip_s = 1.0 / 1e5 * 1                                                  # placeholder, replaced below
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    im = axs[0].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    axs[0].set_xticks(range(8), NAMES, rotation=60, ha="right", fontsize=8)
    axs[0].set_yticks(range(8), NAMES, fontsize=8)
    for i in range(8):
        for j in range(8):
            axs[0].text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7)
    axs[0].set_title("Correlation of the per-skip noise between channels\n(simultaneous traces, pixel means removed)",
                     loc="left", fontsize=9)
    for i, nm in enumerate(NAMES):
        if not coh_acc[i]:
            continue
        L = min(len(x[0]) for x in coh_acc[i])
        Sxy = np.mean([x[0][:L] for x in coh_acc[i]], 0)
        Sxx = np.mean([x[1][:L] for x in coh_acc[i]], 0)
        Syy = np.mean([x[2][:L] for x in coh_acc[i]], 0)
        coh = np.abs(Sxy) ** 2 / (Sxx * Syy)
        f = np.fft.rfftfreq(2 * (L - 1))[:L]                              # cycles per skip
        k = 25
        sm = np.convolve(coh, np.ones(k) / k, mode="same")
        axs[1].plot(f, sm, lw=1, label=nm, ls="-" if nm.startswith("101") else "--")
    axs[1].set_xlabel("frequency (cycles per skip; 0.5 = Nyquist)")
    axs[1].set_ylabel("coherence with the board's other channels")
    axs[1].set_title("Which frequencies are shared", loc="left", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False, ncol=2)
    axs[2].plot(S.hours, S["board-102 temperature change (K, rel. 300 K)"], ".", ms=3, color="#2a78d6")
    axs[2].set_xlabel("hours into the run")
    axs[2].set_ylabel("frontend board 102 temperature change (K)")
    axs[2].set_title("Noise thermometry from the dead channels (Johnson noise of 16.7 k)", loc="left", fontsize=9)
    for ax in axs[1:]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "common_mode.png", dpi=150)
    share = S[[c for c in S.columns if c.endswith("common share")]].median()
    share2 = S[[c for c in S.columns if c.endswith("other-board share")]].median()
    L = ["# Common-mode (setup) noise vs each amplifier's own", "", f"{n_c} simultaneous trace sets.", "",
         "## Mean correlation of the per-skip noise", "", pd.DataFrame(C, index=NAMES, columns=NAMES).round(2).to_markdown(), "",
         "## Share of each channel's per-skip noise variance explained by the others", "",
         "| channel | by its board's other channels | by the other board |", "|---|---|---|"]
    for nm in NAMES:
        L.append(f"| {nm} | {100 * share[f'{nm} common share']:.1f} % | {100 * share2[f'{nm} other-board share']:.1f} % |")
    T = S["board-102 temperature change (K, rel. 300 K)"]
    L += ["", f"Noise thermometry (board 102): temperature spread {T.std():.2f} K (RMS), range {T.quantile(0.05):.2f} to "
          f"{T.quantile(0.95):.2f} K (5-95 %), drift "
          f"{np.polyfit(S.hours[np.isfinite(S.hours) & np.isfinite(T)], T[np.isfinite(S.hours) & np.isfinite(T)], 1)[0]:+.3f} K/h."]
    (out / "common_mode.md").write_text("\n".join(L) + "\n")
    S.to_csv(out / "common_mode_per_trace.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()

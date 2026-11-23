"""
setup_noise_index.py -- single numbers for how good a detector SETUP is, from one simultaneous trace set (~1 min, all
channels of a board digitised together), to compute and minimise while changing grounding / cabling / filtering.

Per channel, the per-skip CDS values of each pixel (pixel mean removed; the first --skip0 skips dropped) are averaged
over blocks of n consecutive skips (n = 1, 10, 40, 120: the noise a pixel sees when read with ~n skips) and split into
  shared  = the part explained by the mean of the board's OTHER channels (setup: ground, cabling, shared drivers),
  own     = the rest (the amplifier + its own frontend channel).
Outputs per board and trace set, median over trace sets:
  SNI_n   = sigma_shared / sigma_own at averaging n.  THE SETUP NOISE INDEX: 0 = no setup noise; 1 = the setup adds as
            much as the amplifier.  Use the n closest to the operating skip count (SNI_120 for many-skip readout).
  X_n     = sigma_total / sigma_own = sqrt(1 + SNI_n^2): the factor by which the setup inflates the noise at that n.
  --remove-pattern  subtracts the droop curve that repeats in every pixel (the frontend's AC coupling): what remains
            is random. Use SNI with --remove-pattern to compare setups.
  e_w     = white input noise density (nV/rtHz at the video input): per-skip noise with a cubic per pixel removed,
            converted with the chain gain (ADU/uV, chain_calibration.py) and the CDS window: e_w = sigma x sqrt(tau)/G.
            Absolute floor for comparisons between setups (17.1 nV/rtHz = the open frontend: 20k || 100k + preamp).

    .venv/bin/python amp_phys/setup_noise_index.py --data processed_v3n --gain-adu-per-uv 0.875 --tau-us 10 \\
        --out amp_phys/results/v3n/setup_noise_index
"""

from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

NS = (1, 10, 40, 120)


def load_cds(f, skip0):
    z = np.load(f, allow_pickle=True)
    if "cds" not in z.files:
        return None
    return z["cds"].astype(float)[:, skip0:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--skip0", type=int, default=20)
    ap.add_argument("--gain-adu-per-uv", type=float, default=0.875)
    ap.add_argument("--tau-us", type=float, default=10.0)
    ap.add_argument("--kinds", default="ref", help="trace kinds to use (prefix; 'ref' = nominal reference traces)")
    ap.add_argument("--remove-pattern", action="store_true",
                    help="subtract the pixel-averaged curve per channel (deterministic AC-coupling droop): keeps only "
                         "the random part; use this for setup comparisons")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    groups = defaultdict(dict)
    for f in glob.glob(str(Path(args.data) / "*" / "traces" / "*_acm*_ch*.npz")):
        p = Path(f)
        m = re.match(r"(\w+?)_acm(\d+)_ch(\d)\.npz", p.name)
        if not m.group(1).startswith(args.kinds):
            continue
        groups[(p.parts[-3], m.group(1), int(m.group(2)))][int(m.group(3))] = f
    rows = []
    for (pk, kind, board), files in sorted(groups.items()):
        chans = sorted(files)
        C = {c: load_cds(files[c], args.skip0) for c in chans}
        C = {c: v for c, v in C.items() if v is not None}
        if args.remove_pattern:                       # the deterministic droop: same curve in every pixel of a trace
            C = {c: (v - v.mean(0, keepdims=True)) * np.sqrt(v.shape[0] / (v.shape[0] - 1)) for c, v in C.items()}
        if len(C) < 3 or len({v.shape for v in C.values()}) != 1:
            continue
        L = next(iter(C.values())).shape[1]
        t = np.linspace(-1, 1, L)
        Vd = np.vander(t, 4)
        for c in C:
            x = C[c]
            r = {"packet_id": pk, "kind": kind, "board": board, "channel": c}
            det = x - (x @ np.linalg.pinv(Vd).T) @ Vd.T
            r["e_w (nV/rtHz)"] = det.std() / args.gain_adu_per_uv * 1e-6 * np.sqrt(args.tau_us * 1e-6) * 1e9
            for n in NS:
                nb = L // n
                if nb < 2:
                    continue
                def blocks(a):
                    a = a[:, :nb * n] - a[:, :nb * n].mean(1, keepdims=True)
                    return a.reshape(a.shape[0], nb, n).mean(2).ravel()
                y = blocks(x)
                ref = np.mean([blocks(C[o]) for o in C if o != c], 0)
                beta = (y @ ref) / (ref @ ref)
                own = y - beta * ref
                tot, ow = y.var(), own.var()
                r[f"SNI_{n}"] = np.sqrt(max(tot - ow, 0) / ow)
                r[f"X_{n}"] = np.sqrt(tot / ow)
                r[f"sigma_total_{n} (ADU)"] = np.sqrt(tot)
            rows.append(r)
    R = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(out.with_suffix(".csv"), index=False)
    S = R.groupby(["board", "channel"])[[c for c in R.columns if c.startswith(("SNI", "X_", "e_w"))]].median()
    B = R.groupby("board")[[c for c in R.columns if c.startswith(("SNI", "X_", "e_w"))]].median()
    L = [f"# Setup noise index: {args.data} ({args.kinds} traces, {R.packet_id.nunique()} packets)", "",
         "SNI_n = shared / own noise after averaging n skips (0 = no setup noise); X_n = total / own;",
         "e_w = white input noise density at the video input (open-frontend floor 17.1 nV/rtHz).", "",
         "## per board", "", B.round(3).to_markdown(), "", "## per channel", "", S.round(3).to_markdown()]
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:8 + len(B)]))


if __name__ == "__main__":
    main()

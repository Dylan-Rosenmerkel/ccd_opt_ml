"""
gain_features.py -- per packet and amplifier, the deterministic signals of the packet's own traces whose size in ADU
is a known charge times the conversion gain (gain = A G q / C_sn), for the gain test (test 2):

  reset jump        the reset edge in the median horizontal-transfer waveform (clock_edges.py edge 0, ~278 samples
                    before the H flag): the injected charge s (RG_high - VR - V*) times the gain / q
  CDS offset        the empty-pixel CDS (signal - pedestal window of the median skip cycle): SW -> sense-node coupling
                    times the SW swing, times the gain
  skip pulse        the largest per-skip pulse of the median skip cycle and its settling
Median over the packet's theta traces.

    .venv/bin/python amp_phys/gain_features.py --data processed_v3n --board 101 --channels 0 1 3 --out <csv>
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clock_edges as ce  # noqa: E402
from trace_anatomy import load, rising  # noqa: E402


def one(path):
    v, fl = load(path)
    out = {}
    e = ce.flag_edges(fl, 6)
    e = e[(e > ce.PRE) & (e < len(v) - ce.POST)]
    if len(e) >= 10:
        H = np.array([v[k - ce.PRE:k + ce.POST] for k in e])
        H -= np.median(H[:, ce.PRE - 250:ce.PRE - 50], axis=1, keepdims=True)
        w = np.median(H, 0)
        ed = [k for k in ce.find_edges(w) if abs((k - ce.PRE) + 278) < 40]
        out["reset jump (ADU)"] = ce.jump(w, ed[0]) if ed else np.nan
        out["H waveform p2p (ADU)"] = np.ptp(w)
    p0 = rising(fl, 0)
    P = int(np.median(np.diff(p0)))
    s0 = int(np.median([s for k in p0[:300] for s in rising(fl[k:k + P], 1)[:1]]))
    C = np.array([v[k:k + P] for k in p0[50:2000] if k + P <= len(v)])
    tm = np.median(C, 0)
    base = np.median(tm[5:145])
    out["CDS offset (ADU)"] = np.median(tm[s0 + 5:s0 + 145]) - base
    mask = np.ones(P, bool)
    mask[:150] = False
    mask[s0:s0 + 150] = False
    pk = int(np.argmax(np.where(mask, np.abs(tm - base), -1)))
    out["skip pulse (ADU)"] = abs(tm[pk] - base)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--board", type=int, default=101)
    ap.add_argument("--channels", type=int, nargs="+", default=[0, 1, 3])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for pdir in sorted(Path(a.data).glob("*/traces")):
        for c in a.channels:
            R = []
            for f in sorted(glob.glob(str(pdir / f"theta*_acm{a.board}_ch{c}.npz"))):
                try:
                    R.append(one(f))
                except Exception:
                    pass
            if R:
                m = pd.DataFrame(R).median().to_dict()
                m.update({"packet_id": pdir.parent.name, "board": a.board, "channel": c})
                rows.append(m)
        print(pdir.parent.name, flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

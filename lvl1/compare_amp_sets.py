"""
compare_amp_sets.py -- what one extra amplifier does (board 101 ch2, the noisy EXT3), from two scorecard runs: without
it (lvl1/scorecard.py, ch0/1/3) and with it (--channels 0 1 2 3).

1. The offset: on the new amplifier, the mean error (prediction - truth, log x 100) of the absolute models on its rows
   against the mean error of the change models on its pairs.  A change model compares two traces of one amplifier, so
   an amplifier's offset should cancel; an absolute model carries it.
2. The extra amplifier as a NEW amplifier (trained on the other three): direction and size of its changes against the
   other amplifiers' (same run), per pair type; absolute error for contrast.
3. The extra amplifier IN TRAINING: the other amplifiers' scores without and with it, on the same test pairs / rows,
   trained-amplifier and new-amplifier cells; packet bootstrap 95 % of the difference.

    .venv/bin/python lvl1/compare_amp_sets.py [--base ...2026-11-04_scorecard] [--with ...2026-11-04_scorecard_with_ch2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

import change_test as ct  # noqa: E402
import scorecard as sc  # noqa: E402

T = sc.T
CLEAR = dict(zip(T, sc.CLEAR))
KEY = ["type", "packet A", "packet B", "channel"]


def rel_models(cols, cell):
    return sorted({c.split(" | ")[1] for c in cols if c.startswith(f"{cell} | ")})


def boot(pa, pb, n=1000):
    return sc.boot_weights(np.asarray(pa), np.asarray(pb), n, seed=3)


def fmt_iv(v, iv, pct=True):
    tag = sc.verdict(np.array(iv), lower_is_better=False)
    return f"{v:+.0f} [{iv[0]:+.0f}, {iv[1]:+.0f}]" if pct else f"{v:+.1f} [{iv[0]:+.1f}, {iv[1]:+.1f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default=str(ROOT / "processed/lvl1_results/2026-11-04_scorecard"))
    parser.add_argument("--with", dest="with_", default=str(ROOT / "processed/lvl1_results/2026-11-04_scorecard_with_ch2"))
    parser.add_argument("--amp", type=int, default=2)
    args = parser.parse_args()
    base, wth = Path(args.base), Path(args.with_)
    rb, rw = pd.read_csv(base / "relative_predictions.csv"), pd.read_csv(wth / "relative_predictions.csv")
    ab, aw = pd.read_csv(base / "absolute_predictions.csv"), pd.read_csv(wth / "absolute_predictions.csv")
    lab = pd.read_csv(ROOT / "processed" / "dataset.csv", low_memory=False)
    lab = lab[lab.board == 101][["packet_id", "channel"] + list(ct.LABELS.values())]
    for t, c in ct.LABELS.items():
        lab[f"true log {t}"] = np.log(lab[c])
    aw = aw.merge(lab, on=["packet_id", "channel"], how="left")
    ab = ab.merge(lab, on=["packet_id", "channel"], how="left")
    L = [f"# One extra amplifier: board 101 ch{args.amp}", "",
         f"Runs: without `{base.name}`, with `{wth.name}`. Direction = right on clear changes (|change| > "
         + ", ".join(f"{t} {100 * v:.1f} %" for t, v in CLEAR.items()) + "); size = RMS error of the change (%).", ""]

    # 1. the offset
    L += [f"## 1. The amplifier offset on ch{args.amp} as a new amplifier", "",
          "Mean error (prediction - truth, % in log) on its rows (absolute) or pairs (change models, n -> n+1). An "
          "absolute model carries the amplifier's offset; a change model should not.", "",
          "| model | sigma_e | lambda | gain |", "|---|---|---|---|"]
    m_amp = aw.channel == args.amp
    for name in [c.split(" | ")[1] for c in aw.columns if c.startswith("new | ") and c.endswith("log sigma_e")]:
        L.append(f"| absolute {name} | " + " | ".join(
            f"{100 * np.nanmean(aw.loc[m_amp, f'new | {name} | log {t}'] - aw.loc[m_amp, f'true log {t}']):+.1f}" for t in T) + " |")
    pairs = rw[(rw.channel == args.amp) & (rw.type == "n -> n+1")]
    for name in rel_models(rw.columns, "new"):
        L.append(f"| change: {name} | " + " | ".join(
            f"{100 * np.nanmean(pairs[f'new | {name} | {t}'] - pairs[f'true change {t}']):+.1f}" for t in T) + " |")
    L.append("")

    # 2. as a new amplifier
    L += [f"## 2. ch{args.amp} as a new amplifier (trained on the other three) against the other amplifiers", ""]
    for ty in ("within packet", "n -> n+1", "n -> n+2"):
        r = rw[rw.type == ty]
        mine, others = r[r.channel == args.amp], r[r.channel != args.amp]
        L += [f"### {ty}", "",
              "clear changes going up: " + ", ".join(
                  f"{t} ch{args.amp} {100 * np.mean(mine[f'true change {t}'][mine[f'true change {t}'].abs() > CLEAR[t]] > 0):.0f} % "
                  f"/ others {100 * np.mean(others[f'true change {t}'][others[f'true change {t}'].abs() > CLEAR[t]] > 0):.0f} %" for t in T), "",
              "| model | " + " | ".join(f"{t} direction ch{args.amp} / others | {t} size ch{args.amp} / others" for t in T) + " |",
              "|---|" + "---|---|" * len(T)]
        for name in rel_models(rw.columns, "new"):
            cells = []
            for t in T:
                col = f"new | {name} | {t}"
                d1 = sc.direction(mine[f"true change {t}"].to_numpy(), mine[col].to_numpy(), CLEAR[t])
                d2 = sc.direction(others[f"true change {t}"].to_numpy(), others[col].to_numpy(), CLEAR[t])
                s1 = 100 * sc.rms((mine[col] - mine[f"true change {t}"]).to_numpy())
                s2 = 100 * sc.rms((others[col] - others[f"true change {t}"]).to_numpy())
                cells.append(f"{100 * d1:.0f} / {100 * d2:.0f} % | {s1:.1f} / {s2:.1f}")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["### absolute, new amplifier: typical error (%)", "", "| model | " + " | ".join(f"{t} ch{args.amp} / others" for t in T) + " |",
          "|---|" + "---|" * len(T)]
    for name in [c.split(" | ")[1] for c in aw.columns if c.startswith("new | ") and c.endswith("log sigma_e")]:
        L.append(f"| {name} | " + " | ".join(
            f"{100 * sc.rms((aw.loc[m_amp, f'new | {name} | log {t}'] - aw.loc[m_amp, f'true log {t}']).to_numpy()):.1f} / "
            f"{100 * sc.rms((aw.loc[~m_amp, f'new | {name} | log {t}'] - aw.loc[~m_amp, f'true log {t}']).to_numpy()):.1f}" for t in T) + " |")
    L.append("")

    # 3. in training
    L += [f"## 3. ch{args.amp} in training: the other amplifiers' scores without -> with it (same test pairs and rows)", "",
          "Change (with - without) and its packet-bootstrap 95 % interval; direction in percentage points (+ = better), "
          "size and absolute error in % (- = better).", ""]
    for cell in sc.CELLS:
        for ty in ("within packet", "n -> n+1"):
            j = rb[rb.type == ty].merge(rw[(rw.type == ty) & (rw.channel != args.amp)], on=KEY, suffixes=("", " W"))
            W = boot(j["packet A"], j["packet B"])
            L += [f"### {sc.CELL_NAME[cell]}, {ty} ({len(j)} pairs)", "",
                  "| model | " + " | ".join(f"{t} direction | {t} size" for t in T) + " |", "|---|" + "---|---|" * len(T)]
            for name in rel_models(rw.columns, cell):
                if f"{cell} | {name} | sigma_e" not in rb.columns:
                    continue
                cells = []
                for t in T:
                    y = j[f"true change {t}"].to_numpy()
                    p0, p1 = j[f"{cell} | {name} | {t}"].to_numpy(), j[f"{cell} | {name} | {t} W"].to_numpy()
                    dd = 100 * (sc.direction(y, p1, CLEAR[t]) - sc.direction(y, p0, CLEAR[t]))
                    ds = 100 * (sc.rms(p1 - y) - sc.rms(p0 - y))
                    bd = sc.interval([100 * (sc.direction(y, p1, CLEAR[t], w) - sc.direction(y, p0, CLEAR[t], w)) for w in W])
                    bs = sc.interval([100 * (sc.rms(p1 - y, w) - sc.rms(p0 - y, w)) for w in W])
                    cells.append(f"{fmt_iv(dd, bd)} | {fmt_iv(ds, bs, False)}")
                L.append(f"| {name} | " + " | ".join(cells) + " |")
            L.append("")
        j = ab.merge(aw[aw.channel != args.amp], on=["packet_id", "channel"], suffixes=("", " W"))
        W = boot(j.packet_id, j.packet_id)
        L += [f"### {sc.CELL_NAME[cell]}, absolute ({len(j)} rows): typical error change (%)", "",
              "| model | " + " | ".join(T) + " |", "|---|" + "---|" * len(T)]
        for name in [c.split(" | ")[1] for c in aw.columns if c.startswith(f"{cell} | ") and c.endswith("log sigma_e")]:
            if f"{cell} | {name} | log sigma_e" not in ab.columns:
                continue
            cells = []
            for t in T:
                y = j[f"true log {t}"].to_numpy()
                p0, p1 = j[f"{cell} | {name} | log {t}"].to_numpy(), j[f"{cell} | {name} | log {t} W"].to_numpy()
                v = 100 * (sc.rms(p1 - y) - sc.rms(p0 - y))
                iv = sc.interval([100 * (sc.rms(p1 - y, w) - sc.rms(p0 - y, w)) for w in W])
                cells.append(fmt_iv(v, iv, False))
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    (wth / f"ch{args.amp}_comparison.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

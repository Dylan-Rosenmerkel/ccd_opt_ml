"""
check_rehearsal.py -- compare the labels of processed simulated packets with their known truth.

After ``simulate_manual_run.py`` and ``process_packet.py``, this reads ``<processed>/dataset.csv`` and the simulation's
``truth.json`` and reports, per packet and channel, the labels against the truth:

    sigma_e, lambda, gain             WADERs (panaSKImg or run_WADERs) against the truth
    pull                              (label - truth) / label error: |pull| < 2 for ~95 % if the errors are right
    dark current, CIC                 image_rows.py's straight line through the 16 rows against the truth
    flags                             what the label extraction flagged (e.g. the noisy channel's unresolved peaks)

    python lvl1/check_rehearsal.py <processed dir> <truth.json> [--out report.md]

Exit code 1 if any unflagged label is more than 5 of its own errors away from the truth (a broken pipeline), else 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLS = {"sigma_e": ("label.sigma_e", "label.sigma_e_err", "sigma_e"), "lambda": ("label.lambda_e_per_bin", "label.lambda_err", "lambda"),
        "gain": ("label.gain_adu_per_e", "label.gain_err", "gain"), "dark": ("label.lambda_dark", "label.lambda_dark_err", "dark"),
        "CIC": ("label.lambda_cic", "label.lambda_cic_err", "cic")}


def compare(processed: Path, truth_path: Path) -> pd.DataFrame:
    df = pd.read_csv(processed / "dataset.csv")
    truth = json.loads(Path(truth_path).read_text())
    rows = []
    for _, r in df.iterrows():
        t = truth.get(str(r["packet_id"]))
        if t is None:
            continue
        tc = t["channels"][int(r["channel"])]
        row = {"packet": r["packet_id"], "channel": int(r["channel"]), "flags": r.get("label.flags") if isinstance(r.get("label.flags"), str) else ""}
        for name, (v, e, tk) in COLS.items():
            val = pd.to_numeric(r.get(v), errors="coerce")
            err = pd.to_numeric(r.get(e), errors="coerce")
            row[f"{name} label"], row[f"{name} truth"] = val, tc[tk]
            row[f"{name} rel. error %"] = 100 * (val / tc[tk] - 1) if tc[tk] else np.nan
            row[f"{name} pull"] = (val - tc[tk]) / err if err and err > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summary(res: pd.DataFrame) -> str:
    ok = res["flags"].fillna("") == ""
    lines = [f"{len(res)} labels ({ok.sum()} without flags)", "",
             "| quantity | median rel. error (unflagged) | RMS rel. error | median |pull| | max |pull| |", "|---|---|---|---|---|"]
    for name in COLS:
        e = res.loc[ok, f"{name} rel. error %"].dropna()
        p = res.loc[ok, f"{name} pull"].dropna().abs()
        if len(e):
            lines.append(f"| {name} | {np.median(e):+.1f} % | {np.sqrt(np.mean(e ** 2)):.1f} % | {np.median(p):.2f} | {p.max():.2f} |")
        else:
            lines.append(f"| {name} | – | – | – | – |")
    flagged = res.loc[~ok, ["packet", "channel", "flags"]]
    if len(flagged):
        lines += ["", "Flagged labels:"] + [f"* {r.packet} ch{r.channel}: {r.flags}" for r in flagged.itertuples()]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("processed")
    parser.add_argument("truth")
    parser.add_argument("--out")
    args = parser.parse_args()
    res = compare(Path(args.processed), Path(args.truth))
    pd.set_option("display.width", 250)
    show = ["packet", "channel"] + [c for c in res.columns if c.endswith("rel. error %") or c.endswith("pull")]
    print(res[show].round(2).to_string(index=False))
    text = summary(res)
    print("\n" + text)
    if args.out:
        table = res.round(4).fillna("")
        md = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
        md += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
        Path(args.out).write_text(text + "\n\n" + "\n".join(md) + "\n")
    ok = res["flags"].fillna("") == ""
    pulls = res.loc[ok, [f"{n} pull" for n in ("sigma_e", "lambda", "gain")]].abs()
    sys.exit(1 if (pulls > 5).any().any() else 0)


if __name__ == "__main__":
    main()

"""
merge_processed.py -- several processed data sets (nights) as one, for training on all of them: dataset.csv and every
model_inputs/*.npz concatenated row by row, and the plans merged into one plan.json.

* curves: only the curves every set has, in the first set's order (v1 has 16 curve kinds, v2 17: 'horizontal: mean
  shape' is new in v2); pretrain.npz likewise;
* dataset.csv: the union of the columns (a column a set does not have is empty there, e.g. v1 has no strips);
* arrays whose other dimensions differ (sets.npz: v1 took 10 trace sets per packet, v2 3) are padded with NaN;
* packet ids must be unique across the sets (v1 L1_*, v2 L2_*, v3 L3_*); sessions keep their names (A-C, D-F, G-H),
  so pairs of packets never cross nights in the change models (pairs are made within a session).

    .venv/bin/python lvl1/merge_processed.py --data processed_v1 processed_v2 --plans lvl1/packet_plan/plan.json \\
        lvl1/packet_plan_v2/plan.json --out processed_v1v2 --plan-out lvl1/packet_plan_v1v2/plan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ml import real_inputs  # noqa: E402

NAME_KEYS = ("feature_names", "theta_names", "targets", "curve_names")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--plans", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan-out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "model_inputs").mkdir(parents=True, exist_ok=True)
    dirs = [Path(d) for d in args.data]

    ds = [pd.read_csv(d / "dataset.csv", low_memory=False) for d in dirs]
    ids = pd.concat([d.packet_id for d in ds])
    assert not set.intersection(*[set(d.packet_id) for d in ds]) if len(ds) > 1 else True, "packet ids repeat across sets"
    pd.concat(ds, ignore_index=True, sort=False).to_csv(out / "dataset.csv", index=False)
    print(f"dataset.csv: {sum(map(len, ds))} rows from {len(ds)} sets, {ids.nunique()} packets")

    fams = sorted(set.intersection(*[{p.stem for p in (d / "model_inputs").glob("*.npz")} for d in dirs]))
    for fam in fams:
        zs = [real_inputs.load(d / "model_inputs" / f"{fam}.npz") for d in dirs]
        merged = {}
        keep = None
        if "curve_names" in zs[0]:
            names = [list(map(str, z["curve_names"])) for z in zs]
            common = [n for n in names[0] if all(n in nn for nn in names[1:])]
            keep = [[nn.index(n) for n in common] for nn in names]
            merged["curve_names"] = np.array(common)
        for k in zs[0]:
            if k == "curve_names":
                continue
            vals = [z[k] for z in zs]
            if isinstance(vals[0], str) or k in NAME_KEYS or np.ndim(vals[0]) == 0:
                if k in NAME_KEYS:
                    assert all(np.array_equal(np.asarray(v), np.asarray(vals[0])) for v in vals), f"{fam}: {k} differ"
                merged[k] = np.array(vals[0]) if isinstance(vals[0], str) else vals[0]
                continue
            if keep is not None and np.ndim(vals[0]) == 3:          # (rows, curves, points): the common curves only
                vals = [v[:, kk, :] for v, kk in zip(vals, keep)]
            if len({np.shape(v)[1:] for v in vals}) > 1:            # e.g. sets.npz: v1 10 trace sets, v2 3: pad
                shape = tuple(max(np.shape(v)[i] for v in vals) for i in range(1, np.ndim(vals[0])))
                fill = lambda v: (np.nan if np.issubdtype(np.asarray(v).dtype, np.floating) else 0)
                padded = []
                for v in vals:
                    w = np.full((len(v),) + shape, fill(v), dtype=np.asarray(v).dtype)
                    w[tuple(slice(0, n) for n in np.shape(v))] = v
                    padded.append(w)
                vals = padded
            merged[k] = np.concatenate(vals, axis=0)
        np.savez_compressed(out / "model_inputs" / f"{fam}.npz", **merged)
        print(f"{fam}.npz: {len(merged[next(k for k in merged if np.ndim(merged[k]) > 0 and k not in NAME_KEYS)])} rows"
              + (f", {len(merged['curve_names'])} common curves" if "curve_names" in merged else ""))

    packets = []
    for pl in args.plans:
        packets += json.loads(Path(pl).read_text())["packets"]
    Path(args.plan_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.plan_out).write_text(json.dumps({"merged_from": args.plans, "packets": packets}, indent=1, default=str))
    print(f"plan: {len(packets)} packets -> {args.plan_out}")


if __name__ == "__main__":
    main()

"""
feature_tolerance.py -- how many extra feature columns can each model take before it gets worse?  (Step 2 of the
feature plan, docs/trace_feature_candidates.md.)

The synthetic world has only 17 trace features, so it cannot tell which REAL features are best.  It can tell how each
model copes with more inputs.  For every synthetic dataset, k extra columns are appended to the own-trace and reference
features (X_trace and X_ref of features.npz and physics.npz, so every model family sees them; the CNNs see X_ref as
scalars), of three kinds:

* noise    independent Gaussian columns;
* copies   near-copies of existing columns (a random existing column + 10 % of its spread in noise);
* real     columns of real 2026-10-12 trace features (docs/slides/data/oct12_trace_features.csv), values drawn at
           random per row: realistic distributions and outliers, but no information.

Then lvl1/train_lvl1.py is run on each copy (same folds), and the errors are compared with k = 0.

    .venv/bin/python lvl1/feature_tolerance.py --datasets <model_inputs dirs> --out <dir> [--ks 0 20 50 100]
        [--kinds noise copies real] [--models ridge trees gp nn physics physics_joint sets cnn] [--quick] [--jobs 3]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
REAL = ROOT / "docs" / "slides" / "data" / "oct12_trace_features.csv"


def extra_columns(kind: str, k: int, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(X)
    if kind == "noise":
        return rng.normal(0, 1, (n, k))
    if kind == "copies":
        cols = rng.integers(0, X.shape[1], k)
        base = X[:, cols]
        sd = np.nanstd(base, axis=0)
        return base + rng.normal(0, 1, (n, k)) * 0.1 * np.where(sd > 0, sd, 1)
    if kind == "real":
        real = pd.read_csv(REAL).select_dtypes("number").drop(columns=["channel", "NCOLPRE", "repeat"], errors="ignore")
        real = real.loc[:, real.notna().mean() > 0.9]
        cols = rng.choice(real.columns, k, replace=k > real.shape[1])
        out = np.empty((n, k))
        for j, c in enumerate(cols):
            v = real[c].dropna().to_numpy(float)
            out[:, j] = rng.choice(v, n)
        return out
    raise ValueError(kind)


def make_copy(src: Path, dst: Path, kind: str, k: int, seed: int) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    if k == 0:
        return
    rng = np.random.default_rng(seed)
    base = dict(np.load(src / "features.npz"))
    add_t = extra_columns(kind, k, base["X_trace"], rng)
    add_r = extra_columns(kind, k, base["X_ref"], rng)
    names = np.array([f"extra {kind} {i}" for i in range(k)])
    for fam in ("features", "physics"):
        p = dst / f"{fam}.npz"
        if not p.exists():
            continue
        d = dict(np.load(p))
        d["X_trace"] = np.hstack([d["X_trace"], add_t])
        d["X_ref"] = np.hstack([d["X_ref"], add_r])
        d["feature_names"] = np.r_[d["feature_names"].astype(str), names]
        np.savez_compressed(p, **d)


def run(job) -> dict:
    data, out, models, quick = job
    cmd = [PY, "-W", "ignore", str(ROOT / "lvl1" / "train_lvl1.py"), "--inputs", str(data), "--out", str(out),
           "--models", *models, "--n-boot", "200"] + (["--quick"] if quick else [])
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode:
        return {"error": done.stderr[-2000:]}
    return json.loads((out / "report.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", required=True, help="model_inputs directories")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[0, 20, 50, 100])
    parser.add_argument("--kinds", nargs="+", default=["noise", "copies", "real"])
    parser.add_argument("--models", nargs="+", default=["ridge", "trees", "gp", "nn", "physics", "physics_joint", "sets", "cnn"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs, keys = [], []
    for di, src in enumerate(args.datasets):
        src = Path(src)
        for kind in args.kinds:
            for k in args.ks:
                if k == 0 and kind != args.kinds[0]:
                    continue                                   # k = 0 is the same for every kind
                tag = f"d{di}_{'base' if k == 0 else kind}_{k}"
                data = out / "inputs" / tag
                make_copy(src, data, kind, k, seed=1000 * di + k)
                jobs.append((data, out / "runs" / tag, args.models, args.quick))
                keys.append((di, "base" if k == 0 else kind, k))
    with ThreadPoolExecutor(args.jobs) as ex:
        reports = list(ex.map(run, jobs))
    rows = []
    for (di, kind, k), rep in zip(keys, reports):
        if "error" in rep:
            print("FAILED", di, kind, k, rep["error"][-500:])
            continue
        for model, r in rep["results"].items():
            if "scores" not in r:
                continue
            for t in ("sigma_e", "lambda", "gain"):
                rows.append({"dataset": di, "kind": kind, "k": k, "model": model, "target": t,
                             "error": 100 * r["scores"][t]["error"], "crps": r["scores"][t].get("crps"),
                             "coverage68": r["scores"][t].get("calibrated_coverage68"), "seconds": r.get("seconds")})
    df = pd.DataFrame(rows)
    df.to_csv(out / "tolerance_results.csv", index=False)
    base = df[df.kind == "base"].set_index(["dataset", "model", "target"])["error"]
    df["change vs k=0 (pp)"] = df.apply(lambda r: r["error"] - base.get((r["dataset"], r["model"], r["target"]), np.nan), axis=1)
    summary = df[df.kind != "base"].groupby(["model", "target", "kind", "k"])["change vs k=0 (pp)"].mean().unstack("k").round(2)
    summary.to_csv(out / "tolerance_summary.csv")
    print(summary.to_string())
    print(f"wrote {out / 'tolerance_results.csv'}, tolerance_summary.csv")


if __name__ == "__main__":
    main()

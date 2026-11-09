"""
amp_fingerprint.py -- which trace features identify the AMPLIFIER rather than its state or the setting?

On the board-101 A+B rows (labels OK, a usable trace), for every trace feature in the own-trace block and in the
reference-trace block:

* eta2: the share of the feature's variance between amplifiers (1 - within-amplifier sum of squares / total): 1 = it
  only says which amplifier this is, 0 = it does not differ between amplifiers;
* acc: how often the feature ALONE names the right amplifier of a held-out packet (nearest amplifier median of the other
  packets; chance 0.25).

No target label is used; the amplifier label is (that is the point).  Writes, into lvl1/feature_study/real_AB/:

    amp_fingerprint.csv               every feature column, sorted by eta2
    amp_fingerprint_eta90.txt         'own: <name>' / 'ref: <name>' with eta2 > 0.9 (train_lvl1.py --exclude-features)
    amp_fingerprint_eta50.txt         the same with eta2 > 0.5

    .venv/bin/python lvl1/amp_fingerprint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "processed_AB"   # processed data folder; set with --data
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

import train_lvl1 as tl  # noqa: E402
from ml import real_inputs  # noqa: E402

OUT = ROOT / "lvl1" / "feature_study" / "real_AB"
THRESHOLDS = (0.9, 0.5)


def eta2(x: np.ndarray, amp: np.ndarray) -> float:
    ok = np.isfinite(x)
    x, a = x[ok], amp[ok]
    if len(x) < 20 or x.std() == 0:
        return float("nan")
    within = sum(((x[a == k] - x[a == k].mean()) ** 2).sum() for k in np.unique(a))
    return float(1 - within / ((x - x.mean()) ** 2).sum())


def accuracy(x: np.ndarray, amp: np.ndarray, packet: np.ndarray) -> float:
    ok = np.isfinite(x)
    x, a, p = x[ok], amp[ok], packet[ok]
    if len(x) < 20:
        return float("nan")
    hits = []
    for q in np.unique(p):
        train, test = p != q, p == q
        centre = {k: np.median(x[train & (a == k)]) for k in np.unique(a[train])}
        hits += [min(centre, key=lambda k: abs(centre[k] - v)) == t for v, t in zip(x[test], a[test])]
    return float(np.mean(hits))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--data", default=str(DATA), help="processed data folder (dataset.csv, model_inputs/); default %(default)s")
    parser.add_argument("--out", default=str(OUT), help="folder for the fingerprint table and lists; default %(default)s")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    d = real_inputs.load(Path(args.data).resolve() / "model_inputs" / "features.npz")
    rows = (d["board"] == 101) & np.isfinite(d["Y"]).all(1) & d["label_ok"] & ~d["fake"] & d["trace_ok"].astype(bool)
    amp, packet = d["channel"][rows], d["packet_id"][rows].astype(str)
    tiers = tl.load_tiers(OUT / "evidence_tiers.json")
    names = [str(n) for n in d["feature_names"]]
    table = []
    for block, X in (("own", d["X_trace"][rows]), ("ref", d["X_ref"][rows])):
        for j, n in enumerate(names):
            table.append({"column": f"{block}: {n}", "block": block, "feature": n, "eta2": eta2(X[:, j], amp),
                          "accuracy": accuracy(X[:, j], amp, packet),
                          "tiers": "+".join(sorted(t for t in tiers.get(n, set()) if t in ("P", "A1", "A2", "B", "C")))})
    t = pd.DataFrame(table).dropna(subset=["eta2"]).sort_values("eta2", ascending=False)
    t.to_csv(out / "amp_fingerprint.csv", index=False)
    for th in THRESHOLDS:
        cols = t.loc[t.eta2 > th, "column"].tolist()
        (out / f"amp_fingerprint_eta{int(th * 100)}.txt").write_text(
            f"# trace features whose variance is > {th:.0%} between amplifiers (lvl1/amp_fingerprint.py; board 101, {Path(args.data).name})\n"
            + "\n".join(cols) + "\n")
        print(f"eta2 > {th}: {len(cols)} columns ({sum(c.startswith('own') for c in cols)} own, "
              f"{sum(c.startswith('ref') for c in cols)} reference)")
    print(f"{len(t)} feature columns; written {out / 'amp_fingerprint.csv'}")


if __name__ == "__main__":
    main()

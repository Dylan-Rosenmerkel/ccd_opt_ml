"""
feature_study.py -- how informative, stable and redundant is every trace feature?  (Step 1 of the feature plan.)

Uses the 59 usable board-101 traces of the 2026-10-12 packet tests (5 settings x 3 repeats x 4 amplifiers; one damaged
ch3 trace left out), with all 134 scalar features (docs/slides/data/oct12_trace_features.csv, made by
docs/slides/oct12_features.py).  For every feature:

* settings F   one-way ANOVA F over the 4 CHANGED settings (all taken with NCOLPRE 185, so the reference's different
               NCOLPRE cannot fake a separation), per amplifier ch0/ch1/ch3; min and median.  F > ~4: p < 0.05
               (df 3, 8).
* amplifier F  ANOVA F over the 4 amplifiers within each of the 5 settings; median.
* repeat share the fraction of the variance that is repeat scatter: within-(setting, amplifier) variance / total variance
               (0 = perfectly repeatable, 1 = pure noise).
* NaN share    fraction of traces where the feature is NaN.
* cluster      features whose |Spearman rho| across the 59 traces exceeds 0.95 form one cluster (average-linkage
               clustering on 1 - |rho|); the cluster's representative is its most informative member.

Proposed evidence tiers (for the models; NOT the implementation tiers 0/1/2 of trace_features.FEATURES):
  A  core        informative (settings F min > 10 or amplifier F median > 10), repeat share < 0.5, NaN share <= 0.2,
                 and the representative of its cluster;
  B  supporting  informative but a non-representative cluster member, or moderately informative (F 4-10);
  C  store only  not informative on this data (both F < 4), mostly NaN, or constant.
Caveat: these 4 settings did not change the noise, so noise features can only show amplifier separation here; the tiers
are to be revisited on session A (delayInt changes the noise).

    .venv/bin/python lvl1/feature_study.py [--out lvl1/feature_study]
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
import trace_features as tf  # noqa: E402

CHANGED = ["VDD −20.5 V", "V low rails + TG", "all H low rails", "H3_A_L only"]
GROUPS = ["reference"] + CHANGED


def anova_F(groups) -> float:
    groups = [np.asarray(g, float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan")
    allv = np.concatenate(groups)
    k, n = len(groups), len(allv)
    msb = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n - k)
    if msw <= 0:
        return float("inf") if msb > 0 else float("nan")
    return float(msb / msw)


def study(df: pd.DataFrame) -> pd.DataFrame:
    feats = [n for n in tf.scalar_names() if n in df.columns]
    rows = []
    for f in feats:
        x = pd.to_numeric(df[f], errors="coerce")
        Fset = [anova_F([x[(df.channel == ch) & (df.group == g)] for g in CHANGED]) for ch in (0, 1, 3)]
        Famp = [anova_F([x[(df.channel == ch) & (df.group == g)] for ch in (0, 1, 2, 3)]) for g in GROUPS]
        cells = df.assign(v=x).groupby(["group", "channel"])["v"]
        within = np.nansum(cells.var(ddof=1) * (cells.count() - 1)) / max(np.nansum(cells.count() - 1), 1)
        total = np.nanvar(x, ddof=1)
        rows.append({"feature": f, "group": tf.FEATURES[f].group, "implementation tier": tf.FEATURES[f].tier,
                     "settings F min": np.nanmin(Fset) if np.isfinite(Fset).any() else np.nan,
                     "settings F median": np.nanmedian(Fset) if np.isfinite(Fset).any() else np.nan,
                     "amplifier F median": np.nanmedian(Famp) if np.isfinite(Famp).any() else np.nan,
                     "repeat share": within / total if total > 0 else np.nan,
                     "NaN share": float(x.isna().mean()), "constant": bool(np.nanstd(x) == 0)})
    t = pd.DataFrame(rows).set_index("feature")
    # redundancy clusters
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    usable = [f for f in feats if X[f].notna().mean() > 0.8 and np.nanstd(X[f]) > 0]
    rho = X[usable].rank().corr().abs().fillna(0).to_numpy()
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    dist = np.clip(1 - rho, 0, None)
    np.fill_diagonal(dist, 0)
    labels = fcluster(linkage(squareform(dist, checks=False), "average"), t=0.05, criterion="distance")
    t["cluster"] = np.nan
    t.loc[usable, "cluster"] = labels
    info = np.log1p(t[["settings F min", "amplifier F median"]].clip(lower=0).fillna(0)).sum(1)
    t["informativeness"] = info
    t["cluster size"] = t.groupby("cluster")["cluster"].transform("count")
    rep = t.dropna(subset=["cluster"]).groupby("cluster")["informativeness"].idxmax()
    t["representative"] = t.index.isin(rep.values) | t["cluster"].isna()

    def tier(r):
        informative = (r["settings F min"] > 10) or (r["amplifier F median"] > 10)
        moderate = (r["settings F min"] > 4) or (r["amplifier F median"] > 4)
        if r["constant"] or r["NaN share"] > 0.2 or not moderate:
            return "C"
        if informative and r["repeat share"] < 0.5 and r["representative"]:
            return "A"
        return "B"
    t["evidence tier"] = t.apply(tier, axis=1)
    return t


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=str(ROOT / "docs" / "slides" / "data" / "oct12_trace_features.csv"))
    parser.add_argument("--out", default=str(ROOT / "lvl1" / "feature_study"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    t = study(df)
    t.to_csv(out / "feature_table.csv")
    counts = pd.crosstab(t["implementation tier"], t["evidence tier"])
    print(counts)
    tiers = {k: list(v) for k, v in t.groupby("evidence tier").groups.items()}
    (out / "evidence_tiers.json").write_text(json.dumps(tiers, indent=1))
    print(f"wrote {out / 'feature_table.csv'} and evidence_tiers.json")


if __name__ == "__main__":
    main()

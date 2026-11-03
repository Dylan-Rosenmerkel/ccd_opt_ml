"""
feature_study.py -- how informative, stable and redundant is every trace feature?  (Step 1 of the feature plan.)

Uses the 59 usable board-101 traces of the 2026-10-12 packet tests (5 settings x 3 repeats x 4 amplifiers; one damaged
ch3 trace left out), with all 141 scalar features (docs/slides/data/oct12_trace_features.csv, made by
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

Proposed evidence tiers (for the models; NOT the implementation tiers 0/1/2 of trace_features.FEATURES); A is then split
by informativeness into A1 (top half) and A2, so that A1, A2, B and C hold similar numbers of features:
  A  core        informative (settings F min > 10 or amplifier F median > 10), repeat share < 0.5, NaN share <= 0.2,
                 and the representative of its cluster;
  B  supporting  informative but a non-representative cluster member, or moderately informative (F 4-10);
  C  store only  not informative on this data (both F < 4), mostly NaN, or constant.
Guard (a), added 2026-10-21: PROTECTED lists the direct physical estimators of the labels.  They are written as tier
"P" in evidence_tiers.json, besides their evidence tier, and lvl1/train_lvl1.py gives them to every model.  The
evidence tiers reward separating the traces, not predicting the labels, and on these 59 traces the amplifier dominates
every correlation, so without this guard the most direct sigma_e estimator (skip: CDS rms (detrended)) and the kTC
noise (noise: reset-level scatter) end up in B and tier-A models would not see them.
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
#: guard (a): direct physical estimators of the labels, always given to every model (tier "P"); chosen by physics, not by
#: these data: sigma_e (CDS scatter, white noise, jump-robust and null-CDS noise, Allan deviation) and gain (kTC noise,
#: the amplification of the RG and SW clock feedthrough).  No direct lambda estimator exists in a trace yet.
PROTECTED = ["skip: CDS rms (detrended)", "fit: white noise per skip (ADU)", "noise: jump-robust CDS sigma (ADU)",
             "noise: null-CDS noise (ADU)", "noise: Allan deviation n=50 (ADU)",
             "noise: reset-level scatter (ADU)", "edge: pulse per volt RG (ADU/V)", "edge: dip per volt SW (ADU/V)"]
GROUPS = ["reference"] + CHANGED


def anova_F(groups) -> float:
    """One-way ANOVA F of the groups (NaN removed; groups with fewer than 2 values dropped)."""
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
    """Per feature: settings F, amplifier F, repeat share, NaN share, redundancy cluster, informativeness and
    evidence tier."""
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
    return _cluster_and_tier(t, df[feats].apply(pd.to_numeric, errors="coerce"), feats)


def packet_traces(processed: Path, plan: Path, sessions: list[str], board: int = 101) -> pd.DataFrame:
    """
    One row per usable trace at the packet's own settings (kind "theta") of ``board``, from every
    ``<processed>/<packet>/result.json`` of the planned packets in ``sessions``: packet, setting (the plan's changes;
    packets with the same changes share it), channel, trace set, and the 141 scalar features.  No labels are read.
    """
    planned = {p["packet_id"]: p for p in json.loads(plan.read_text())["packets"] if p["session"] in sessions}
    rows = []
    for pid, p in sorted(planned.items()):
        path = processed / pid / "result.json"
        if not path.exists():
            continue
        for t in json.loads(path.read_text())["traces"]:
            if t["board"] == board and t["kind"] == "theta" and t["usable"]:
                rows.append({"packet": pid, "setting": json.dumps(p["changes"], sort_keys=True),
                             "channel": t["channel"], "set": t["repeat"], **t["tools"]})
    return pd.DataFrame(rows)


def study_packets(df: pd.DataFrame) -> pd.DataFrame:
    """
    The study of :func:`study` on real packets.  The unit of the settings comparison is the PACKET (mean over its
    trace sets), because the trace sets of one packet are minutes apart and would hide the drift between packets:
    * settings F   variance of the packet means over all packets / pooled variance of the packet means within the
                   settings taken more than once (the replicate packets: nominal, the delayInt replicates), per
                   amplifier; min and median over the amplifiers.  This asks whether the feature changes with the
                   settings more than it wanders between two packets at the same settings.
    * repeat share that within-setting variance / the total variance of the packet means (pooled over amplifiers);
    * amplifier F  ANOVA F over the amplifiers within each packet (its trace sets as replicates), median over packets;
    * clusters     |Spearman rho| > 0.95 over the packet means of every amplifier.
    The tier rule is the one of :func:`study`.
    """
    feats = [n for n in tf.scalar_names() if n in df.columns]
    means = df.groupby(["packet", "setting", "channel"])[feats].mean().reset_index()
    replicated = means.groupby("setting")["packet"].nunique()
    replicated = replicated[replicated > 1].index
    rows = []
    for f in feats:
        Fset, within_ss, within_df, total = [], 0.0, 0, []
        for ch, m in means.groupby("channel"):
            x = m[f].astype(float)
            ok = x.notna()
            rep = m[ok & m.setting.isin(replicated)].groupby("setting")[f]
            ss, dof = float((rep.var(ddof=1) * (rep.count() - 1)).sum()), int((rep.count() - 1).sum())
            within_ss, within_df = within_ss + ss, within_df + dof
            total.append(x[ok] - x[ok].mean())
            msw = ss / dof if dof > 0 else np.nan
            msb = float(np.var(x[ok], ddof=1)) if ok.sum() > 2 else np.nan
            Fset.append(msb / msw if msw and msw > 0 else (np.inf if msb and msb > 0 else np.nan))
        Famp = [anova_F([g[f] for _, g in d.groupby("channel")]) for _, d in df.groupby("packet")]
        tot = np.concatenate(total) if total else np.array([])
        total_var = float(np.var(tot, ddof=1)) if len(tot) > 2 else np.nan
        x_all = pd.to_numeric(df[f], errors="coerce")
        rows.append({"feature": f, "group": tf.FEATURES[f].group, "implementation tier": tf.FEATURES[f].tier,
                     "settings F min": np.nanmin(Fset) if np.isfinite(Fset).any() else np.nan,
                     "settings F median": np.nanmedian(Fset) if np.isfinite(Fset).any() else np.nan,
                     "amplifier F median": np.nanmedian(Famp) if np.isfinite(Famp).any() else np.nan,
                     "repeat share": (within_ss / within_df) / total_var if within_df and total_var > 0 else np.nan,
                     "NaN share": float(x_all.isna().mean()), "constant": bool(np.nanstd(x_all) == 0)})
    t = pd.DataFrame(rows).set_index("feature")
    return _cluster_and_tier(t, means[feats].apply(pd.to_numeric, errors="coerce"), feats)


def _cluster_and_tier(t: pd.DataFrame, X: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Redundancy clusters on the rows of ``X``, informativeness, the evidence tiers A1/A2/B/C and the protected flag
    (the second half of :func:`study`)."""
    usable = [f for f in feats if X[f].notna().mean() > 0.8 and np.nanstd(X[f]) > 0]
    rho = X[usable].rank().corr().abs().fillna(0).to_numpy()
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    dist = np.clip(1 - rho, 0, None)
    np.fill_diagonal(dist, 0)
    labels = fcluster(linkage(squareform(dist, checks=False), "average"), t=0.05, criterion="distance")
    t["cluster"] = np.nan
    t.loc[usable, "cluster"] = labels
    t["informativeness"] = np.log1p(t[["settings F min", "amplifier F median"]].clip(lower=0).fillna(0)
                                    .replace(np.inf, 1e6)).sum(axis=1)
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
    a = t[t["evidence tier"] == "A"].sort_values("informativeness", ascending=False)
    t["evidence subtier"] = t["evidence tier"]
    t.loc[a.index[: (len(a) + 1) // 2], "evidence subtier"] = "A1"
    t.loc[a.index[(len(a) + 1) // 2:], "evidence subtier"] = "A2"
    t["protected"] = t.index.isin(PROTECTED)
    return t


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=str(ROOT / "docs" / "slides" / "data" / "oct12_trace_features.csv"))
    parser.add_argument("--packets", help="processed directory: study the real planned packets instead (label-free)")
    parser.add_argument("--sessions", nargs="+", default=["A", "B"], help="plan sessions used with --packets")
    parser.add_argument("--plan", default=str(ROOT / "lvl1" / "packet_plan" / "plan.json"))
    parser.add_argument("--out", default=str(ROOT / "lvl1" / "feature_study"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.packets:
        df = packet_traces(Path(args.packets), Path(args.plan), args.sessions)
        print(f"{df.packet.nunique()} packets, {df.setting.nunique()} settings, {len(df)} traces (sessions "
              f"{'+'.join(args.sessions)}, board 101, own settings only)")
        t = study_packets(df)
    else:
        t = study(pd.read_csv(args.data))
    t.to_csv(out / "feature_table.csv")
    counts = pd.crosstab(t["implementation tier"], t["evidence subtier"])
    print(counts)
    # A1, A2, B, C and P (A = A1 + A2 is formed by the reader, lvl1/train_lvl1.py load_tiers)
    tiers = {k: list(v) for k, v in t.groupby("evidence subtier").groups.items()}
    tiers["P"] = [n for n in PROTECTED if n in t.index]                  # guard (a): given to every model
    (out / "evidence_tiers.json").write_text(json.dumps(tiers, indent=1))
    print(f"wrote {out / 'feature_table.csv'} and evidence_tiers.json")


if __name__ == "__main__":
    main()

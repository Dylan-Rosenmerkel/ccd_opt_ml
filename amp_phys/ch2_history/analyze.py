"""Board-101 ch2 vs ch0/1/3: per-trace features from the stored decoded traces, compared packet by packet.

Inputs: processed/L1_*/traces/*_acm101_ch*.npz (2-3 Nov 2026) and lvl1/demo/processed/legacy_* (12 Oct 2026),
labels.json (image sigma_e, gain). Output: scratch CSVs + printed summary.
"""
import glob, json, os, sys
import numpy as np, pandas as pd

ROOT = sys.argv[1]
OUT = sys.argv[2]
SKIP0 = 20  # drop the first-skip transient


def feats(path):
    z = np.load(path)
    cds = np.asarray(z["cds"], float)[:, SKIP0:]
    n = cds.shape[1]
    d = np.diff(cds, axis=1)
    s_pix = d.std(axis=1, ddof=1) / np.sqrt(2)
    ok = np.isfinite(s_pix) & (s_pix > 0)
    cds, d, s_pix = cds[ok], d[ok], s_pix[ok]
    sig1 = np.median(s_pix)
    cv = s_pix.std(ddof=1) / s_pix.mean()
    cv_stat = 1 / np.sqrt(2 * (n - 2))          # statistical CV of a std from n-1 diffs (approx, ignores diff correlation)
    dn = (d / (s_pix[:, None] * np.sqrt(2))).ravel()
    kurt = pd.Series(dn).kurt()
    burst = np.mean(np.abs(dn) > 5)
    k = np.arange(n)
    slope = np.array([np.polyfit(k, row, 1)[0] for row in cds])  # ADU per skip
    wl = np.asarray(z["curve: noise: noise vs window length"], float)
    shape = np.asarray(z["curve: skip: mean shape"], float)
    return dict(sigma1=sig1, pix_cv=cv, pix_cv_stat=cv_stat, kurt=kurt, burst=burst,
                drift_adu_per_pixel=np.median(slope) * n, wl_ratio=wl[0] / wl[-1], wl_short=wl[0], wl_long=wl[-1],
                skip_p2p=shape.max() - shape.min(), n_pix=int(ok.sum()))


def labels(pdir):
    p = os.path.join(pdir, "labels.json")
    out = {}
    if not os.path.exists(p):
        return out
    L = json.load(open(p))
    for im in L.get("images", []):
        if im.get("board") != 101:
            continue
        for lab in im.get("labels", []):
            out[lab["channel"]] = dict(sigma_e=lab.get("sigma_e"), gain=lab.get("gain_adu_per_e"),
                                       lam=lab.get("lambda_e_per_bin"), flags=";".join(lab.get("flags", [])))
    return out


rows = []
dirs = sorted(glob.glob(os.path.join(ROOT, "processed/L1_*"))) + sorted(glob.glob(os.path.join(ROOT, "lvl1/demo/processed/legacy_*")))
for pdir in dirs:
    pk = os.path.basename(pdir)
    lab = labels(pdir)
    for f in sorted(glob.glob(os.path.join(pdir, "traces/*_acm101_ch*.npz"))):
        name = os.path.basename(f)[:-4]
        tr, _, ch = name.split("_")
        ch = int(ch[2:])
        try:
            r = feats(f)
        except Exception as e:
            print("skip", f, e)
            continue
        r.update(packet=pk, trace=tr, kind="ref" if tr.startswith("ref") else "theta", channel=ch)
        r.update({k: v for k, v in lab.get(ch, {}).items()})
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "ch2_trace_features.csv"), index=False)
print("traces:", len(df), df.groupby("channel").size().to_dict())

# per-channel medians
cols = ["sigma1", "pix_cv", "kurt", "burst", "drift_adu_per_pixel", "wl_ratio", "skip_p2p", "sigma_e", "gain"]
print("\nmedian by channel (all traces):")
print(df.groupby("channel")[cols].median().round(4).T.to_string())

# paired comparison: same packet, same trace name -> ch2 vs each other channel
pv = {c: df.pivot_table(index=["packet", "trace"], columns="channel", values=c) for c in cols}
print("\nch2 vs others, same packet and trace (median [IQR]); ratio for sizes, difference for shapes")
for c in cols:
    p = pv[c].dropna()
    if 2 not in p:
        continue
    res = []
    for o in [0, 1, 3]:
        if o not in p:
            continue
        if c in ("sigma1", "wl_ratio", "skip_p2p", "sigma_e", "gain", "pix_cv"):
            x = p[2] / p[o]; tag = "ratio"
        else:
            x = p[2] - p[o]; tag = "diff"
        res.append("vs ch%d %s %.3f [%.3f, %.3f]" % (o, tag, x.median(), x.quantile(.25), x.quantile(.75)))
    rank = p.rank(axis=1, ascending=False)[2]
    print("%-22s n=%4d  %s   ch2 highest in %3.0f%%" % (c, len(p), "  ".join(res), 100 * (rank == 1).mean()))

# history: per packet, ref traces only, ch2 sigma1 / median(others) and modulation
ref = df[df.kind == "ref"]
h = []
for pk, g in ref.groupby("packet"):
    m = g.groupby("channel")[["sigma1", "pix_cv", "kurt", "sigma_e"]].median()
    if 2 not in m.index:
        continue
    o = m.drop(index=2).median()
    h.append(dict(packet=pk, sig1_ratio=m.loc[2, "sigma1"] / o["sigma1"], pixcv_ch2=m.loc[2, "pix_cv"], pixcv_oth=o["pix_cv"],
                  kurt_ch2=m.loc[2, "kurt"], kurt_oth=o["kurt"], sige_ratio=m.loc[2, "sigma_e"] / o["sigma_e"] if pd.notna(o["sigma_e"]) else np.nan))
H = pd.DataFrame(h)
H.to_csv(os.path.join(OUT, "ch2_history.csv"), index=False)
print("\nhistory (ref traces), first/last packets:")
print(H.round(3).to_string(index=False))

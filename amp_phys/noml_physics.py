import json, sys
import numpy as np, pandas as pd
D = sys.argv[1]
d = pd.read_csv(f"{D}/dataset.csv", low_memory=False)
packets = json.load(open(f"{D}/plan.json"))["packets"]
session = {q["packet_id"]: str(q.get("session", "?")) for q in packets}
def planned_id(pid, planned):
    if pid in planned: return pid
    for q in planned:
        if pid.startswith(q) and not pid[len(q)].isdigit(): return q
    return None
d["session"] = [session.get(planned_id(str(x), session), "?") for x in d.packet_id]
t = lambda c: d[c].astype(str).str.lower().eq("true")
ok = d.board.eq(101) & d.channel.isin([0, 1, 3]) & d.session.isin(["G", "H"]) & t("label.ok")
if "label.fractional_window" in d: ok &= ~t("label.fractional_window")
d = d[ok].copy()
F = pd.read_csv(f"{D}/pf.csv")
d = d.merge(F, on=["packet_id", "board", "channel"], how="inner")
d = d[(d["S_w"] > 0) & np.isfinite(d["sigma_500 pred (ADU)"])].reset_index(drop=True)
y = np.log(pd.to_numeric(d["label.sigma_e"])); g = np.log(pd.to_numeric(d["label.gain_adu_per_e"]))
s5 = np.log(d["sigma_500 pred (ADU)"])
nominal = d.theta_id.astype(str).eq(d.reference_id.astype(str))
print(len(d), "rows;", (d.session == "H").sum(), "H;", "nominal G rows per amp:", d[nominal & d.session.eq("G")].groupby("channel").size().to_dict())
res = {}
def add(name, e): res.setdefault(name, []).append(e)
for amp in (0, 1, 3):
    te = d.session.eq("H") & d.channel.eq(amp); tr = d.session.eq("G") & ~d.channel.eq(amp)
    own_nom = d.session.eq("G") & d.channel.eq(amp) & nominal
    for cal in (False, True):
        sfx = " + 1 constant fitted on the other amplifiers" if cal else ""
        # (a) gain from the same image (oracle)
        c = np.median((y - (s5 - g))[tr]) if cal else 0.0
        add("σ500 ÷ image gain of that same image" + sfx, (s5 - g + c - y)[te])
        # (b) gain from one nominal image of the new amplifier (session G)
        gn = np.median(g[own_nom])
        c = np.median((y - (s5 - g))[tr]) if cal else 0.0
        add("σ500 ÷ gain from the new amplifier's nominal images" + sfx, (s5 - gn + c - y)[te])
        # (c) gain = median of the other amplifiers (nothing known about the new one)
        gf = np.median(g[tr])
        c = np.median((y - (s5 - gf))[tr]) if cal else 0.0
        add("σ500 ÷ typical gain of the other amplifiers" + sfx, (s5 - gf + c - y)[te])
for k, v in res.items():
    e = np.concatenate([np.asarray(x) for x in v]); print("%-80s n=%d  RMS %.1f %%  median|e| %.1f %%  mean %.1f %%" % (k, len(e), 100*np.sqrt(np.mean(e**2)), 100*np.median(np.abs(e)), 100*np.mean(e)))

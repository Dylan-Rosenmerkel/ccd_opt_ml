"""Pilot: the dip just before the per-cycle (reset) pulse, from the equivalent-time pulse, on the 59 traces."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
ROOT=Path("/Users/dylanrosenmerkel/Documents/Software/ccd_opt_ml"); sys.path.insert(0,str(ROOT/"src"))
from decode_trace import decode_trace; from trace_features import trace_features; from trace_regions import split_regions
RUN=ROOT/"data"/"2026-10-12-PA03_PQ04_Packet_Tests"
def group(m):
    if m["VDD1"] < -20: return "VDD −20.5 V"
    if m["V1_A_L"] < 7.6: return "V low rails + TG"
    if m["H1_A_L"] < 6.5: return "all H low rails"
    if m["H3_A_L"] < 6.5: return "H3_A_L only"
    return "reference"
bins=np.arange(-15,60.05,0.1); centers=bins[:-1]+0.05
rows=[]; curves={}
for mp in sorted(RUN.glob("trace_*_101_*.meta")):
    m=json.loads(mp.read_text()); data=max(mp.parent.glob(mp.stem+".RJ45ADCDATA*"),key=lambda p:p.stat().st_size)
    t=decode_trace(data)
    if t.quality_problems(): continue
    f=trace_features(t,meta=m,regions=split_regions(t,m))
    e=np.asarray(f.curves["edge: equivalent-time pulse"],float); curves[mp.stem]=e
    base=np.nanmean(e[(centers>-15)&(centers<-8)])
    win=(centers>-6)&(centers<0)
    i=np.flatnonzero(win)[np.nanargmin(e[win])]
    depth=e[i]-base; peak=np.nanmax(e)-base
    half=base+depth/2; below=np.flatnonzero(win & (e<half))
    rows.append({"file":mp.stem,"channel":int(data.suffix[-1]),"group":group(m),
        "pre-pulse dip depth (ADU)":depth,"pre-pulse dip time before edge (ns)":-centers[i]/15e6*1e9,
        "pre-pulse dip width (ns)":(centers[below[-1]]-centers[below[0]]+0.1)/15e6*1e9 if len(below) else np.nan,
        "pre-pulse dip / pulse height":depth/peak,"pulse height (ADU)":peak,
        "dip per volt RG (ADU/V)":depth/(m["RG_1_H"]-m["RG_1_L"]),"SW dip (skip: dip height)":f.scalars["skip: dip height"],
        "edge: cycle variation at reset (ADU)":f.scalars["edge: cycle variation at reset (ADU)"]})
    print(mp.stem[-18:],rows[-1]["channel"],rows[-1]["group"],round(depth),round(rows[-1]["pre-pulse dip time before edge (ns)"]),round(peak),flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1]+"/predip.csv",index=False); np.save(sys.argv[1]+"/ets.npy",curves,allow_pickle=True)

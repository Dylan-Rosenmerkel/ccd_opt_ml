"""Pilot: clocking-region features (V-block level slope, H-V level difference, first H block of a row) on the 59 traces."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
ROOT=Path("/Users/dylanrosenmerkel/Documents/Software/ccd_opt_ml"); sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"docs"/"slides")); sys.path.insert(0,str(ROOT/"lvl1"))
from decode_trace import decode_trace
from trace_regions import _rising
RUN=ROOT/"data"/"2026-10-12-PA03_PQ04_Packet_Tests"
def group(m):
    if m["VDD1"] < -20: return "VDD −20.5 V"
    if m["V1_A_L"] < 7.6: return "V low rails + TG"
    if m["H1_A_L"] < 6.5: return "all H low rails"
    if m["H3_A_L"] < 6.5: return "H3_A_L only"
    return "reference"
def per_transfer(x, edges, period):
    ends=np.r_[edges[1:], edges[-1]+period]; ends=np.minimum(ends, edges+int(period*1.05))
    seg=[x[int(a):int(b)] for a,b in zip(edges,ends)]
    return np.array([s.mean() for s in seg]), np.array([np.ptp(s) for s in seg]), np.array([np.median(s) for s in seg])
rows=[]; curves={}
for mp in sorted(RUN.glob("trace_*_101_*.meta")):
    m=json.loads(mp.read_text()); data=max(mp.parent.glob(mp.stem+".RJ45ADCDATA*"),key=lambda p:p.stat().st_size)
    t=decode_trace(data)
    if t.quality_problems(): continue
    x=np.asarray(t.values,float); fl=np.asarray(t.flags)
    v=_rising(fl,7); h=_rising(fl,6)
    vb=np.split(v,np.flatnonzero(np.diff(v)>20000)+1); hb=np.split(h,np.flatnonzero(np.diff(h)>5000)+1)
    vp=np.median(np.diff(vb[-1])); hp=np.median(np.diff(hb[1]))
    r={"file":mp.stem,"channel":int(data.suffix[-1]),"group":group(m),"NCOLPRE":m.get("NCOLPRE"),"n V block1":len(vb[0]),"n V block2":len(vb[1]),"n H blocks":len(hb),"H block sizes":"/".join(str(len(b)) for b in hb[:3])}
    lev=[]; ppv=[]
    for i,b in enumerate(vb[:2]):
        mean,pp,med=per_transfer(x,b,vp); mean,pp,med=mean[:-1],pp[:-1],med[:-1]   # last overlaps the H block
        # skip the first transfer of block 2 (it starts inside the end of the previous pixel)
        k=np.arange(len(med)); sel=k>=1
        r[f"V{i+1}: level slope (ADU/transfer)"]=np.polyfit(k[sel],med[sel],1)[0]
        r[f"V{i+1}: level change, last 10 transfers (ADU)"]=med[-1]-med[-10] if len(med)>=10 else np.nan
        r[f"V{i+1}: median level (ADU)"]=np.median(med[sel])
        r[f"V{i+1}: transfer peak-to-peak (ADU)"]=np.median(pp[sel])
        lev.append(med); ppv.append(pp)
    # first H block of row 2 = the first H block after V block 2
    after=[j for j,b in enumerate(hb) if b[0]>vb[1][-1]][0]
    ppH=[]; medH=[]
    for b in hb:
        mean,pp,med=per_transfer(x,b,hp); ppH.append(pp[:-1]); medH.append(np.median(med[:-1]))
    ppH=np.array([p[:min(map(len,ppH))] for p in ppH]); others=[j for j in range(len(hb)) if j not in (0,after)]
    base=np.median(ppH[others],axis=0)
    for lab,j in (("row 1",0),("row 2",after)):
        r[f"H first block {lab}: pp excess, transfers 0-4 (ADU)"]=np.mean(ppH[j,:5]-base[:5])
    r["H: transfer-1 dip, other blocks (ADU)"]=np.median(ppH[others,1]-np.median(ppH[others,10:],axis=1))
    r["H: transfer-1 dip, first block row 2 (ADU)"]=ppH[after,1]-np.median(ppH[after,10:])
    r["H-V level difference row 2 (ADU)"]=medH[after]-np.median(lev[1][1:])
    r["H-V peak-to-peak ratio"]=np.median(ppH[others])/np.median(ppv[1][1:])
    rows.append(r); curves[mp.stem]=(lev,ppH)
    print(mp.stem[-18:], r["channel"], r["group"], r["n V block1"], r["n V block2"], r["H block sizes"], round(r["V1: level slope (ADU/transfer)"]), round(r["V2: level slope (ADU/transfer)"]), flush=True)
df=pd.DataFrame(rows); df.to_csv(sys.argv[1]+"/clock_pilot.csv",index=False)
np.save(sys.argv[1]+"/clock_curves.npy",curves,allow_pickle=True)

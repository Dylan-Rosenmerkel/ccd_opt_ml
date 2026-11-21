import glob, os, sys, re
import numpy as np, pandas as pd
ROOT=sys.argv[1]; sys.path.insert(0, os.path.join(ROOT,'src'))
from decode_trace import decode_trace
rows=[]
for day in ['2026-10-06-PA03_PQ04_Installation_Check130K_1','2026-10-08-PA03_PQ04_traces','2026-10-12-PA03_PQ04_Packet_Tests']:
    for meta in sorted(glob.glob(os.path.join(ROOT,'data',day,'trace_*_ch?_101_*.meta'))):
        ch=int(re.search(r'_ch(\d)_101_',meta).group(1))
        f=meta[:-5]+'.RJ45ADCDATA%d'%ch
        if not os.path.exists(f): continue
        try:
            t=decode_trace(f); s=t.skips()
        except Exception as e:
            print('fail',f,e); continue
        df=pd.DataFrame({k:s[k] for k in ('pixel','skip','cds')}); df=df[df.skip>=20]
        sp=[]; dn=[]
        for p,g in df.groupby('pixel'):
            if len(g)<50: continue
            d=np.diff(g.cds.values); sd=d.std(ddof=1)/np.sqrt(2); sp.append(sd); dn.append(d/(sd*np.sqrt(2)))
        if not sp: continue
        sp=np.array(sp); dn=np.concatenate(dn)
        rows.append(dict(day=day[:10],file=os.path.basename(meta)[:-5],channel=ch,sigma1=np.median(sp),
             pix_cv=sp.std(ddof=1)/sp.mean(),kurt=pd.Series(dn).kurt(),frac_gt5=np.mean(np.abs(dn)>5),npix=len(sp)))
d=pd.DataFrame(rows); d.to_csv(os.path.join(sys.argv[2],'october_traces.csv'),index=False)
pd.set_option('display.width',200)
print(d.groupby(['day','channel'])[['sigma1','pix_cv','kurt','frac_gt5']].median().round(3).to_string())
print(d.groupby(['day','channel']).size().unstack())

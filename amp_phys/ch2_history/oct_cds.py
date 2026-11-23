import glob, os, sys
import numpy as np, pandas as pd
ROOT=sys.argv[1]; sys.path.insert(0, os.path.join(ROOT,'src'))
from decode_trace import decode_trace
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(3,2,figsize=(12,9))
for row,(day,ch) in enumerate([('2026-10-06-PA03_PQ04_Installation_Check130K_1',2),('2026-10-12-PA03_PQ04_Packet_Tests',2),('2026-10-06-PA03_PQ04_Installation_Check130K_1',0)]):
    meta=sorted(glob.glob(os.path.join(ROOT,'data',day,'trace_*_ch%d_101_*.meta'%ch)))[0]
    s=decode_trace(meta[:-5]+'.RJ45ADCDATA%d'%ch).skips()
    df=pd.DataFrame({k:s[k] for k in ('pixel','skip','cds')})
    lag1=[];bimod=[];specs=[]
    for p,g in df.groupby('pixel'):
        x=g.cds.values[20:]
        if len(x)<100: continue
        x=x-x.mean(); lag1.append(np.corrcoef(x[:-1],x[1:])[0,1])
        specs.append(np.abs(np.fft.rfft(x*np.hanning(len(x))))**2)
        if len(lag1)<=3: ax[row,0].plot(g.skip.values[20:],g.cds.values[20:],lw=.6)
    n=min(len(v) for v in specs); sp=np.mean([v[:n] for v in specs],0); nu=np.arange(n)/(2*(n-1))
    ax[row,1].semilogy(nu[1:],sp[1:]); ax[row,1].set_xlabel('cycles per skip')
    ax[row,0].set_title('%s ch%d: CDS vs skip (3 pixels)'%(day[:10],ch)); ax[row,0].set_xlabel('skip')
    k=np.argsort(sp[1:])[::-1][:3]+1
    print(day[:10],'ch%d'%ch,'lag-1 corr of CDS %.2f'%np.median(lag1),'top nu',np.round(nu[k],3),'peak/median %.0f'%(sp[k[0]]/np.median(sp[1:])),
          'low/high power ratio (nu<0.05 vs >0.3) %.1f'%(sp[(nu>0)&(nu<0.05)].mean()/sp[nu>0.3].mean()))
plt.tight_layout(); plt.savefig(os.path.join(sys.argv[2],'oct_ch2_cds.png'),dpi=80)

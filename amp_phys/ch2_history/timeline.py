import sys, os, numpy as np, pandas as pd, datetime as dt
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S=sys.argv[1]; ROOT=sys.argv[2]
o=pd.read_csv(os.path.join(S,'october_traces.csv')); o['t']=pd.to_datetime(o.file.str.extract(r'_(\d{8}_\d{6})_')[0],format='%Y%m%d_%H%M%S')
# pair each ch2 trace with the other channels' traces taken within the same minute group
o=o.sort_values('t'); pts=[]
ref=o[o.channel!=2].groupby(o.t.dt.floor('2min')).sigma1.median()
for _,r in o[o.channel==2].iterrows():
    k=r.t.floor('2min')
    if k in ref.index: pts.append((r.t, r.sigma1/ref[k], 'raw traces, Oct'))
n=pd.read_csv(os.path.join(S,'ch2_trace_features.csv')); n=n[n.packet.str.startswith('L1_')]
ds=pd.read_csv(os.path.join(ROOT,'processed/dataset.csv'),usecols=['packet_id','date_start']).groupby('packet_id').date_start.min()
for (pk,tr),g in n.groupby(['packet','trace']):
    m=g.set_index('channel').sigma1
    if 2 in m and len(m)==4: pts.append((pd.to_datetime(ds[pk]), m[2]/m.drop(2).median(), 'decoded packets, Nov 2-3'))
P=pd.DataFrame(pts,columns=['t','ratio','src'])
P.to_csv(os.path.join(S,'ch2_timeline.csv'),index=False)
fig,ax=plt.subplots(figsize=(9,4.2))
for src,mk in [('raw traces, Oct','o'),('decoded packets, Nov 2-3','.')]:
    q=P[P.src==src]; ax.plot(q.t,q.ratio,mk,ms=5 if mk=='o' else 3,alpha=.7,label=src)
ax.plot([pd.Timestamp('2026-11-11')],[4.28/np.median([3.66,3.90,4.34])],'s',ms=8,label='v3 physics sheet, Nov 10-12 (median)')
ax.plot([pd.Timestamp('2026-10-15')],[2.0],'D',ms=8,label='Oct images: single-sample noise ~2x')
ax.axhline(1,color='k',lw=.8); ax.set_yscale('log'); ax.set_ylabel('ch2 noise per skip / other channels')
ax.axvspan(pd.Timestamp('2026-11-07'),pd.Timestamp('2026-11-10'),color='0.85',label='cabling change')
ax.set_title('Board 101 ch2: excess single-skip noise over time (same packets, other channels as reference)')
ax.legend(fontsize=7,loc='upper right'); fig.autofmt_xdate(); plt.tight_layout(); plt.savefig(os.path.join(S,'ch2_timeline.png'),dpi=110)
print(P.groupby([P.t.dt.date]).ratio.describe()[['count','min','50%','max']].round(2).to_string())

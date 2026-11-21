"""Electrons per pixel from the unbinned Nov 2-3 traces (2 rows x 5 cols x 500 skips, 1x1), per channel.
Per trace: pixel means over skips 20-499 for pixels 1-4, 6-9; zero = mean of the pixels within 0.5 e of the lowest;
counts k = round((m - zero)/gain). lambda = mean k over pixels (zero-defining minimum pixel excluded)."""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT, OUT = sys.argv[1], sys.argv[2]
rows = []
for pdir in sorted(glob.glob(os.path.join(ROOT, 'processed/L1_*'))):
    pk = os.path.basename(pdir)
    lab = {}
    L = json.load(open(os.path.join(pdir, 'labels.json')))
    for im in L['images']:
        if im['board'] == 101:
            for l in im['labels']:
                lab[l['channel']] = (l.get('gain_adu_per_e'), l.get('lambda_e_per_bin'))
    R = json.load(open(os.path.join(pdir, 'result.json')))
    meta = {t['name']: t for t in R['traces']}
    for f in sorted(glob.glob(os.path.join(pdir, 'traces/*_acm101_ch*.npz'))):
        name = os.path.basename(f)[:-4]; ch = int(name[-1])
        z = np.load(f)
        if 'cds' not in z.files: continue
        cds = np.asarray(z['cds'], float)
        if cds.shape != (10, 500) or ch not in lab or lab[ch][0] is None: continue
        g = lab[ch][0]
        m = cds[[1, 2, 3, 4, 6, 7, 8, 9], 20:].mean(1)
        sd = cds[[1, 2, 3, 4, 6, 7, 8, 9], 20:].std(1) / np.sqrt(480)
        e = (m - m.min()) / g
        zero = m[e < 0.5].mean()
        x = (m - zero) / g
        k = np.round(x)
        i0 = np.argmin(m)
        keep = np.ones(8, bool); keep[i0] = False
        rows.append(dict(packet=pk, trace=name.split('_')[0], channel=ch, gain=g, lam_image=lab[ch][1],
                         lam_trace=k[keep].clip(0).mean(), frac0=np.mean(k[keep] <= 0), max_e=x.max(),
                         resid=np.abs(x - k)[keep].mean(), sd_e=np.median(sd) / g,
                         **{f'x{j}': v for j, v in enumerate(x)}))
d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'trace_charge.csv'), index=False)
pd.set_option('display.width', 200)
print('traces per channel', d.groupby('channel').size().to_dict())
print('per-pixel mean noise (e):', d.groupby('channel').sd_e.median().round(3).to_dict(),
      ' mean |x - nearest integer| (0.25 = no quantisation):', d.groupby('channel').resid.mean().round(3).to_dict())
s = d.groupby('channel').agg(lam_trace=('lam_trace', 'mean'), frac0=('frac0', 'mean'), lam_image=('lam_image', 'median'))
s['lam_from_frac0'] = -np.log(s.frac0)
print(s.round(3).to_string())
X = d[[f'x{j}' for j in range(8)]].values
for ch in range(4):
    v = X[d.channel.values == ch].ravel()
    h, b = np.histogram(v, bins=np.arange(-0.5, 4.51, 0.1))
    print('ch%d histogram of pixel charge (e), 0.1 e bins from -0.5:' % ch, ' '.join(str(c) for c in h[:30]))
# per packet: trace lambda vs image lambda
p = d.groupby(['packet', 'channel']).agg(lt=('lam_trace', 'mean'), li=('lam_image', 'median')).reset_index()
for ch in range(4):
    q = p[p.channel == ch]
    print('ch%d: packets %d, corr(trace lambda, image lambda) = %.2f, ratio median %.2f' % (ch, len(q), q.lt.corr(q.li), (q.lt / q.li).median()))

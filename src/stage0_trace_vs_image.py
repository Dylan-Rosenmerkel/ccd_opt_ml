"""
stage0_trace_vs_image.py -- stage 0 of the real-data test plan (docs/overleaf_ml_models/ml_models.tex): does the
trace physics predict the image's read noise?

For every trace set of one board in a run directory, the Allan deviation of the CDS is fitted with
a^2/n + b^2 (only n <= N_skips / 5: longer blocks are too few and biased low), and the read noise after
500 skips sqrt(a^2/500 + b^2) is compared with the robust noise of the image's overscan (last 6 columns,
first 2 rows left out) at the same settings.  A ratio near 1 means model A works almost without
training; a larger ratio is the image-only noise that A's learned correction must supply.

    cd src && ../.venv/bin/python stage0_trace_vs_image.py [run_dir] [board]

Written for the 2026-10-12 packet tests (settings grouped by identical .meta parameters; images
avg_img_*.fits.bz2 next to the raw files).  First results (2026-10-14): with the detrended Allan curve the overscan noise of
the binned 20x20 images is 1.2-2.4x the trace prediction on ch0 and 4-9x on ch1/ch3; with the curve that
keeps the slow pixel-to-pixel noise ("common pattern removed"), see the last column.
"""
import sys, json, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_features import trace_features, ALLAN_N
from ml.structured import allan_fit
from astropy.io import fits
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / 'data' / '2026-10-12-PA03_PQ04_Packet_Tests'
BOARD = int(sys.argv[2]) if len(sys.argv) > 2 else 101
metas = sorted(((json.loads(p.read_text()), p) for p in RUN.glob("*.meta")), key=lambda x: x[0]["DATEini"])
KEYS = [k for k in metas[0][0] if k[:2] in ("V1","V2","V3","H1","H2","H3","TG","SW","OG","RG","DG") or k in ("VSUB","VDD1","VDD2","VDRAIN1","VDRAIN2","VR1","VR2") or k.startswith("delay")]
groups, sig = [], None
for m, p in metas:
    if m["ACMID"] != BOARD: continue
    s = json.dumps({k: m.get(k) for k in KEYS})
    if s != sig: groups.append([]); sig = s
    groups[-1].append((m, p))
rows = []
for g, files in enumerate(groups, start=1):
    img = next((RUN / f"avg_{p.stem}.fits.bz2" for m, p in files if p.name.startswith("img_") and (RUN / f"avg_{p.stem}.fits.bz2").exists()), None)
    over = {}
    if img is not None:
        h = fits.open(img)
        for ch in range(4):
            d = h[ch + 1].data.astype(float)[2:, -6:]
            over[ch] = 1.4826 * np.median(np.abs(d - np.median(d)))
    for m, p in files:
        if not p.name.startswith("trace"): continue
        ch = int(p.name.split("_ch")[1][0])
        try:
            f = trace_features(p.with_suffix(f".RJ45ADCDATA{ch}"), meta=m)
        except ValueError:
            continue
        al = f.curves["noise: Allan deviation"]
        a, b = allan_fit(al[None], ALLAN_N, max_n=100)
        alc = f.curves["noise: Allan deviation, common pattern removed"]
        ac, bc = allan_fit(alc[None], ALLAN_N)
        rows.append(dict(g=g, ch=ch, cds=f.scalars["skip: CDS rms (detrended)"], a=a[0], b=b[0], pred=np.sqrt(a[0]**2/500 + b[0]**2),
                         ac=ac[0], bc=bc[0], predc=np.sqrt(ac[0]**2/500 + bc[0]**2),
                         white=a[0]/np.sqrt(500), allan160=al[-1], over=over.get(ch, np.nan), ncolpre=m.get("NCOLPRE")))
print("detrended curve (n <= 100) | curve with the common pattern removed (all n) | image")
print(f"{'set':>3s} {'ch':>2s} {'NCOLPRE':>7s} {'CDS rms':>8s} {'a':>7s} {'b':>6s} {'→ 500':>7s} | {'a':>6s} {'b':>6s} {'→ 500':>7s} | {'overscan':>8s} {'ratio':>6s} {'ratio':>6s}")
for g in sorted({r['g'] for r in rows}):
    for ch in range(4):
        rr = [r for r in rows if r['g'] == g and r['ch'] == ch]
        if not rr: continue
        mean = lambda k: np.mean([r[k] for r in rr])
        print(f"{g:3d} {ch:2d} {rr[0]['ncolpre']:7d} {mean('cds'):8.2f} {mean('a'):7.2f} {mean('b'):6.3f} {mean('pred'):7.3f} | "
              f"{mean('ac'):6.2f} {mean('bc'):6.3f} {mean('predc'):7.3f} | {mean('over'):8.3f} {mean('over')/mean('pred'):6.2f} {mean('over')/mean('predc'):6.2f}")

# the Allan curves themselves: ratio to the white-noise expectation CDS rms / sqrt(n)
print("\nAllan deviation / (CDS rms / sqrt(n)), mean over the trace sets of settings 2-4")
print("n:      " + " ".join(f"{n:6d}" for n in ALLAN_N))
for ch in range(4):
    rr = [r for r in rows if r['ch'] == ch and r['g'] in (2, 3, 4)]
    curves_ = []
    for g, files in enumerate(groups, start=1):
        if g not in (2, 3, 4): continue
        for m, p in files:
            if p.name.startswith("trace") and int(p.name.split("_ch")[1][0]) == ch:
                try:
                    f = trace_features(p.with_suffix(f".RJ45ADCDATA{ch}"), meta=m)
                except ValueError:
                    continue
                curves_.append(f.curves["noise: Allan deviation"] / (f.scalars["skip: CDS rms (detrended)"] / np.sqrt(ALLAN_N)))
    print(f"ch{ch}:    " + " ".join(f"{v:6.2f}" for v in np.mean(curves_, 0)))

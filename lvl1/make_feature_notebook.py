"""
Build lvl1/new_trace_features.ipynb: the trace features added on 2026-10-21 (src/trace_features_extra.py), shown on the
real 2026-10-12 traces, with the feature study (lvl1/feature_study.py) and the synthetic tolerance test
(lvl1/feature_tolerance.py).

    .venv/bin/python lvl1/make_feature_notebook.py
    .venv/bin/jupyter execute --inplace lvl1/new_trace_features.ipynb     (or run it in Jupyter, from lvl1/)

Needs data/2026-10-12-PA03_PQ04_Packet_Tests (not in the repository) and docs/slides/data/oct12_trace_features.csv.
"""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# The new trace features (2026-10-21)

On 2026-10-21, 49 scalar features and 4 curves were added to the 85 + 14 of `src/trace_features.py`. They are computed
in `src/trace_features_extra.py`, and every feature now has a **tier**:

* 0: the original set;
* 1: implement-first;
* 2: candidates.

They came from a search of signal analysis in 21 fields (`docs/literature/signal_analysis_features.md`). The candidate
list and the reasons are in `docs/trace_feature_candidates.md`.

| Section | Family | Borrowed from |
|---|---|---|
| 1 | the start-of-pixel transient (`start:`) | CDS theory, Prony / matrix-pencil (power systems), matched filters (radar) |
| 2 | telegraph switching inside one window, null CDS | patch-clamp electrophysiology, gravitational-wave null streams |
| 3 | jump-robust noise | high-frequency finance (median realized variance) |
| 4 | noise shape: 1/f corner, Hadamard and modified Allan deviation, complexity measures | semiconductor noise, time-frequency metrology, cardiology, nonlinear dynamics |
| 5 | pickup lines: multitaper F-test and their phase | oceanography, gravitational-wave line tracking |
| 6 | the skip-cycle noise budget | ECG beat analysis, machine diagnostics |
| 7 | results on all 59 real traces, evidence tiers | `lvl1/feature_study.py` |
| 8 | how many features each model can take | `lvl1/feature_tolerance.py` (synthetic) |

**Data.** Real traces from the 2026-10-12 packet tests, board 101. There are 5 settings (reference, VDD −20.5 V,
V low rails + TG, all H low rails, H3_A_L only), each traced 3 times on all 4 amplifiers. The traces are 20×20-binned,
but structured like the planned 1×1 traces. ch2 is the noisy amplifier.
""")

code(r"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "lvl1" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))
import trace_features as tf, trace_features_extra as tx
from decode_trace import decode_trace, SAMPLE_RATE_HZ
from trace_regions import split_regions
from compare_trace import usable_pixels
RUN = ROOT / "data" / "2026-10-12-PA03_PQ04_Packet_Tests"
TABLE = pd.read_csv(ROOT / "docs" / "slides" / "data" / "oct12_trace_features.csv")
COLORS = {"reference": "#4a5568", "VDD −20.5 V": "#dd6b20", "V low rails + TG": "#2f855a", "all H low rails": "#c53030", "H3_A_L only": "#6b46c1"}

def load(group, channel, repeat=1):
    stem = TABLE[(TABLE.group == group) & (TABLE.channel == channel) & (TABLE.repeat == repeat)]["file"].iloc[0]
    meta = json.loads((RUN / f"{stem}.meta").read_text())
    t = decode_trace(RUN / f"{stem}.RJ45ADCDATA{channel}")
    r = split_regions(t, meta)
    return t, r, meta, tf.trace_features(t, meta=meta, regions=r)

print("tiers:", {k: len(tf.names_by_tier((k,))) for k in (0, 1, 2)}, "scalars;",
      len(tf.names_by_tier((0, 1, 2), "curve")), "curves")
t0, r0, m0, f0 = load("reference", 0)
t2, r2, m2, f2 = load("reference", 2)
pd.DataFrame([{"feature": n, "tier": tf.FEATURES[n].tier, "ch0": f0.scalars[n], "ch2": f2.scalars[n], "what": tf.FEATURES[n].description[:90]}
              for n in tf.names_by_tier((1, 2))]).round(4)
""")

md(r"""
## 1. The start-of-pixel transient (`start:`)

The first skips of every pixel are offset: skip 0 is about +290 ADU on board 101, which is why skips 0–9 are left out
of the noise features. The pilot study found that this offset is **the CDS reading the slope of a decaying baseline**.

The reasoning: CDS = mean(pedestal window) − mean(signal window), and the two windows' centres are about 207 samples
apart. If the baseline falls with slope s (ADU per sample) during one skip, the CDS picks up about −207 · s.
"""

r"""
Two features test this on every trace:

* `start: CDS-slope lever arm`: the regression slope of the CDS excess on the in-window slope excess, over skips 0–19.
  A value near −207 means pure baseline slope.
* `start: slope-explained fraction`: the R² of that regression.

The plot below (the "differentiator plot") shows it for one trace: the points lie on one line, with the slope expected
from the window geometry.
""")

code(r"""
def first_skip_points(r):
    good = usable_pixels(r)
    n_ped, sig = r.n_ped, r.sig_start
    xs, ys, ks = [], [], []
    for p in good:
        Y = r.skip[p].astype(float)
        delta = 0.5 * (tx.slopes(Y[:, :n_ped]) + tx.slopes(Y[:, sig:sig + n_ped]))
        cds = np.asarray(r.cds[p], float)
        xs.append(delta[:20] - np.median(delta[20:60])); ys.append(cds[:20] - np.median(cds[10:])); ks.append(np.arange(20))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ks)

x, y, k = first_skip_points(r0)
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(x, y, c=k, cmap="viridis", s=18)
xx = np.linspace(x.min(), x.max(), 10)
ax.plot(xx, f0.scalars["start: CDS-slope lever arm (samples)"] * xx, "k--",
        label=f"fit: lever arm {f0.scalars['start: CDS-slope lever arm (samples)']:.0f} samples, R² {f0.scalars['start: slope-explained fraction']:.2f}")
ax.plot(xx, -(r0.sig_start) * xx, ":", color="grey", label=f"window-centre separation: −{r0.sig_start}")
plt.colorbar(sc, label="skip number")
ax.set_xlabel("in-window baseline slope − late [ADU/sample]"); ax.set_ylabel("CDS − median of later skips [ADU]")
ax.set_title("Differentiator plot, ch0 reference: the first-skip offset is a baseline slope"); ax.legend()
plt.show()
""")

md(r"""
Inside skip 0, the transient itself is an exponential. Each skip has its pedestal mean removed, and the late mean
waveform is subtracted; the remainder is fitted on the pedestal and signal windows only. This gives
`start: transient time constant` and `start: transient amplitude`.

Below is the **mean transient curve** (`start: mean transient`: the CDS excess of skips 0–29) for ch0 at all 5
settings. Only the H-rail change moves it: lowering all H low rails raises the transient by about 21 %, i.e.
horizontal-clock feedthrough.
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for g in COLORS:
    _, _, _, f = load(g, 0)
    axes[0].plot(f.curves["start: mean transient"], "o-", ms=3, color=COLORS[g], label=g)
axes[0].set_xlabel("skip number"); axes[0].set_ylabel("CDS − median of later skips [ADU]"); axes[0].set_title("start: mean transient (ch0)")
axes[0].legend(fontsize=9)
med = TABLE.groupby("group")[["start: transient amplitude (ADU)", "start: in-cycle slope at skip 0 (ADU/us)"]].median()
med.loc[list(COLORS)].plot.bar(ax=axes[1], color=["#2b6cb0", "#a0aec0"], subplots=False, rot=20)
axes[1].set_title("median over all amplifiers and repeats"); plt.tight_layout(); plt.show()
med
""")

md(r"""
## 2. Telegraph switching inside one window: the null CDS and in-window steps

ch2's excess noise turned out to sit **inside a single 10 µs window**, with no charge transfer involved. Two
measurements show it:

* the **null CDS**: the first half of the pedestal window minus its second half. It has the same arithmetic as a CDS,
  but no charge moves in between (gravitational-wave "null streams" use the same trick);
* the **in-window step detector**: the best single step in 10-sample block means of each window, in units of its
  standard error. A window counts as switching if t > 6.

Random telegraph switching in the output transistor gives steps of equal size, up and down. Patch-clamp
electrophysiology has the best tools for analysing it.
""")

code(r"""
half = r0.n_ped // 2
def nulls(r):
    from trace_features import residual_cycles
    return np.concatenate([(lambda R: R[:, :half].mean(1) - R[:, half:2 * half].mean(1))(residual_cycles(r.skip[p].astype(float), 10))
                           for p in usable_pixels(r)])
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
for r, f, lab, c in ((r0, f0, "ch0", "#2b6cb0"), (r2, f2, "ch2", "#c53030")):
    axes[0].hist(nulls(r), bins=80, histtype="step", lw=2, color=c, density=True,
                 label=f"{lab}: sd {f.scalars['noise: null-CDS noise (ADU)']:.1f}, kurtosis {f.scalars['discrete: null-CDS kurtosis']:.1f}")
axes[0].set_yscale("log"); axes[0].set_xlabel("null CDS [ADU]"); axes[0].legend(); axes[0].set_title("null CDS (no charge transfer)")
# one ch2 window with a detected step
from trace_features import residual_cycles
p = usable_pixels(r2)[0]
R = residual_cycles(r2.skip[p].astype(float), 10)
W = R[:, :r2.n_ped]
m = W.shape[1] // 10
b = W[:, :m * 10].reshape(len(W), m, 10).mean(2)
jump = np.abs(b[:, m // 2:].mean(1) - b[:, :m // 2].mean(1))
i = int(np.argmax(jump))
axes[1].plot(np.arange(r2.n_ped) / 15, W[i], color="#c53030", lw=1); axes[1].plot(np.arange(r2.n_ped) / 15, np.repeat(b[i], 10)[:r2.n_ped], "k", lw=2)
axes[1].set_xlabel("time in the pedestal window [µs]"); axes[1].set_ylabel("ADU (mean waveform removed)"); axes[1].set_title("a ch2 window with a step")
TABLE.groupby("channel")["discrete: in-window step rate (%)"].median().plot.bar(ax=axes[2], color="#4a5568", rot=0)
axes[2].set_ylabel("% of windows with a step"); axes[2].set_title("discrete: in-window step rate (median)")
plt.tight_layout(); plt.show()
""")

md(r"""
## 3. Jump-robust noise (from high-frequency finance)

In finance, a price series' variance is split into its continuous part and its **jumps**. The median realized variance
MedRV (Andersen, Dobrev & Schaumburg 2012) uses the median of three neighbouring |e|, so it ignores isolated jumps:

* `noise: jump-robust CDS sigma` = √(MedRV/N);
* `discrete: jump share of CDS variance` = 1 − MedRV / Σe².

On the good amplifiers the share is about 0; on ch2 about 27 % of the CDS variance is in jumps.
""")

code(r"""
fig, ax = plt.subplots(figsize=(14, 3.8))
for f, r, lab, c in ((f0, r0, "ch0", "#2b6cb0"), (f2, r2, "ch2", "#c53030")):
    e = tf.detrend(np.asarray(r.cds[usable_pixels(r)[0]][10:], float))
    ax.plot(e, lw=0.8, color=c, label=f"{lab}: rms {e.std():.1f}, jump-robust sigma {f.scalars['noise: jump-robust CDS sigma (ADU)']:.1f}, "
                                       f"jump share {f.scalars['discrete: jump share of CDS variance']:.2f}")
ax.set_xlabel("skip"); ax.set_ylabel("detrended CDS [ADU]"); ax.legend(); plt.show()
TABLE.groupby("channel")[["noise: jump-robust CDS sigma (ADU)", "skip: CDS rms (detrended)", "discrete: jump share of CDS variance",
                          "discrete: jump count (Lee-Mykland, 1 %)"]].median().round(3)
""")

md(r"""
## 4. Noise shape

* **Hadamard deviation** (time-frequency metrology): the third difference of n-skip means. It is insensitive to a
  linear drift, so it needs no polynomial detrending, which biases the long-n Allan points low.
* **Modified Allan deviation**: separates anticorrelated noise from white noise.
* **1/f corner and exponent**: a fit of W + A f^−γ to the log-frequency CDS spectrum. On these traces the corner falls
  below the band, so there is no 1/f noise to speak of in 0.2 s. delayInt in session A should move it.
* **Complexity measures** (sample and permutation entropy, DFA, spectral slope and centroid, red-noise β): they carry
  information that the old noise numbers do not, and they separate the amplifiers.
""")

code(r"""
ns = tf.ALLAN_N
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
for f, lab, c in ((f0, "ch0", "#2b6cb0"), (f2, "ch2", "#c53030")):
    axes[0].loglog(ns, f.curves["noise: Allan deviation"], "o-", color=c, label=f"{lab} Allan (detrended)")
    axes[0].loglog(ns, f.curves["noise: Hadamard deviation"], "s--", color=c, label=f"{lab} Hadamard", alpha=0.7)
    axes[0].loglog(ns, f.curves["noise: modified Allan deviation"], "^:", color=c, label=f"{lab} modified Allan", alpha=0.7)
axes[0].set_xlabel("skips averaged n"); axes[0].set_ylabel("ADU"); axes[0].legend(fontsize=9)
spec = f0.curves["noise: CDS spectrum, log frequency"]
from trace_features import spectrum
axes[1].loglog(np.geomspace(1, 15350, len(spec)), spec, "o-", color="#2b6cb0", label="ch0 CDS spectrum (log bins)")
axes[1].set_xlabel("frequency [Hz] (approximate bin centres)"); axes[1].set_ylabel("ADU²/Hz")
axes[1].set_title(f"1/f corner {f0.scalars['noise: 1/f corner frequency (kHz)']:.3f} kHz (clipped to the band), exponent {f0.scalars['noise: 1/f exponent']:.2f}")
axes[1].legend(); plt.tight_layout(); plt.show()
""")

md(r"""
## 5. Pickup lines: a multitaper F-test and the line's phase

The old line count compared spectrum peaks with their neighbours. The **Thomson multitaper F-test** gives a stated
false-alarm rate instead (here p < 0.001). Each pixel's phase at its first skip also tells us where the line comes
from:

* if it is the same for every pixel (*pixel-clock coherence* ≈ 1), our own readout sequence makes the line;
* if it is the same in absolute time (*absolute-time coherence* ≈ 1), it is an external source.

The study in section 7 shows that on these traces the line features vary between repeats. The pickup changes from trace
to trace, so these are tier C for now.
""")

code(r"""
series = [tf.detrend(np.asarray(r0.cds[p][10:], float)) for p in usable_pixels(r0)]
freqs, F, mu, Fcrit = tx.multitaper_lines(series, SAMPLE_RATE_HZ / r0.period_exact)
fig, ax = plt.subplots(figsize=(12, 3.8))
ax.semilogy(freqs / 1e3, np.median(F, 0), color="#2b6cb0", label="median F over pixels (ch0)")
ax.axhline(Fcrit, color="#c53030", ls="--", label=f"F critical, p = 0.001 ({Fcrit:.1f})")
ax.set_xlabel("frequency [kHz]"); ax.set_ylabel("Thomson F"); ax.legend(); plt.show()
{k: round(f0.scalars[k], 3) for k in tf.names_by_tier((1, 2)) if "line" in k}
""")

md(r"""
## 6. The skip-cycle noise budget

Each skip cycle (≈ 489 samples) varies from skip to skip. A principal-component analysis of the residual cycles (ECG
beat analysis does the same with heartbeats) splits that variation into **modes**. For every mode we can compute how much
CDS variance it carries, because the CDS is a fixed weighting of the samples: +1/n over the pedestal window and −1/n
over the signal window.

The result: the largest waveform modes, the big clock and reset fluctuations, **do not reach the CDS at all**. They
carry about 1 % of its variance, and the CDS noise is spread over about 100 small modes.
""")

code(r"""
from trace_features import residual_cycles
good = usable_pixels(r0)
R = np.vstack([residual_cycles(r0.skip[p].astype(float), 10) for p in good])
w = np.zeros(R.shape[1]); w[:r0.n_ped] = 1 / r0.n_ped; w[r0.sig_start:r0.sig_start + r0.n_ped] = -1 / r0.n_ped
U, sv, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
wave = sv ** 2 / np.sum(sv ** 2); cds = (sv ** 2) * (Vt @ w) ** 2; cds /= cds.sum()
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(np.arange(1, 61), wave[:60], "o-", label="share of the waveform variance")
ax.semilogy(np.arange(1, 61), cds[:60], "s-", label="share of the CDS variance")
ax.set_xlabel("mode (ordered by waveform variance)"); ax.legend(); ax.set_title(
    f"ch0: 3 largest modes carry {f0.scalars['noise: CDS variance share of the 3 largest waveform modes']:.1%} of the CDS variance; "
    f"{f0.scalars['noise: waveform modes for 90 % of the CDS variance']:.0f} modes for 90 %")
plt.show()
""")

md(r"""
## 7. All 59 traces: what separates settings, what separates amplifiers, what is redundant

`lvl1/feature_study.py` scores every one of the 134 features:

* **settings F**: ANOVA F over the 4 changed settings (all taken with the same NCOLPRE), per amplifier ch0/1/3;
* **amplifier F**: ANOVA F over the 4 amplifiers, within each setting;
* **repeat share**: within-(setting, amplifier) variance ÷ total variance;
* **redundancy clusters**: |Spearman ρ| > 0.95.

It proposes evidence tiers for the models:

* **A, core**: informative, stable, the representative of its cluster;
* **B, supporting**;
* **C, store only**.

Caveat: these 4 settings did not change the noise, so noise features can only separate amplifiers here. The tiers are to
be revisited on session A.
""")

code(r"""
study = pd.read_csv(ROOT / "lvl1" / "feature_study" / "feature_table.csv", index_col=0)
print(pd.crosstab(study["implementation tier"], study["evidence tier"]))
top = study.sort_values("informativeness", ascending=False)
top[["implementation tier", "settings F min", "amplifier F median", "repeat share", "cluster size", "evidence tier"]].head(30).round(2)
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 6))
for tier, c in ((0, "#a0aec0"), (1, "#2b6cb0"), (2, "#2f855a")):
    s = study[study["implementation tier"] == tier]
    ax.scatter(np.log10(1 + s["amplifier F median"].clip(lower=0)), np.log10(1 + s["settings F min"].clip(lower=0)), s=25, c=c,
               label={0: "original", 1: "new, tier 1", 2: "new, tier 2"}[tier], alpha=0.8)
ax.axvline(np.log10(11), color="k", ls=":"); ax.axhline(np.log10(11), color="k", ls=":")
ax.set_xlabel("log10(1 + amplifier F)"); ax.set_ylabel("log10(1 + settings F, min over amplifiers)")
ax.set_title("every feature: separation of amplifiers vs of settings (dotted: F = 10)"); ax.legend(); plt.show()
""")

md(r"""
## 8. How many features can each model take? (synthetic)

The synthetic world has only 17 trace features, so it cannot say which *real* features are best. It can say how each
model copes with more inputs. `lvl1/feature_tolerance.py` adds 20, 50 or 100 extra columns of three kinds:

* pure noise;
* near-copies of existing features;
* real 2026-10-12 feature values drawn at random: realistic distributions and outliers, but no information.

It then reruns the training on 3 synthetic 41-packet datasets. The table is the change in error (percentage points)
against no extra columns.
""")

code(r"""
p = ROOT / "lvl1" / "feature_study" / "tolerance_summary.csv"
if p.exists():
    tol = pd.read_csv(p)
    display(tol.round(1))
else:
    print("run lvl1/feature_tolerance.py first (see lvl1/feature_study/README.md)")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent / "new_trace_features.ipynb"
nbf.write(nb, out)
print("wrote", out)

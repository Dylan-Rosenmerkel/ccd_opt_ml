import glob, json, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.io import fits
ROOT, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.size": 13, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
f = sorted(glob.glob(os.path.join(ROOT, "processed/L1_23/fits/*_101_*.fits")))[0]
X = np.array(fits.open(f)[1].data, float)[:, 10:860]
L = json.load(open(os.path.join(ROOT, "processed/L1_23/labels.json")))
g = [l for im in L["images"] if im["board"] == 101 for l in im["labels"] if l["channel"] == 0][0]["gain_adu_per_e"]
X = X - np.median(X)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(X.ravel(), bins=np.arange(-1.5, 5.5, 0.04), color="C0")
ax.set_yscale("log")
for k in range(4):
    ax.axvline(k * g, color="C3", ls=":", lw=1)
    ax.text(k * g, 3e3, f"{k} e⁻", color="C3", ha="center", fontsize=11)
ax.annotate("", xy=(g, 250), xytext=(0, 250), arrowprops=dict(arrowstyle="<->", color="k"))
ax.text(g / 2, 330, f"g = {g:.2f} ADU/e⁻", ha="center", fontsize=12)
ax.set_xlabel("pixel value (ADU)"); ax.set_ylabel("pixels"); ax.set_ylim(0.8, 6e3)
ax.set_title("Image gain g: the spacing of the electron peaks")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "gain_peaks.png")); print("g", g)

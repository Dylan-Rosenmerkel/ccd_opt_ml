"""
make_models_2026_11_05.py -- the figures of the models deck (docs/overleaf_models_2026-11-05/models.tex): how every level-1
model works, not how well.  Toy figures (boosting, GP, ensembles, kernels, convolution, the direction classifier, the
Pareto shortlist) are drawn from small made-up examples; the physics, state-vector, drift and pair figures use the first
night (board 101 ch0/1/3) to show what the models' inputs and targets look like.

    .venv/bin/python docs/slides/make_models_2026_11_05.py

Writes docs/overleaf_models_2026-11-05/figures/m_*.png.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lvl1")]
OUT = ROOT / "docs" / "overleaf_models_2026-11-05" / "figures"

INK, INK2, GRID, SURFACE = "#1b2430", "#4a5462", "#e2e0d8", "#f7f6f2"
BLUE, ORANGE, GREEN, RED, GREY, PURPLE, GOLD = "#2a78d6", "#c9551f", "#13865d", "#b3372e", "#9a9890", "#7a4fb5", "#b8860b"
AMP_COLOR = {0: BLUE, 1: ORANGE, 3: GREEN}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------------------------------------ toy data
RNG = np.random.default_rng(3)


def toy_truth(x):
    """A made-up 'read noise vs one knob' curve: a valley with a shoulder."""
    return 0.18 + 0.10 * (x - 0.45) ** 2 / 0.09 + 0.025 * np.sin(9 * x)


X1 = np.sort(RNG.uniform(0, 1, 40))
Y1 = toy_truth(X1) + RNG.normal(0, 0.012, len(X1))
XG = np.linspace(0, 1, 400)


def tree_partition():
    """One regression tree on two inputs: the plane cut into boxes, one value per box."""
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (120, 2))
    y = 0.2 + 0.12 * (X[:, 0] > 0.55) + 0.06 * (X[:, 1] > 0.3) * (X[:, 0] < 0.55) + rng.normal(0, 0.015, 120)
    t = DecisionTreeRegressor(max_leaf_nodes=4, min_samples_leaf=10).fit(X, y)
    g = np.linspace(0, 1, 300)
    G = np.array(np.meshgrid(g, g)).reshape(2, -1).T
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = ax.imshow(t.predict(G).reshape(300, 300), origin="lower", extent=(0, 1, 0, 1), cmap="Blues", alpha=0.75,
                   vmin=0.12, vmax=0.36, aspect="auto")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Blues", vmin=0.12, vmax=0.36, edgecolor=INK, lw=0.4, s=22)
    tr = t.tree_
    for node in range(tr.node_count):
        if tr.children_left[node] != -1:
            pass
    # the leaf values, written in their boxes
    leaves = t.apply(G)
    for lf in np.unique(leaves):
        c = G[leaves == lf].mean(0)
        ax.text(*c, f"{tr.value[lf][0][0]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold", color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.8))
    ax.set_xlabel("input 1 (e.g. a timing knob)")
    ax.set_ylabel("input 2 (e.g. a trace feature)")
    ax.set_title("one tree = boxes, one number per box")
    fig.colorbar(im, ax=ax, fraction=0.045, label="predicted value")
    save(fig, "m_tree")


def boosting():
    """Gradient boosting, step by step: each new small tree fits what the sum so far still gets wrong."""
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.3, max_depth=1, random_state=0).fit(X1[:, None], Y1)
    staged = list(gb.staged_predict(XG[:, None]))
    staged_tr = list(gb.staged_predict(X1[:, None]))
    for k, m in enumerate((1, 2, 5, 200), 1):
        prev = np.full(len(XG), Y1.mean()) if m == 1 else staged[m - 2]
        prev_tr = np.full(len(X1), Y1.mean()) if m == 1 else staged_tr[m - 2]
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
        a.scatter(X1, Y1, s=16, color=INK2, zorder=3, label="training points")
        a.plot(XG, prev, color=GREY, lw=1.6, ls="--", label=f"sum of the first {m - 1} trees" if m > 1 else "start: the mean")
        a.plot(XG, staged[m - 1], color=BLUE, lw=2.2, label=f"after tree {m}")
        a.set_title(f"the model after {m} tree{'s' if m > 1 else ''}")
        a.set_xlabel("knob")
        a.set_ylabel(r"$\sigma_e$ (toy)")
        a.legend(fontsize=8, loc="upper center")
        grid(a)
        res = Y1 - prev_tr
        b.axhline(0, color=GRID, lw=1)
        b.scatter(X1, res, s=16, color=ORANGE, zorder=3, label="what is still wrong (residuals)")
        b.plot(XG, staged[m - 1] - prev, color=GREEN, lw=2.2, label=f"tree {m} (scaled by $\\eta$ = 0.3)")
        b.set_ylim(-0.12, 0.12)
        b.set_title(f"tree {m} is fitted to these residuals")
        b.set_xlabel("knob")
        b.legend(fontsize=8, loc="upper center")
        grid(b)
        save(fig, f"m_boost_{k}")


def matern52(A, B, ls, s2=1.0):
    r = np.abs(A[:, None] - B[None, :]) / ls
    return s2 * (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)


def gp_steps():
    """GP: prior, then the posterior after 2, 5 and 15 points."""
    ls, s2, noise = 0.18, 0.06 ** 2, 0.012 ** 2
    mu0 = 0.22
    idx = RNG.permutation(len(X1))
    for k, n in enumerate((0, 2, 5, 15)):
        fig, ax = plt.subplots(figsize=(7.4, 3.3))
        if n == 0:
            K = matern52(XG, XG, ls, s2) + 1e-9 * np.eye(len(XG))
            L = np.linalg.cholesky(K)
            for j, c in enumerate((BLUE, ORANGE, GREEN)):
                ax.plot(XG, mu0 + L @ np.random.default_rng(j).normal(size=len(XG)), color=c, lw=1.4, alpha=0.9)
            ax.fill_between(XG, mu0 - 2 * np.sqrt(s2), mu0 + 2 * np.sqrt(s2), color=BLUE, alpha=0.12)
            ax.set_title("before any data: smooth curves of a typical wiggle size (3 random draws)")
        else:
            xi, yi = X1[idx[:n]], Y1[idx[:n]]
            K = matern52(xi, xi, ls, s2) + noise * np.eye(n)
            ks = matern52(XG, xi, ls, s2)
            alpha = np.linalg.solve(K, yi - mu0)
            m = mu0 + ks @ alpha
            v = s2 - np.einsum("ij,ji->i", ks, np.linalg.solve(K, ks.T))
            sd = np.sqrt(np.maximum(v, 0))
            ax.fill_between(XG, m - 2 * sd, m + 2 * sd, color=BLUE, alpha=0.15, label=r"$\pm 2$ std (the model's own error bar)")
            ax.plot(XG, m, color=BLUE, lw=2.2, label="posterior mean (the prediction)")
            ax.scatter(xi, yi, color=INK, s=26, zorder=4, label=f"{n} measured points")
            ax.set_title(f"after {n} points: the curve is pinned near the data, the band opens away from it")
            ax.legend(fontsize=8, loc="upper center", ncol=3)
        if n:
            ax.plot(XG, toy_truth(XG), color=GREY, lw=1, ls=":")
        ax.set_ylim(0.05, 0.5)
        ax.set_xlabel("knob")
        ax.set_ylabel(r"$\sigma_e$ (toy)")
        grid(ax)
        save(fig, f"m_gp_{k}")


def kernel_and_priors():
    """Matern-5/2 correlation vs distance; the dimension-scaled prior on l; a SAAS prior draw."""
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.3))
    a = axs[0]
    r = np.linspace(0, 1.2, 300)
    for ls, c in ((0.1, ORANGE), (0.3, BLUE), (1.0, GREEN)):
        a.plot(r, matern52(np.array([0.0]), r, ls)[0], color=c, lw=2, label=f"$\\ell$ = {ls:g}")
    a.set_xlabel(r"distance $|x_i - x'_i|$ along one input")
    a.set_ylabel("correlation of the two outputs")
    a.set_title("Matérn-5/2: how far one point 'reaches'")
    a.legend(fontsize=8)
    grid(a)
    b = axs[1]
    from scipy.stats import lognorm
    l = np.logspace(-1, 3, 400)
    for d, c in ((3, ORANGE), (30, BLUE), (300, GREEN)):
        mu, s = np.sqrt(2) + np.log(d) / 2, np.sqrt(3)
        b.plot(l, lognorm.pdf(l, s, scale=np.exp(mu)) * l, color=c, lw=2, label=f"d = {d} inputs")
    b.set_xscale("log")
    b.set_xlabel(r"length scale $\ell$")
    b.set_ylabel("prior density (per log $\\ell$)")
    b.set_title("gp_dim: typical $\\ell$ grows like $\\sqrt{d}$")
    b.legend(fontsize=8)
    grid(b)
    c_ = axs[2]
    rng = np.random.default_rng(7)
    tau = abs(rng.standard_cauchy()) * 0.1
    inv = np.abs(rng.standard_cauchy(30)) * tau
    order = np.argsort(inv)[::-1]
    c_.bar(np.arange(30), inv[order], color=[BLUE if v > 0.1 * inv.max() else GREY for v in inv[order]], width=0.75)
    c_.set_xlabel("inputs, sorted")
    c_.set_ylabel(r"$1/\ell_i^2$ (relevance)")
    c_.set_title("SAAS: one prior draw, few inputs 'on'")
    grid(c_)
    fig.tight_layout()
    save(fig, "m_priors")


def ensemble():
    """A deep ensemble: five networks agree where there are data and disagree where there are none."""
    from sklearn.neural_network import MLPRegressor
    keep = (X1 < 0.35) | (X1 > 0.7)
    x, y = X1[keep], Y1[keep]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    preds = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s, c in zip(range(5), (BLUE, ORANGE, GREEN, PURPLE, GOLD)):
            m = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=4000, random_state=s, alpha=1e-3,
                             learning_rate_init=3e-3).fit(x[:, None] * 2 - 1, (y - 0.22) * 10)
            p = m.predict(XG[:, None] * 2 - 1) / 10 + 0.22
            preds.append(p)
            ax.plot(XG, p, color=c, lw=1.3, alpha=0.9, label=f"network {s + 1}")
    P = np.array(preds)
    ax.fill_between(XG, P.mean(0) - 2 * P.std(0), P.mean(0) + 2 * P.std(0), color=GREY, alpha=0.2,
                    label="ensemble spread ($\\pm 2$ std)")
    ax.axvspan(0.35, 0.7, color=GOLD, alpha=0.07)
    ax.text(0.525, 0.39, "no training data here", ha="center", fontsize=9, color=GOLD)
    ax.scatter(x, y, color=INK, s=18, zorder=4)
    ax.set_ylim(0.05, 0.42)
    ax.set_xlabel("knob")
    ax.set_ylabel(r"$\sigma_e$ (toy)")
    ax.set_title("five networks from five random starts")
    ax.legend(fontsize=7.5, ncol=3, loc="lower left")
    grid(ax)
    save(fig, "m_ensemble")


# ------------------------------------------------------------------------------------------------ real data
def load_night():
    import scorecard as sc
    files, rows, d = sc.load((0, 1, 3))
    return files, rows, d


def convolution(files, rows, d):
    """A 1D convolution filter slid along one real curve ('skip: mean shape'): one output per position."""
    c = files["curves"]
    names = [str(n) for n in c["curve_names"]]
    curve = np.nan_to_num(c["curves_theta"][rows][0, names.index("skip: mean shape")])
    curve = (curve - curve.mean()) / (curve.std() + 1e-12)
    filt = np.array([-1, -1, -1, 0, 1, 1, 1], float) / 3            # an edge detector, width 7 as in the first layer
    out = np.convolve(curve, filt[::-1], mode="same")
    for k, pos in enumerate((60, 86, 196), 1):
        fig, (a, b) = plt.subplots(2, 1, figsize=(8.6, 3.9), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
        a.plot(curve, color=INK2, lw=1.4)
        a.axvspan(pos - 3, pos + 3, color=GOLD, alpha=0.35)
        a.annotate("filter: 7 weights\n(-1 -1 -1 0 +1 +1 +1)/3", (pos, curve[pos]), xytext=(pos + 25, curve.max() * 0.8),
                   fontsize=8, color=GOLD, arrowprops=dict(arrowstyle="->", color=GOLD))
        a.set_ylabel("curve\n(standardized)")
        a.set_title("input: one of the 16 curves (here 'skip: mean shape', 256 points)")
        grid(a)
        b.plot(np.arange(pos + 1), out[:pos + 1], color=BLUE, lw=1.8)
        b.scatter([pos], [out[pos]], color=GOLD, s=40, zorder=4)
        b.set_xlim(0, 255)
        b.set_ylim(out.min() * 1.1, out.max() * 1.1)
        b.set_ylabel("feature map")
        b.set_xlabel("position along the curve")
        b.set_title("output: the weighted sum under the window, one per position (large where the curve rises)")
        grid(b)
        fig.tight_layout()
        save(fig, f"m_conv_{k}")


def masking(files, rows, d):
    """Self-supervised pre-training: hide blocks (or a whole curve) and ask the encoder to fill them in."""
    c = files["curves"]
    names = [str(n) for n in c["curve_names"]]
    show = ["noise: Allan deviation", "noise: CDS spectrum, log frequency", "skip: mean shape", "start: mean transient"]
    X = np.nan_to_num(c["curves_theta"][rows][5])
    fig, axs = plt.subplots(1, 4, figsize=(12.6, 2.6))
    for j, (ax, nm) in enumerate(zip(axs, show)):
        y = X[names.index(nm)]
        y = (y - y.mean()) / (y.std() + 1e-12)
        if j == 1:
            ax.plot(y, color=GREY, lw=1.2, alpha=0.35)
            ax.axvspan(0, 255, color=RED, alpha=0.08)
            ax.text(128, 0, "whole curve hidden\nrebuild it from the other 15", ha="center", va="center", fontsize=8.5, color=RED)
        else:
            ax.plot(y, color=INK2, lw=1.2)
            for s in (np.random.default_rng(j).choice(7, 2, replace=False) * 32 + 16):
                ax.axvspan(s, s + 32, color=RED, alpha=0.18)
        ax.set_title(nm.split(": ")[1], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("masked reconstruction: red = hidden from the encoder, the decoder must fill it in", fontsize=10, y=1.03)
    save(fig, "m_mask")


def hier_kernel():
    """The hierarchical kernel built term by term: shared + module + amplifier (+ noise)."""
    n_per, mods, amps = 6, [0, 0, 1, 1], [0, 1, 2, 3]
    x = np.tile(np.linspace(0, 1, n_per), 4)
    mod = np.repeat(mods, n_per)
    amp = np.repeat(amps, n_per)
    shared = matern52(x, x, 0.4, 1.0)
    module = (mod[:, None] == mod[None, :]) * (0.5 + 0.3 * matern52(x, x, 0.4))
    ampk = (amp[:, None] == amp[None, :]) * (0.4 + 0.2 * matern52(x, x, 0.4))
    terms = [("shared by every amplifier\n$s^2 M(x,x')$", shared), ("+ same module\n$c_m^2 + s_m^2 M$", module),
             ("+ same amplifier\n$c_a^2 + s_a^2 M$", ampk)]
    total = np.zeros_like(shared)
    for k, (lab, T) in enumerate(terms, 1):
        total = total + T
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.6))
        for ax, M, t in ((a, T, "term added: " + lab.split("\n")[0]), (b, total, "kernel so far")):
            ax.imshow(M, cmap="Blues", vmin=0, vmax=3.0 if ax is b else 1.2)
            for e in range(n_per, 4 * n_per, n_per):
                ax.axhline(e - 0.5, color=SURFACE, lw=1.2 if e != 2 * n_per else 2.5)
                ax.axvline(e - 0.5, color=SURFACE, lw=1.2 if e != 2 * n_per else 2.5)
            ax.set_xticks([2.5 + n_per * i for i in range(4)], ["amp 1", "amp 2", "amp 3", "amp 4"], fontsize=8)
            ax.set_yticks([2.5 + n_per * i for i in range(4)], ["amp 1", "amp 2", "amp 3", "amp 4"], fontsize=8)
            ax.set_title(t, fontsize=10)
        a.text(0.5, -0.2, lab.split("\n")[1], transform=a.transAxes, ha="center", fontsize=10)
        b.text(0.5, -0.2, "modules: amps 1-2 | amps 3-4", transform=b.transAxes, ha="center", fontsize=9, color=INK2)
        fig.tight_layout()
        save(fig, f"m_hier_{k}")


def physics_parts(files, rows, d):
    """The physics model's three pieces on the first night: what the trace measures (ADU), the gain, the correction."""
    ph = files["physics"]
    names = [str(n) for n in ph["feature_names"]]
    w = ph["X_trace"][rows][:, names.index("fit: white noise per skip (ADU)")].astype(float)
    N = ph["n_skips"][rows].astype(float)
    Y = ph["Y"][rows]
    amp = d.channel.to_numpy()
    trace_term = 0.5 * np.log(w ** 2 / N)                     # log sigma_ADU predicted by the trace
    image_adu = Y[:, 0] + Y[:, 2]                             # log(sigma_e g): the image's read noise in ADU
    r = Y[:, 0] - (trace_term - Y[:, 2])
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 3.5), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    a = axs[0]
    for ch in (0, 1, 3):
        m = amp == ch
        a.scatter(np.exp(trace_term[m]), np.exp(image_adu[m]), s=16, color=AMP_COLOR[ch], label=f"ch{ch}", alpha=0.85)
    lim = np.array([np.exp(trace_term).min() * 0.95, np.exp(trace_term).max() * 1.05])
    a.plot(lim, lim, color=GREY, ls="--", lw=1)
    a.text(lim[1], lim[1], " image = trace", fontsize=8, color=GREY, va="bottom", ha="right")
    a.set_xlabel(r"trace: $w/\sqrt{N}$ (ADU)")
    a.set_ylabel(r"image: $\sigma_e\, g$ (ADU)")
    a.set_title("1  measured on the trace (ADU)")
    a.legend(fontsize=8)
    grid(a, "both")
    for ax, val, title, ylab in ((axs[1], np.exp(Y[:, 2]), "2  the gain $g$: learned", "gain (ADU per e$^-$)"),
                                 (axs[2], r, "3  the correction $r$: learned", r"$r = \log\sigma_e - \log(w/\sqrt{N}) + \log g$")):
        for k, ch in enumerate((0, 1, 3)):
            m = amp == ch
            ax.scatter(k + RNG.uniform(-0.18, 0.18, m.sum()), val[m], s=12, color=AMP_COLOR[ch], alpha=0.8)
        ax.set_xticks([0, 1, 2], ["ch0", "ch1", "ch3"])
        ax.set_title(title)
        ax.set_ylabel(ylab, fontsize=9)
        grid(ax)
    fig.tight_layout()
    save(fig, "m_physics_parts")


def allan(files, rows, d):
    """One real Allan-deviation curve with the two physics terms: a^2/n + b^2 (Allan fit) and w^2/n (white)."""
    import ml.structured as st
    ph = files["physics"]
    ns = np.asarray(ph["allan_ns"], float)
    A = np.asarray(ph["allan"][rows], float)
    names = [str(n) for n in ph["feature_names"]]
    w = ph["X_trace"][rows][:, names.index("fit: white noise per skip (ADU)")].astype(float)
    i = int(np.nanargmin(np.abs(np.nanmedian(A, 1) - np.nanmedian(A))))
    a, b = st.allan_fit(A[i:i + 1], ns)
    nn = np.logspace(0, np.log10(500), 200)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.loglog(ns, A[i], "o", color=INK, ms=5, label=r"measured Allan deviation $\sigma_A(n)$")
    ax.loglog(nn, np.sqrt(a[0] ** 2 / nn + b[0] ** 2), color=BLUE, lw=2, label=f"Allan fit $\\sqrt{{a^2/n + b^2}}$ (a = {a[0]:.1f}, b = {b[0]:.2f})")
    ax.loglog(nn, w[i] / np.sqrt(nn), color=ORANGE, lw=2, ls="--", label=f"white term $w/\\sqrt{{n}}$ (w = {w[i]:.1f} ADU)")
    ax.axvline(500, color=GREY, lw=1, ls=":")
    ax.text(500, ax.get_ylim()[1] * 0.8, "N = 500\n(the image)", fontsize=8, color=GREY, ha="right")
    ax.set_xlabel("n = number of skips averaged")
    ax.set_ylabel("noise (ADU)")
    ax.set_title("how the trace predicts the image's read noise in ADU")
    ax.legend(fontsize=8, loc="lower left")
    grid(ax, "both")
    save(fig, "m_allan")


def state_vector(files, rows, d):
    """The packet state vector, built from one reference feature: per amplifier, minus its own average, averaged."""
    ph = files["features"]
    names = [str(n) for n in ph["feature_names"]]
    f = "skip: CDS rms (detrended)"
    R = np.log(ph["X_ref"][rows][:, names.index(f)].astype(float))
    amp, h, pk = d.channel.to_numpy(), d.hours.to_numpy(), d.packet_id.to_numpy()
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 3.3), sharex=True)
    cen = R.copy()
    for ch in (0, 1, 3):
        m = amp == ch
        o = np.argsort(h[m])
        axs[0].plot(h[m][o], R[m][o], "o-", ms=3, lw=1, color=AMP_COLOR[ch], label=f"ch{ch}")
        cen[m] = R[m] - R[m].mean()
        axs[1].plot(h[m][o], cen[m][o], "o-", ms=3, lw=1, color=AMP_COLOR[ch])
    st_ = {p: cen[pk == p].mean() for p in np.unique(pk)}
    hp = {p: h[pk == p].mean() for p in st_}
    o = sorted(st_, key=hp.get)
    axs[2].plot([hp[p] for p in o], [st_[p] for p in o], "o-", ms=3, lw=1.4, color=INK)
    axs[0].set_title("1  reference feature per amplifier")
    axs[1].set_title("2  minus each amplifier's own average")
    axs[2].set_title("3  averaged over the packet: the state")
    axs[0].set_ylabel(f"log '{f}'", fontsize=8.5)
    axs[0].legend(fontsize=8)
    for ax in axs:
        ax.set_xlabel("hours into the night")
        grid(ax)
    fig.tight_layout()
    save(fig, "m_state")


def drift(files, rows, d):
    """Drift correction: the nominal label interpolated in time; a packet's setting effect is its distance from it."""
    import change_test as ct
    Y = files["physics"]["Y"][rows]
    amp, h = d.channel.to_numpy(), d.hours.to_numpy()
    nom = d.packet_id.isin(ct.NOMINAL).to_numpy()
    m = amp == 0
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    hn, yn = h[m & nom], np.exp(Y[m & nom, 1])
    o = np.argsort(hn)
    hh = np.linspace(h[m].min(), h[m].max(), 300)
    ax.plot(hh, np.interp(hh, hn[o], yn[o]), color=INK, lw=1.4, label=r"nominal, interpolated in time: $\lambda_{\rm nom}(t)$")
    ax.scatter(hn, yn, color=INK, s=40, zorder=4, label="nominal packets")
    ax.scatter(h[m & ~nom], np.exp(Y[m & ~nom, 1]), color=ORANGE, s=16, zorder=3, label="other settings")
    j = np.flatnonzero(m & ~nom)[np.argmax(np.abs(Y[m & ~nom, 1] - np.interp(h[m & ~nom], hn[o], np.log(yn[o]))))]
    ln = np.exp(np.interp(h[j], hn[o], np.log(yn[o])))
    ax.annotate("", (h[j], np.exp(Y[j, 1])), (h[j], ln), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(h[j] + 0.1, np.sqrt(ln * np.exp(Y[j, 1])), r" $e = \log\lambda - \log\lambda_{\rm nom}(t)$", color=RED, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("hours into the night (ch0)")
    ax.set_ylabel(r"dark charge $\lambda$")
    ax.legend(fontsize=8, loc="upper right")
    grid(ax)
    save(fig, "m_drift")


def antisymmetry(files, rows, d):
    """The within-packet changes (nominal -> new setting) as given, and after adding every pair in both orders."""
    import scorecard as sc
    Y = files["physics"]["Y"][rows]
    P = sc.make_pairs(d, Y)
    y = 100 * P["Y"][P["type"] == 0, 0]
    bins = np.linspace(-60, 60, 31)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.1), sharey=True)
    axs[0].hist(y, bins, color=ORANGE, alpha=0.85)
    axs[0].set_title(f"as measured: {np.mean(y > 0) * 100:.0f} % go up")
    axs[1].hist(np.concatenate([y, -y]), bins, color=BLUE, alpha=0.85)
    axs[1].set_title("each pair also used backwards: 50 % up by construction")
    for ax in axs:
        ax.axvline(0, color=INK, lw=1)
        ax.set_xlabel(r"change of $\log\sigma_e$ (%)")
        grid(ax)
    axs[0].set_ylabel("pairs")
    fig.tight_layout()
    save(fig, "m_antisym")


# ------------------------------------------------------------------------------------------------ change models
def classifier():
    """The direction classifier on a toy: label = did it go up; weight = how far beyond the label noise."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(4)
    s = 0.12
    true = rng.normal(0, 0.3, 160)
    x = true + rng.normal(0, 0.08, 160)                     # a trace feature that sees the change
    meas = true + rng.normal(0, s, 160)                      # the label carries noise of the size s
    lab = (meas > 0).astype(int)
    w = 1 - np.exp(-meas ** 2 / (2 * s ** 2))
    clf = LogisticRegression(C=10).fit(x[:, None], lab, sample_weight=w)
    xx = np.linspace(-0.9, 0.9, 300)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.scatter(x, lab + rng.uniform(-0.04, 0.04, 160), s=6 + 60 * w, color=np.where(lab, ORANGE, BLUE), alpha=0.6,
               edgecolor="none")
    ax.plot(xx, clf.predict_proba(xx[:, None])[:, 1], color=INK, lw=2.2, label="P(B worse than A)")
    ax.axhline(0.5, color=GREY, lw=0.8, ls=":")
    ax.text(0.9, 1.1, "label 1: went up (worse)", ha="right", fontsize=8.5, color=ORANGE)
    ax.text(-0.9, -0.17, "label 0: went down (better)", ha="left", fontsize=8.5, color=BLUE)
    ax.text(0.02, 0.62, "small dots = changes of the size\nof the label noise: weight ~ 0", fontsize=8, color=INK2)
    ax.set_ylim(-0.25, 1.2)
    ax.set_xlabel("what the traces see (toy: a feature difference)")
    ax.set_ylabel("label / probability")
    ax.legend(fontsize=8, loc="center right")
    grid(ax)
    save(fig, "m_classifier")


def pareto():
    """Shortlist rules on toy candidates: the J(w) order for three weights, and the Pareto front."""
    import shortlist as sl
    rng = np.random.default_rng(11)
    ds, dl = rng.normal(0.05, 0.08, 18), rng.normal(0.0, 0.25, 18)
    front = sl.pareto_front(ds, dl)
    for k, w in enumerate((1.0, 0.5, 0.0), 1):
        fig, ax = plt.subplots(figsize=(5.6, 3.9))
        j = sl.objective(ds, dl, w)
        top = np.argsort(j)[:5]
        ax.scatter(100 * ds, 100 * dl, s=30, color=GREY, zorder=3)
        ax.scatter(100 * ds[front], 100 * dl[front], s=60, facecolor="none", edgecolor=PURPLE, lw=1.6, zorder=4,
                   label="Pareto front")
        ax.scatter(100 * ds[top], 100 * dl[top], s=30, color=GREEN, zorder=5, label="first 5 by $J(w)$")
        xs = np.linspace(100 * ds.min() - 5, 100 * ds.max() + 5, 10)
        for c in np.quantile(j, (0.1, 0.5, 0.9)):
            if w < 1:
                ax.plot(xs, 100 * (c - w * xs / 100) / (1 - w), color=GREEN, lw=0.8, ls="--", alpha=0.6)
            else:
                ax.axvline(100 * c, color=GREEN, lw=0.8, ls="--", alpha=0.6)
        ax.set_xlim(100 * ds.min() - 4, 100 * ds.max() + 4)
        ax.set_ylim(100 * dl.min() - 8, 100 * dl.max() + 8)
        ax.set_xlabel(r"predicted change of $\log\sigma_e$ (%)")
        ax.set_ylabel(r"predicted change of $\log\lambda$ (%)")
        ax.set_title(f"w = {w:g}: lines of equal $J$")
        ax.legend(fontsize=8, loc="upper right")
        grid(ax, "both")
        save(fig, f"m_pareto_{k}")


def bradley_terry():
    """Bradley-Terry on six toy settings: the soft-win table, and the strengths after each MM iteration."""
    import shortlist as sl
    names = list("ABCDEF")
    true = np.array([0.6, -0.5, 0.0, 0.1, 0.3, -0.2])           # made-up setting effects (lower = better)
    rng = np.random.default_rng(2)
    D = np.full((6, 6), np.nan)
    for i in range(6):
        for j in range(6):
            if i != j:
                D[i, j] = np.clip(0.5 / (1 + np.exp(-(true[j] - true[i]) / 0.15)) - 0.25 + rng.normal(0, 0.05), -0.5, 0.5)
    M = sl.antisymmetric(D)
    beats = np.nan_to_num(np.clip(0.5 + M, 0, 1))
    played = np.isfinite(M).astype(float)
    W = beats.sum(1) + 0.5
    p, hist = np.ones(6), [np.zeros(6)]
    for _ in range(40):
        p = W / ((played / (p[:, None] + p[None, :])).sum(1) + 1.0 / (p + 1.0))
        p /= np.exp(np.mean(np.log(p)))
        hist.append(-np.log(p))
    for k, it in enumerate((0, 1, 2, 40)):
        fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.5), gridspec_kw={"width_ratios": [1.05, 1]})
        show = np.where(np.isfinite(M), 0.5 + M, np.nan)
        a.imshow(show, cmap="RdBu", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                if i != j:
                    a.text(j, i, f"{show[i, j]:.2f}", ha="center", va="center", fontsize=8.5)
        a.set_xticks(range(6), [f"vs {n}" for n in names])
        a.set_yticks(range(6), [f"setting {n}" for n in names])
        a.set_title("soft wins: $0.5 + M_{ij}$ = how sure 'row beats column'", fontsize=10)
        for sp in a.spines.values():
            sp.set_visible(False)
        b.bar(range(6), hist[it], color=[GREEN if v == hist[it].min() else BLUE for v in hist[it]] if it else GREY)
        b.axhline(0, color=INK2, lw=0.8)
        b.set_xticks(range(6), names)
        b.set_ylim(-1.1 * np.abs(hist[-1]).max(), 1.1 * np.abs(hist[-1]).max())
        b.set_ylabel(r"score $-\log p_i$ (lower = better)")
        b.set_title("start: every strength $p_i = 1$" if it == 0 else f"after {it} MM iteration{'s' if it > 1 else ''}",
                    fontsize=10)
        grid(b)
        fig.tight_layout()
        save(fig, f"m_bt_{k}")


def difference_curves(files, rows, d):
    """A within-packet pair: the reference trace (A, nominal) and the own trace (B), and the difference curve B - A."""
    c = files["curves"]
    names = [str(n) for n in c["curve_names"]]
    show = ["skip: mean shape", "noise: noise vs window length"]
    timing = np.flatnonzero(~d.packet_id.str.contains("nominal", case=False).to_numpy())
    r = timing[len(timing) // 3]
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 4.2), sharex=True)
    for j, nm in enumerate(show):
        A = np.nan_to_num(c["curves_ref"][rows][r, names.index(nm)])
        B = np.nan_to_num(c["curves_theta"][rows][r, names.index(nm)])
        axs[0, j].plot(A, color=BLUE, lw=1.5, label="A: reference trace (nominal)")
        axs[0, j].plot(B, color=ORANGE, lw=1.5, ls="--", label="B: trace at the new setting")
        axs[0, j].set_title(nm, fontsize=10)
        axs[1, j].plot(B - A, color=PURPLE, lw=1.5)
        axs[1, j].axhline(0, color=GREY, lw=0.8)
        axs[1, j].set_xlabel("point along the curve (0-255)")
        grid(axs[0, j]), grid(axs[1, j])
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].set_ylabel("B $-$ A")
    axs[0, 0].set_ylabel("curve")
    fig.suptitle(f"{d.packet_id[r]}, ch{d.channel[r]}: the difference curve keeps only what the setting changed", fontsize=10)
    fig.tight_layout()
    save(fig, "m_diffcurve")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tree_partition()
    boosting()
    gp_steps()
    kernel_and_priors()
    ensemble()
    hier_kernel()
    classifier()
    pareto()
    bradley_terry()
    files, rows, d = load_night()
    convolution(files, rows, d)
    masking(files, rows, d)
    physics_parts(files, rows, d)
    allan(files, rows, d)
    state_vector(files, rows, d)
    drift(files, rows, d)
    antisymmetry(files, rows, d)
    difference_curves(files, rows, d)


if __name__ == "__main__":
    main()

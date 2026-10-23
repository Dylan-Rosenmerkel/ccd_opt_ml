"""
plot_guide_figures.py -- the teaching figures of docs/overleaf_ml_guide/ml_guide.tex (toy examples, no CCD data).

Every figure illustrates one idea with a small, fully known example:

    guide_fitting.png     under- and over-fitting; training against test error
    guide_trees.png       gradient boosting: the fit after 1, 20 and 300 trees
    guide_gp.png          a Gaussian process: prior samples, posterior, the length scale
    guide_ei.png          Bayesian optimization: the posterior and the expected improvement
    guide_allan.png       the Allan deviation of white + 1/f noise and the fit a^2/n + b^2
    guide_hier.png        a hierarchical GP: a new amplifier with 0 and 1 of its points
    guide_rows.png        dark current and CIC from a line through the image rows; their correlated errors
    guide_pareto.png      a Pareto front and the hypervolume
    guide_conv.png        a 1D convolution: a filter slid along a curve

    python -m ml.plot_guide_figures ../docs/overleaf_ml_guide/figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------- #
# a tiny Gaussian process (numpy), written out so the figures follow the text  #
# --------------------------------------------------------------------------- #


def matern52(a, b, ell):
    r = np.abs(a[:, None] - b[None, :]) / ell
    return (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r)


def gp_posterior(x, y, xs, ell=1.0, s2=1.0, noise=1e-3, mean=0.0):
    K = s2 * matern52(x, x, ell) + noise * np.eye(len(x))
    Ks = s2 * matern52(xs, x, ell)
    Kss = s2 * matern52(xs, xs, ell)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - mean))
    mu = mean + Ks @ alpha
    v = np.linalg.solve(L, Ks.T)
    cov = Kss - v.T @ v
    return mu, cov


# --------------------------------------------------------------------------- #
# figures                                                                      #
# --------------------------------------------------------------------------- #


def fitting(out):
    rng = np.random.default_rng(3)
    f = lambda x: np.sin(2 * x) + 0.3 * x
    x = np.sort(np.linspace(0.1, 2.9, 14) + rng.uniform(-0.08, 0.08, 14))
    y = f(x) + 0.25 * rng.normal(size=x.size)
    xt = rng.uniform(x.min(), x.max(), 300)                      # new points in the same range (no extrapolation)
    yt = f(xt) + 0.25 * rng.normal(size=xt.size)
    grid = np.linspace(0, 3, 400)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
    for ax, deg, title in zip(axes[:3], (1, 4, 12), ("too simple (degree 1)", "about right (degree 4)", "too flexible (degree 12)")):
        p = np.polynomial.Polynomial.fit(x, y, deg)
        ax.plot(grid, f(grid), color="0.6", lw=1.2, ls="--", label="true function")
        ax.plot(grid, p(grid), color="C0", lw=2, label="fitted model")
        ax.scatter(x, y, color="k", s=18, zorder=3, label="training points")
        ax.set_ylim(-1.8, 2.3)
        ax.set_title(title)
        ax.set_xlabel("input $x$")
    axes[0].set_ylabel("output $y$")
    axes[0].legend(fontsize=8, loc="lower right")
    degs = np.arange(0, 13)
    tr, te = [], []
    for d in degs:
        p = np.polynomial.Polynomial.fit(x, y, d)
        tr.append(np.sqrt(np.mean((p(x) - y) ** 2)))
        te.append(np.sqrt(np.mean((p(xt) - yt) ** 2)))
    ax = axes[3]
    ax.plot(degs, tr, "o-", label="training error")
    ax.plot(degs, te, "s-", label="test error (new points)")
    ax.set_yscale("log")
    ax.set_xlabel("model flexibility (polynomial degree)")
    ax.set_ylabel("RMS error")
    ax.set_title("error against flexibility")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "guide_fitting.png", dpi=160)
    plt.close(fig)


def trees(out):
    from sklearn.ensemble import GradientBoostingRegressor
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 6, 120))
    y = np.sin(x) + 0.5 * (x > 3.5) + 0.15 * rng.normal(size=x.size)
    grid = np.linspace(0, 6, 600)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=2, min_samples_leaf=10, random_state=0).fit(x[:, None], y)
    stages = list(model.staged_predict(grid[:, None]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
    for ax, m in zip(axes, (1, 20, 300)):
        ax.scatter(x, y, s=8, color="0.5")
        ax.plot(grid, stages[m - 1], color="C3", lw=2)
        ax.set_title(f"after {m} tree{'s' if m > 1 else ''}")
        ax.set_xlabel("input $x$")
    axes[0].set_ylabel("output $y$")
    fig.tight_layout()
    fig.savefig(out / "guide_trees.png", dpi=160)
    plt.close(fig)


def gp(out):
    rng = np.random.default_rng(1)
    xs = np.linspace(0, 10, 300)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), sharey=True)
    K = matern52(xs, xs, 1.5) + 1e-8 * np.eye(xs.size)
    samples = rng.multivariate_normal(np.zeros(xs.size), K, 4)
    ax = axes[0]
    ax.fill_between(xs, -2, 2, color="C0", alpha=0.15, label="$\\pm 2$ standard deviations")
    for s in samples:
        ax.plot(xs, s, lw=1.2)
    ax.set_title("prior: functions before seeing data")
    ax.legend(fontsize=8, loc="lower left")
    x = np.array([1.0, 2.5, 4.0, 6.5, 7.2])
    y = np.array([0.3, -0.8, 0.6, 1.2, 0.7])
    for ax, ell, title in ((axes[1], 1.5, "posterior after 5 points ($\\ell = 1.5$)"),
                           (axes[2], 0.4, "same points, short length scale ($\\ell = 0.4$)")):
        mu, cov = gp_posterior(x, y, xs, ell=ell, noise=1e-4)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color="C0", alpha=0.18)
        for s in rng.multivariate_normal(mu, cov + 1e-8 * np.eye(xs.size), 3):
            ax.plot(xs, s, lw=0.8, alpha=0.7)
        ax.plot(xs, mu, color="C0", lw=2, label="posterior mean $\\mu(x)$")
        ax.scatter(x, y, color="k", zorder=3, s=22, label="observations")
        ax.set_title(title)
    axes[1].legend(fontsize=8, loc="lower left")
    for ax in axes:
        ax.set_xlabel("input $x$")
        ax.set_ylim(-3, 3)
    axes[0].set_ylabel("$f(x)$")
    fig.tight_layout()
    fig.savefig(out / "guide_gp.png", dpi=160)
    plt.close(fig)


def ei(out):
    f = lambda x: 0.6 * np.sin(1.3 * x) + 0.08 * (x - 5) ** 2 - 0.2
    xs = np.linspace(0, 10, 500)
    x = np.array([0.8, 3.2, 6.0, 8.8])
    y = f(x)
    mu, cov = gp_posterior(x, y, xs, ell=1.6, s2=0.6, noise=1e-4, mean=y.mean())
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    best = y.min()
    z = (best - mu) / sd
    e = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(xs, f(xs), "--", color="0.6", label="true objective $J(\\theta)$ (unknown)")
    ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color="C0", alpha=0.18, label="$\\mu \\pm 2s$")
    ax.plot(xs, mu, color="C0", lw=2, label="surrogate mean $\\mu(\\theta)$")
    ax.scatter(x, y, color="k", zorder=3, label="packets taken")
    ax.axhline(best, color="C2", lw=1, ls=":", label="best so far $J_\\mathrm{best}$")
    ax.set_ylabel("objective $J$ (lower is better)")
    ax.legend(fontsize=8, ncol=2, loc="upper center")
    ax.set_ylim(-1.5, 2.6)
    ax = axes[1]
    ax.plot(xs, e, color="C3", lw=2)
    i = int(np.argmax(e))
    ax.axvline(xs[i], color="C3", ls=":")
    ax.annotate("next packet here", (xs[i], e[i]), xytext=(xs[i] + 0.6, e[i] * 0.8), fontsize=9, arrowprops=dict(arrowstyle="->"))
    ax.set_ylabel("expected\nimprovement")
    ax.set_xlabel("setting $\\theta$")
    fig.tight_layout()
    fig.savefig(out / "guide_ei.png", dpi=160)
    plt.close(fig)


def allan(out):
    rng = np.random.default_rng(2)
    n_pix, n_skip = 400, 2000
    white = rng.normal(0, 1.0, (n_pix, n_skip))
    freqs = np.fft.rfftfreq(n_skip)
    shape = np.zeros_like(freqs)
    shape[1:] = 1 / np.sqrt(freqs[1:])                          # amplitude ~ f^-1/2: power ~ 1/f
    spec = (rng.normal(size=(n_pix, freqs.size)) + 1j * rng.normal(size=(n_pix, freqs.size))) * shape
    flicker = np.fft.irfft(spec, n=n_skip, axis=1)
    flicker *= 0.3 / flicker.std()
    x = white + flicker
    ns = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    sig = []
    for n in ns:
        m = n_skip // n
        blocks = x[:, :m * n].reshape(n_pix, m, n).mean(2)
        sig.append(np.sqrt(0.5 * np.mean(np.diff(blocks, axis=1) ** 2)))
    sig = np.array(sig)
    A = np.column_stack([1 / ns, np.ones_like(ns, dtype=float)])
    w = 1 / sig ** 2
    coef = np.linalg.lstsq(A * w[:, None], sig ** 2 * w, rcond=None)[0]
    a, b = np.sqrt(coef)
    grid = np.geomspace(1, 300, 200)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.loglog(ns, sig, "ko", label="Allan deviation of the simulated skips")
    ax.loglog(grid, np.sqrt(a ** 2 / grid + b ** 2), "C3", lw=2, label=f"fit $\\sqrt{{a^2/n + b^2}}$: $a$={a:.2f}, $b$={b:.3f}")
    ax.loglog(grid, a / np.sqrt(grid), "C0--", label="white part $a/\\sqrt{n}$ (averages down)")
    ax.loglog(grid, b + 0 * grid, "C2--", label="floor $b$ (does not average down)")
    ax.set_xlabel("block length $n$ (skips averaged)")
    ax.set_ylabel("$\\sigma_A(n)$  [ADU]")
    ax.set_title("white noise ($\\sigma_1$ = 1) + 1/f noise")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "guide_allan.png", dpi=160)
    plt.close(fig)


def hier(out):
    rng = np.random.default_rng(4)
    shared = lambda x: np.sin(x) + 0.2 * x
    offsets = [0.9, -0.6, 0.2, 1.4]
    xs = np.linspace(0, 8, 300)
    s2, c2, sa2, noise, ell = 1.0, 0.8, 0.02, 0.01, 1.8

    def K(xa, aa, xb, ab):
        M = matern52(xa, xb, ell)
        same = (aa[:, None] == ab[None, :]).astype(float)
        return s2 * M + same * (c2 + sa2 * M)

    X, A, Y = [], [], []
    for a in range(3):
        x = np.sort(rng.uniform(0, 8, 9))
        X.append(x), A.append(np.full(x.size, a)), Y.append(shared(x) + offsets[a] + 0.1 * rng.normal(size=x.size))
    X0, A0, Y0 = np.concatenate(X), np.concatenate(A), np.concatenate(Y)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for ax, k_new in zip(axes, (0, 1)):
        x_new = np.array([5.0])[:k_new]
        Xt = np.concatenate([X0, x_new])
        At = np.concatenate([A0, np.full(k_new, 3)])
        Yt = np.concatenate([Y0, shared(x_new) + offsets[3]])
        Kt = K(Xt, At, Xt, At) + noise * np.eye(Xt.size)
        L = np.linalg.cholesky(Kt)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Yt - Yt.mean()))
        for a in range(4):
            aa = np.full(xs.size, a)
            Ks = K(xs, aa, Xt, At)
            mu = Yt.mean() + Ks @ alpha
            v = np.linalg.solve(L, Ks.T)
            sd = np.sqrt(np.clip(s2 + c2 + sa2 - (v * v).sum(0), 0, None))
            if a < 3:
                ax.plot(xs, mu, color=f"C{a}", lw=1.5)
                ax.scatter(X[a], Y[a], color=f"C{a}", s=14)
            else:
                ax.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color="C3", alpha=0.15)
                ax.plot(xs, mu, color="C3", lw=2.2, label="new amplifier: prediction $\\pm 2s$")
                ax.plot(xs, shared(xs) + offsets[3], "k:", lw=1.2, label="new amplifier: truth")
        if k_new:
            ax.scatter(x_new, shared(x_new) + offsets[3], color="C3", marker="*", s=160, zorder=4, label="its one packet")
        ax.set_title(f"new amplifier with {k_new} of its own point{'s' if k_new != 1 else ''}")
        ax.set_xlabel("setting $x$")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("$y$ (e.g. $\\log\\sigma_e$)")
    fig.tight_layout()
    fig.savefig(out / "guide_hier.png", dpi=160)
    plt.close(fig)


def rows(out):
    rng = np.random.default_rng(5)
    R, dc, cic, rel = 16, 0.012, 0.010, 0.12
    xr = (np.arange(R) + 0.5) / (R / 2)
    lam = cic + dc * xr
    A = np.column_stack([xr, np.ones(R)])
    P = np.linalg.solve(A.T @ A, A.T)
    cov = P @ np.diag((rel * lam) ** 2) @ P.T
    obs = lam * (1 + rel * rng.normal(size=R))
    fit = P @ obs
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    ax.errorbar(np.arange(R), obs, rel * lam, fmt="ko", ms=4, label="$\\lambda_r$ measured in each row")
    ax.plot(np.arange(R), lam, "C0", lw=2, label="truth: CIC + DC$\\cdot x_r$")
    ax.plot(np.arange(R), fit[1] + fit[0] * xr, "C3--", label="least-squares line")
    ax.axhline(cic, color="C2", ls=":", label="CIC (intercept)")
    ax.set_xlabel("row $r$ of the image")
    ax.set_ylabel("$\\lambda_r$  [e$^-$/pixel]")
    ax.legend(fontsize=8)
    ax = axes[1]
    draws = rng.multivariate_normal([dc, cic], cov, 2000)
    ax.scatter(draws[:, 0], draws[:, 1], s=3, alpha=0.3, color="C0", label="(DC, CIC) from repeated images")
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ax.plot([dc], [cic], "r*", ms=14, label="truth")
    t = np.linspace(dc - 0.012, dc + 0.012, 10)
    ax.plot(t, (dc + cic) - t, "k:", lw=1, label="DC + CIC = const")
    ax.set_xlim(draws[:, 0].min(), draws[:, 0].max())
    ax.set_ylim(draws[:, 1].min(), draws[:, 1].max())
    ax.set_xlabel("dark-current estimate DC")
    ax.set_ylabel("CIC estimate")
    ax.set_title(f"the two estimates are anti-correlated (correlation {corr:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "guide_rows.png", dpi=160)
    plt.close(fig)


def pareto(out):
    rng = np.random.default_rng(6)
    xs_ = rng.uniform(0.05, 1.0, 40)
    pts = np.column_stack([xs_, np.minimum(0.1 / xs_ + rng.uniform(0, 0.6, 40), 1.1)])
    order = np.argsort(pts[:, 0])
    front, best = [], np.inf
    for i in order:
        if pts[i, 1] < best:
            front.append(i)
            best = pts[i, 1]
    F = pts[front]
    ref = np.array([1.05, 1.15])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for k, (x, y) in enumerate(F):
        x_next = F[k + 1, 0] if k + 1 < len(F) else ref[0]
        ax.fill_between([x, x_next], y, ref[1], color="C2", alpha=0.18, lw=0)
    ax.scatter(pts[:, 0], pts[:, 1], color="0.55", s=18, label="settings tried")
    ax.step(np.r_[F[:, 0], ref[0]], np.r_[F[:, 1], F[-1, 1]], where="post", color="C3", lw=1.5)
    ax.scatter(F[:, 0], F[:, 1], color="C3", s=40, zorder=3, label="Pareto front (not beaten in both)")
    ax.plot(*ref, "ks", label="reference point")
    ax.text(0.62, 0.85, "hypervolume\n(shaded area)", color="C2", fontsize=10)
    ax.set_xlabel("$\\log\\sigma_e$  (lower is better)")
    ax.set_ylabel("$\\log\\lambda$  (lower is better)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out / "guide_pareto.png", dpi=160)
    plt.close(fig)


def conv(out):
    rng = np.random.default_rng(7)
    t = np.arange(120)
    curve = 0.15 * rng.normal(size=t.size)
    curve[30:34] += [0.5, 1.0, 1.0, 0.5]
    curve[80:84] += [0.5, 1.0, 1.0, 0.5]
    curve[55:60] -= 0.8
    w = np.array([-0.5, 0.5, 1.0, 1.0, 0.5, -0.5, -1.0]) / 2
    outp = np.convolve(curve, w[::-1], mode="same")              # cross-correlation, as in a CNN layer
    fig, axes = plt.subplots(1, 3, figsize=(14, 3), gridspec_kw={"width_ratios": [3, 1, 3]})
    axes[0].plot(t, curve, color="0.3")
    axes[0].set_title("input curve $x[t]$")
    axes[1].bar(np.arange(w.size) - 3, w, color="C1")
    axes[1].set_title("learned filter $w[k]$ (7 taps)")
    axes[2].plot(t, outp, color="C1")
    axes[2].set_title("output $(w \\star x)[t]$: large where the shape matches")
    for ax in (axes[0], axes[2]):
        ax.set_xlabel("position $t$")
    axes[1].set_xlabel("$k$")
    fig.tight_layout()
    fig.savefig(out / "guide_conv.png", dpi=160)
    plt.close(fig)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "../docs/overleaf_ml_guide/figures")
    out.mkdir(parents=True, exist_ok=True)
    for make in (fitting, trees, gp, ei, allan, hier, rows, pareto, conv):
        make(out)
        print("wrote", make.__name__)


if __name__ == "__main__":
    main()

"""Content of the model tutorials 00-05 (foundations, ridge, trees, networks, CNN, Gaussian processes).  Built by make_tutorials.py."""

NOTEBOOKS = {}


def notebook(name):
    cells = []
    NOTEBOOKS[name] = cells
    return (lambda s: cells.append(("md", s))), (lambda s: cells.append(("code", s)))


SETUP = r"""
from tutorial_utils import *
"""

# =====================================================================================================================
md, code = notebook("00_foundations")
md(r"""
# 00 · Foundations: data, losses, over-fitting and honest testing

*Level: beginner. Pairs with Part I of the learning guide (`docs/overleaf_ml_guide/ml_guide.tex`, Sections 1–4).*

Every later notebook builds on six ideas:

1. what one **training example** is in this project;
2. why we predict **logarithms**;
3. **learning = minimizing a loss**, done by **gradient descent**;
4. **over-fitting**: why the training error says little;
5. **honest testing**: split by packet, never by row;
6. the bit of **probability** needed for error bars.

We use the **toy CCD world** (`src/ml/synthetic.py`). It is a simulator that produces packets like the real ones: settings, trace features,
curves, and image labels with realistic noise. Its physics is invented, so it can be used to understand methods, not the real CCD.
""")
code(SETUP)
md(r"""
## 1. One example = one amplifier's traces in one packet

A synthetic scan of 60 packets. Each packet is read by 4 amplifiers, so there are 240 rows. Each row holds:
* `theta.*`: the settings (10 grouped parameters in the toy world);
* `trace.*` / `ref.*`: hand-made trace features at the settings and at the reference;
* `label.*`: the image's three labels.

The `truth.*` columns are the simulator's hidden truth. They are never inputs; we use them only to look behind the scenes.
""")
code(r"""
df, curves, Y, groups = toy_scan(60, seed=1)
print(df.shape, "rows x columns;", df.packet_id.nunique(), "packets")
show = ["packet_id", "board", "channel", "theta.VDD", "theta.VSUB", data.columns(df, "trace")[11], "label.sigma_e", "label.lambda_e_per_bin", "label.gain_adu_per_e"]
df[show].head(8)
""")
md(r"""
## 2. Why logarithms

σe, λ and the gain are positive and vary by **factors** (one amplifier is ~4× noisier). In logs:

* an error is a **relative** error: $\log\hat y - \log y = \log(\hat y / y) \approx (\hat y - y)/y$;
* products become sums: $\log\sigma_e = \log\sigma_\text{ADU} - \log g$.

`Y` therefore holds $(\log\sigma_e, \log\lambda, \log g)$.
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].hist(df["label.sigma_e"], 40); ax[0].set_xlabel("σe (e⁻)"); ax[0].set_title("linear: one noisy amplifier squeezes the rest")
ax[1].hist(Y[:, 0], 40); ax[1].set_xlabel("log σe"); ax[1].set_title("log: every amplifier resolved")
plt.tight_layout(); plt.show()
yhat, y = 1.05, 1.00
print(f"a 5 % over-prediction: log(1.05/1.00) = {np.log(yhat / y):.4f}  ≈ 0.05")
""")
md(r"""
## 3. Learning = minimizing a loss: gradient descent from scratch

The simplest model is a straight line, $f(x) = w_1 x + w_0$. Take as input $x$ the trace feature most correlated with $\log\sigma_e$ (standardized),
and as target $y = \log\sigma_e$. The **mean squared error**

$$L(w_0, w_1) = \frac1n\sum_i \bigl(y_i - w_1 x_i - w_0\bigr)^2$$

is minimized by **gradient descent**, $w \leftarrow w - \eta\,\nabla L$. Here
* $\eta$ is the learning rate;
* the gradient has two components: $\partial L/\partial w_1 = -\frac2n\sum_i x_i r_i$ and $\partial L/\partial w_0 = -\frac2n\sum_i r_i$, with residuals $r_i = y_i - f(x_i)$.
""")
code(r"""
F = df[data.columns(df, "trace")].to_numpy(float)
best = int(np.argmax([abs(np.corrcoef(F[:, k], Y[:, 0])[0, 1]) for k in range(F.shape[1])]))
x = (F[:, best] - F[:, best].mean()) / F[:, best].std()
y = Y[:, 0]
print("input:", data.columns(df, "trace")[best])

w0, w1, eta, losses = 0.0, 0.0, 0.1, []
for step in range(100):
    r = y - (w1 * x + w0)                       # residuals
    losses.append(np.mean(r ** 2))
    g1, g0 = -2 * np.mean(x * r), -2 * np.mean(r)  # the gradient
    w1, w0 = w1 - eta * g1, w0 - eta * g0       # one step downhill
A = np.column_stack([x, np.ones_like(x)])
w_exact = np.linalg.lstsq(A, y, rcond=None)[0]   # the closed-form least-squares answer
print(f"gradient descent: w1 = {w1:.4f}, w0 = {w0:.4f}   |   exact: w1 = {w_exact[0]:.4f}, w0 = {w_exact[1]:.4f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(losses); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss L"); ax[0].set_yscale("log")
ax[1].scatter(x, y, s=6); xs = np.linspace(x.min(), x.max(), 2); ax[1].plot(xs, w1 * xs + w0, "C3")
ax[1].set_xlabel("standardized feature"); ax[1].set_ylabel("log σe"); plt.tight_layout(); plt.show()
""")
md(r"""
Gradient descent reaches the same answer as the exact formula. Its advantage is that it works for **any** model whose loss can be
differentiated: neural networks are trained exactly like this, with millions of steps and parameters.

Try: set `eta = 1.1` and watch the loss blow up (too large a step overshoots).
""")
md(r"""
## 4. Over-fitting: the training error lies

Polynomials of growing degree are fitted to 14 noisy points of a known curve. The **test error** is measured on 300 fresh points.
""")
code(r"""
rng = np.random.default_rng(3)
f = lambda x: np.sin(2 * x) + 0.3 * x
xt = np.sort(np.linspace(0.1, 2.9, 14) + rng.uniform(-0.08, 0.08, 14)); yt = f(xt) + 0.25 * rng.normal(size=14)
xn = rng.uniform(xt.min(), xt.max(), 300); yn = f(xn) + 0.25 * rng.normal(size=300)
train_err, test_err = [], []
for deg in range(13):
    p = np.polynomial.Polynomial.fit(xt, yt, deg)
    train_err.append(np.sqrt(np.mean((p(xt) - yt) ** 2))); test_err.append(np.sqrt(np.mean((p(xn) - yn) ** 2)))
plt.figure(figsize=(5, 3)); plt.plot(train_err, "o-", label="training error"); plt.plot(test_err, "s-", label="test error")
plt.yscale("log"); plt.xlabel("polynomial degree (flexibility)"); plt.ylabel("RMS error"); plt.legend(); plt.show()
print("best degree on new data:", int(np.argmin(test_err)), "| training error keeps falling:", np.round(train_err[-3:], 3))
""")
md(r"""
## 5. Honest testing: split by packet, never by row

The 4 rows of a packet share the CCD's state at that moment. If some went to training and others to testing, a model could
recognize the packet from its reference features and look better than it is (**leakage**).

Below, the same model (boosted trees on the trace and reference features, predicting log λ) is scored twice:
* with folds that put each **row** anywhere;
* with folds that keep each **packet** together.
""")
code(r"""
X, cols = data.make_inputs(df, ("theta", "trace", "ref"), drop_constant=False)
rows_as_groups = np.arange(len(df)).astype(str)            # every row its own group = a random split by row
m_row, s_row = grouped_cv(lambda: models.GBT(max_iter=150), X, Y, rows_as_groups)
m_pkt, s_pkt = grouped_cv(lambda: models.GBT(max_iter=150), X, Y, groups)
e_row = report("split by ROW (leaky)", Y, m_row, s_row)
e_pkt = report("split by PACKET (honest)", Y, m_pkt, s_pkt)
d = 100 * (e_pkt["lambda"]["error"] - e_row["lambda"]["error"])
print(f"\nλ: the row split looks {abs(d):.1f} percentage points {'better' if d > 0 else 'worse'} than the honest packet split.")
""")
md(r"""
In every notebook, scores come from **packet-grouped cross-validation**: the packets are split into 5 folds, and each fold is predicted by a
model trained on the other four (`tutorial_utils.grouped_cv`).

**Standardization.** Inputs come in volts, clock cycles and ADU. Most models first rescale each column to mean 0 and standard
deviation 1, using the **training rows only**. Every model in `src/ml/models.py` does this inside `fit`.
""")
md(r"""
## 6. The probability you need

A **Gaussian** with mean μ and standard deviation σ puts 68 % of its values within μ ± σ and 95 % within μ ± 1.96σ. Our models predict a
mean **and** a standard deviation. A model is **calibrated** if its ±σ bars contain the truth 68 % of the time; this fraction is its
**coverage**.

**Why squared error?** If the noise is Gaussian with a fixed σ, the negative log-likelihood is
$\frac{1}{2\sigma^2}\sum_i (y_i-\mu_i)^2 + \text{const}$. Minimizing it *is* least squares.
""")
code(r"""
rng = np.random.default_rng(0)
z = rng.normal(0, 1, 100_000)
print(f"fraction within ±1σ: {np.mean(abs(z) <= 1):.3f}   within ±1.96σ: {np.mean(abs(z) <= 1.96):.3f}")
# NLL and MSE pick the same line: scan the slope of the line in section 3
slopes = np.linspace(w_exact[0] - 0.2, w_exact[0] + 0.2, 201)
mse = [np.mean((y - s * x - w_exact[1]) ** 2) for s in slopes]
sig = 0.3
nll = [np.sum(0.5 * np.log(2 * np.pi * sig ** 2) + (y - s * x - w_exact[1]) ** 2 / (2 * sig ** 2)) for s in slopes]
print(f"slope minimizing MSE: {slopes[np.argmin(mse)]:.4f}   slope minimizing Gaussian NLL: {slopes[np.argmin(nll)]:.4f}")
""")
md(r"""
## Summary

| Idea | Where it is used |
|---|---|
| one example = one amplifier's traces in a packet | every model |
| log targets: errors are relative | every report in these notebooks (errors in %) |
| loss + gradient descent | networks (03, 04, 12–14), GP hyperparameters (05) |
| over-fitting | choosing model size; regularization (01) |
| packet-grouped CV | every score |
| mean + standard deviation, coverage | GPs (05), ensembles (03), calibration (11) |

Next: **01 · Linear regression and Bayesian ridge**, the simplest model.
""")

# =====================================================================================================================
md, code = notebook("01_linear_bayesian_ridge")
md(r"""
# 01 · Linear regression, ridge, and Bayesian ridge

*Level: beginner. Guide Section 5. Code: `models.BayesRidge`.*

**Idea.** Each output is a weighted sum of the inputs, $f(x) = \mathbf w^\top x + w_0$.
* **Ridge** keeps the weights small.
* **Bayesian ridge** also says how sure it is.

It is the simplest model, and the yardstick every other model must beat.
""")
code(SETUP)
md(r"""
## 1. Ordinary least squares, by hand

Stack the inputs as rows of a matrix $X$ (with a column of ones for $w_0$) and the targets as $\mathbf y$. The squared error is
minimized by

$$\hat{\mathbf w} = (X^\top X)^{-1} X^\top \mathbf y .$$
""")
code(r"""
rng = np.random.default_rng(0)
x = np.concatenate([rng.uniform(0, 1.5, 25), rng.uniform(2.5, 3, 3)])
y = 0.8 * x + 0.3 + 0.3 * rng.normal(size=x.size)
X = np.column_stack([x, np.ones_like(x)])
w = np.linalg.solve(X.T @ X, X.T @ y)
print(f"slope {w[0]:.3f} (truth 0.8), intercept {w[1]:.3f} (truth 0.3)")
""")
md(r"""
## 2. Ridge: when inputs are near-copies

Trace features are often nearly the same quantity measured twice. Then $X^\top X$ is almost impossible to invert, and the weights blow up
in opposite directions. Ridge adds $\alpha\lVert\mathbf w\rVert^2$ to the loss:

$$\hat{\mathbf w}_\text{ridge} = (X^\top X + \alpha I)^{-1} X^\top \mathbf y .$$
""")
code(r"""
n, p = 40, 6
base = rng.normal(size=n)
Xc = np.column_stack([base + 0.01 * rng.normal(size=n) for _ in range(p)])   # 6 near-copies of one input
yc = base + 0.3 * rng.normal(size=n)
alphas = np.logspace(-6, 2, 50)
W = np.array([np.linalg.solve(Xc.T @ Xc + a * np.eye(p), Xc.T @ yc) for a in alphas])
plt.figure(figsize=(6, 3)); plt.semilogx(alphas, W); plt.axhline(1 / p, color="k", ls=":", label="sensible: 1/6 each")
plt.xlabel("ridge strength α"); plt.ylabel("weights"); plt.legend(); plt.title("tiny α: wild weights; larger α: shared, steady"); plt.show()
print("weights at α=1e-6:", W[0].round(2), "\nweights at α=1:   ", W[np.argmin(abs(alphas - 1))].round(3))
""")
md(r"""
## 3. Bayesian ridge: the same model with an error bar

The model is written with probabilities:
* **prior** $\mathbf w \sim \mathcal N(0, \alpha^{-1}I)$ (weights are probably small);
* **noise** $y_i \sim \mathcal N(\mathbf w^\top x_i, \beta^{-1})$.

Bayes' theorem gives the posterior of the weights, $\mathcal N(\mathbf m, S)$, with

$$S = (\alpha I + \beta X^\top X)^{-1},\qquad \mathbf m = \beta S X^\top \mathbf y,$$

and the prediction at $x_*$ has mean $\mathbf m^\top x_*$ and variance $\frac1\beta + x_*^\top S x_*$:
* the first term is the noise (**aleatoric**);
* the second is the uncertainty of the weights (**epistemic**), which grows far from the data.
""")
code(r"""
alpha, beta = 1.0, 1 / 0.3 ** 2
S = np.linalg.inv(alpha * np.eye(2) + beta * X.T @ X)
m = beta * S @ X.T @ y
xs = np.linspace(-1, 5, 200); Xs = np.column_stack([xs, np.ones_like(xs)])
mean = Xs @ m
sd_total = np.sqrt(1 / beta + np.einsum("ij,jk,ik->i", Xs, S, Xs))
sd_weights = np.sqrt(np.einsum("ij,jk,ik->i", Xs, S, Xs))
plt.figure(figsize=(7, 3.3))
plt.fill_between(xs, mean - 2 * sd_total, mean + 2 * sd_total, alpha=0.15, label="±2 sd (noise + weights)")
plt.fill_between(xs, mean - 2 * sd_weights, mean + 2 * sd_weights, alpha=0.3, label="±2 sd (weights only)")
plt.plot(xs, mean, "C0"); plt.scatter(x, y, color="k", s=12); plt.legend(fontsize=8); plt.xlabel("x"); plt.show()
print("the weight uncertainty is smallest where the data are, and grows outside 0..3")
""")
md(r"""
Our `models.BayesRidge` uses scikit-learn's `BayesianRidge`, which also *learns* α and β by maximizing the evidence (the marginal likelihood).
**Careful with names:** scikit-learn calls the noise precision `alpha_` and the weight precision `lambda_`, the reverse of the guide's α, β.
""")
code(r"""
from sklearn.linear_model import BayesianRidge
br = BayesianRidge().fit(x[:, None], y)
print(f"learned noise sd = {1 / np.sqrt(br.alpha_):.3f} (truth 0.3);  weight prior sd = {1 / np.sqrt(br.lambda_):.2f}")
print("slope, intercept:", br.coef_.round(3), round(br.intercept_, 3))
""")
md(r"""
## 4. Its limit: no curvature

Read noise has a minimum in some settings. A weighted sum of the inputs has no minimum inside the range, but it can if we give it
**nonlinear features** (e.g. $x^2$). Below is the toy world's true $\log\sigma_e$ of one amplifier as VDD is varied, with everything else at nominal.
""")
code(r"""
amps = synthetic.make_amplifiers(np.random.default_rng(1234))
i = synthetic.NAMES.index("VDD")
vdd = np.linspace(synthetic.LOW[i], synthetic.HIGH[i], 40)
th = np.tile(synthetic.NOMINAL, (40, 1)); th[:, i] = vdd
true = np.log([synthetic.physics(t, amps[0], 0.0)["sigma_e"] for t in th])
lin = models.BayesRidge().fit(vdd[:, None], true[:, None])
quad = models.BayesRidge().fit(np.column_stack([vdd, vdd ** 2]), true[:, None])
plt.figure(figsize=(6, 3)); plt.plot(vdd, true, "k", lw=3, alpha=0.4, label="truth")
plt.plot(vdd, lin.predict(vdd[:, None])[0], "C3--", label="ridge on VDD")
plt.plot(vdd, quad.predict(np.column_stack([vdd, vdd ** 2]))[0], "C0:", lw=2, label="ridge on VDD and VDD²")
plt.xlabel("VDD (V)"); plt.ylabel("log σe"); plt.legend(); plt.show()
""")
md(r"""
## 5. On the toy CCD: settings + each amplifier's own traces → the three labels

There are no amplifier labels: each row's own trace and reference features carry its identity.
""")
code(r"""
df, curves, Y, groups = toy_scan(120, seed=1)
X, cols = data.make_inputs(df, ("theta", "trace", "ref"), drop_constant=False)
mean, std = grouped_cv(lambda: models.BayesRidge(), X, Y, groups)
_ = report("Bayesian ridge (settings + own traces)", Y, mean, std)
""")
md(r"""
## When to use it

* **Good:** very little data (from ~20 examples); a first baseline; readable weights; never wild.
* **Bad:** optima and interactions, unless you build the nonlinear features yourself.

**Try:** add `"diff"` to the input groups; train on 30 packets instead of 120.

Next: **02 · Decision trees and gradient boosting.**
""")

# =====================================================================================================================
md, code = notebook("02_trees_boosting")
md(r"""
# 02 · Decision trees and gradient boosting

*Level: beginner. Guide Section 6. Code: `models.GBT`.*

**Idea.** A tree asks yes/no questions ("is VDD below −19.4 V?") and returns a number per answer. **Boosting** adds hundreds of small
trees, each correcting what the previous ones got wrong.
""")
code(SETUP)
code(r"""
rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 6, 150))
y = np.sin(x) + 0.5 * (x > 3.5) + 0.15 * rng.normal(size=x.size)
""")
md(r"""
## 1. One split, by hand

A split at threshold $t$ predicts the mean of $y$ on each side. Its loss is the summed squared error of both sides:

$$\text{SSE}(t) = \sum_{x_i \le t}(y_i - \bar y_\text{left})^2 + \sum_{x_i > t}(y_i - \bar y_\text{right})^2 .$$

The tree tries every threshold and keeps the best one.
""")
code(r"""
def best_split(x, r):
    order = np.argsort(x); xs, rs = x[order], r[order]
    best = (np.inf, None, None, None)
    for k in range(1, len(xs)):
        if xs[k] == xs[k - 1]:
            continue
        left, right = rs[:k], rs[k:]
        sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
        if sse < best[0]:
            best = (sse, (xs[k] + xs[k - 1]) / 2, left.mean(), right.mean())
    return best

sse_curve = [(t, ((y[x <= t] - y[x <= t].mean()) ** 2).sum() + ((y[x > t] - y[x > t].mean()) ** 2).sum()) for t in np.linspace(0.2, 5.8, 200)]
sse, t, lo, hi = best_split(x, y)
fig, ax = plt.subplots(1, 2, figsize=(11, 3))
ax[0].plot(*zip(*sse_curve)); ax[0].axvline(t, color="C3"); ax[0].set_xlabel("threshold t"); ax[0].set_ylabel("SSE(t)")
ax[1].scatter(x, y, s=6, color="0.5"); ax[1].step([0, t, 6], [lo, lo, hi], where="post", color="C3"); ax[1].set_title(f"best split at x = {t:.2f}")
plt.tight_layout(); plt.show()
""")
md(r"""
## 2. Boosting, by hand

Start from the mean, $F_0 = \bar y$. Then repeat:
* fit a stump (a one-split tree) $t_m$ to the current **residuals** $r_i = y_i - F_{m-1}(x_i)$;
* add a fraction of it: $F_m = F_{m-1} + \nu\, t_m$.

$\nu$ (the learning rate, 0.05 in our code) makes each tree correct only part of the error, which smooths the result.
""")
code(r"""
nu, F, snaps, mse = 0.05, np.full_like(y, y.mean()), {}, []
grid = np.linspace(0, 6, 600); Fg = np.full_like(grid, y.mean())
for m_ in range(1, 301):
    _, t, lo, hi = best_split(x, y - F)
    F += nu * np.where(x <= t, lo, hi); Fg += nu * np.where(grid <= t, lo, hi)
    mse.append(np.mean((y - F) ** 2))
    if m_ in (1, 20, 100, 300):
        snaps[m_] = Fg.copy()
fig, ax = plt.subplots(1, 5, figsize=(17, 3))
for a, (m_, g) in zip(ax, snaps.items()):
    a.scatter(x, y, s=5, color="0.6"); a.plot(grid, g, "C3"); a.set_title(f"{m_} stumps")
ax[4].plot(mse); ax[4].set_yscale("log"); ax[4].set_xlabel("number of trees"); ax[4].set_title("training MSE")
plt.tight_layout(); plt.show()
""")
md(r"""
## 3. Error bars from quantiles: the pinball loss

The $\tau$-quantile minimizes the **pinball loss**
$\rho_\tau(u) = \tau u$ for $u \ge 0$ and $(\tau-1)u$ for $u < 0$, where $u = y - q$.

For a Gaussian, the 16 % and 84 % quantiles are μ ∓ σ, so $\hat s = (\hat q_{0.84} - \hat q_{0.16})/2$. Our `GBT` fits these two extra
quantile models per target.
""")
code(r"""
z = np.random.default_rng(1).normal(2.0, 0.5, 5000)
qs = np.linspace(1, 3, 401)
pinball = lambda q, tau: np.mean(np.where(z - q >= 0, tau * (z - q), (tau - 1) * (z - q)))
for tau in (0.16, 0.5, 0.84):
    q_best = qs[np.argmin([pinball(q, tau) for q in qs])]
    print(f"tau = {tau}: pinball minimum at {q_best:.3f};  the data's {tau:.0%} quantile is {np.quantile(z, tau):.3f}")
""")
md(r"""
## 4. On the toy CCD, and its weak point: error bars

The quantile error bars are often too narrow. Below, training is on one scan and testing on a new scan, comparing the raw error bars
with **conformal calibration** (notebook 11), which rescales them using held-out packets.
""")
code(r"""
from ml.calibrate import Conformal
dfa, ca, Ya, ga = toy_scan(120, seed=1); dfb, cb, Yb, gb = toy_scan(60, seed=2)
Xa, _ = data.make_inputs(dfa, ("theta", "trace", "ref"), drop_constant=False)
Xb, _ = data.make_inputs(dfb, ("theta", "trace", "ref"), drop_constant=False)
raw = models.GBT().fit(Xa, Ya)
m, s = raw.predict(Xb)
_ = report("boosted trees, raw error bars", Yb, m, s)
cal = Conformal(lambda: models.GBT()).fit(Xa, Ya, ga)
m2, s2 = cal.predict(Xb)
_ = report("boosted trees, conformal error bars", Yb, m2, s2)
""")
md(r"""
## When to use it

* **Good:** many inputs of different kinds and scales; thresholds and interactions; robust; from ~100 examples.
* **Bad:** error bars (calibrate them); the fit is piecewise constant and flat outside the data, so it is a poor surrogate for optimization.

Next: **03 · Neural networks and deep ensembles.**
""")

# =====================================================================================================================
md, code = notebook("03_neural_networks")
md(r"""
# 03 · Neural networks, mean–variance outputs, and deep ensembles

*Level: intermediate. Guide Section 7. Code: `models.MLPEnsemble`.*

**Idea.** A network stacks layers $\mathbf h_\text{out} = \mathrm{act}(W\mathbf h_\text{in} + \mathbf b)$, where $W$ and $\mathbf b$ are learned
and `act` is a fixed bend. It is trained by gradient descent (notebook 00). Our networks also predict their own error bar, and an
ensemble of five tells where they disagree.
""")
code(SETUP + r"""
import torch
_ = torch.manual_seed(0)
""")
md(r"""
## 1. Activation functions

Without a nonlinear activation, stacked layers collapse into one linear map. We use **SiLU**, $u/(1+e^{-u})$, a smooth version of ReLU, $\max(0, u)$.
""")
code(r"""
u = np.linspace(-5, 5, 200)
plt.figure(figsize=(5, 3)); plt.plot(u, u / (1 + np.exp(-u)), label="SiLU"); plt.plot(u, np.maximum(0, u), "--", label="ReLU")
plt.legend(); plt.xlabel("u"); plt.show()
""")
md(r"""
## 2. A small network, trained step by step

The data have noise that **grows** with $x$. We first train a plain network (one output, squared error) with a hand-written loop:
forward pass → loss → `backward()` (backpropagation computes every gradient) → an Adam step.
""")
code(r"""
rng = np.random.default_rng(0)
x = rng.uniform(0, 3, 400); noise_sd = 0.05 + 0.15 * x
y = np.sin(2 * x) + noise_sd * rng.normal(size=x.size)
xt, yt = torch.tensor(x[:, None], dtype=torch.float32), torch.tensor(y[:, None], dtype=torch.float32)

net = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
for step in range(1500):
    loss = ((net(xt) - yt) ** 2).mean()   # forward pass and loss
    opt.zero_grad(); loss.backward()       # backpropagation: d loss / d every weight
    opt.step()                             # one Adam step
    if step % 500 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")
print("number of weights:", sum(p.numel() for p in net.parameters()))
""")
md(r"""
## 3. Predicting an error bar: the Gaussian negative log-likelihood

Two outputs, $\mu(x)$ and $\log\sigma^2(x)$, trained with

$$\text{loss} = \tfrac12\left[\log\sigma^2(x_i) + \frac{(y_i - \mu(x_i))^2}{\sigma^2(x_i)}\right].$$

The first term stops it claiming huge error bars everywhere; the second stops it claiming small ones where it is wrong.
""")
code(r"""
class MeanVar(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.body = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU())
        s.mu, s.logvar = torch.nn.Linear(64, 1), torch.nn.Linear(64, 1)
    def forward(s, x):
        h = s.body(x); return s.mu(h), s.logvar(h)

def train_meanvar(seed):
    torch.manual_seed(seed); m = MeanVar(); o = torch.optim.Adam(m.parameters(), lr=3e-3)
    for step in range(2000):
        mu, lv = m(xt); loss = 0.5 * (lv + (yt - mu) ** 2 / lv.exp()).mean()
        o.zero_grad(); loss.backward(); o.step()
    return m

mv = train_meanvar(0)
xs = torch.linspace(0, 3, 200)[:, None]
with torch.no_grad():
    mu, lv = mv(xs)
sd = lv.exp().sqrt().numpy()[:, 0]
plt.figure(figsize=(7, 3)); plt.scatter(x, y, s=4, color="0.6")
plt.plot(xs[:, 0], mu[:, 0], "C0"); plt.fill_between(xs[:, 0], mu[:, 0] - 2 * sd, mu[:, 0] + 2 * sd, alpha=0.25)
plt.title("learned ±2σ(x): the network found that the noise grows"); plt.show()
print("learned σ at x = 0.5, 2.5:", sd[33].round(3), sd[166].round(3), "| truth:", 0.05 + 0.15 * 0.5, 0.05 + 0.15 * 2.5)
""")
md(r"""
## 4. A deep ensemble: knowing what you don't know

One network extrapolates confidently. **Five** networks from different random starts agree where data pin the function down and disagree
elsewhere. The ensemble's variance is

$$s^2 = \underbrace{\overline{\sigma_k^2}}_\text{aleatoric} + \underbrace{\mathrm{Var}_k(\mu_k)}_\text{epistemic}.$$
""")
code(r"""
nets = [train_meanvar(k) for k in range(5)]
xs = torch.linspace(-1.5, 4.5, 300)[:, None]
with torch.no_grad():
    outs = [n(xs) for n in nets]
mus = np.array([o[0][:, 0].numpy() for o in outs]); vars_ = np.array([o[1][:, 0].exp().numpy() for o in outs])
mean, epi, ale = mus.mean(0), mus.var(0), vars_.mean(0)
plt.figure(figsize=(8, 3)); plt.scatter(x, y, s=4, color="0.6")
for mk in mus: plt.plot(xs[:, 0], mk, lw=0.8)
plt.fill_between(xs[:, 0], mean - 2 * np.sqrt(ale + epi), mean + 2 * np.sqrt(ale + epi), alpha=0.15)
plt.axvspan(-1.5, 0, color="k", alpha=0.05); plt.axvspan(3, 4.5, color="k", alpha=0.05)
plt.ylim(-4, 4); plt.title("5 networks: they agree on 0..3 and split outside (grey)"); plt.show()
""")
md(r"""
## 5. On the toy CCD

Our `models.MLPEnsemble` wraps this:
* 5 members, 2 hidden layers of 64 units;
* AdamW (Adam plus weight decay);
* early stopping on 15 % of the training rows.
""")
code(r"""
df, curves, Y, groups = toy_scan(120, seed=1)
X, cols = data.make_inputs(df, ("theta", "trace", "ref"), drop_constant=False)
mean, std = grouped_cv(lambda: models.MLPEnsemble(), X, Y, groups)
_ = report("NN ensemble (settings + own traces)", Y, mean, std)
""")
md(r"""
## When to use it

* **Good:** smooth nonlinear relations with many inputs, and a few hundred or more examples.
* **Bad:** tens of examples; error bars far from the data are less trustworthy than a GP's.

Next: **04 · Convolutional networks on the trace curves.**
""")

# =====================================================================================================================
md, code = notebook("04_cnn_curves")
md(r"""
# 04 · Convolutional networks on the trace curves

*Level: intermediate. Guide Section 8. Code: `models.CurveCNN`, `data.curve_tensor`, `data.log_frequency`.*

**Idea.** The hand-made features are numbers *we* chose. A convolutional network gets the **curves** themselves (Allan deviation,
noise spectrum, pulse shapes) and learns which shapes matter, so it can find effects missing from our list.
""")
code(SETUP + r"""
import torch
from ml.benchmark import ALL_CURVES, inputs_for
""")
md(r"""
## 1. The curves

Each example comes with 7 curves of the traces at θ and 7 of the reference traces.
""")
code(r"""
df, curves, Y, groups = toy_scan(60, seed=1)
fig, ax = plt.subplots(1, 3, figsize=(14, 3))
for r in range(4):
    ax[0].loglog(synthetic.ALLAN_N, curves["noise: Allan deviation"][r]); ax[1].semilogy(curves["noise: CDS spectrum"][r]); ax[2].plot(curves["skip: mean shape"][r])
ax[0].set_title("Allan deviation (one line per amplifier)"); ax[1].set_title("CDS noise spectrum (linear frequency)"); ax[2].set_title("mean skip waveform")
plt.tight_layout(); plt.show()
""")
md(r"""
## 2. Convolution by hand

A filter $w[k]$ slides along a curve $x[t]$:

$$(w \star x)[t] = \sum_k w[k]\, x[t+k] + b .$$

The output is large where the curve locally looks like the filter. In a CNN, the filter weights are learned. The same filter is used at every
position, so a feature is found wherever it appears.
""")
code(r"""
xcurve = np.zeros(60); xcurve[20:24] = [0.5, 1, 1, 0.5]; xcurve[40:44] = [0.5, 1, 1, 0.5]
w = np.array([0.25, 0.5, 0.5, 0.25])
by_hand = np.array([np.sum(w * xcurve[t:t + 4]) for t in range(len(xcurve) - 3)])
by_torch = torch.nn.functional.conv1d(torch.tensor(xcurve)[None, None], torch.tensor(w)[None, None])[0, 0].numpy()
print("hand-made and PyTorch results equal:", np.allclose(by_hand, by_torch))
plt.figure(figsize=(7, 2.5)); plt.plot(xcurve, label="curve"); plt.plot(np.arange(len(by_hand)) + 1.5, by_hand, label="filter output"); plt.legend(); plt.show()
print("max pooling (pairs):", np.max(by_hand[:20].reshape(-1, 2), axis=1).round(2))
""")
md(r"""
## 3. The representation: a logarithmic frequency axis

On a linear frequency axis, the lowest frequencies (where slow noise that does not average down lives) are a handful of points out of
hundreds. `data.log_frequency` resamples the spectrum so that **every decade gets the same number of points**.
""")
code(r"""
spec = curves["noise: CDS spectrum"][0]
logf = data.log_frequency(spec)[0]
f = np.linspace(0, 1, len(spec))
print(f"points below 1 % of the top frequency: linear axis {np.sum((f > 0) & (f < 0.01))}, log axis {np.sum(np.r_[0, np.geomspace(f[1], 1, len(spec))[:-1]] < 0.01) - 1}")
fig, ax = plt.subplots(1, 2, figsize=(11, 3))
ax[0].semilogy(spec); ax[0].set_title("as computed (linear frequency)")
ax[1].semilogy(logf); ax[1].set_title("resampled on a log-frequency axis")
plt.tight_layout(); plt.show()
""")
md(r"""
## 4. The CNN on the toy CCD: linear against logarithmic frequency

Train on one scan (120 packets) and test on a new scan, once with each representation. Our `CurveCNN` has:
* 3 convolution layers (7, 7 and 5 taps; 16, 32 and 32 filters), with pooling;
* the settings appended after the convolutions;
* a dense layer that outputs a mean and a variance;
* an ensemble (2 members here, to keep it quick).
""")
code(r"""
dfa, ca, Ya, ga = toy_scan(120, seed=1); dfb, cb, Yb, gb = toy_scan(60, seed=2)
for logf_ in (False, True):
    Xa, _ = inputs_for(dfa, ca, ("theta",), True, ALL_CURVES, log_frequency_spectra=logf_)
    Xb, _ = inputs_for(dfb, cb, ("theta",), True, ALL_CURVES, log_frequency_spectra=logf_)
    cnn = models.CurveCNN(n_models=2).fit(Xa, Ya)
    m, s = cnn.predict(Xb)
    _ = report(f"CNN, {'log' if logf_ else 'linear'}-frequency spectra", Yb, m, s)
""")
md(r"""
In the project's full tests (240 packets) the logarithmic axis cut the CNN's σe error from 10.7 % to 1.8 %: the largest single gain we found.
**How the data are presented often matters more than the model.**

## When to use it

* **Good:** curves are available and there are enough labelled packets (a few hundred); to find effects the features miss.
* **Bad:** few labels. Then pre-train (12) or put the CNN inside a GP (13).

Next: **05 · Gaussian processes**, the workhorse of the project.
""")

# =====================================================================================================================
md, code = notebook("05_gaussian_processes")
md(r"""
# 05 · Gaussian processes

*Level: intermediate. Guide Section 9. Code: `models.GP` (scikit-learn), `structured.HierGP` (PyTorch, extended in 09).*

**Idea.** A GP predicts at new inputs by averaging the examples it has seen, weighted by how **similar** their inputs are.
* Far from all examples, it falls back to the average and its error bar grows.
* It learns, for every input, how far "similar" reaches, so inputs that do not matter get ignored.
""")
code(SETUP)
md(r"""
## 1. The kernel: similarity between two inputs

The Matérn-5/2 kernel:

$$k(x, x') = \sigma_f^2\left(1 + \sqrt5 r + \tfrac53 r^2\right)e^{-\sqrt5 r},\qquad r = |x - x'|/\ell .$$

* $\sigma_f^2$ is the signal variance;
* $\ell$ is the **length scale**, i.e. how far apart two inputs can be and still give similar values.
""")
code(r"""
def matern52(a, b, ell=1.0, sf2=1.0):
    r = np.abs(a[:, None] - b[None, :]) / ell
    return sf2 * (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r)

d = np.linspace(0, 5, 200)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.2))
for ell in (0.5, 1, 2):
    ax[0].plot(d, matern52(np.array([0.0]), d, ell)[0], label=f"ℓ = {ell}")
ax[0].legend(); ax[0].set_xlabel("distance |x − x'|"); ax[0].set_title("k: similarity falls with distance")
xg = np.linspace(0, 10, 150); K = matern52(xg, xg, 1.5)
ax[1].imshow(K, extent=[0, 10, 10, 0]); ax[1].set_title("kernel matrix K for 150 inputs")
L = np.linalg.cholesky(K + 1e-8 * np.eye(len(xg)))
for s in range(4):
    ax[2].plot(xg, L @ np.random.default_rng(s).normal(size=len(xg)))
ax[2].set_title("4 functions drawn from the prior"); plt.tight_layout(); plt.show()
""")
md(r"""
Drawing from the prior uses the **Cholesky factor** $L$ ($K = LL^\top$): if $\mathbf z$ is standard normal, $L\mathbf z$ has covariance $K$.

## 2. The posterior, by hand

With noisy observations $y_i = f(x_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal N(0, \sigma_n^2)$, the prediction at $x_*$ is

$$\mu(x_*) = \mathbf k_*^\top (K + \sigma_n^2 I)^{-1}\mathbf y,\qquad s^2(x_*) = k_{**} - \mathbf k_*^\top (K + \sigma_n^2 I)^{-1}\mathbf k_* ,$$

* $K$: the kernel matrix of the training inputs;
* $\mathbf k_*$: the kernel between $x_*$ and every training input;
* $k_{**} = k(x_*, x_*)$.
""")
code(r"""
x = np.array([1.0, 2.5, 4.0, 6.5, 7.2]); y = np.array([0.3, -0.8, 0.6, 1.2, 0.7]); sn2 = 1e-3
def posterior(x, y, xs, ell, sf2=1.0, sn2=1e-3):
    K = matern52(x, x, ell, sf2) + sn2 * np.eye(len(x)); Ks = matern52(xs, x, ell, sf2)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))          # (K + σn² I)⁻¹ y via two triangular solves
    v = np.linalg.solve(L, Ks.T)
    return Ks @ alpha, np.sqrt(np.clip(sf2 - (v * v).sum(0), 0, None))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.2), sharey=True)
for a, ell in zip(ax, (0.4, 1.5, 6.0)):
    mu, s = posterior(x, y, xg, ell)
    a.fill_between(xg, mu - 2 * s, mu + 2 * s, alpha=0.2); a.plot(xg, mu); a.scatter(x, y, color="k", zorder=3); a.set_title(f"ℓ = {ell}")
plt.tight_layout(); plt.show()
""")
md(r"""
## 3. Choosing ℓ: the log marginal likelihood

$$\log p(\mathbf y) = \underbrace{-\tfrac12\mathbf y^\top K_y^{-1}\mathbf y}_\text{data fit} \underbrace{-\tfrac12\log\det K_y}_\text{complexity penalty} - \tfrac n2\log 2\pi,\qquad K_y = K + \sigma_n^2 I.$$

* A short $\ell$ fits anything, so the data-fit term is good but the penalty is large.
* A long $\ell$ is simple but fits badly.

Maximizing the sum is an automatic Occam's razor. Here it is evaluated on 25 noisy points of a known function.
""")
code(r"""
rng = np.random.default_rng(2)
x2 = np.sort(rng.uniform(0, 10, 25)); y2 = np.sin(x2) + 0.1 * rng.normal(size=25)
ells = np.geomspace(0.1, 20, 80); fit_, pen_ = [], []
for ell in ells:
    Ky = matern52(x2, x2, ell) + 0.01 * np.eye(25); L = np.linalg.cholesky(Ky)
    a = np.linalg.solve(L.T, np.linalg.solve(L, y2))
    fit_.append(-0.5 * y2 @ a); pen_.append(-np.sum(np.log(np.diag(L))))   # log det K_y = 2 Σ log L_ii
lml = np.array(fit_) + np.array(pen_)
plt.figure(figsize=(6, 3)); plt.semilogx(ells, fit_, label="data fit"); plt.semilogx(ells, pen_, label="complexity penalty")
plt.semilogx(ells, lml, "k", lw=2, label="log marginal likelihood"); plt.axvline(ells[np.argmax(lml)], color="C3", ls=":")
plt.legend(fontsize=8); plt.xlabel("length scale ℓ"); plt.show()
print(f"best ℓ = {ells[np.argmax(lml)]:.2f}")
""")
md(r"""
## 4. One length scale per input (ARD): finding what matters

With two inputs where only $x_1$ matters, the fitted length scale of $x_2$ grows large, so $x_2$ is effectively ignored. Relevance = $1/\ell$.
""")
code(r"""
X2 = rng.uniform(0, 1, (120, 2)); y3 = np.sin(6 * X2[:, 0]) + 0.05 * rng.normal(size=120)
gp = models.GP().fit(X2, y3[:, None])
print("fitted length scales (inputs standardized): x1 =", gp.length_scales()[0].round(2), " x2 =", gp.length_scales()[1].round(1))
""")
md(r"""
## 5. On the toy CCD: which settings matter for one amplifier?

The GP is fitted on the 10 settings for amplifier 0 only. Its relevances show which knobs move σe and which move λ.
""")
code(r"""
df, curves, Y, groups = toy_scan(120, seed=1)
one = (df.channel == 0).to_numpy()
Xs, names = data.make_inputs(df[one], ("theta",), drop_constant=False)
mean, std = grouped_cv(lambda: models.GP(), Xs, Y[one], groups[one])
_ = report("GP on the settings (amplifier 0)", Y[one], mean, std)
g = models.GP().fit(Xs, Y[one])
fig, ax = plt.subplots(1, 2, figsize=(12, 3))
for a, t in zip(ax, (0, 1)):
    a.barh([n.split(".")[1] for n in names], 1 / g.length_scales(t)); a.set_title(f"relevance (1/ℓ) for log {TARGETS[t]}")
plt.tight_layout(); plt.show()
""")
md(r"""
## When to use it

* **Good:** small data (tens to ~1000 examples); honest error bars; Bayesian optimization (06); reading off what matters.
* **Bad:** many thousands of rows (cost grows as $n^3$); very many inputs, because one length scale each can over-fit.

Next: **06 · Bayesian optimization.**
""")

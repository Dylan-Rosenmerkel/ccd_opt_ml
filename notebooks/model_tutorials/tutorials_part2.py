"""Content of the model tutorials 06-14 (optimization, and the project's advanced models).  Built by make_tutorials.py."""

NOTEBOOKS = {}


def notebook(name):
    cells = []
    NOTEBOOKS[name] = cells
    return (lambda s: cells.append(("md", s))), (lambda s: cells.append(("code", s)))


SETUP = r"""
from tutorial_utils import *
"""

# =====================================================================================================================
md, code = notebook("06_bayesian_optimization")
md(r"""
# 06 · Bayesian optimization: choosing the next settings

*Level: intermediate. Guide Section 10. Code: `optimize.expected_improvement`, `optimize.propose`, `optimize.run_loop`.*

**Idea.** Each packet costs ~16 minutes, so settings cannot be tried blindly. Bayesian optimization keeps a GP **surrogate** of the
objective $J(\theta) = \log\sigma_e + w\log\lambda$. It takes the next packet where the GP predicts a good value **or** is very unsure,
updates the GP, and repeats.

(This is level 2 of the project. Here it is only explained and simulated; nothing touches the hardware.)
""")
code(SETUP + r"""
from scipy.stats import norm
from ml import optimize
""")
md(r"""
## 1. Expected improvement, by hand

Let $J_\text{best}$ be the best value measured so far. If the true value at θ is $J$, the improvement is $\max(0, J_\text{best} - J)$.
The GP says $J \sim \mathcal N(\mu, s^2)$, so the expected improvement is

$$\mathrm{EI} = (J_\text{best} - \mu)\,\Phi(z) + s\,\phi(z),\qquad z = \frac{J_\text{best}-\mu}{s},$$

where $\Phi$ is the standard Gaussian's cumulative distribution and $\phi$ its density.
* The first term is large where the mean is good (**exploitation**).
* The second is large where $s$ is large (**exploration**).
""")
code(r"""
def ei_by_hand(mu, s, best):
    z = (best - mu) / s
    return (best - mu) * norm.cdf(z) + s * norm.pdf(z)
mu, s = np.array([0.0, 0.5, 0.5, 1.0]), np.array([0.1, 0.1, 1.0, 1.0])
print("μ:", mu, "  s:", s, "  J_best = 0.2")
print("EI by hand:      ", ei_by_hand(mu, s, best=0.2).round(4))
print("optimize.py's EI:", optimize.expected_improvement(mu, s, 0.2).round(4))
print("→ a slightly worse mean with a large s (third) can beat a better mean that is certain (first).")
""")
md(r"""
## 2. The loop on a 1D function

Start with 3 measurements. At every step: fit the GP, compute EI on a grid, measure where EI is largest.
""")
code(r"""
f = lambda t: 0.6 * np.sin(1.3 * t) + 0.08 * (t - 5) ** 2 - 0.2
grid = np.linspace(0, 10, 400)
X = np.array([0.8, 4.0, 8.8]); Jm = f(X)
fig, axes = plt.subplots(2, 3, figsize=(15, 6))
for step, ax in enumerate(axes.flat):
    gp = models.GP(restarts=2).fit(X[:, None], Jm[:, None])
    m, s = gp.predict(grid[:, None]); m, s = m[:, 0], s[:, 0]
    ei = ei_by_hand(m, np.maximum(s, 1e-9), Jm.min())
    ax.plot(grid, f(grid), "--", color="0.6"); ax.plot(grid, m, "C0"); ax.fill_between(grid, m - 2 * s, m + 2 * s, alpha=0.15)
    ax.scatter(X, Jm, color="k", zorder=3)
    ax2 = ax.twinx(); ax2.plot(grid, ei, "C3", lw=1); ax2.set_yticks([])
    nxt = grid[np.argmax(ei)]; ax.axvline(nxt, color="C3", ls=":"); ax.set_title(f"step {step + 1}: next θ = {nxt:.2f}")
    X, Jm = np.append(X, nxt), np.append(Jm, f(nxt))
plt.tight_layout(); plt.show()
print(f"best found {Jm.min():.3f} at θ = {X[np.argmin(Jm)]:.2f};  true minimum {f(grid).min():.3f} at θ = {grid[np.argmin(f(grid))]:.2f}")
""")
md(r"""
## 3. A whole campaign on the toy CCD (10 settings)

`run_loop` starts with 10 random packets, then uses EI. `allowed` is the toy world's rule between parameters, so candidates that break it
are never proposed. The comparison is against packets taken at random.
""")
code(r"""
rng = np.random.default_rng(0)
measure = lambda th: synthetic.true_objective(th) + 0.01 * rng.normal()
plt.figure(figsize=(6.5, 3.5))
for strategy, color in (("bo", "C0"), ("random", "C1")):
    for seed in (0, 1):
        res = optimize.run_loop(measure, synthetic.LOW, synthetic.HIGH, n_initial=10, n_total=40, seed=seed,
                                allowed=synthetic.allowed, strategy=strategy)
        plt.plot(res["best_so_far"], color=color, label=strategy if seed == 0 else None)
plt.axvline(10, color="k", ls=":", lw=1); plt.xlabel("packets taken"); plt.ylabel("best J so far (lower is better)"); plt.legend(); plt.show()
""")
md(r"""
## When to use it

* **Good:** each evaluation is expensive; up to ~20 parameters; noisy measurements.
* **Bad:** cheap evaluations (then just try many points); very many parameters at once (use a trust region, notebook 07).

**Try:** `xi=0.05` in `expected_improvement` (more exploration); a larger measurement noise.

Next: **07 · Better optimization strategies** (trust region, constraints, cheap traces, Pareto fronts).
""")

# =====================================================================================================================
md, code = notebook("07_better_optimization")
md(r"""
# 07 · Better optimization strategies (model F)

*Level: advanced. Guide Section 11. Code: `bo_advanced.py`.*

Four upgrades of plain Bayesian optimization:
1. a **trust region**: search near the best point, with a box that grows and shrinks;
2. a **learned constraint**: learn where packets fail;
3. **multi-fidelity**: use cheap trace-only packets;
4. **several objectives**: the Pareto front of read noise against dark charge.

All four use the **decomposed surrogate**: one GP for log σe and one for log λ, combined into J. Each part is a simpler function than their
sum.
""")
code(SETUP + r"""
from scipy.stats import norm
from ml import bo_advanced
ei = lambda mu, s, best: (best - mu) * norm.cdf((best - mu) / s) + s * norm.pdf((best - mu) / s)
""")
md(r"""
## 1. A trust region, by hand (2 settings)

* Candidates are drawn only in a box of side $L$ (a fraction of each range) around the best point.
* $L$ doubles after 3 improvements in a row and halves after 5 failures.
* The search stays local and moves with the best point, like a CCD whose voltages change in bounded steps.
""")
code(r"""
f2 = lambda p: (p[..., 0] - 0.7) ** 2 + 2 * (p[..., 1] - 0.3) ** 2 + 0.15 * np.sin(8 * p[..., 0])
rng = np.random.default_rng(0)
P = rng.random((6, 2)); J = f2(P); L, succ, fail, boxes = 0.4, 0, 0, []
for step in range(20):
    best = P[np.argmin(J)]
    lo, hi = np.clip(best - L / 2, 0, 1), np.clip(best + L / 2, 0, 1); boxes.append((lo, hi))
    cand = lo + (hi - lo) * rng.random((2000, 2))
    m, s = models.GP().fit(P, J[:, None]).predict(cand)
    nxt = cand[np.argmax(ei(m[:, 0], np.maximum(s[:, 0], 1e-9), J.min()))]
    improved = f2(nxt) < J.min() - 1e-3
    P, J = np.vstack([P, nxt]), np.append(J, f2(nxt))
    succ, fail = (succ + 1, 0) if improved else (0, fail + 1)
    if succ >= 3: L, succ = min(2 * L, 1), 0
    if fail >= 5: L, fail = L / 2, 0
g = np.linspace(0, 1, 100); G = np.stack(np.meshgrid(g, g), -1)
plt.figure(figsize=(5, 4.5)); plt.contourf(g, g, f2(G), 30, cmap="Greys")
for k, (lo, hi) in enumerate(boxes[::4]):
    plt.gca().add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, color=plt.cm.viridis(k / 5), lw=1.5))
plt.plot(*P[6:].T, "C3.-", lw=0.6); plt.scatter(*P[:6].T, color="C0", label="start"); plt.legend(); plt.title("trust-region boxes (every 4th step)")
plt.show(); print(f"best J {J.min():.3f}")
""")
md(r"""
## 2. A constraint that must be learned

If the safe region were unknown, packets outside it would fail. A GP regression on outcomes $\pm1$ gives
$P(\text{feasible}) = \Phi(\mu_c/s_c)$, and the acquisition becomes $\mathrm{EI} \times P(\text{feasible})$.

We **know** the CCD's rules (`ccd_limits.py`), so the project uses the known constraint. This is for limits that cannot be written down.
""")
code(r"""
Pc = rng.random((40, 2)); ok = (Pc[:, 0] + Pc[:, 1] < 1.2)
gpc = models.GP().fit(Pc, np.where(ok, 1.0, -1.0)[:, None])
mc, sc = gpc.predict(G.reshape(-1, 2)); pf = norm.cdf(mc[:, 0] / np.maximum(sc[:, 0], 1e-9)).reshape(100, 100)
plt.figure(figsize=(5, 4.2)); plt.contourf(g, g, pf, 20, cmap="RdYlGn"); plt.colorbar(label="P(feasible)")
plt.scatter(*Pc[ok].T, c="k", marker="o", s=12); plt.scatter(*Pc[~ok].T, c="k", marker="x"); plt.plot([0.2, 1], [1, 0.2], "w--", label="true boundary")
plt.legend(); plt.show()
""")
md(r"""
## 3. Cheap trace-only packets (multi-fidelity)

A trace-only packet takes ~1 minute and measures the noise in ADU, but not the gain or λ. The classic way to combine a cheap and an
expensive measurement is **co-kriging**:

$$f_\text{expensive}(\theta) = \rho\,f_\text{cheap}(\theta) + \delta(\theta),$$

with $\rho$ a learned factor and $\delta$ a GP correction. Our version uses model A (notebook 08): trace noise minus the predicted log gain is
an observation of $\log\sigma_e$. It pays off only if the trace predicts the image well. That is exactly what level 1 tests.

## 4. Several objectives: the Pareto front and the hypervolume

* Settings **dominate** others if they are at least as good in both objectives and better in one.
* The non-dominated ones form the **Pareto front**: the best available trade-offs.
* The **hypervolume** (the area they dominate, up to a reference point) scores a front.
* **ParEGO** draws a random weight $w$ for $J_w = \log\sigma_e + w\log\lambda$ at each step, so the packets spread along the front.
""")
code(r"""
pts = np.column_stack([rng.uniform(0.05, 1, 40), np.zeros(40)]); pts[:, 1] = np.minimum(0.1 / pts[:, 0] + rng.uniform(0, 0.6, 40), 1.1)
def pareto(p):
    keep = [i for i in range(len(p)) if not np.any(np.all(p <= p[i], 1) & np.any(p < p[i], 1))]
    return p[sorted(keep, key=lambda i: p[i, 0])]
front = pareto(pts); ref = np.array([1.05, 1.15])
print(f"{len(front)} of {len(pts)} points are on the front; hypervolume = {bo_advanced.hypervolume_2d(pts, ref):.3f}")
plt.figure(figsize=(5, 3.8)); plt.scatter(*pts.T, color="0.6"); plt.step(*front.T, where="post", color="C3"); plt.scatter(*front.T, color="C3")
plt.plot(*ref, "ks"); plt.xlabel("log σe"); plt.ylabel("log λ"); plt.show()
""")
md(r"""
## 5. On the toy CCD: global EI against the trust region

`bo_advanced.run` simulates whole campaigns with the decomposed surrogate. The score is the *true* objective at the best point found.
""")
code(r"""
plt.figure(figsize=(6.5, 3.5))
for strategy in ("global", "trust region", "random"):
    res = bo_advanced.run(strategy, seed=0, n_total=30, n_initial=10)
    plt.plot(res["best_true"], label=strategy)
plt.xlabel("packets"); plt.ylabel("true J at the best point so far"); plt.legend(); plt.show()
""")
md(r"""
One campaign is noisy: which strategy is ahead changes from seed to seed. Over several campaigns in the project's tests
(`docs/ml_results/advanced_bo.json`), the trust region found the best settings.

## When to use which

| Strategy | Use when |
|---|---|
| trust region | many settings; bounded steps (best in our tests) |
| learned constraint | limits unknown (ours are known) |
| multi-fidelity | the trace predicts the image well and trace packets are much cheaper |
| ParEGO | to see the σe–λ trade-off before fixing $w$ |

Next: **08 · Model A, physics-structured outputs.**
""")

# =====================================================================================================================
md, code = notebook("08_physics_structured")
md(r"""
# 08 · Model A: physics-structured outputs (the default model)

*Level: advanced. Guide Section 12. Code: `structured.PhysicsStructured`, `structured.StructuredPrep`, `structured.allan_fit`.*

**Idea.** Do not learn σe from scratch: the traces **measure** the noise in ADU, including how it averages down with skips and the part that does
not. Only the gain cannot be seen on a trace. So model A:
* takes the noise from the trace;
* learns the gain;
* learns a small correction for what the image sees and the trace does not.

A model that learns less needs fewer packets.
""")
code(SETUP + r"""
from ml import structured as st
""")
md(r"""
## 1. Skipper averaging and the noise floor

Averaging $N$ independent skips of noise $\sigma_1$ gives $\sigma_1/\sqrt N$. Slow ($1/f$) noise is correlated from skip to skip and does **not**
average down, which leaves a floor:

$$\sigma_e^2 = \frac{\sigma_1^2}{N} + \sigma_\text{floor}^2 .$$
""")
code(r"""
rng = np.random.default_rng(0)
n_pix, n_skip = 400, 2000
white = rng.normal(0, 1.0, (n_pix, n_skip))
freqs = np.fft.rfftfreq(n_skip); shape = np.r_[0, 1 / np.sqrt(freqs[1:])]
flick = np.fft.irfft((rng.normal(size=(n_pix, freqs.size)) + 1j * rng.normal(size=(n_pix, freqs.size))) * shape, n=n_skip, axis=1)
flick *= 0.3 / flick.std()
skips = white + flick
Ns = np.array([1, 4, 16, 64, 256, 1000])
sd_pixel = [skips[:, :N].mean(1).std() for N in Ns]
plt.figure(figsize=(5.5, 3.2)); plt.loglog(Ns, sd_pixel, "ko", label="noise of the N-skip average")
plt.loglog(Ns, 1 / np.sqrt(Ns), "C0--", label="1/√N (white only)"); plt.xlabel("N skips"); plt.legend(); plt.show()
""")
md(r"""
## 2. The Allan deviation, by hand

Split a pixel's skips into consecutive blocks of $n$, with block means $\bar y_k(n)$. Then

$$\sigma_A^2(n) = \tfrac12\,\mathbb E\bigl[(\bar y_{k+1}(n) - \bar y_k(n))^2\bigr].$$

* White noise gives $\sigma_A^2 = \sigma_1^2/n$.
* $1/f$ noise gives a flat $\sigma_A^2$.

Hence the fit $\sigma_A^2(n) = a^2/n + b^2$: it is linear in $1/n$, so it can be solved by (weighted) least squares.
""")
code(r"""
def allan(x, ns):
    out = []
    for n in ns:
        m = x.shape[1] // n
        blocks = x[:, :m * n].reshape(x.shape[0], m, n).mean(2)
        out.append(np.sqrt(0.5 * np.mean(np.diff(blocks, axis=1) ** 2)))
    return np.array(out)
ns = synthetic.ALLAN_N
sA = allan(skips, ns)
A_ = np.column_stack([1 / ns, np.ones_like(ns, dtype=float)]); w = 1 / sA ** 2          # weights for a constant relative scatter
coef = np.linalg.lstsq(A_ * w[:, None], sA ** 2 * w, rcond=None)[0]
a_hand, b_hand = np.sqrt(coef)
a_lib, b_lib = st.allan_fit(sA[None], ns)
print(f"by hand: a = {a_hand:.3f}, b = {b_hand:.3f}   |   structured.allan_fit: a = {a_lib[0]:.3f}, b = {b_lib[0]:.3f}")
g = np.geomspace(1, 200, 100)
plt.figure(figsize=(5.5, 3.2)); plt.loglog(ns, sA, "ko"); plt.loglog(g, np.sqrt(a_hand ** 2 / g + b_hand ** 2), "C3")
plt.loglog(g, a_hand / np.sqrt(g), "C0--"); plt.axhline(b_hand, color="C2", ls="--"); plt.xlabel("block length n"); plt.show()
""")
md(r"""
## 3. Model A step by step on the toy CCD

1. $a, b$ from each trace's Allan curve, and from them the noise in ADU after $N$ skips: $\tfrac12\log(a^2/N + b^2)$.
2. Divide by the gain, $\log\sigma_e = \tfrac12\log(a^2/N + b^2) - \log g$.
3. With the **true** gain, how good is the trace alone? With the average gain (no knowledge of the amplifier)?
""")
code(r"""
df, curves, Y, groups = toy_scan(60, seed=1)
a, b = st.allan_fit(curves["noise: Allan deviation"], synthetic.ALLAN_N)
phys = 0.5 * np.log(a ** 2 / synthetic.N_SKIPS + b ** 2)
truth = np.log(df["truth.sigma_e"].to_numpy())
err = lambda p: 100 * np.sqrt(np.mean((p - truth) ** 2))
print(f"trace physics with the TRUE gain:    σe error {err(phys - np.log(df['truth.gain'])):.1f} %")
print(f"trace physics with the MEAN gain:    σe error {err(phys - Y[:, 2].mean()):.1f} %   (the gains differ between amplifiers)")
""")
md(r"""
So model A learns:
* **the gain** $\hat g(x)$: a GP;
* **a correction** $\hat r(x)$: a GP trained on the residual $\log\sigma_e - \tfrac12\log(a^2/N+b^2) + \log g$ (with the *labelled* gain), i.e.
  whatever the image sees and the trace does not;
* λ: a GP on $\log\lambda$.

$$\log\sigma_e = \tfrac12\log\!\Bigl(\frac{a^2}{N} + b^2\Bigr) - \log\hat g(x) + \hat r(x).$$

**Inputs** $x$:
* the settings;
* the **packet state vector**: each reference feature minus that amplifier's average, averaged over the packet's amplifiers (the CCD's drifting state);
* the amplifier's **own** trace and reference features.

There is **no amplifier label**.
""")
code(r"""
dft, ct, Yt, gt = toy_scan(60, seed=2)                         # an independent test scan
truth_t = np.log(dft[["truth.sigma_e", "truth.lambda", "truth.gain"]].to_numpy())
for n in (15, 60):
    dtr, ctr, Ytr, _ = toy_scan(n, seed=1)
    prep = st.StructuredPrep().fit(dtr)
    A = st.PhysicsStructured().fit(prep.transform(dtr, ctr), Ytr)
    m, s = A.predict(prep.transform(dft, ct))
    _ = report(f"model A, {n} training packets (vs truth)", truth_t, m, s)
    Xtr, _ = data.make_inputs(dtr, ("theta", "trace", "ref"), drop_constant=False)
    Xte, _ = data.make_inputs(dft, ("theta", "trace", "ref"), drop_constant=False)
    m2, s2 = models.GP().fit(Xtr, Ytr).predict(Xte)
    _ = report(f"plain GP on features, {n} packets", truth_t, m2, s2)
print("\npacket state vector of the first test packet (first 5 numbers):", prep.state_vector(dft)[0, :5].round(3))
""")
md(r"""
**Reading the numbers:**
* For σe, model A is far ahead with few packets (here ~3 % from 15 packets, against ~20 % for a GP that must learn everything).
* λ is hard for every model with 15 packets: it follows the CCD's drifting state, which takes more packets to learn. By 60 packets
  both models are close.
* The gain is learned well by both.

## When to use it

* **Good:** our first choice. It learns little, so it needs few packets. $N$ appears explicitly, so it predicts other skip numbers too.
* **Limits:**
  * it relies on the trace measuring the noise the image sees; level 1 tests exactly this on real data;
  * a never-seen amplifier needs one imaged packet for its gain, though the dependence on the settings is right without it.

Next: **09 · Hierarchical GP and amplifier labels.**
""")

# =====================================================================================================================
md, code = notebook("09_hierarchical_gp")
md(r"""
# 09 · Model B: hierarchical Gaussian processes, and whether amplifier labels are needed

*Level: advanced. Guide Section 13. Code: `structured.HierGP`, `PhysicsStructured(kind="hier")`.*

**Idea.** Amplifiers differ but are related. A hierarchical model learns:
* what they share;
* how much each deviates.

A new module then needs a few packets, not a whole scan.
""")
code(SETUP + r"""
from ml import structured as st
from scipy.stats import spearmanr
""")
md(r"""
## 1. Shrinkage: borrowing strength

An amplifier's offset $\delta$ is drawn with spread $c_a$. With $k$ of its points (noise $\sigma_n$), averaging $\bar r$, the best estimate is

$$\hat\delta = \frac{c_a^2}{c_a^2 + \sigma_n^2/k}\,\bar r ,$$

which is pulled towards 0 when there are few points. A simulation compares it with the raw mean.
""")
code(r"""
rng = np.random.default_rng(0); c_a, s_n = 0.3, 0.5
for k in (1, 2, 5, 20):
    delta = rng.normal(0, c_a, 20000); rbar = delta + rng.normal(0, s_n / np.sqrt(k), 20000)
    shrunk = c_a ** 2 / (c_a ** 2 + s_n ** 2 / k) * rbar
    print(f"k = {k:2d}: RMS error raw mean {np.sqrt(np.mean((rbar - delta) ** 2)):.3f}   shrunk {np.sqrt(np.mean((shrunk - delta) ** 2)):.3f}")
""")
md(r"""
## 2. The kernel: sums of independent pieces

For amplifier $j$ on module $m$: $f_j(x) = f(x) + \delta_m(x) + \delta_j(x)$. Independent GPs add their kernels:

$$k(i,i') = \sigma_f^2 M + \mathbb 1[m=m'](c_m^2 + \sigma_m^2 M) + \mathbb 1[j=j'](c_a^2 + \sigma_a^2 M) + \sigma_n^2\mathbb 1[i=i'] .$$

* $M$ is the Matérn correlation.
* The indicators switch a term on only for pairs on the same module or the same amplifier.
""")
code(r"""
x = np.tile(np.linspace(0, 5, 10), 3); amp = np.repeat([0, 1, 2], 10)
M = (lambda r: (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r))(np.abs(x[:, None] - x[None, :]) / 1.5)
K = 1.0 * M + (amp[:, None] == amp[None, :]) * (0.5 + 0.1 * M)
plt.figure(figsize=(4.2, 3.6)); plt.imshow(K); plt.colorbar(); plt.title("3 amplifiers × 10 settings: shared + same-amplifier blocks"); plt.show()
""")
md(r"""
## 3. A new amplifier with 0 and 1 of its own points

`HierGP(levels=("amp",))` learns all the amplitudes from the data by maximizing the marginal likelihood.
""")
code(r"""
shared = lambda x: np.sin(x) + 0.2 * x; offsets = [0.9, -0.6, 0.2, 1.4]
X, A_, Yv = [], [], []
for j in range(3):
    xx = np.sort(rng.uniform(0, 8, 9)); X.append(xx); A_.append(np.full(9, j)); Yv.append(shared(xx) + offsets[j] + 0.1 * rng.normal(size=9))
X, A_, Yv = map(np.concatenate, (X, A_, Yv)); xs = np.linspace(0, 8, 200)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
for a_, k in zip(ax, (0, 1)):
    Xk, Ak, Yk = (np.append(X, 5.0), np.append(A_, 3), np.append(Yv, shared(5.0) + offsets[3])) if k else (X, A_, Yv)
    gp = st.HierGP(levels=("amp",)).fit({"x": Xk[:, None], "amp": Ak}, Yk)
    for j in range(4):
        m, s = gp.predict({"x": xs[:, None], "amp": np.full(200, j)})
        a_.plot(xs, m[:, 0], color=f"C{j}", lw=2.5 if j == 3 else 1)
        if j == 3: a_.fill_between(xs, m[:, 0] - 2 * s[:, 0], m[:, 0] + 2 * s[:, 0], color="C3", alpha=0.15)
    a_.plot(xs, shared(xs) + offsets[3], "k:"); a_.scatter(X, Yv, c=[f"C{j}" for j in A_], s=10)
    a_.set_title(f"new amplifier (red) with {k} of its own points")
plt.tight_layout(); plt.show()
print("fitted amplitudes (in units of the target sd):", {k: round(v, 3) for k, v in gp.hyperparameters().items()})
""")
md(r"""
## 4. Do the models need amplifier labels? A new module on the toy CCD

Training uses a whole scan of module 100 plus $k$ packets of module 101; testing uses new packets of module 101. Two models are compared:
* **A+B**: told the module and amplifier (the hierarchical levels);
* **A, no labels**: each amplifier's own traces instead (the default).

Besides the error, two scores matter for optimization:
* **shape**: the error after removing each amplifier's mean error;
* **rank**: the within-amplifier Spearman correlation, i.e. whether the settings are put in the right order.
""")
code(r"""
def scores(df_, m):
    t = np.log(df_["truth.sigma_e"].to_numpy()); r = m[:, 0] - t
    amp = (df_.board.astype(str) + df_.channel.astype(str)).to_numpy()
    shape = r.copy()
    for a in np.unique(amp): shape[amp == a] -= r[amp == a].mean()
    rank = np.mean([spearmanr(m[amp == a, 0], t[amp == a])[0] for a in np.unique(amp)])
    return 100 * np.sqrt(np.mean(r ** 2)), 100 * np.sqrt(np.mean(shape ** 2)), rank

tr, ctr = synthetic.make_dataset(80, seed=1, n_boards=2); te, cte = synthetic.make_dataset(40, seed=99, n_boards=2)
te_new = (te.board == 101).to_numpy(); te, cte = te[te_new].reset_index(drop=True), {k: v[te_new] for k, v in cte.items()}
pk = tr.packet_id.str[4:].astype(int)
for k in (0, 2):
    keep = ((tr.board == 100) | (pk < k)).to_numpy()
    d, c = tr[keep].reset_index(drop=True), {kk: v[keep] for kk, v in ctr.items()}
    Yd = data.make_targets(d)[0]
    for label, kind in (("A+B, amplifier/module labels", "hier"), ("A, no labels (own traces)", "none")):
        prep = st.StructuredPrep().fit(d)
        m, s = st.PhysicsStructured(kind).fit(prep.transform(d, c), Yd).predict(prep.transform(te, cte))
        e, sh, rk = scores(te, m)
        print(f"new module, {k} of its packets | {label:30s}: σe error {e:5.1f} %, shape {sh:4.1f} %, rank {rk:.2f}")
""")
md(r"""
With 0 packets of the new module, every model is off in *level* (its gains are unknown), but the shape and the order of the settings are right.

After 2 packets both are within a few percent. In this single run the labelled model is ahead; averaged over 3 training scans in the project's
test (`benchmark_amp.py`) they were 2.0 % against 2.2 %. Both put the settings in the right order, and the label-free model works for **any**
amplifier. That is why the default uses no labels.

## When to use it

* **Good:**
  * several amplifiers or modules;
  * a new module with 1–2 packets;
  * honest error bars for an amplifier never seen.
* **Not needed:** when each amplifier's own traces are inputs. The module level only means something with ≥ 2 modules (model G, several modules, is this GP fitted across modules).

Next: **09b · Model A+B**, the two combined (A's physics on B's hierarchical GPs).
""")

# =====================================================================================================================
md, code = notebook("10_joint_dark_cic")
md(r"""
# 10 · Dark current and CIC: the joint fit

*Level: advanced. Guide Section 14. Code: `structured.JointLambdaGP`, `structured.line_fit_covariance`, `src/image_rows.py`.*

**Idea.** The dark charge λ has two sources:
* **dark current**, which grows with the time a pixel waits;
* **clock-induced charge (CIC)**, from the clock swings.

Rows read later wait longer, so the image separates the two. Their estimates are strongly **anti-correlated**, and a model must use that.
""")
code(SETUP + r"""
from ml import structured as st
import image_rows as ir
""")
md(r"""
## 1. λ row by row

$$\lambda_r = \mathrm{CIC} + \mathrm{DC}\,x_r,\qquad x_r = \frac{r+\tfrac12}{R/2}.$$

Each row's λ is fitted from its pixels by maximum likelihood (`image_rows.row_lambda`), with the gain and noise fixed. Here it is tried on
simulated pixels.
""")
code(r"""
rng = np.random.default_rng(0); R, C, g, s, dc, cic = 16, 844, 1.3, 0.25, 0.02, 0.01
xr = (np.arange(R) + 0.5) / (R / 2); lam = cic + dc * xr
img = np.array([100 + g * rng.poisson(l, C) + s * rng.normal(size=C) for l in lam])
plt.figure(figsize=(5, 3)); plt.hist(img[8] - 100, 120); plt.yscale("log"); plt.xlabel("pixel value − pedestal (ADU)"); plt.title("one row: 0-, 1-, 2-electron peaks"); plt.show()
fit = np.array([ir.row_lambda(row, g, s) for row in img])
print("row 0: λ = %.4f ± %.4f (truth %.4f)" % (*fit[0], lam[0]))
""")
md(r"""
## 2. The straight line and its covariance, by hand

With $A$ the $R\times2$ matrix of rows $(x_r, 1)$, least squares gives

$$\begin{pmatrix}\widehat{\mathrm{DC}}\\ \widehat{\mathrm{CIC}}\end{pmatrix} = P\boldsymbol\lambda,\quad P = (A^\top A)^{-1}A^\top,\qquad \Sigma = P\,\mathrm{diag}(\sigma_r^2)\,P^\top .$$

Repeating the simulated image 2000 times shows the anti-correlation.
""")
code(r"""
A = np.column_stack([xr, np.ones(R)]); P = np.linalg.solve(A.T @ A, A.T)
sig_r = 0.12 * lam                                           # each row's λ to ~12 %
Sigma = P @ np.diag(sig_r ** 2) @ P.T
draws = np.array([P @ (lam + sig_r * rng.normal(size=R)) for _ in range(2000)])
print("covariance by formula (×1e6):\n", (Sigma * 1e6).round(3), "\ncorrelation:", (Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])).round(3))
print("library, structured.line_fit_covariance (×1e6):\n", (st.line_fit_covariance(np.array([dc]), np.array([cic]))[0] * 1e6).round(3))
plt.figure(figsize=(4.5, 3.5)); plt.scatter(*draws.T, s=2, alpha=0.3); plt.xlabel("DC estimate"); plt.ylabel("CIC estimate"); plt.show()
var_sum = Sigma[0, 0] + Sigma[1, 1] + 2 * Sigma[0, 1]
print(f"sd of DC + CIC: with the covariance {np.sqrt(var_sum):.5f}; ignoring it {np.sqrt(Sigma[0, 0] + Sigma[1, 1]):.5f}  (much worse)")
""")
md(r"""
**Fitting the two parts separately** throws this correlation away: each part looks uncertain, while their sum (the image's mean λ) is precise.

**The joint model** has two GPs, $f_\text{DC}$ and $f_\text{CIC}$, observed together with each image's known $2\times2$ covariance. In logs
the covariance is converted by the delta method, $\Sigma_\text{log} = J\Sigma J$ with $J = \mathrm{diag}(1/\mathrm{DC}, 1/\mathrm{CIC})$.

## 3. On the toy CCD: direct, separate, and joint λ

The toy world v3 has per-row λ and an image-only noise term. Errors are against the truth.
""")
code(r"""
kw = dict(per_row_lambda=True, image_only_noise=True)
dft, ct = synthetic.make_dataset(60, seed=2, **kw)
truth_t = np.log(dft[["truth.sigma_e", "truth.lambda", "truth.gain"]].to_numpy())
for n in (60, 120):
    dtr, ctr = synthetic.make_dataset(n, seed=1, **kw); Ytr = data.make_targets(dtr)[0]
    prep = st.StructuredPrep().fit(dtr); Xtr, Xte = prep.transform(dtr, ctr), prep.transform(dft, ct)
    for label, model, comp in (("direct GP on log λ", st.PhysicsStructured(), None),
                               ("separate DC and CIC GPs", st.PhysicsStructured(use_components=True), st.component_targets(dtr)),
                               ("joint DC + CIC (with covariance)", st.PhysicsStructured(lambda_model="joint"), None)):
        m, s = model.fit(Xtr, Ytr, comp).predict(Xte)
        print(f"{n:3d} packets | {label:34s}: λ error {100 * np.sqrt(np.mean((m[:, 1] - truth_t[:, 1]) ** 2)):5.1f} %")
""")
md(r"""
## When to use it

**Reading the numbers:** the joint fit beats fitting the two parts separately. Against the direct fit it varies: in this single
run, without amplifier labels, the direct fit is ahead. Over 3 training scans in the project's test (with the hierarchical GP) they were equal
within the scatter (1.6 against 1.8 % at 240 packets).

* **Good:** when λ row by row is available and you want dark current and CIC **separately** (which settings drive which). For the total λ the
  direct fit is as good or better.
* **Avoid:** fitting the parts with separate models.

Next: **11 · Calibrated error bars (conformal prediction).**
""")

# =====================================================================================================================
md, code = notebook("11_conformal_calibration")
md(r"""
# 11 · Calibrated error bars: conformal prediction (model H)

*Level: advanced (but short). Guide Section 15. Code: `calibrate.Conformal`, `calibrate.conformal_factor`.*

**Idea.** Measure, on packets the model did not train on, how many of its own error bars it takes to cover the truth. Then rescale the error bars by
that factor. It works for **any** model.
""")
code(SETUP + r"""
from ml.calibrate import Conformal, conformal_factor
""")
md(r"""
## 1. The recipe, by hand

1. For held-out examples: $z_i = |y_i - \mu_i|/s_i$, the error in units of the model's own standard deviation.
2. The factor for coverage $c$ is the $\lceil (n+1)c\rceil$-th smallest $z_i$.
3. Predict with $\mu \pm q_c\, s$.

Below, a deliberately **over-confident** model claims σ = 0.5 when the truth scatters by 1.
""")
code(r"""
rng = np.random.default_rng(0)
cal_err, test_err, claimed = rng.normal(0, 1, 200), rng.normal(0, 1, 5000), 0.5
z = np.abs(cal_err) / claimed
n = len(z); k = int(np.ceil((n + 1) * 0.68)) - 1
q_hand = np.sort(z)[k]
print(f"factor by hand {q_hand:.3f}; calibrate.conformal_factor {conformal_factor(z, 0.68):.3f}  (ideal 1/0.5 = 2)")
print(f"coverage on new data: raw {np.mean(np.abs(test_err) <= claimed):.2f}, calibrated {np.mean(np.abs(test_err) <= q_hand * claimed):.2f} (target 0.68)")
""")
md(r"""
**The guarantee.** If new examples are **exchangeable** with the calibration ones (they could have been any of them), the coverage is at least
$c$, whatever the model. Our version uses 5 packet-grouped folds, so every training packet serves for calibration.

## 2. On the toy CCD: boosted trees and ridge, raw and calibrated
""")
code(r"""
dfa, ca, Ya, ga = toy_scan(120, seed=1); dfb, cb, Yb, gb = toy_scan(60, seed=2)
Xa, _ = data.make_inputs(dfa, ("theta", "trace", "ref"), drop_constant=False)
Xb, _ = data.make_inputs(dfb, ("theta", "trace", "ref"), drop_constant=False)
for name, make in (("boosted trees", lambda: models.GBT()), ("Bayesian ridge", lambda: models.BayesRidge())):
    m, s = make().fit(Xa, Ya).predict(Xb); _ = report(f"{name}, raw", Yb, m, s)
    cal = Conformal(make).fit(Xa, Ya, ga); m, s = cal.predict(Xb); _ = report(f"{name}, conformal", Yb, m, s)
    print("   factors (68 %):", cal.factors[0.68].round(2))
""")
md(r"""
## 3. What it cannot do: a new situation

Calibrated on module 100 and tested on a module never seen, the coverage is no longer guaranteed: the new module is **not exchangeable** with the old one.
""")
code(r"""
tr, _ = synthetic.make_dataset(120, seed=1, n_boards=2); te, _ = synthetic.make_dataset(40, seed=99, n_boards=2)
tr = tr[tr.board == 100].reset_index(drop=True); te_old, te_new = te[te.board == 100], te[te.board == 101]
Xt, _ = data.make_inputs(tr, ("theta", "trace", "ref"), drop_constant=False)
cal = Conformal(lambda: models.GBT()).fit(Xt, data.make_targets(tr)[0], data.groups_of(tr))
for label, d in (("same module", te_old), ("NEW module", te_new)):
    X_, _ = data.make_inputs(d, ("theta", "trace", "ref"), drop_constant=False); m, s = cal.predict(X_)
    _ = report(f"conformal trees, {label}", data.make_targets(d)[0], m, s)
""")
md(r"""
## When to use it

Always, as the last step: it costs 5 extra fits. It is essential for boosted trees and a check for every other model. Remember the
exchangeability condition.

Next: **12 · Pre-training on unlabelled traces (D).**
""")

# =====================================================================================================================
md, code = notebook("12_pretraining")
md(r"""
# 12 · Model D: self-supervised pre-training on unlabelled traces

*Level: advanced. Guide Section 16. Code: `deep.Pretrainer`, `deep.PretrainedCNN`, `deep.PretrainedHead`.*

**Idea.** Labels need images and are expensive; traces are cheap. A network first learns "what traces look like" from traces alone, using
tasks that need no labels:
* fill in hidden parts of the curves;
* predict the settings;
* recognize two noisy copies of the same curves.

Then a few labelled packets suffice.
""")
code(SETUP + r"""
import torch
from ml import deep
from ml.benchmark import THETA_CURVES, inputs_for
""")
md(r"""
## 1. The encoder and the masks

The **encoder** $g_\omega$ is the CNN body of notebook 04, ending in 16 numbers: the **embedding** $\mathbf z$. In **masked reconstruction**,
part of the input is hidden and a decoder must rebuild it. The hidden part is either:
* blocks of 32 positions; or
* one whole curve.

Whole curves matter: short gaps can be filled by interpolation, but rebuilding the noise spectrum from the other curves forces the encoder to
learn how the curves are related.
""")
code(r"""
df, curves, Y, groups = toy_scan(30, seed=1)
Xc, _ = inputs_for(df, curves, ("theta",), True, THETA_CURVES, log_frequency_spectra=True)
c = Xc["curves"][0]                                  # (7 curves, 256 points)
rng = np.random.default_rng(0)
block = np.repeat(rng.random(8) < 0.25, 32)[None, :].repeat(7, 0)
whole = np.zeros_like(block); whole[3] = True
fig, ax = plt.subplots(1, 3, figsize=(15, 3))
for a, mask, title in zip(ax, (np.zeros_like(block), block, whole), ("the 7 curves (standardized)", "blocks of 32 hidden", "one whole curve hidden")):
    a.imshow(np.where(mask, np.nan, (c - c.mean(1, keepdims=True)) / (c.std(1, keepdims=True) + 1e-9)), aspect="auto", cmap="viridis"); a.set_title(title)
    a.set_yticks(range(7)); a.set_yticklabels([n[:22] for n in THETA_CURVES], fontsize=7)
plt.tight_layout(); plt.show()
""")
md(r"""
## 2. Contrastive learning (InfoNCE), by hand

Two noisy views of each example. For view $i$ with partner $i^+$:

$$\ell_i = -\log\frac{\exp(\mathrm{sim}(\mathbf v_i,\mathbf v_{i^+})/\tau)}{\sum_{k\ne i}\exp(\mathrm{sim}(\mathbf v_i,\mathbf v_k)/\tau)},\qquad \mathrm{sim} = \text{cosine similarity}.$$

* The loss is small when each view's partner is the most similar of all the other views.
* τ (the temperature) sharpens the contrast.
""")
code(r"""
emb = rng.normal(size=(4, 8)); views = np.concatenate([emb + 0.1 * rng.normal(size=emb.shape), emb + 0.1 * rng.normal(size=emb.shape)])
v = views / np.linalg.norm(views, axis=1, keepdims=True); sim = v @ v.T
partner = np.r_[np.arange(4, 8), np.arange(0, 4)]
for tau in (1.0, 0.2, 0.05):
    logits = sim / tau - 1e9 * np.eye(8)
    loss = np.mean([-(logits[i, partner[i]] - np.log(np.exp(logits[i]).sum())) for i in range(8)])
    print(f"τ = {tau}: InfoNCE loss {loss:.3f}")
shuffled = np.r_[np.arange(1, 8), 0]
logits = sim / 0.2 - 1e9 * np.eye(8)
print("with WRONG partners the loss is much larger:", np.mean([-(logits[i, shuffled[i]] - np.log(np.exp(logits[i]).sum())) for i in range(8)]).round(3))
""")
md(r"""
## 3. Pre-training on 400 trace-only packets (1600 traces, no labels used)

The total loss is $L = L_\text{rec} + L_\theta + 0.3\,L_\text{con}$ (reconstruction, settings prediction, contrastive).
""")
code(r"""
du, cu, _, _ = toy_scan(400, seed=7)
Xu, _ = inputs_for(du, cu, ("theta",), True, THETA_CURVES, log_frequency_spectra=True)
pre = deep.Pretrainer(epochs=20).fit(Xu["curves"], du[data.columns(du, "theta")].to_numpy(float))
fig, ax = plt.subplots(1, 3, figsize=(13, 2.8))
for a, (k, v_) in zip(ax, pre.losses.items()): a.plot(v_); a.set_title(k); a.set_xlabel("epoch")
plt.tight_layout(); plt.show()
""")
md(r"""
**What did it learn?** A linear "probe" predicts hidden quantities of the toy world from the 16-number embedding. The comparison is
against an untrained (random) encoder.
""")
code(r"""
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
z_pre = pre.encode(Xu["curves"])
rand = deep.Pretrainer(epochs=0).fit(Xu["curves"][:10]); z_rand = rand.encode(Xu["curves"])
for q in ("truth.floor", "truth.sigma1", "truth.state"):
    t = du[q].to_numpy()
    r2 = lambda z: cross_val_score(RidgeCV(), z, t, cv=5, scoring="r2").mean()
    print(f"{q:14s}: probe R² from the pre-trained embedding {r2(z_pre):5.2f}   from a random encoder {r2(z_rand):5.2f}")
""")
md(r"""
## 4. With few labels: from scratch against pre-trained

There are two ways to use the encoder:
* **fine-tuning** (`PretrainedCNN`): start the CNN from the pre-trained weights and train everything with a small learning rate;
* **frozen** (`PretrainedHead`): keep the encoder fixed and train a small network on the embedding.

Below, fine-tuning is compared with a CNN trained from scratch.
""")
code(r"""
dft, ct, Yt, _ = toy_scan(60, seed=2)
Xt, _ = inputs_for(dft, ct, ("theta", "ref"), True, THETA_CURVES, log_frequency_spectra=True)   # settings + own reference features
for n in (30, 60):
    dtr, ctr, Ytr, _ = toy_scan(n, seed=1)
    Xtr, _ = inputs_for(dtr, ctr, ("theta", "ref"), True, THETA_CURVES, log_frequency_spectra=True)
    m, s = models.CurveCNN(n_models=2).fit(Xtr, Ytr).predict(Xt); _ = report(f"CNN from scratch, {n} labelled packets", Yt, m, s)
    m, s = deep.PretrainedCNN(pre, n_models=2).fit(Xtr, Ytr).predict(Xt); _ = report(f"pre-trained + fine-tuned, {n} packets", Yt, m, s)
""")
md(r"""
**Reading the numbers:**
* The probe shows what pre-training bought: the embedding knows the single-skip noise and, unlike a random encoder, part of the noise
  floor, without seeing a single label.
* On the labels, pre-training did **not** help in this run (σe 9.7 % against 7.5 % from scratch at 30 packets; equal at 60). One training
  scan is noisy: over 5 scans in the project's test, pre-training helped at 60 packets (6.6 against 8.9 %) and no longer at 240. The
  benefit is small and not guaranteed. On real data it has to earn its place against the CNN trained from scratch.

## When to use it

* **Good:** many unlabelled traces and few labelled packets (it helped most at 30–120 packets in our tests).
* **Less useful:** with many labels, a CNN from scratch catches up.

The pre-training tasks must make the encoder notice what matters. It only learned the noise floor once the spectra were on a log-frequency
axis.

Next: **13 · Deep kernel learning (C).**
""")

# =====================================================================================================================
md, code = notebook("13_deep_kernel")
md(r"""
# 13 · Model C: deep kernel learning (a network inside a Gaussian process)

*Level: advanced. Guide Section 17. Code: `deep.DeepKernelGP`.*

**Idea.**
* The CNN finds features but needs data and has weaker error bars.
* The GP has honest error bars and needs little data, but it only uses the inputs it is given.

A deep kernel puts the network in front of the GP: the GP works on the network's 16-number embedding (plus the settings),

$$k(i,i') = \sigma_f^2\,M\bigl((g_\omega(\text{curves}_i), x_i), (g_\omega(\text{curves}_{i'}), x_{i'})\bigr) + \sigma_n^2\mathbb 1[i=i'] .$$

There is no amplifier term here (`amp_term=False`): **no amplifier labels**.
""")
code(SETUP + r"""
from ml import deep
from ml.benchmark import THETA_CURVES, inputs_for
du, cu, _, _ = toy_scan(400, seed=7)
Xu, _ = inputs_for(du, cu, ("theta",), True, THETA_CURVES, log_frequency_spectra=True)
pre = deep.Pretrainer(epochs=20).fit(Xu["curves"], du[data.columns(du, "theta")].to_numpy(float))   # as in notebook 12
""")
md(r"""
## 1. Frozen or fine-tuned encoder

* **Fine-tuned:** the encoder's weights and the GP's hyperparameters maximize the marginal likelihood **together**.
* **Frozen:** only the GP is fitted, on the fixed pre-trained embedding.

A flexible encoder can "over-fit" the marginal likelihood: it can map the training examples so that they look perfectly explained, which makes
the error bars too small. Watch the **coverage** column (target 0.68).
""")
code(r"""
dft, ct, Yt, _ = toy_scan(60, seed=2)
Xt, _ = inputs_for(dft, ct, ("theta", "ref"), True, THETA_CURVES, log_frequency_spectra=True)   # GP inputs: settings + own reference features
dtr, ctr, Ytr, _ = toy_scan(60, seed=1)
Xtr, _ = inputs_for(dtr, ctr, ("theta", "ref"), True, THETA_CURVES, log_frequency_spectra=True)
for label, finetune in (("deep kernel, encoder frozen", False), ("deep kernel, encoder fine-tuned", True)):
    dk = deep.DeepKernelGP(pre, finetune=finetune, amp_term=False)
    m, s = dk.fit({"curves": Xtr["curves"], "x": Xtr["scalars"]}, Ytr).predict({"curves": Xt["curves"], "x": Xt["scalars"]})
    _ = report(label, Yt, m, s)
m, s = models.GP().fit(Xtr["scalars"], Ytr).predict(Xt["scalars"]); _ = report("plain GP, settings + own reference features", Yt, m, s)
""")
md(r"""
**Reading the numbers:** the fine-tuned version can have a small error but claims far too small error bars (its coverage is far below
0.68). That is the over-fitting of the marginal likelihood described above. The frozen version's error bars are honest.

## When to use it

* **Good:** with a frozen pre-trained encoder, it was the best network-based model for λ and the gain in our tests, with calibrated error bars. It is a
  candidate surrogate once traces are part of the inputs.
* **Bad:** fine-tuned with few packets (over-confident).

Next: **14 · Set models for the repeated trace sets (E).**
""")

# =====================================================================================================================
md, code = notebook("14_set_models")
md(r"""
# 14 · Model E: set models for the repeated trace sets

*Level: advanced. Guide Section 18. Code: `setmodels.SetPooling`, `setmodels.SetStructured`, `setmodels.combine_sets`.*

**Idea.** A packet takes 10 trace sets at the same settings, and today a fixed rule combines them. A set network **learns** the combination. It
must do two things no fixed rule does together:
* ignore a damaged set;
* keep real variation between sets that the image also sees.
""")
code(SETUP + r"""
from ml import setmodels, structured as st
""")
md(r"""
## 1. Permutation invariance and attention pooling

The sets have no order, so the result must not depend on how they are listed. The **DeepSets** form is

$$F(\{\mathbf c_k\}) = \rho\Bigl(\tfrac1K\textstyle\sum_k \psi(\mathbf c_k)\Bigr),$$

* $\psi$ is a network applied to each set separately;
* the average pools them;
* $\rho$ is a network applied to the pooled result.

**Attention** replaces the plain average by learned weights $a_k = e^{\mathbf v^\top\psi_k}/\sum_l e^{\mathbf v^\top\psi_l}$, so a damaged set can get
weight ≈ 0.
""")
code(r"""
rng = np.random.default_rng(0)
W = rng.normal(size=(13, 6)); psi = lambda c: np.tanh(c @ W)
sets = rng.normal(size=(10, 13)); sets[4] += 4                   # set 4 is "damaged"
pool = lambda S: psi(S).mean(0)
print("same result for any order:", np.allclose(pool(sets), pool(sets[rng.permutation(10)])))
others = np.delete(psi(sets), 4, axis=0).mean(0)
v = -(psi(sets)[4] - others)                                     # a scoring vector that dislikes what makes set 4 different
score = psi(sets) @ v; a = np.exp(score) / np.exp(score).sum()
print("attention weights:", a.round(3), " → set 4 gets", a[4].round(4))
""")
md(r"""
## 2. Why a fixed rule can be wrong: intermittent pickup

A pickup with variance $\sigma_p^2$ is on during a fraction $d$ of the time.
* The 5-minute image sees $\sigma_0^2 + d\,\sigma_p^2$.
* Each 0.2-second set sees either $\sigma_0^2$ or $\sigma_0^2+\sigma_p^2$.

The **mean of the sets' variances** estimates the image's value; a **median** picks "on" or "off".
""")
code(r"""
s0, sp = 1.0, 2.0; res = {"median of sd": [], "mean of sd": [], "mean of variances": []}; true = []
for _ in range(3000):
    d = rng.uniform(0.2, 1); on = rng.random(10) < d
    sd_sets = np.sqrt(s0 ** 2 + on * sp ** 2) * (1 + 0.03 * rng.normal(size=10))
    true.append(np.sqrt(s0 ** 2 + d * sp ** 2))
    res["median of sd"].append(np.median(sd_sets)); res["mean of sd"].append(sd_sets.mean()); res["mean of variances"].append(np.sqrt(np.mean(sd_sets ** 2)))
true = np.array(true)
for k, v_ in res.items():
    r = np.log(np.array(v_) / true); print(f"{k:18s}: bias {100 * r.mean():+5.1f} %, RMS {100 * np.sqrt(np.mean(r ** 2)):4.1f} %")
""")
md(r"""
The **robust** rules drop values more than 5 robust standard deviations from the median. The robust standard deviation is 1.4826 × the median
absolute deviation (MAD), which equals σ for a Gaussian.

## 3. On the toy CCD (world v4: intermittent pickup, 10 % damaged sets)

Model A is used with the sets combined by each fixed rule, and with the learned set network. There are no amplifier labels (`kind="none"`).
""")
code(r"""
V4 = dict(within_packet_pickup=True, damaged_fraction=0.1)
te, cte = synthetic.make_dataset(60, seed=99, **V4); truth_t = np.log(te[["truth.sigma_e", "truth.lambda", "truth.gain"]].to_numpy())
tr, ctr = synthetic.make_dataset(60, seed=1, **V4); Ytr = data.make_targets(tr)[0]
prep = st.StructuredPrep().fit(tr)
Xtr = dict(prep.transform(tr, ctr), sets=ctr["sets: noise: Allan deviation"]); Xte = dict(prep.transform(te, cte), sets=cte["sets: noise: Allan deviation"])
print("sets per example:", Xtr["sets"].shape[1:], "(10 sets × 13 Allan points)")
for comb in setmodels.COMBINERS:
    m, s = setmodels.SetStructured(comb, kind="none").fit(Xtr, Ytr).predict(Xte)
    print(f"sets combined by {comb:26s}: σe error {100 * np.sqrt(np.mean((m[:, 0] - truth_t[:, 0]) ** 2)):5.1f} % (vs truth)")
""")
md(r"""
## When to use it

* **Good:** trace sets that genuinely differ (intermittent pickup, glitches), with ~60 or more labelled packets. In our 5-scan tests it was then
  better and steadier than every fixed rule.
* **Bad:** ~30 packets, where a robust fixed rule is better.

---

**That is every model in the project.** Back to the overview: `README.md` in this folder. The full test results are in
`docs/overleaf_ml_models/ml_models.tex`; level 1 runs all these models on real packets (`lvl1/`).
""")

# =====================================================================================================================
md, code = notebook("09b_physics_plus_hierarchical")
md(r"""
# 09b · Model A+B: physics-structured outputs on hierarchical Gaussian processes

*Level: advanced. Guide Sections 12–13. Code: `structured.PhysicsStructured(kind="hier")`; in level 1: `train_lvl1.py --models physics_hier`.*

**Idea.** A+B combines the two previous notebooks:
* **A (08)** gives the *structure*: the noise is measured on the trace, and only the gain, a small correction and λ are learned;
* **B (09)** gives the *learner*: each of those three functions is a hierarchical GP that knows the data come from amplifiers on modules.

$$\log\sigma_e = \tfrac12\log\!\Bigl(\frac{a^2}{N}+b^2\Bigr) - \log\hat g(x; m, j) + \hat r(x; m, j),\qquad \log\hat\lambda(x; m, j),$$

where $m$ is the module and $j$ the amplifier. Each hat is a GP with the kernel of notebook 09:

$$k = \sigma_f^2 M + \mathbb 1[m=m'](c_m^2+\sigma_m^2M) + \mathbb 1[j=j'](c_a^2+\sigma_a^2M) + \sigma_n^2\,\mathbb 1[i=i'] .$$

**Its history.**
* It was the project's first recommended model (2026-10-14): σe to 2–3 % from 15 packets, and a new module learned from 1–2 packets.
* The default then became **A without labels** (`PhysicsStructured()`): giving each amplifier's own traces instead of its identity was as good, and works for any amplifier.

A+B stays as a **labelled reference**. Level 1 runs it next to A to measure what labels add on the real CCD.

This notebook:
1. looks inside A+B;
2. compares how A+B and A represent the amplifiers;
3. compares them on known amplifiers and on a new module, including how honest their error bars are.
""")
code(SETUP + r"""
from ml import structured as st
from scipy.stats import spearmanr
TRUTH = ["truth.sigma_e", "truth.lambda", "truth.gain"]
""")
md(r"""
## 1. Inside A+B: what is shared and what belongs to each amplifier

A+B is trained on 60 packets from **two modules**; with only one module the module terms could not be learned. Each of its three GPs reports its
fitted amplitudes, in units of that target's standard deviation:
* `s` is the shared function;
* `c_module`, `s_module` are a module's offset and own shape;
* `c_amp`, `s_amp` are an amplifier's offset and own shape;
* `noise` is the measurement noise.

The smallest possible value, 0.0009, means "switched off".
""")
code(r"""
tr, ctr = synthetic.make_dataset(60, seed=1, n_boards=2); Ytr = data.make_targets(tr)[0]
prep = st.StructuredPrep().fit(tr)
AB = st.PhysicsStructured(kind="hier").fit(prep.transform(tr, ctr), Ytr)
table = pd.DataFrame({"gain  ĝ": AB.gain.hyperparameters(), "correction  r̂": AB.resid.hyperparameters(), "dark charge  λ̂": AB.lam.hyperparameters()}).T
display(table.round(3))
for name, gp in (("gain", AB.gain), ("correction", AB.resid), ("dark charge", AB.lam)):
    h = gp.hyperparameters(); total = sum(v ** 2 for k, v in h.items() if k != "noise")
    amp, mod = (h["c_amp"] ** 2 + h["s_amp"] ** 2) / total, (h["c_module"] ** 2 + h["s_module"] ** 2) / total
    print(f"{name:12s}: {100 * amp:5.1f} % of its variance belongs to the amplifier, {100 * mod:5.1f} % to the module, the rest is shared")
""")
md(r"""
**Reading it:**
* **The gain** is almost entirely an amplifier **offset** ($c_a$): each amplifier has its own gain, nearly independent of the settings. This is the
  one thing a label really helps with.
* **The correction and λ** are almost entirely **shared**: how the image's extra noise and the dark charge depend on the settings is the same
  for all amplifiers. That is why a model without labels can learn them as well.

## 2. Two ways to know the amplifier: a label, or its own traces

A+B gets the gain of each amplifier from its label. A (no labels) gets it from the amplifier's **own trace and reference features**, which
differ between amplifiers like a fingerprint. On a new scan of the same amplifiers, the two compare as follows.
""")
code(r"""
te, cte = synthetic.make_dataset(60, seed=2, n_boards=2)
truth_t = np.log(te[TRUTH].to_numpy())
A = st.PhysicsStructured().fit(prep.transform(tr, ctr), Ytr)
mA, sA = A.predict(prep.transform(te, cte)); mAB, sAB = AB.predict(prep.transform(te, cte))
amp = (te.board.astype(str) + "_ch" + te.channel.astype(str)).to_numpy()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
for a_, m_, title in ((ax[0], mAB, "A+B (told the amplifier)"), (ax[1], mA, "A (own traces, no label)")):
    for k, j in enumerate(np.unique(amp)):
        sel = amp == j; a_.scatter(np.exp(truth_t[sel, 2]), np.exp(m_[sel, 2]), s=10, label=j)
    a_.plot([0.6, 2.6], [0.6, 2.6], "k--", lw=1); a_.set_xlabel("true gain (ADU/e⁻)"); a_.set_title(title)
ax[0].set_ylabel("predicted gain"); ax[1].legend(fontsize=7, ncol=2); plt.tight_layout(); plt.show()
_ = report("A+B (labels)", truth_t, mAB, sAB); _ = report("A (no labels)", truth_t, mA, sA)
""")
md(r"""
## 3. Known amplifiers: how many packets?

Here there is one module. Three models are compared against the truth on a new scan:
* A+B;
* A without labels;
* A with the amplifier as a plain one-hot input (`kind="plain"`).
""")
code(r"""
te1, cte1 = synthetic.make_dataset(60, seed=2); truth1 = np.log(te1[TRUTH].to_numpy())
rows = []
for n in (15, 30, 60):
    d, c = synthetic.make_dataset(n, seed=1); Yd = data.make_targets(d)[0]; p = st.StructuredPrep().fit(d)
    for label, kind in (("A+B (labels)", "hier"), ("A, one-hot label", "plain"), ("A, no labels", "none")):
        m, s = st.PhysicsStructured(kind).fit(p.transform(d, c), Yd).predict(p.transform(te1, cte1))
        e = 100 * np.sqrt(np.mean((m - truth1) ** 2, axis=0))
        rows.append({"packets": n, "model": label, "σe %": e[0], "λ %": e[1], "gain %": e[2],
                     "σe coverage": np.mean(np.abs(m[:, 0] - truth1[:, 0]) <= s[:, 0])})
display(pd.DataFrame(rows).set_index(["packets", "model"]).round(2))
""")
md(r"""
## 4. A new module: 0, 1, 2 and 5 of its packets

* **Training:** a scan of module 100 plus $k$ packets of module 101.
* **Testing:** new packets of module 101.

Besides the error, three scores are reported:
* **shape**: error after removing each amplifier's mean;
* **rank**: whether the settings are put in the right order, within each amplifier;
* **coverage** of the ±1σ error bars, i.e. whether the model *knows* when it is unsure.
""")
code(r"""
def scores(df_, m, s):
    t = np.log(df_["truth.sigma_e"].to_numpy()); r = m[:, 0] - t
    a = (df_.board.astype(str) + df_.channel.astype(str)).to_numpy(); shape = r.copy()
    for j in np.unique(a): shape[a == j] -= r[a == j].mean()
    rank = np.mean([spearmanr(m[a == j, 0], t[a == j])[0] for j in np.unique(a)])
    return {"σe %": 100 * np.sqrt(np.mean(r ** 2)), "shape %": 100 * np.sqrt(np.mean(shape ** 2)), "rank": rank,
            "coverage": np.mean(np.abs(r) <= s[:, 0]), "mean σ predicted %": 100 * np.mean(s[:, 0])}

trn, ctrn = synthetic.make_dataset(80, seed=1, n_boards=2); tst, ctst = synthetic.make_dataset(40, seed=99, n_boards=2)
new = (tst.board == 101).to_numpy(); tst, ctst = tst[new].reset_index(drop=True), {k: v[new] for k, v in ctst.items()}
pk = trn.packet_id.str[4:].astype(int)
rows = []
for k in (0, 1, 2, 5):
    keep = ((trn.board == 100) | (pk < k)).to_numpy()
    d, c = trn[keep].reset_index(drop=True), {kk: v[keep] for kk, v in ctrn.items()}
    Yd = data.make_targets(d)[0]; p = st.StructuredPrep().fit(d)
    for label, kind in (("A+B (labels)", "hier"), ("A, no labels", "none")):
        m, s = st.PhysicsStructured(kind).fit(p.transform(d, c), Yd).predict(p.transform(tst, ctst))
        rows.append({"packets of the new module": k, "model": label, **scores(tst, m, s)})
res = pd.DataFrame(rows).set_index(["packets of the new module", "model"])
display(res.round(2))
""")
md(r"""
The 0-packet error is large, but it is an average. Below, the same 0-packet models are broken down **amplifier by amplifier** of the new module.
""")
code(r"""
keep = (trn.board == 100).to_numpy()
d, c = trn[keep].reset_index(drop=True), {kk: v[keep] for kk, v in ctrn.items()}
p = st.StructuredPrep().fit(d); Yd = data.make_targets(d)[0]; t = np.log(tst["truth.sigma_e"].to_numpy())
rows = []
for label, kind in (("A+B (labels)", "hier"), ("A, no labels", "none")):
    m, s = st.PhysicsStructured(kind).fit(p.transform(d, c), Yd).predict(p.transform(tst, ctst))
    for ch in range(4):
        sel = (tst.channel == ch).to_numpy(); r = m[sel, 0] - t[sel]
        rows.append({"model": label, "new amplifier": f"101_ch{ch}", "σe error %": 100 * np.sqrt(np.mean(r ** 2)),
                     "predicted σ %": 100 * np.mean(s[sel, 0]), "coverage": np.mean(np.abs(r) <= s[sel, 0])})
display(pd.DataFrame(rows).set_index(["model", "new amplifier"]).round(2))
""")
md(r"""
**Reading it:**
* **With 0 packets** of the new module, nothing can know a new amplifier's gain. Both models keep the shape and the order of the settings.
* **One amplifier causes most of the 0-packet error.** Three of the four new amplifiers are predicted within a few percent by *both* models: an
  amplifier's own traces already place it well. The fourth has an unusual gain, and both models miss it badly.
* **The error bars differ in kind:**
  * A+B knows the amplifiers are new, so it gives each of them the same wide error bar (the offset variance $c_a^2$). That is honest on
    average but too wide for the three typical amplifiers, and still too narrow for the odd one.
  * A has no notion of "new amplifier": narrow error bars, right for the typical amplifiers, and confidently wrong on the odd one.
* **After 1–2 packets** of the new module, both learn it. In this run A+B is ahead at 2 and 5 packets. Over 3 training scans in the project's
  test they were 2.0 % (A+B) and 2.2 % (A) after 2 packets.

## When to use it

| | A+B (labels) | A (no labels, default) |
|---|---|---|
| known amplifiers, few packets (≤ 30) | **better** (here σe 2.2 vs 3.7 %, gain 1.0 vs 2.6 % at 30) | worse |
| known amplifiers, 60 packets | equal for σe and λ; gain better | equal |
| a new module after 1–5 packets | equal or slightly better | equal or slightly worse |
| error bars for an amplifier never seen | wide for every new amplifier | narrow; confidently wrong on an unusual one |
| works for any amplifier without retraining its identity | no | **yes** |
| needs | the module and amplifier of every row | each amplifier's own traces |

* **A+B is worth it** when the set of amplifiers is fixed and known and packets are few. Its main gain is the **gain** itself: a per-amplifier
  constant that a label pins down directly (section 1).
* **A is the default** because it needs no labels, works for any amplifier, and catches up by ~60 packets.

The gap with few packets is the price of having no labels. Whether it exists on the real CCD, where the gains differ by only ~2 % on module 101,
is what level 1 will show.

In level 1, `physics_hier` runs A+B next to `physics` on the real packets, so the first real data will show which holds for the real CCD.

Next: **10 · Dark current and CIC, the joint fit.**
""")

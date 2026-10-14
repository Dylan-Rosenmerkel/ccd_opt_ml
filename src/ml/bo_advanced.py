"""
bo_advanced.py -- model F of docs/overleaf_ml_models/ml_models.tex: better ways to choose the next settings.

All strategies use the DECOMPOSED surrogate (benchmark_structured.decomposed_propose): one hierarchical GP for
log sigma_e and one for log lambda per good amplifier, combined into J = mean(log sigma_e + w log lambda).

* ``trust region`` (TuRBO-like): candidates only inside a box around the best point so far.  The box doubles
  after 3 improvements in a row, halves after 5 failures, and restarts when it gets too small.  It keeps
  the search local, which also suits a CCD whose voltages are changed in bounded steps.
* ``learned constraint``: the relation between parameters is NOT given to the optimizer.  A packet in the
  forbidden region fails (its noise explodes and its labels are flagged); a GP classifier (GP regression on
  +-1) learns the feasible region and the acquisition is EI x P(feasible).  Compared with a known constraint
  (ccd_limits.py) and with no constraint at all.
* ``multi-fidelity``: a trace-only packet costs a fraction ``trace_cost`` of a full packet.  It measures
  sigma_1 * g on the trace, which with the gain model gives sigma_e (model A), but no lambda.  Each round
  screens the 3 best candidates with trace-only packets, then takes ONE full packet at the best of them
  after the update.  Progress is counted in cost (full-packet equivalents).
* ``ParEGO`` (multi-objective): every step uses a random weight w for J_w = log sigma_e + w log lambda; the
  result is a Pareto front of read noise against dark charge instead of one point.  Scored by the
  hypervolume it dominates.

The simulator (``Simulator``) measures the toy world at state 0 without pickup (as benchmark.exp_bo).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from . import structured as st, synthetic
from .optimize import expected_improvement

GOOD = (0, 1, 3)


class Simulator:
    """One packet of the toy world (state 0, no pickup) on the good amplifiers; fails outside ``allowed``."""

    def __init__(self, seed: int = 0, trace_noise: float = 0.01):
        self.amps = synthetic.make_amplifiers(np.random.default_rng(1234))
        self.rng, self.trace_noise = np.random.default_rng(seed), trace_noise

    def full(self, theta):
        """(ok, labels (3, 3): log sigma_e, log lambda, log gain per good amplifier)."""
        if not synthetic.allowed(theta)[0]:
            return False, None
        out = []
        for ch in GOOD:
            ph = synthetic.physics(np.asarray(theta, float), self.amps[ch], 0.0)
            out.append((np.log(ph["sigma_e"] * (1 + 0.015 * self.rng.normal())), np.log(ph["lambda"] * (1 + 0.03 * self.rng.normal())),
                        np.log(ph["gain"] * (1 + 0.008 * self.rng.normal()))))
        return True, np.array(out)

    def trace(self, theta):
        """(ok, log of sigma_1 * g / sqrt(N) in ADU per good amplifier): what a trace-only packet measures."""
        if not synthetic.allowed(theta)[0]:
            return False, None
        out = []
        for ch in GOOD:
            ph = synthetic.physics(np.asarray(theta, float), self.amps[ch], 0.0)
            out.append(np.log(ph["sigma1"] * ph["gain"] / np.sqrt(synthetic.N_SKIPS) * (1 + self.trace_noise * self.rng.normal())))
        return True, np.array(out)

    def true_parts(self, theta):
        """Noise-free (mean log sigma_e, mean log lambda) over the good amplifiers."""
        v = np.array([[np.log(synthetic.physics(np.asarray(theta, float), self.amps[ch], 0.0)[k]) for k in ("sigma_e", "lambda")]
                      for ch in GOOD])
        return v.mean(0)


def _rows(thetas):
    return {"x": np.repeat(thetas, len(GOOD), axis=0), "amp": np.tile(np.arange(len(GOOD)), len(thetas))}


class Decomposed:
    """The decomposed surrogate; sigma_e can also learn from trace-only packets through the gain GP (model A)."""

    def fit(self, thetas, labels, trace_thetas=None, trace_logs=None):
        self.lam = st.HierGP(levels=("amp",), iters=60).fit(_rows(thetas), labels[..., 1].reshape(-1))
        X_s, y_s = _rows(thetas), labels[..., 0].reshape(-1)
        if trace_thetas is not None and len(trace_thetas):
            self.gain = st.HierGP(levels=("amp",), iters=60).fit(_rows(thetas), labels[..., 2].reshape(-1))
            g, _ = self.gain.predict(_rows(trace_thetas))
            y_t = trace_logs.reshape(-1) - g[:, 0]
            X_s = {k: np.concatenate([X_s[k], _rows(trace_thetas)[k]]) for k in X_s}
            y_s = np.concatenate([y_s, y_t])
        self.sig = st.HierGP(levels=("amp",), iters=60).fit(X_s, y_s)
        return self

    def parts(self, cand):
        (ms, ss), (ml, sl) = self.sig.predict(_rows(cand)), self.lam.predict(_rows(cand))
        k = len(GOOD)
        return (ms.reshape(-1, k), ss.reshape(-1, k), ml.reshape(-1, k), sl.reshape(-1, k))

    def J(self, cand, w=0.5):
        ms, ss, ml, sl = self.parts(cand)
        return (ms + w * ml).mean(1), np.sqrt(ss ** 2 + w ** 2 * sl ** 2).mean(1)


def _candidates(rng, low, high, n, center=None, length=1.0):
    lo, hi = low, high
    if center is not None:
        half = 0.5 * length * (high - low)
        lo, hi = np.maximum(low, center - half), np.minimum(high, center + half)
    return lo + (hi - lo) * rng.random((n, len(low)))


def run(strategy: str, seed: int, n_total: int = 50, n_initial: int = 10, trace_cost: float = 0.25, budget: float | None = None,
        n_candidates: int = 3000) -> dict:
    """
    One campaign.  strategy: 'global', 'trust region', 'known constraint', 'learned constraint', 'no constraint',
    'multi-fidelity', 'parego', 'random'.  Returns the tried points, the cost after each full packet,
    the best true J so far, the failures, and (for ParEGO) the true (log sigma_e, log lambda) of every point.
    """
    rng = np.random.default_rng(seed)
    sim = Simulator(seed + 1000)
    low, high = synthetic.LOW, synthetic.HIGH
    known = strategy not in ("learned constraint", "no constraint")
    thetas, labels, tried, ok_flags = [], [], [], []
    trace_t, trace_y = [], []
    cost, costs, failures = 0.0, [], 0
    tr = {"length": 0.4, "succ": 0, "fail": 0}
    budget = budget if budget is not None else float(n_total)

    def take(theta):
        nonlocal cost, failures
        cost += 1.0
        ok, lab = sim.full(theta)
        tried.append(theta)
        ok_flags.append(ok)
        if ok:
            thetas.append(theta)
            labels.append(lab)
        else:
            failures += 1
        costs.append(cost)

    start = synthetic.sample_theta(rng, n_initial) if known else _candidates(rng, low, high, n_initial)
    for t in start:
        take(t)
    while cost < budget - 1e-9 and len(tried) < 10 * n_total:
        T, L = np.array(thetas), np.array(labels)
        Jobs = (L[..., 0] + 0.5 * L[..., 1]).mean(1)
        best_i = int(np.argmin(Jobs))
        if strategy == "random":
            take(synthetic.sample_theta(rng, 1)[0])
            continue
        if strategy == "parego":
            w = rng.uniform(0.1, 2.0)
            Jw = (L[..., 0] + w * L[..., 1]).mean(1)
            sur = Decomposed().fit(T, L)
            cand = _candidates(rng, low, high, n_candidates)
            cand = cand[synthetic.allowed(cand)]
            mu, s = sur.J(cand, w)
            take(cand[int(np.argmax(expected_improvement(mu, s, float(Jw.min()))))])
            continue
        if strategy == "multi-fidelity":
            sur = Decomposed().fit(T, L, np.array(trace_t) if trace_t else None, np.array(trace_y) if trace_y else None)
            cand = _candidates(rng, low, high, n_candidates)
            cand = cand[synthetic.allowed(cand)]
            mu, s = sur.J(cand)
            ei = expected_improvement(mu, s, float(Jobs.min()))
            top = cand[np.argsort(-ei)[:3]]
            for t in top:
                if cost + trace_cost + 1.0 > budget + 1e-9:
                    break
                cost += trace_cost
                ok, y = sim.trace(t)
                if ok:
                    trace_t.append(t)
                    trace_y.append(y)
            sur = Decomposed().fit(T, L, np.array(trace_t), np.array(trace_y))
            mu, s = sur.J(top)
            if cost + 1.0 > budget + 1e-9:
                break
            take(top[int(np.argmax(expected_improvement(mu, s, float(Jobs.min()))))])
            continue
        sur = Decomposed().fit(T, L)
        if strategy == "trust region":
            cand = _candidates(rng, low, high, n_candidates, T[best_i], tr["length"])
        else:
            cand = _candidates(rng, low, high, n_candidates)
        if known:
            cand = cand[synthetic.allowed(cand)]
        mu, s = sur.J(cand)
        acq = expected_improvement(mu, s, float(Jobs.min()))
        if strategy == "learned constraint":
            feas = st.HierGP(levels=(), iters=60).fit({"x": np.array(tried)}, np.where(ok_flags, 1.0, -1.0))
            fm, fs = feas.predict({"x": cand})
            acq = acq * norm.cdf(fm[:, 0] / np.maximum(fs[:, 0], 1e-9))
        before = Jobs.min()
        take(cand[int(np.argmax(acq))])
        if strategy == "trust region" and ok_flags[-1]:
            new = (labels[-1][:, 0] + 0.5 * labels[-1][:, 1]).mean()
            if new < before - 1e-3:
                tr["succ"], tr["fail"] = tr["succ"] + 1, 0
            else:
                tr["succ"], tr["fail"] = 0, tr["fail"] + 1
            if tr["succ"] >= 3:
                tr["length"], tr["succ"] = min(2 * tr["length"], 1.0), 0
            if tr["fail"] >= 5:
                tr["length"], tr["fail"] = tr["length"] / 2, 0
            if tr["length"] < 0.02:
                tr["length"] = 0.4
    T, L = np.array(thetas), np.array(labels)
    Jobs = (L[..., 0] + 0.5 * L[..., 1]).mean(1)
    best_true = [synthetic.true_objective(T[int(np.argmin(Jobs[:k + 1]))]) for k in range(len(T))]
    out = {"costs": [c for c, ok in zip(costs, ok_flags) if ok], "best_true": best_true, "failures": failures,
           "n_full": len(tried), "n_trace": len(trace_t)}
    if strategy in ("parego", "random", "global"):
        out["parts"] = [sim.true_parts(t).tolist() for t in T]
    return out


def hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """Area dominated by the points (minimization of both coordinates) inside the box up to ``ref``."""
    p = points[(points[:, 0] < ref[0]) & (points[:, 1] < ref[1])]
    if not len(p):
        return 0.0
    p = p[np.argsort(p[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in p:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return float(hv)

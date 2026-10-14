"""
optimize.py -- Bayesian optimization: choose the next operating parameters to try.

The loop (docs/ml_models.tex, "Optimization"):

    1. a surrogate model (here a Gaussian process) is fitted to all packets taken so far:
       theta -> objective J = log sigma_e + w * log lambda  (smaller is better);
    2. for many candidate thetas inside the allowed region, the model gives a predicted mean mu and
       uncertainty s of J;
    3. the candidate with the largest EXPECTED IMPROVEMENT over the best J so far is chosen:

           EI = (J_best - mu) * Phi(z) + s * phi(z),   z = (J_best - mu) / s

       EI is large where the prediction is good (exploitation) OR uncertain (exploration);
    4. that packet is taken (traces + image), its labels are added, and the loop repeats.

Safety: candidates are drawn only inside the box [low, high] and, optionally, within a trust
region of +-``step`` around the best point so far (the voltages are changed in bounded steps);
``allowed`` is a hook for the relations between parameters (ccd_limits.py, still to be written):
candidates it rejects are never proposed.  On the real DAQ every proposal goes through
take_packet's limit checks again before anything is sent.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm

from .models import GP


def expected_improvement(mu: np.ndarray, s: np.ndarray, best: float, xi: float = 0.0) -> np.ndarray:
    """EI for minimization (see the module docstring); ``xi`` > 0 favours exploration."""
    s = np.maximum(s, 1e-12)
    z = (best - mu - xi) / s
    return (best - mu - xi) * norm.cdf(z) + s * norm.pdf(z)


def propose(thetas: np.ndarray, J: np.ndarray, low: np.ndarray, high: np.ndarray, rng: np.random.Generator,
            n_candidates: int = 4000, step: np.ndarray | None = None,
            allowed: Callable[[np.ndarray], np.ndarray] | None = None, model=None):
    """
    The next theta to try, given the thetas tried so far and their objective values J.

    Returns (theta_next, info) with the fitted model and the EI of the chosen point.
    """
    model = (model or GP(restarts=1)).fit(thetas, J[:, None])
    best_i = int(np.argmin(J))
    lo, hi = low, high
    if step is not None:                                            # trust region around the incumbent
        lo = np.maximum(low, thetas[best_i] - step)
        hi = np.minimum(high, thetas[best_i] + step)
    cand = lo + (hi - lo) * rng.random((n_candidates, len(low)))
    if allowed is not None:
        cand = cand[allowed(cand)]
    mu, s = model.predict(cand)
    ei = expected_improvement(mu[:, 0], s[:, 0], float(J[best_i]))
    i = int(np.argmax(ei))
    return cand[i], {"model": model, "ei": float(ei[i]), "mu": float(mu[i, 0]), "s": float(s[i, 0])}


def run_loop(measure: Callable[[np.ndarray], float], low: np.ndarray, high: np.ndarray, n_initial: int,
             n_total: int, seed: int = 0, step=None, allowed=None, strategy: str = "bo") -> dict:
    """
    Simulate an optimization campaign: ``measure(theta)`` returns the (noisy) objective of one packet.

    strategy 'bo' (Bayesian optimization after n_initial random packets) or 'random'.
    Returns the thetas, the measured J and the best-so-far curve.
    """
    rng = np.random.default_rng(seed)

    def random_allowed(k):
        out = np.empty((0, len(low)))
        while len(out) < k:
            c = low + (high - low) * rng.random((4 * k, len(low)))
            out = np.vstack([out, c[allowed(c)] if allowed is not None else c])
        return out[:k]

    thetas = list(random_allowed(n_initial))
    J = [measure(t) for t in thetas]
    while len(thetas) < n_total:
        if strategy == "bo":
            nxt, _ = propose(np.array(thetas), np.array(J), low, high, rng, step=step, allowed=allowed)
        else:
            nxt = random_allowed(1)[0]
        thetas.append(nxt)
        J.append(measure(nxt))
    J = np.array(J)
    return {"thetas": np.array(thetas), "J": J, "best_so_far": np.minimum.accumulate(J)}

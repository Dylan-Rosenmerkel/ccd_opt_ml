"""
shortlist.py -- the joint sigma_e / lambda objective and the shortlist rules of the funnel (level-2 screening), shared by
lvl1/scorecard.py (tested on the first night) and lvl1/funnel_shortlist.py (session F).

Objective of a setting, from its (predicted or measured) changes against the same reference, in log units:

    J(w) = w * dlog sigma_e + (1 - w) * dlog lambda,        0 <= w <= 1

w = 1: sigma_e only; w = 0: lambda only; w = 0.5: a given relative change of either counts the same.  Lower is better.

Rules for ordering the candidates (the shortlist is the first k):
* "weighted": by J(w);
* "pareto":   the predicted Pareto front first (no other candidate predicted lower in both sigma_e and lambda), each part
              ordered by J(w).  It keeps settings that trade one for the other near the top even when w favours one.
"""

from __future__ import annotations

import numpy as np

RULES = ("weighted", "pareto")
DEFAULT_WEIGHTS = (1.0, 0.75, 0.5, 0.25, 0.0)


def objective(d_sigma: np.ndarray, d_lambda: np.ndarray, w: float) -> np.ndarray:
    """J(w) = w dlog sigma_e + (1 - w) dlog lambda."""
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"the sigma_e weight must be in [0, 1], not {w}")
    return w * np.asarray(d_sigma, float) + (1.0 - w) * np.asarray(d_lambda, float)


def pareto_front(d_sigma: np.ndarray, d_lambda: np.ndarray) -> np.ndarray:
    """True for the candidates no other candidate beats in both (lower or equal in both, lower in one)."""
    s, l = np.asarray(d_sigma, float), np.asarray(d_lambda, float)
    dominated = ((s[None, :] <= s[:, None]) & (l[None, :] <= l[:, None])
                 & ((s[None, :] < s[:, None]) | (l[None, :] < l[:, None]))).any(1)
    return ~dominated


def order(d_sigma, d_lambda, w: float, rule: str = "weighted", rng=None) -> np.ndarray:
    """Candidate indices, best first; ties broken at random (rng) or by index."""
    j = objective(d_sigma, d_lambda, w)
    tie = rng.random(len(j)) if rng is not None else np.arange(len(j))
    if rule == "weighted":
        return np.lexsort((tie, j))
    if rule == "pareto":
        return np.lexsort((tie, j, ~pareto_front(d_sigma, d_lambda)))
    raise ValueError(f"rule must be one of {RULES}")


def weight_label(w: float) -> str:
    return f"{w:g}:{1 - w:g}"

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


# --------------------------------------------------------------------------- sorting pairwise comparisons
#: ways of turning the pairwise predictions of many screens into one order (lvl1/rank_methods.py compares them)
SORTS = ("mean change", "wins", "bradley-terry")


def antisymmetric(D: np.ndarray) -> np.ndarray:
    """D[i, j] = predicted change i -> j (log y_j - log y_i; NaN = not compared).  Averages the two directions where
    both were predicted: M = (D - D^T) / 2."""
    M = np.where(np.isfinite(D) & np.isfinite(D.T), (D - D.T) / 2, np.where(np.isfinite(D), D, -D.T))
    np.fill_diagonal(M, np.nan)
    return M


def sort_scores(D: np.ndarray, method: str = "mean change", n_iter: int = 200) -> np.ndarray:
    """One score per screen from the pairwise predicted changes D (n x n, see antisymmetric); lower = better.
    * "mean change": the mean predicted change from every other screen to this one (the least-squares ranking of the
      differences; uses their sizes);
    * "wins": the fraction of the other screens predicted better than this one minus those predicted worse (a
      round-robin tournament; direction only);
    * "bradley-terry": the strength p_i of P(i beats j) = p_i / (p_i + p_j) fitted to the predicted wins (Hunter's MM
      iteration, half a win against a virtual average opponent so a screen that wins or loses everything stays
      finite); score = -log p."""
    M = antisymmetric(D)
    if method == "mean change":
        return np.nanmean(M, 0)
    if method == "wins":
        return np.nanmean(np.sign(M), 0)
    if method == "bradley-terry":
        n = len(M)
        beats = np.nan_to_num(M > 0)               # M[i, j] > 0: j is worse, i beats j
        played = np.isfinite(M).astype(float)
        W = beats.sum(1) + 0.5
        p = np.ones(n)
        for _ in range(n_iter):
            denom = (played / (p[:, None] + p[None, :])).sum(1) + 1.0 / (p + 1.0)
            p = W / denom
            p /= np.exp(np.mean(np.log(p)))
        return -np.log(p)
    raise ValueError(f"sort must be one of {SORTS}")

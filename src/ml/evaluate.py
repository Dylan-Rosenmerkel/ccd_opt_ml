"""
evaluate.py -- honest scores for the models: cross-validation by packet, errors, uncertainty calibration.

Rules (docs/ml_training_plan.tex, "Validation rules"):
* the rows of one packet (its amplifiers) are always together, in training OR in test
  (``grouped_cv``); otherwise a model is tested on a packet it has partly seen;
* a harder test holds out whole amplifiers (``groups_of(df, 'amplifier')``): can the model predict
  an amplifier it has never seen?

Scores per target (targets are logs, so errors are relative):
* ``rel_error``  root-mean-square error of log(prediction / truth): 0.05 = 5 %
* ``r2``         fraction of the spread of the target explained (1 = perfect, 0 = no better than the mean)
* ``coverage``   fraction of test rows whose truth lies within mean +- 1 std (0.68 if calibrated)
* ``nll``        mean negative log-likelihood of the truth under N(mean, std^2) (lower = better)
* ``crps``       continuous ranked probability score of N(mean, std^2) (``crps_gaussian``; lower = better): a proper
                 score in the units of the target that rewards both accuracy and honest error bars

Stronger validation (docs/literature/landscape_grounded.md, H1):
* ``setting_groups``      packets taken at identical settings (the nominal repeats, the replicates) form one group,
                          so a test packet's twin is never in the training set;
* ``bootstrap_weights``   packet-bootstrap resamples of the out-of-fold predictions (no refitting), for intervals on
                          every score and for paired differences between models on the same resamples;
* ``percentile_interval`` the 68 % (16-84) and 95 % (2.5-97.5) ranges of a bootstrap distribution.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.stats import norm

from .models import n_rows, take


def grouped_folds(groups: np.ndarray, n_splits: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Split the unique groups at random into n_splits folds; return the test-row indices of each fold."""
    unique = np.unique(groups)
    order = np.random.default_rng(seed).permutation(unique)
    folds = np.array_split(order, n_splits)
    return [np.flatnonzero(np.isin(groups, f)) for f in folds if len(f)]


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """CRPS of N(mu, sigma^2) at y, element-wise (Gneiting & Raftery 2007): sigma [z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)].

    In the units of y (here log units, i.e. relative); it reduces to |y - mu| as sigma -> 0.  Lower is better."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-12)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def setting_groups(theta: np.ndarray, packet_id: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """One group label per row: packets whose settings are equal share a group (nominal repeats, replicates).

    The settings of a packet are the median of its rows' ``theta``.  Two packets are "equal" when every setting
    column differs by at most ``tol`` times that column's range over all packets (or by a negligible absolute and
    relative amount; columns that never change are ignored; NaN equals NaN only).  Equality is made transitive
    (a chain of equal packets is one group).  The label is the group's first packet id, joined with '+' to the
    others, e.g. 'L1_01+L1_07+L1_14' for the nominal packets; a packet with unique settings keeps its own id."""
    theta, packet_id = np.asarray(theta, float), np.asarray(packet_id).astype(str)
    if theta.ndim == 1:
        theta = theta[:, None]
    uniq = np.unique(packet_id)
    with warnings.catch_warnings():                                # all-NaN columns
        warnings.simplefilter("ignore")
        T = np.array([np.nanmedian(theta[packet_id == p], 0) for p in uniq]).reshape(len(uniq), -1)
        span = np.nanmax(T, 0) - np.nanmin(T, 0) if len(T) else np.zeros(T.shape[1])
    span = np.where(np.isfinite(span), span, 0.0)
    parent = list(range(len(uniq)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(uniq)):
        a = T[i]
        d = np.abs(T[i + 1:] - a)
        lim = np.maximum(tol * span, 1e-9 + 1e-6 * np.abs(a))
        both_nan = np.isnan(T[i + 1:]) & np.isnan(a)
        same = np.where(both_nan, True, d <= lim)                 # NaN vs a number: different
        for j in np.flatnonzero(same.all(1)) + i + 1:
            parent[find(j)] = find(i)
    roots = np.array([find(i) for i in range(len(uniq))])
    label = {}
    for r in np.unique(roots):
        label[r] = "+".join(uniq[roots == r])
    by_packet = {p: label[r] for p, r in zip(uniq, roots)}
    return np.array([by_packet[p] for p in packet_id])


def bootstrap_weights(packet_id: np.ndarray, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Packet-bootstrap resamples as row weights, shape (n_boot, n_rows).

    Each resample draws as many packets as there are, with replacement; a row's weight is how often its packet was
    drawn (0, 1, 2, ...).  A weighted score over the rows equals the score on the resampled data set.  Use the same
    weights for every model, so that differences between models are paired."""
    packet_id = np.asarray(packet_id).astype(str)
    uniq, inv = np.unique(packet_id, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), (n_boot, len(uniq)))
    counts = np.zeros((n_boot, len(uniq)))
    np.add.at(counts, (np.repeat(np.arange(n_boot), len(uniq)), draws.ravel()), 1)
    return counts[:, inv]


def percentile_interval(samples: np.ndarray) -> dict[str, list[float] | None]:
    """{'68': [16th, 84th percentile], '95': [2.5th, 97.5th]} of a bootstrap distribution (NaN draws ignored)."""
    x = np.asarray(samples, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"68": None, "95": None}
    q = np.percentile(x, [16, 84, 2.5, 97.5])
    return {"68": [float(q[0]), float(q[1])], "95": [float(q[2]), float(q[3])]}


def grouped_cv(make_model, X, Y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 0):
    """Out-of-fold predictions (mean, std) for every row: each row is predicted by a model that never saw its group."""
    mean, std = np.full(Y.shape, np.nan), np.full(Y.shape, np.nan)
    for test in grouped_folds(groups, n_splits, seed):
        train = np.setdiff1d(np.arange(n_rows(X)), test)
        model = make_model().fit(take(X, train), Y[train])
        m, s = model.predict(take(X, test))
        mean[test], std[test] = m, s
    return mean, std


def scores(Y: np.ndarray, mean: np.ndarray, std: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    """The five scores per target (see the module docstring)."""
    out = {}
    for t, name in enumerate(names):
        y, m, s = Y[:, t], mean[:, t], np.maximum(std[:, t], 1e-9)
        ok = np.isfinite(y) & np.isfinite(m)
        y, m, s = y[ok], m[ok], s[ok]
        err = m - y
        out[name] = {"rel_error": float(np.sqrt(np.mean(err ** 2))),
                     "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
                     "coverage": float(np.mean(np.abs(err) <= s)),
                     "nll": float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + 0.5 * (err / s) ** 2)),
                     "crps": float(np.mean(crps_gaussian(y, m, s)))}
    return out


def evaluate(make_model, X, Y, groups, names, n_splits=5, seed=0) -> dict:
    """grouped_cv + scores + the wall time."""
    t0 = time.time()
    mean, std = grouped_cv(make_model, X, Y, groups, n_splits, seed)
    return {"scores": scores(Y, mean, std, names), "seconds": time.time() - t0, "mean": mean, "std": std}


def subset_packets(groups: np.ndarray, n_packets: int, seed: int = 0) -> np.ndarray:
    """Row indices of a random subset of n_packets packets (for learning curves)."""
    unique = np.unique(groups)
    chosen = np.random.default_rng(seed).choice(unique, min(n_packets, len(unique)), replace=False)
    return np.flatnonzero(np.isin(groups, chosen))


def train_test_by_packets(make_model, X_train, Y_train, X_test, Y_test, names):
    """Fit on one scan, score on another (an independent test set, e.g. a second synthetic scan)."""
    t0 = time.time()
    model = make_model().fit(X_train, Y_train)
    mean, std = model.predict(X_test)
    return {"scores": scores(Y_test, mean, std, names), "seconds": time.time() - t0, "model": model}

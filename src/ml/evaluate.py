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
"""

from __future__ import annotations

import time

import numpy as np

from .models import n_rows, take


def grouped_folds(groups: np.ndarray, n_splits: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Split the unique groups at random into n_splits folds; return the test-row indices of each fold."""
    unique = np.unique(groups)
    order = np.random.default_rng(seed).permutation(unique)
    folds = np.array_split(order, n_splits)
    return [np.flatnonzero(np.isin(groups, f)) for f in folds if len(f)]


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
    """The four scores per target (see the module docstring)."""
    out = {}
    for t, name in enumerate(names):
        y, m, s = Y[:, t], mean[:, t], np.maximum(std[:, t], 1e-9)
        ok = np.isfinite(y) & np.isfinite(m)
        y, m, s = y[ok], m[ok], s[ok]
        err = m - y
        out[name] = {"rel_error": float(np.sqrt(np.mean(err ** 2))),
                     "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
                     "coverage": float(np.mean(np.abs(err) <= s)),
                     "nll": float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + 0.5 * (err / s) ** 2))}
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

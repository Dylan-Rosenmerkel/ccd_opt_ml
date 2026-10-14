"""
calibrate.py -- model H: conformal calibration of any model's error bars.

A model's std is only useful if it is honest: 68 % of the truths inside mean +- std, 95 % inside
mean +- 1.96 std.  The first round showed the boosted trees' quantile std too narrow (coverage 0.4-0.6)
and the GP and networks roughly right.  Conformal calibration fixes any model without changing it:

    1. split the training packets into K folds (by packet, as everywhere);
    2. fit on K-1 folds, predict the held-out fold, and keep the normalized errors |y - mean| / std;
    3. the factor q_c = the c-quantile of those normalized errors (with the finite-sample correction
       ceil((n+1) c) / n) is what the std must be multiplied by for coverage c;
    4. fit once more on all training packets; predict with std * q.

Under exchangeability (new packets like the training packets) the coverage is guaranteed to be at
least c on average, whatever the model.  It costs K extra fits.  It does not fix a model that is
wrong in a NEW situation (a new module): there the calibration packets are not like the test packets.
"""

from __future__ import annotations

import numpy as np

from .evaluate import grouped_folds
from .models import n_rows, take


def conformal_factor(z: np.ndarray, coverage: float) -> float:
    """The finite-sample conformal quantile of the normalized errors z (>= 0) for the given coverage."""
    z = np.sort(z[np.isfinite(z)])
    n = len(z)
    k = int(np.ceil((n + 1) * coverage)) - 1
    return float(z[min(max(k, 0), n - 1)])


class Conformal:
    """
    Wrap a model factory: ``Conformal(make_model).fit(X, Y, groups, extra)``; ``predict`` returns
    (mean, std68) where std68 is the model's std times the 68 % factor; ``interval(X, 0.95)`` gives
    the half-width for 95 %.  ``extra`` (e.g. the dark/CIC labels) is passed on to the model's fit.
    """

    def __init__(self, make_model, n_folds: int = 5, seed: int = 0):
        self.make_model, self.n_folds, self.seed = make_model, n_folds, seed

    def _fit(self, X, Y, extra, idx):
        model = self.make_model()
        args = (take(X, idx), Y[idx]) + ((extra[idx],) if extra is not None else ())
        return model.fit(*args)

    def fit(self, X, Y, groups, extra=None):
        z = np.full(Y.shape, np.nan)
        for test in grouped_folds(np.asarray(groups), self.n_folds, self.seed):
            train = np.setdiff1d(np.arange(n_rows(X)), test)
            m, s = self._fit(X, Y, extra, train).predict(take(X, test))
            z[test] = np.abs(Y[test] - m) / np.maximum(s, 1e-12)
        self.z = z
        self.factors = {c: np.array([conformal_factor(z[:, t], c) for t in range(Y.shape[1])]) for c in (0.68, 0.95)}
        self.model = self._fit(X, Y, extra, np.arange(n_rows(X)))
        return self

    def predict(self, X):
        m, s = self.model.predict(X)
        return m, s * self.factors[0.68]

    def interval(self, X, coverage: float = 0.95):
        m, s = self.model.predict(X)
        return m, s * self.factors[coverage]


def coverage_report(Y, mean, std_raw, factors=None) -> dict:
    """Fraction of rows within the 68 % and 95 % intervals, raw (Gaussian 1 and 1.96 std) and conformal."""
    out = {}
    for c, g in ((0.68, 1.0), (0.95, 1.96)):
        err = np.abs(Y - mean)
        out[f"raw {c}"] = np.mean(err <= g * std_raw, axis=0).tolist()
        if factors is not None:
            out[f"conformal {c}"] = np.mean(err <= factors[c] * std_raw, axis=0).tolist()
    return out

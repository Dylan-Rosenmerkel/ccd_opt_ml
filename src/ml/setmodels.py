"""
setmodels.py -- model E of docs/overleaf_ml_models/ml_models.tex: learn how to combine the repeated trace sets.

A packet has 10 trace sets at theta.  Today they are combined by hand before any model sees them (the
median of the curves; for the features, a mean after rejecting values beyond 5 x 1.4826 MAD).  Two things
can make any fixed rule wrong at once:

* damaged sets (a glitch, a burst) must be ignored;
* real variation between sets must be KEPT: an outside source that is on during some sets and off during
  others is seen by the 4.8-min image as its time average, so the right combination of the sets' noise is
  the mean of their VARIANCES, not a median (which picks "on" or "off") nor a mean of deviations.

Model E puts a small permutation-invariant network (DeepSets with attention pooling) where the fixed rule
was.  It reads the Allan deviation of every set and outputs the read noise in ADU that the image will see,
log sigma_ADU(N).  It sits inside model A: log sigma_e = (set network) - log g_hat + r_hat(x), and it is
trained on log sigma_e + log g with the LABELED gain, so it learns only the combination (the same for every
setting and amplifier), which takes few packets.

``combine_sets`` has the fixed rules for comparison; ``SetStructured`` is A+B with any of them.
"""

from __future__ import annotations

import numpy as np

from . import structured as st
from .models import _device, _MeanVarHead, _Scaler, _torch

COMBINERS = ("mean", "median", "robust mean", "mean of variances", "robust mean of variances", "learned")


def combine_sets(sets: np.ndarray, how: str) -> np.ndarray:
    """
    Combine per-set curves (n, n_sets, m) into one curve (n, m) by a fixed rule:
    'mean', 'median', 'robust mean' (process_packet's rule: per point, drop values beyond 5 x 1.4826 MAD
    from the median, then the mean), 'mean of variances' (sqrt of the mean of squares) and its robust form.
    """
    if how == "mean":
        return sets.mean(1)
    if how == "median":
        return np.median(sets, axis=1)
    squared = how.endswith("variances")
    v = sets ** 2 if squared else sets
    if how.startswith("robust"):
        med = np.median(v, axis=1, keepdims=True)
        mad = 1.4826 * np.median(np.abs(v - med), axis=1, keepdims=True)
        keep = np.abs(v - med) <= 5 * np.maximum(mad, 1e-12)
        out = np.where(keep, v, 0).sum(1) / np.maximum(keep.sum(1), 1)
    else:
        out = v.mean(1)
    return np.sqrt(out) if squared else out


class SetPooling(_MeanVarHead):
    """
    The set network: per set psi(log Allan curve) -> 64 numbers; pooled by a mean and by an attention
    (softmax over the sets of a learned score); rho(pooled, base) -> mean and log variance of log sigma_ADU(N).
    ``base`` is the physics value from the median curve, so the network learns a correction.  Ensemble of
    ``n_models``; trained with the Gaussian likelihood (models._MeanVarHead).
    """

    def __init__(self, n_models: int = 5, epochs: int = 400, lr: float = 2e-3, weight_decay: float = 1e-3, batch: int = 64,
                 patience: int = 40, val_fraction: float = 0.15, hidden: int = 64):
        self.n_models, self.epochs, self.lr, self.weight_decay = n_models, epochs, lr, weight_decay
        self.batch, self.patience, self.val_fraction, self.hidden = batch, patience, val_fraction, hidden
        self.device = _device(prefer_gpu=False)

    def _net(self, m):
        torch = _torch()
        nn, h = torch.nn, self.hidden

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.psi = nn.Sequential(nn.Linear(m, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
                s.score = nn.Linear(h, 1)
                s.rho = nn.Sequential(nn.Linear(2 * h + 1, h), nn.SiLU())
                s.mean, s.logvar = nn.Linear(h, 1), nn.Linear(h, 1)

            def forward(s, sets, base):
                e = s.psi(sets)                                   # (n, n_sets, h)
                w = torch.softmax(s.score(e), dim=1)              # attention over the sets
                pooled = torch.cat([e.mean(1), (w * e).sum(1), base], dim=1)
                z = s.rho(pooled)
                return s.mean(z), s.logvar(z).clamp(-12, 6)
        return Net()

    def _prep(self, sets, base, fit=False):
        torch = _torch()
        logs = np.log(np.maximum(sets, 1e-12))
        if fit:
            self.s_mean, self.s_sd = logs.mean((0, 1)), logs.std((0, 1)) + 1e-9
            self.sb = _Scaler().fit(base[:, None])
        st_ = torch.tensor((logs - self.s_mean) / self.s_sd, dtype=torch.float32)
        bt = torch.tensor(self.sb(base[:, None]), dtype=torch.float32)
        return st_, bt

    def fit(self, sets, base, y, groups=None):
        """``groups``: optional group id per row (packet ids) for the early-stopping split (``_train_one``)."""
        self.sy = _Scaler().fit(y[:, None])
        st_, bt = self._prep(sets, base, fit=True)
        self.nets = [self._train_one(self._net(sets.shape[2]), [st_, bt], self.sy(y[:, None]), seed, groups)
                     for seed in range(self.n_models)]
        return self

    def predict(self, sets, base):
        m, s = self._ensemble_predict(list(self._prep(sets, base)))
        return m[:, 0], s[:, 0]


class SetStructured(st.PhysicsStructured):
    """
    A+B (structured.PhysicsStructured) with the trace sets combined by ``combiner`` (one of COMBINERS).
    X needs "sets" (n, n_sets, 13): every set's Allan deviation (synthetic: curves["sets: noise: Allan deviation"]).
    """

    def __init__(self, combiner: str = "learned", kind: str = "hier", allan_ns=None, allan_max_n: float | None = None,
                 pool_kw: dict | None = None, **kw):
        """allan_max_n: fit the Allan curves only up to this n (real traces: n_skips / 5, see structured.allan_fit);
        pool_kw: arguments of the set network (SetPooling), e.g. fewer networks and epochs for a quick run."""
        super().__init__(kind=kind, use_components=False, **kw)
        from . import synthetic
        self.combiner, self.allan_ns = combiner, synthetic.ALLAN_N if allan_ns is None else np.asarray(allan_ns)
        self.allan_max_n, self.pool_kw = allan_max_n, pool_kw or {}
        self.name = f"A+B, sets combined: {combiner}"

    def _fixed(self, sets, how, X=None):
        a, b = st.allan_fit(combine_sets(sets, how), self.allan_ns, max_n=self.allan_max_n)
        n = self.skips(X) if X is not None else self.n_skips
        return 0.5 * np.log(np.maximum(a ** 2 / n + b ** 2, 1e-30))

    def physics_log_sigma_adu(self, X):
        if self.combiner != "learned":
            return self._fixed(X["sets"], self.combiner, X)
        return self.pool.predict(X["sets"], self._fixed(X["sets"], "median", X))[0]

    def fit(self, X, Y, components=None):
        if self.combiner == "learned":
            self.pool = SetPooling(**self.pool_kw).fit(X["sets"], self._fixed(X["sets"], "median", X), Y[:, 0] + Y[:, 2],
                                                       groups=X.get("packet"))
        return super().fit(X, Y, components)

    def predict(self, X):
        mean, std = super().predict(X)
        if self.combiner == "learned":
            _, s_pool = self.pool.predict(X["sets"], self._fixed(X["sets"], "median", X))
            std[:, 0] = np.sqrt(std[:, 0] ** 2 + s_pool ** 2)
        return mean, std

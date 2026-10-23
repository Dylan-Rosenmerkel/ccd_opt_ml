"""
models.py -- the model types compared in docs/ml_models.tex, all with the same interface:

    model.fit(X, Y)                  X: inputs (n, p) -- or a dict {"curves": (n, C, L), "scalars": (n, p)}
                                     Y: targets (n, T), e.g. log sigma_e, log lambda, log gain
    mean, std = model.predict(X)     both (n, T); std is the model's own uncertainty (1 sigma)

The networks (MLPEnsemble, CurveCNN; also setmodels.SetPooling, deep.PretrainedCNN) take ``fit(X, Y, groups=...)``:
with group ids (packet ids) their early-stopping split holds out whole groups.

| class          | what it is                                      | strength                         |
|----------------|-------------------------------------------------|----------------------------------|
| BayesRidge     | linear in the inputs, Bayesian                  | baseline; tiny data; readable    |
| GBT            | gradient-boosted decision trees                 | nonlinear, robust, any scale     |
| GP             | Gaussian process, one length scale per input    | small data, calibrated std, BO   |
| MLPEnsemble    | 5 small neural networks, each predicts mean+var | smooth nonlinear, moderate data  |
| CurveCNN       | 1D convolutional network on the trace curves    | learns its own features          |

Inputs are standardized inside every model with the TRAINING rows only (no leakage).
"""

from __future__ import annotations

import warnings

import numpy as np

# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


class _Scaler:
    """Standardize columns with the training mean and standard deviation."""

    def fit(self, X):
        self.mean = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0)
        self.sd[~np.isfinite(self.sd) | (self.sd == 0)] = 1.0
        return self

    def __call__(self, X):
        return (X - self.mean) / self.sd


def take(X, idx):
    """Rows ``idx`` of an input (array or dict of arrays)."""
    if isinstance(X, dict):
        return {k: (v[idx] if v is not None else None) for k, v in X.items()}
    return X[idx]


def n_rows(X) -> int:
    """Number of rows of an input (array, or dict of arrays with optional None entries)."""
    if isinstance(X, dict):
        return len(X["curves"]) if "curves" in X else len(next(v for v in X.values() if v is not None))
    return len(X)


# --------------------------------------------------------------------------- #
# scikit-learn models                                                          #
# --------------------------------------------------------------------------- #


class BayesRidge:
    """Bayesian linear regression (one per target): y = w . x + b, with a Gaussian prior on w."""
    name = "Bayesian ridge"

    def fit(self, X, Y):
        from sklearn.linear_model import BayesianRidge
        self.sx = _Scaler().fit(X)
        self.models = [BayesianRidge().fit(self.sx(X), Y[:, t]) for t in range(Y.shape[1])]
        return self

    def predict(self, X):
        out = [m.predict(self.sx(X), return_std=True) for m in self.models]
        return np.column_stack([o[0] for o in out]), np.column_stack([o[1] for o in out])


class GBT:
    """
    Gradient-boosted trees (one per target): a sum of a few hundred small decision trees, each
    fitted to the errors left by the previous ones.  The uncertainty comes from two extra models
    fitted to the 16 % and 84 % quantiles: std = (q84 - q16) / 2.
    """
    name = "boosted trees"

    def __init__(self, max_iter: int = 300, learning_rate: float = 0.05, max_leaf_nodes: int = 15,
                 min_samples_leaf: int = 10):
        self.kw = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                       min_samples_leaf=min_samples_leaf, random_state=0)

    def fit(self, X, Y):
        from sklearn.ensemble import HistGradientBoostingRegressor as H
        from threadpoolctl import threadpool_limits
        self.models = []
        # one thread: on a few hundred rows, OpenMP thread start-up costs 100x the fit itself
        with threadpool_limits(1):
            for t in range(Y.shape[1]):
                y = Y[:, t]
                self.models.append((H(**self.kw).fit(X, y),
                                    H(loss="quantile", quantile=0.16, **self.kw).fit(X, y),
                                    H(loss="quantile", quantile=0.84, **self.kw).fit(X, y)))
        return self

    def predict(self, X):
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            mean = np.column_stack([m[0].predict(X) for m in self.models])
            std = np.column_stack([np.abs(m[2].predict(X) - m[1].predict(X)) / 2 for m in self.models])
        return mean, np.maximum(std, 1e-6)


class GP:
    """
    Gaussian process (one per target).  The prediction at a new point is a weighted average of the
    training points, weighted by how similar the inputs are; the kernel

        k(x, x') = c * Matern_5/2(|x - x'| / l)  +  white noise

    has ONE LENGTH SCALE l_i PER INPUT ("automatic relevance determination"): an input whose
    length scale grows large does not matter.  The std is exact under the model's assumptions,
    which is what Bayesian optimization needs.  Cost grows as n^3, so above ``max_points``
    training rows a random subset is used (GPs are for the small-data regime anyway).
    """
    name = "Gaussian process"

    def __init__(self, max_points: int = 400, restarts: int = 0, max_iter: int = 150, seed: int = 0):
        self.max_points, self.restarts, self.max_iter, self.seed = max_points, restarts, max_iter, seed

    def _optimizer(self, obj_func, initial_theta, bounds):
        """L-BFGS on the log marginal likelihood, capped at max_iter steps (many inputs = many length scales)."""
        from scipy.optimize import minimize
        res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
                       options={"maxiter": self.max_iter})
        return res.x, res.fun

    def fit(self, X, Y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        self.sx = _Scaler().fit(X)
        Xs = self.sx(X)
        if len(Xs) > self.max_points:
            idx = np.random.default_rng(self.seed).choice(len(Xs), self.max_points, replace=False)
            Xs, Y = Xs[idx], Y[idx]
        self.models = []
        for t in range(Y.shape[1]):
            kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(Xs.shape[1]), (1e-2, 1e3), nu=2.5)
                      + WhiteKernel(1e-2, (1e-6, 1e1)))
            gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=self.restarts,
                                          random_state=self.seed, optimizer=self._optimizer)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models.append(gp.fit(Xs, Y[:, t]))
        return self

    def predict(self, X):
        out = [m.predict(self.sx(X), return_std=True) for m in self.models]
        return np.column_stack([o[0] for o in out]), np.column_stack([o[1] for o in out])

    def length_scales(self, t: int = 0) -> np.ndarray:
        """The fitted length scale of every (standardized) input for target t: small = important."""
        return np.atleast_1d(self.models[t].kernel_.k1.k2.length_scale)


# --------------------------------------------------------------------------- #
# neural networks (PyTorch)                                                    #
# --------------------------------------------------------------------------- #


def _torch():
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return torch


def _device(prefer_gpu: bool):
    """Apple GPU (MPS) or CUDA if available and wanted, else CPU.  (This PyTorch build has no optimized
    CPU convolutions: a CNN step is ~100x faster on the Mac's GPU.)"""
    torch = _torch()
    if prefer_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _MeanVarHead:
    """Mixin: train an ensemble of networks that output (mean, log variance) with the Gaussian NLL."""

    #: standardized inputs are clipped to +-INPUT_CLIP (a point that hardly varied among few training rows must not
    #: become a huge input); a mean beyond the training labels' range by more than OUTPUT_MARGIN training sd is pulled
    #: back to that edge and the distance added to its sd (with very few rows a network can extrapolate wildly)
    INPUT_CLIP, OUTPUT_MARGIN = 10.0, 3.0

    def _train_one(self, net, inputs, Y, seed, groups=None):
        """Train one network with the Gaussian NLL and early stopping on an internal validation split.

        Without ``groups``: ``val_fraction`` of the rows are held out, and the network of the best validation epoch is
        kept.  With ``groups`` (packet ids): ``val_fraction`` of the GROUPS are held out, because the rows of one packet
        share its noise and settings, so a row split validates on near-copies of training rows.  Holding out whole
        packets removes their settings from training, so the packet split only chooses the NUMBER of epochs; the network
        is then retrained from the same initial weights on all rows for that many epochs.  The stopping criterion is the
        validation mean squared error (``stop_metric``, default "mse"; "nll" = the Gaussian NLL): on unseen packets the
        NLL rises early, as the variance becomes overconfident, while the mean is still improving, and NLL stopping cost
        nn 10.5 -> 15.3 % on lambda (lvl1/test_data); with MSE it is 10.6 %, pretrained 11.6 -> 10.3 % (2026-10-23)."""
        torch = _torch()
        torch.manual_seed(seed)
        self._y_range = (np.nanmin(Y, 0), np.nanmax(Y, 0))          # standardized training labels
        n = len(Y)
        rng = np.random.default_rng(seed)
        if groups is not None:
            groups = np.asarray(groups)
            ug = np.unique(groups)
            pick = rng.random(len(ug)) < self.val_fraction
            if not pick.any() and len(ug) > 2:
                pick[rng.integers(len(ug))] = True                 # at least one validation group
            val = np.isin(groups, ug[pick])
        else:
            val = rng.random(n) < self.val_fraction                # internal early-stopping split
        if val.sum() < 2 or (~val).sum() < 2:
            val = np.zeros(n, bool)
        tr_idx, va_idx = np.flatnonzero(~val), np.flatnonzero(val)
        net = net.to(self.device)
        init_state = {k: x.clone() for k, x in net.state_dict().items()}
        y = torch.tensor(Y, dtype=torch.float32, device=self.device)

        def nll(idx):
            mean, logvar = net(*[t[idx] if t is not None else None for t in inputs])
            return 0.5 * (logvar + (y[idx] - mean) ** 2 / logvar.exp()).mean()

        def train(idx, n_epochs, val_idx):
            """Train on rows ``idx``; with ``val_idx`` stop early and return (best epoch, its weights)."""
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            best, best_epoch, best_state, patience = np.inf, n_epochs - 1, None, 0
            for epoch in range(n_epochs):
                net.train()
                perm = rng.permutation(idx)
                for start in range(0, len(perm), self.batch):
                    loss = nll(perm[start:start + self.batch])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                if len(val_idx):
                    net.eval()
                    with torch.no_grad():
                        if getattr(self, "stop_metric", "mse") == "mse":
                            m_, _ = net(*[t[val_idx] if t is not None else None for t in inputs])
                            v = float(((y[val_idx] - m_) ** 2).mean())
                        else:
                            v = float(nll(val_idx))
                    if v < best - 1e-4:
                        best, best_epoch, best_state, patience = v, epoch, {k: x.clone() for k, x in net.state_dict().items()}, 0
                    else:
                        patience += 1
                        if patience >= self.patience:
                            break
            return best_epoch, best_state

        best_epoch, best_state = train(tr_idx, self.epochs, va_idx)
        if groups is not None and len(va_idx):
            net.load_state_dict(init_state)                        # refit on all rows for the chosen number of epochs
            torch.manual_seed(seed)
            train(np.arange(n), best_epoch + 1, np.array([], int))
        elif best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def _ensemble_predict(self, inputs):
        torch = _torch()
        means, variances = [], []
        with torch.no_grad():
            for net in self.nets:
                m, lv = net(*inputs)
                means.append(m.cpu().numpy())
                variances.append(lv.exp().cpu().numpy())
        means, variances = np.array(means), np.array(variances)
        mean = means.mean(0)
        var = variances.mean(0) + means.var(0)                       # aleatoric + disagreement
        if getattr(self, "_y_range", None) is not None:
            lo, hi = self._y_range[0] - self.OUTPUT_MARGIN, self._y_range[1] + self.OUTPUT_MARGIN
            clipped = np.clip(mean, lo, hi)
            var = var + (mean - clipped) ** 2
            mean = clipped
        return mean * self.sy.sd + self.sy.mean, np.sqrt(var) * self.sy.sd


class MLPEnsemble(_MeanVarHead):
    """
    A "deep ensemble": ``n_models`` small fully connected networks (2 hidden layers), each trained
    from a different random start to predict a mean and a variance per target.  The spread of the
    ensemble plus the predicted variances give the uncertainty.
    """
    name = "neural-net ensemble"

    def __init__(self, hidden: int = 64, n_models: int = 5, epochs: int = 600, lr: float = 3e-3,
                 weight_decay: float = 1e-3, batch: int = 64, patience: int = 60, val_fraction: float = 0.15):
        self.hidden, self.n_models, self.epochs, self.lr = hidden, n_models, epochs, lr
        self.weight_decay, self.batch, self.patience, self.val_fraction = weight_decay, batch, patience, val_fraction
        self.device = _device(prefer_gpu=False)                     # small dense layers: CPU is fine

    def _net(self, p, T):
        torch = _torch()
        nn = torch.nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.body = nn.Sequential(nn.Linear(p, self.hidden), nn.SiLU(), nn.Linear(self.hidden, self.hidden), nn.SiLU())
                s.mean, s.logvar = nn.Linear(self.hidden, T), nn.Linear(self.hidden, T)

            def forward(s, x, _unused=None):
                z = s.body(x)
                return s.mean(z), s.logvar(z).clamp(-12, 6)
        return Net()

    def fit(self, X, Y, groups=None):
        """``groups``: optional group id per row (packet ids) for the early-stopping split (``_train_one``)."""
        torch = _torch()
        self.sx, self.sy = _Scaler().fit(X), _Scaler().fit(Y)
        x = torch.tensor(np.clip(self.sx(X), -self.INPUT_CLIP, self.INPUT_CLIP), dtype=torch.float32)
        y = self.sy(Y)
        self.nets = [self._train_one(self._net(X.shape[1], Y.shape[1]), [x, None], y, seed, groups) for seed in range(self.n_models)]
        return self

    def predict(self, X):
        torch = _torch()
        return self._ensemble_predict([torch.tensor(np.clip(self.sx(X), -self.INPUT_CLIP, self.INPUT_CLIP), dtype=torch.float32), None])


class CurveCNN(_MeanVarHead):
    """
    A 1D convolutional network on the trace curves (input: {"curves": (n, C, L), "scalars": (n, p) or None}).

    Each curve is a channel; the convolutions slide small filters along the curves and LEARN which
    shapes matter (nobody tells it where the reset pulse or the noise floor is).  After three
    convolution + pooling stages, the result is summarized to 8 positions x 32 filters, the optional
    scalar inputs (theta, hand-made features) are appended, and a small dense head predicts
    mean and variance per target.  An ensemble of ``n_models`` gives the uncertainty.
    """
    name = "1D CNN on curves"

    def __init__(self, n_models: int = 3, epochs: int = 400, lr: float = 2e-3, weight_decay: float = 1e-3,
                 batch: int = 256, patience: int = 50, val_fraction: float = 0.15, width: int = 32):
        self.n_models, self.epochs, self.lr, self.weight_decay = n_models, epochs, lr, weight_decay
        self.batch, self.patience, self.val_fraction, self.width = batch, patience, val_fraction, width
        self.device = _device(prefer_gpu=True)

    def _net(self, C, p, T):
        torch = _torch()
        nn, w = torch.nn, self.width

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.conv = nn.Sequential(
                    nn.Conv1d(C, w // 2, 7, padding=3), nn.SiLU(), nn.MaxPool1d(2),
                    nn.Conv1d(w // 2, w, 7, padding=3), nn.SiLU(), nn.MaxPool1d(2),
                    nn.Conv1d(w, w, 5, padding=2), nn.SiLU(), nn.AdaptiveAvgPool1d(8))
                s.head = nn.Sequential(nn.Linear(w * 8 + p, 64), nn.SiLU())
                s.mean, s.logvar = nn.Linear(64, T), nn.Linear(64, T)

            def forward(s, c, x=None):
                z = s.conv(c).flatten(1)
                if x is not None:
                    z = torch.cat([z, x], dim=1)
                z = s.head(z)
                return s.mean(z), s.logvar(z).clamp(-12, 6)
        return Net()

    def _prep(self, X, fit=False):
        torch = _torch()
        c = X["curves"]
        if fit:
            self.c_mean = c.mean(axis=0, keepdims=True)
            self.c_sd = c.std(axis=0, keepdims=True)
            self.c_sd[self.c_sd == 0] = 1.0
            self.sx = _Scaler().fit(X["scalars"]) if X.get("scalars") is not None else None
        k = self.INPUT_CLIP
        ct = torch.tensor(np.clip((c - self.c_mean) / self.c_sd, -k, k), dtype=torch.float32, device=self.device)
        xt = (torch.tensor(np.clip(self.sx(X["scalars"]), -k, k), dtype=torch.float32, device=self.device)
              if self.sx is not None else None)
        return ct, xt

    def fit(self, X, Y, groups=None):
        """``groups``: optional group id per row (packet ids) for the early-stopping split (``_train_one``)."""
        self.sy = _Scaler().fit(Y)
        ct, xt = self._prep(X, fit=True)
        p = 0 if xt is None else xt.shape[1]
        self.nets = [self._train_one(self._net(ct.shape[1], p, Y.shape[1]), [ct, xt], self.sy(Y), seed, groups)
                     for seed in range(self.n_models)]
        return self

    def predict(self, X):
        return self._ensemble_predict(list(self._prep(X)))


MODELS = {"ridge": BayesRidge, "gbt": GBT, "gp": GP, "mlp": MLPEnsemble, "cnn": CurveCNN}

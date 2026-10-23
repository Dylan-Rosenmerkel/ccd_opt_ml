"""
bogp.py -- Gaussian processes from BoTorch with priors meant for MANY inputs and FEW points.

Our GPs see ~230 (features) to ~320 (physics) inputs and at most a few hundred examples.  Fitting one length scale per
input freely (maximum likelihood, as ``models.GP`` does) has more knobs than data.  Two remedies from the literature
(docs/literature/ml_models_literature_review.md, section 7.4):

* ``kind="dim"``: BoTorch's ``SingleTaskGP`` with its default kernel: RBF with every length scale ``l_i`` drawn from
  LogNormal(sqrt(2) + log(d)/2, sqrt(3))  (Hvarfner, Hellsten & Nardi, ICML 2024, arXiv:2402.02229).  With d inputs the
  typical length scale grows like sqrt(d), so that a point's distance to its neighbours (which also grows like sqrt(d))
  stays comparable to the length scale and the GP does not treat every point as unrelated.  Fitted by maximizing the
  marginal likelihood times this prior (MAP).
* ``kind="saas"``: the sparse axis-aligned subspace GP (Eriksson & Jankowiak, UAI 2021, arXiv:2103.00349): a global
  shrinkage tau ~ HalfCauchy(0.1) and per input 1/l_i^2 ~ HalfCauchy(tau), so most inputs are switched off unless the
  data demand them; the hyperparameters are SAMPLED (NUTS, full Bayesian inference) and the prediction is the mixture
  over the samples.  ``relevance()`` gives each input's median inverse length scale (large = used).

Inputs are scaled to the unit cube with the training rows' range (``Normalize``, as both priors assume) and each target is
standardized.  Labels' known variances are not used as fixed noise (that would remove the learned noise, which also
carries everything the inputs do not explain); rows with ``extra_var > 0`` (outliers that PhysicsStructured down-weights)
are left out of the fit.

Same interface as ``structured.HierGP``: ``fit(X, Y, y_var=None, extra_var=None)`` with X an array or {"x": array},
``predict(X) -> (mean, std)``, each (n, n_targets).
"""

from __future__ import annotations

import warnings

import numpy as np


def _x(X):
    return np.asarray(X["x"] if isinstance(X, dict) else X, float)


class BoTorchGP:
    """A BoTorch GP per target with a high-dimension length-scale prior: 'dim' (dimension-scaled, fitted by maximum
    likelihood) or 'saas' (sparsity prior, hyperparameters sampled by NUTS); see the module docstring."""
    def __init__(self, kind: str = "dim", warmup: int = 256, num_samples: int = 128, thinning: int = 16, seed: int = 0,
                 **_ignored):
        if kind not in ("dim", "saas"):
            raise ValueError(f"kind must be 'dim' or 'saas', not {kind!r}")
        self.kind, self.warmup, self.num_samples, self.thinning, self.seed = kind, warmup, num_samples, thinning, seed
        self.name = {"dim": "GP, dimension-scaled prior (BoTorch)", "saas": "SAAS GP (BoTorch, NUTS)"}[kind]

    def fit(self, X, Y, y_var=None, extra_var=None):
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        x = _x(X)
        Y = np.atleast_2d(np.asarray(Y, float).T).T
        if extra_var is not None:
            extra_var = np.atleast_2d(np.nan_to_num(np.asarray(extra_var, float)).T).T
        torch.manual_seed(self.seed)
        self.models = []
        for t in range(Y.shape[1]):
            keep = np.isfinite(Y[:, t])
            if extra_var is not None:
                keep &= extra_var[:, min(t, extra_var.shape[1] - 1)] <= 0
            tx = torch.tensor(x[keep], dtype=torch.float64)
            ty = torch.tensor(Y[keep, t:t + 1], dtype=torch.float64)
            transforms = dict(input_transform=Normalize(d=tx.shape[1]), outcome_transform=Standardize(m=1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.kind == "dim":
                    from botorch.fit import fit_gpytorch_mll
                    from gpytorch.mlls import ExactMarginalLogLikelihood
                    model = SingleTaskGP(tx, ty, **transforms)
                    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
                else:
                    from botorch.fit import fit_fully_bayesian_model_nuts
                    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
                    model = SaasFullyBayesianSingleTaskGP(tx, ty, **transforms)
                    fit_fully_bayesian_model_nuts(model, warmup_steps=self.warmup, num_samples=self.num_samples,
                                                  thinning=self.thinning, disable_progbar=True, seed=self.seed)
            model.eval()
            self.models.append(model)
        return self

    def predict(self, X):
        import torch
        tx = torch.tensor(_x(X), dtype=torch.float64)
        means, stds = [], []
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for model in self.models:
                post = model.posterior(tx, observation_noise=True)
                if self.kind == "saas":                 # mixture over the NUTS samples
                    m, v = post.mixture_mean, post.mixture_variance
                else:
                    m, v = post.mean, post.variance
                means.append(m.reshape(-1).numpy())
                stds.append(np.sqrt(np.maximum(v.reshape(-1).numpy(), 1e-12)))
        return np.column_stack(means), np.column_stack(stds)

    def relevance(self, t: int = 0) -> np.ndarray:
        """Per input (in the given order): the inverse length scale (SAAS: median over the samples).  Large = used."""
        ls = self.models[t].covar_module.lengthscale.detach().numpy() if self.kind == "dim" else \
            self.models[t].median_lengthscale.detach().numpy()
        return 1 / np.asarray(ls).reshape(-1)

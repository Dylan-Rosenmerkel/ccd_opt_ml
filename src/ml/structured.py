"""
structured.py -- the second-round models (docs/overleaf_ml_models/ml_models.tex, "Prototypes of the next models").

The first round (models.py) treated sigma_e, lambda and gain as three unrelated numbers to be learned
from scratch.  Two things we know were left out:

A. PHYSICS-STRUCTURED OUTPUTS (``PhysicsStructured``)
   The model predicts the physical pieces and combines them with the formulas we trust, instead of
   predicting the labels directly:

   * read noise.  A skipper pixel averaged over N skips has
         sigma_e^2 = (sigma_1^2 / N + sigma_floor^2) / g^2          (ADU -> e- through the gain g)
     sigma_1 (single-skip noise) and sigma_floor (noise that does not average down) are MEASURED in
     ADU on the trace, by fitting a^2/n + b^2 to its Allan deviation (``allan_fit``; on real data
     also the features "skip: CDS rms" and "fit: correlated noise floor").  So only the gain has to
     be learned, plus a small learned correction r(x) for whatever the image sees and the trace
     does not:  log sigma_e = 1/2 log(a^2/N + b^2) - log g + r(x).
   * gain: learned (it is not visible on the trace in ADU: the trace measures sigma_1 * g).
   * dark charge.  lambda = dark current (grows with the time a row waits, depends on temperature and
     substrate) + clock-induced charge (CIC, set by the clock swings).  When the image gives lambda
     row by row, the two parts are labeled separately (synthetic.lambda_per_row) and learned by two
     models; lambda = exp(f_dark) + exp(f_cic).  Without per-row labels, log lambda is learned directly.

B. HIERARCHICAL GAUSSIAN PROCESS (``HierGP``)
   Every learned function f(x) above is a GP whose kernel knows that rows come from amplifiers on
   modules:
         k(x, x') = s^2 M(x, x')                                    shared by every amplifier
                  + [same module]    (c_m^2 + s_m^2 M(x, x'))       the module's own offset and shape
                  + [same amplifier] (c_a^2 + s_a^2 M(x, x'))       the amplifier's own offset and shape
                  + noise^2 [same row]
   M is a Matern-5/2 kernel with one length scale per input (ARD).  An amplifier never seen gets the
   shared prediction with the offset variances added to its error bar; after a few of its packets
   the offsets are learned from those packets alone (shrunk toward 0 by the prior).

Inputs for both (``StructuredPrep``): theta, and a PACKET STATE VECTOR: the reference-trace features of
the packet, minus each amplifier's own average (learned on training rows), averaged over the
amplifiers of the packet.  It measures the CCD's drifting state (temperature, history) with the
amplifier signatures removed; for an amplifier never seen it comes from the other amplifiers of the
same packet.  No label is used to build it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .models import _Scaler

# --------------------------------------------------------------------------- #
# physics pieces measured on the trace                                         #
# --------------------------------------------------------------------------- #


def allan_fit(allan: np.ndarray, ns: np.ndarray, rel_sd: float = 0.03, max_n: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit sigma_A(n)^2 = a^2 / n + b^2 to Allan-deviation curves (one per row), a, b >= 0.

    ``max_n``: use only n <= max_n.  On REAL traces use max_n = N_skips / 5: with 500 skips the n = 160
    point comes from 3 blocks and is biased low (~0.43 of the white-noise value on the 2026-10-12 traces),
    which drags the fit to b = 0 and a below the CDS rms.

    Weighted least squares in sigma_A^2 with weights for a constant RELATIVE scatter of every point;
    if b^2 comes out negative the fit is redone with b = 0.  Returns (a, b) in the curve's units (ADU):
    a = single-skip noise, b = noise floor.
    """
    allan = np.atleast_2d(np.asarray(allan, float))
    ns = np.asarray(ns, float)
    if max_n is not None:
        allan, ns = allan[:, ns <= max_n], ns[ns <= max_n]
    v = allan ** 2
    w = 1 / np.maximum(v * 2 * rel_sd, 1e-30) ** 2                     # 1 / var(sigma_A^2)
    x1, x2 = 1 / ns, np.ones_like(ns)
    s11, s12, s22 = (w * x1 * x1).sum(1), (w * x1 * x2).sum(1), (w * x2 * x2).sum(1)
    t1, t2 = (w * x1 * v).sum(1), (w * x2 * v).sum(1)
    det = s11 * s22 - s12 ** 2
    A, B = (t1 * s22 - t2 * s12) / det, (s11 * t2 - s12 * t1) / det
    neg = B < 0
    A[neg], B[neg] = t1[neg] / s11[neg], 0.0                           # b = 0: fit a alone
    return np.sqrt(np.maximum(A, 0)), np.sqrt(np.maximum(B, 0))


# --------------------------------------------------------------------------- #
# inputs                                                                       #
# --------------------------------------------------------------------------- #


def amplifier_key(df: pd.DataFrame) -> np.ndarray:
    return (df["board"].astype(str) + "_ch" + df["channel"].astype(str)).to_numpy()


class StructuredPrep:
    """
    Builds the inputs of ``HierGP`` / ``PhysicsStructured`` from a dataset (synthetic or dataset.csv):

    ``x``       theta + the packet state vector (see the module docstring)
    ``a, b``    single-skip noise and noise floor (ADU) of the traces at theta, from the Allan-deviation
                curve (on real data the features "skip: CDS rms" and "fit: correlated noise floor"
                measure the same; the toy features carry large per-amplifier offsets, so they are not used)
    ``module``, ``amp``  integer ids (the same id for the same board / amplifier in every call)
    ``onehot``  the amplifier one-hot (for the non-hierarchical variant)

    ``fit`` learns, on the training rows only, each amplifier's average reference features; only
    amplifiers with at least ``min_rows`` training rows are used for the state (an average over a few
    packets would still contain those packets' state).
    """

    def __init__(self, allan_ns: np.ndarray | None = None, n_skips: int = 500, min_rows: int = 20,
                 allan_max_n: float | None = None):
        from . import synthetic
        self.allan_ns = synthetic.ALLAN_N if allan_ns is None else np.asarray(allan_ns)
        self.n_skips, self.min_rows, self.allan_max_n = n_skips, min_rows, allan_max_n   # real data: allan_max_n = n_skips / 5

    def fit(self, df: pd.DataFrame):
        self.ref_cols = data.columns(df, "ref")
        self.theta_cols = data.columns(df, "theta")
        R = df[self.ref_cols].to_numpy(float)
        keys = amplifier_key(df)
        self.amp_ids = {k: i for i, k in enumerate(sorted(np.unique(keys)))}
        self.centers = {k: R[keys == k].mean(0) for k in self.amp_ids if (keys == k).sum() >= self.min_rows}
        self.module_ids = {b: i for i, b in enumerate(sorted(df["board"].unique()))}
        return self

    def state_vector(self, df: pd.DataFrame) -> np.ndarray:
        """Per row: mean over the packet's KNOWN amplifiers of (reference features - amplifier average)."""
        R = df[self.ref_cols].to_numpy(float)
        keys = amplifier_key(df)
        known = np.array([k in self.centers for k in keys])
        centered = np.full_like(R, np.nan)
        for k in np.unique(keys[known]):
            centered[keys == k] = R[keys == k] - self.centers[k]
        frame = pd.DataFrame(centered).assign(packet=df["packet_id"].to_numpy())
        state = frame.groupby("packet").transform("mean").to_numpy(float)
        return np.nan_to_num(state)                                      # a packet with no known amplifier: 0

    def ids(self, mapping: dict, values) -> np.ndarray:
        """Integer ids; values never seen in fit get new ids (an unseen module / amplifier)."""
        out, extra = [], {}
        for v in values:
            if v not in mapping:
                extra.setdefault(v, len(mapping) + len(extra))
            out.append(mapping.get(v, extra.get(v)))
        return np.array(out)

    def transform(self, df: pd.DataFrame, curves: dict) -> dict:
        theta = df[self.theta_cols].to_numpy(float)
        x = np.hstack([theta, self.state_vector(df)])
        a, b = allan_fit(curves["noise: Allan deviation"], self.allan_ns, max_n=self.allan_max_n)
        keys = amplifier_key(df)
        amp = self.ids(self.amp_ids, keys)
        onehot = np.zeros((len(df), len(self.amp_ids)))
        seen = amp < len(self.amp_ids)
        onehot[np.flatnonzero(seen), amp[seen]] = 1.0
        return {"x": x, "a": a, "b": b, "amp": amp, "onehot": onehot,
                "module": self.ids(self.module_ids, df["board"].to_numpy())}


def component_targets(df: pd.DataFrame, floor_fraction: float = 0.05) -> np.ndarray | None:
    """log of the per-row dark and CIC labels (if the dataset has them); fits below 5 % of lambda are clipped."""
    if "label.lambda_dark" not in df:
        return None
    lam = df["label.lambda_e_per_bin"].to_numpy(float)
    parts = [np.maximum(df[c].to_numpy(float), floor_fraction * lam) for c in ("label.lambda_dark", "label.lambda_cic")]
    return np.log(np.column_stack(parts))


# --------------------------------------------------------------------------- #
# B: hierarchical Gaussian process                                             #
# --------------------------------------------------------------------------- #


def _torch():
    import torch
    return torch


class HierGP:
    """
    Exact GP with the hierarchical kernel of the module docstring (one GP per target column).

    X is a dict with ``x`` (n, p) and the integer ids ``module`` and ``amp``.  ``levels`` chooses the
    hierarchy: ("module", "amp") is model B; () gives an ordinary ARD GP (then append the amplifier
    one-hot to x yourself, as the first-round GP did).  Hyperparameters maximize the marginal likelihood
    (L-BFGS, PyTorch, float64).  Above ``max_points`` rows a subset is used, sampling rare amplifiers
    first so that a new amplifier's few rows are always kept.
    """
    name = "hierarchical GP"

    def __init__(self, levels: tuple[str, ...] = ("module", "amp"), max_points: int = 1200, iters: int = 120,
                 seed: int = 0, include_noise: bool = True):
        self.levels, self.max_points, self.iters, self.seed = tuple(levels), max_points, iters, seed
        self.include_noise = include_noise

    # kernel pieces ------------------------------------------------------------
    @staticmethod
    def _matern(torch, A, B, ls):
        A, B = A / ls, B / ls
        sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
        r = torch.sqrt(sq.clamp_min(0) + 1e-12)
        s5r = np.sqrt(5.0) * r
        return (1 + s5r + 5.0 / 3.0 * r * r) * torch.exp(-s5r)

    def _K(self, torch, p, xa, xb, ida, idb):
        M = self._matern(torch, xa, xb, p["ls"].exp())
        K = p["s"].exp() ** 2 * M
        for lvl in self.levels:
            same = (ida[lvl][:, None] == idb[lvl][None, :]).to(xa.dtype)
            K = K + same * (p[f"c_{lvl}"].exp() ** 2 + p[f"s_{lvl}"].exp() ** 2 * M)
        return K

    def _prior_var(self, p):
        v = p["s"].exp() ** 2
        for lvl in self.levels:
            v = v + p[f"c_{lvl}"].exp() ** 2 + p[f"s_{lvl}"].exp() ** 2
        return v

    def _subset(self, n, amp):
        if n <= self.max_points:
            return np.arange(n)
        counts = pd.Series(amp).map(pd.Series(amp).value_counts()).to_numpy(float)
        prob = 1 / counts
        return np.sort(np.random.default_rng(self.seed).choice(n, self.max_points, replace=False, p=prob / prob.sum()))

    def _ids(self, X, idx=None):
        torch = _torch()
        sel = slice(None) if idx is None else idx
        return {lvl: torch.as_tensor(np.asarray(X[lvl])[sel]) for lvl in self.levels}

    # fit / predict --------------------------------------------------------------
    def fit(self, X, Y):
        torch = _torch()
        Y = np.atleast_2d(np.asarray(Y, float).T).T
        idx = self._subset(len(Y), np.asarray(X.get("amp", np.zeros(len(Y)))))
        self.sx = _Scaler().fit(X["x"][idx])
        self.xt = torch.tensor(self.sx(X["x"][idx]), dtype=torch.float64)
        self.idt = self._ids(X, idx)
        n, d = self.xt.shape
        self.fits = []
        for t in range(Y.shape[1]):
            y = Y[idx, t]
            ym, ys = y.mean(), y.std() if y.std() > 0 else 1.0
            yt = torch.tensor((y - ym) / ys, dtype=torch.float64)
            p = {"ls": torch.full((d,), np.log(np.sqrt(d)), dtype=torch.float64),
                 "s": torch.tensor(np.log(0.7), dtype=torch.float64),
                 "noise": torch.tensor(np.log(0.1), dtype=torch.float64),
                 "mu": torch.tensor(0.0, dtype=torch.float64)}
            for lvl in self.levels:
                p[f"c_{lvl}"] = torch.tensor(np.log(0.5), dtype=torch.float64)
                p[f"s_{lvl}"] = torch.tensor(np.log(0.2), dtype=torch.float64)
            for v in p.values():
                v.requires_grad_(True)
            opt = torch.optim.LBFGS(list(p.values()), lr=0.5, max_iter=self.iters, line_search_fn="strong_wolfe")
            eye = torch.eye(n, dtype=torch.float64)

            def nll():
                opt.zero_grad()
                q = {k: (v.clamp(-7, 4) if k != "mu" else v) for k, v in p.items()}
                K = self._K(torch, q, self.xt, self.xt, self.idt, self.idt) + (q["noise"].exp() ** 2 + 1e-6) * eye
                L = torch.linalg.cholesky(K)
                r = (yt - q["mu"])[:, None]
                alpha = torch.cholesky_solve(r, L)
                loss = 0.5 * (r * alpha).sum() + torch.log(torch.diagonal(L)).sum()
                loss = loss + 0.5 * ((q["ls"] - np.log(np.sqrt(d))) ** 2).sum() / 9  # weak prior: no extreme length scales
                loss.backward()
                return loss

            try:
                opt.step(nll)
            except RuntimeError:                                         # Cholesky failed: keep the start values
                pass
            with torch.no_grad():
                q = {k: (v.detach().clamp(-7, 4) if k != "mu" else v.detach()) for k, v in p.items()}
                K = self._K(torch, q, self.xt, self.xt, self.idt, self.idt) + (q["noise"].exp() ** 2 + 1e-6) * eye
                L = torch.linalg.cholesky(K)
                alpha = torch.cholesky_solve((yt - q["mu"])[:, None], L)
            self.fits.append({"p": q, "L": L, "alpha": alpha, "ym": ym, "ys": ys})
        return self

    def predict(self, X):
        torch = _torch()
        xs = torch.tensor(self.sx(X["x"]), dtype=torch.float64)
        ids = self._ids(X)
        means, stds = [], []
        with torch.no_grad():
            for f in self.fits:
                q = f["p"]
                Ks = self._K(torch, q, xs, self.xt, ids, self.idt)
                m = q["mu"] + (Ks @ f["alpha"])[:, 0]
                v = torch.linalg.solve_triangular(f["L"], Ks.T, upper=False)
                var = self._prior_var(q) - (v * v).sum(0) + (q["noise"].exp() ** 2 if self.include_noise else 0.0)
                means.append(m.numpy() * f["ys"] + f["ym"])
                stds.append(np.sqrt(var.clamp_min(1e-12).numpy()) * f["ys"])
        return np.column_stack(means), np.column_stack(stds)

    def hyperparameters(self, t: int = 0) -> dict[str, float]:
        """Fitted amplitudes (in units of the target's sd): how much is shared, per module, per amplifier."""
        q = self.fits[t]["p"]
        return {k: float(v.exp()) for k, v in q.items() if k not in ("ls", "mu")}


def _plain_gp(**kw):
    """The non-hierarchical variant: an ARD GP on x + amplifier one-hot (as the first round)."""
    class Plain(HierGP):
        name = "ARD GP (amplifier one-hot)"

        def fit(self, X, Y):
            return super().fit({"x": np.hstack([X["x"], X["onehot"]]), "amp": X["amp"]}, Y)

        def predict(self, X):
            return super().predict({"x": np.hstack([X["x"], X["onehot"]])})
    return Plain(levels=(), **kw)


def make_gp(kind: str, **kw):
    return HierGP(**kw) if kind == "hier" else _plain_gp(**kw)


class DirectGP:
    """Model B alone: the hierarchical GP predicting log sigma_e, log lambda, log gain directly from x."""
    name = "B: hierarchical GP"

    def __init__(self, kind: str = "hier", **kw):
        self.gp = make_gp(kind, **kw)

    def fit(self, X, Y, components=None):
        self.gp.fit(X, Y)
        return self

    def predict(self, X):
        return self.gp.predict(X)


# --------------------------------------------------------------------------- #
# A: physics-structured outputs                                                #
# --------------------------------------------------------------------------- #


class PhysicsStructured:
    """
    Model A (with ``kind='plain'``) or A + B (``kind='hier'``): see the module docstring.

    fit(X, Y, components): X from ``StructuredPrep.transform``; Y = (log sigma_e, log lambda, log gain);
    components = (log dark, log CIC) from ``component_targets`` or None.
    predict(X) -> mean, std of the same three logs.
    """

    def __init__(self, kind: str = "hier", n_skips: int = 500, use_components: bool = True, use_floor: bool = True, **kw):
        self.kind, self.n_skips, self.use_components, self.use_floor, self.kw = kind, n_skips, use_components, use_floor, kw
        self.name = ("A+B" if kind == "hier" else "A") + ": physics-structured"

    def physics_log_sigma_adu(self, X) -> np.ndarray:
        """1/2 log(a^2/N + b^2): the read noise after N skips, in ADU, measured on the trace (b = 0 if not use_floor)."""
        b2 = X["b"] ** 2 if self.use_floor else 0.0
        return 0.5 * np.log(np.maximum(X["a"] ** 2 / self.n_skips + b2, 1e-30))

    def fit(self, X, Y, components=None):
        log_s, log_lam, log_g = Y[:, 0], Y[:, 1], Y[:, 2]
        self.gain = make_gp(self.kind, **self.kw).fit(X, log_g)
        # what the image sees that the trace physics does not (0 if the physics were complete)
        resid = log_s - (self.physics_log_sigma_adu(X) - log_g)
        self.resid = make_gp(self.kind, **self.kw).fit(X, resid)
        self.split = self.use_components and components is not None
        if self.split:
            self.dark = make_gp(self.kind, **self.kw).fit(X, components[:, 0])
            self.cic = make_gp(self.kind, **self.kw).fit(X, components[:, 1])
        else:
            self.lam = make_gp(self.kind, **self.kw).fit(X, log_lam)
        return self

    def predict(self, X):
        mg, sg = self.gain.predict(X)
        mr, sr = self.resid.predict(X)
        m_s = self.physics_log_sigma_adu(X) - mg[:, 0] + mr[:, 0]
        s_s = np.sqrt(sg[:, 0] ** 2 + sr[:, 0] ** 2)
        if self.split:
            (md, sd), (mc, sc) = self.dark.predict(X), self.cic.predict(X)
            d, c = np.exp(md[:, 0]), np.exp(mc[:, 0])
            m_l = np.log(d + c)
            s_l = np.sqrt((d * sd[:, 0]) ** 2 + (c * sc[:, 0]) ** 2) / (d + c)   # delta method
        else:
            ml, sl = self.lam.predict(X)
            m_l, s_l = ml[:, 0], sl[:, 0]
        return np.column_stack([m_s, m_l, mg[:, 0]]), np.column_stack([s_s, s_l, sg[:, 0]])

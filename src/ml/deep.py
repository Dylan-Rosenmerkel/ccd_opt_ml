"""
deep.py -- models C and D of docs/overleaf_ml_models/ml_models.tex: networks that learn their own features from the
trace curves, made to work with few labeled packets.

D. SELF-SUPERVISED PRE-TRAINING (``Pretrainer``)
   A packet has one image (labels) but many traces; trace-only packets are cheap.  A convolutional encoder
   (the CurveCNN's convolutions -> a 16-number embedding) is first trained on curves WITHOUT labels, on
   three tasks:
     * masked reconstruction: blocks of 32 positions, and in half of the examples one whole curve, are
       hidden and must be filled in from the rest.  Hiding whole curves matters: short blocks can be
       interpolated from their neighbours without learning anything, while rebuilding e.g. the CDS
       spectrum from the Allan deviation and the other curves forces the encoder to learn how the curves
       hang together (a flattening Allan deviation goes with a raised low-frequency spectrum);
     * theta prediction: the settings are known for every trace, labels or not;
     * contrastive (InfoNCE): two noisy, slightly shifted copies of the same curves must have close
       embeddings, different packets far apart (the embedding ignores measurement noise).
   Then, with the few labeled packets:
     * ``PretrainedHead``  ('linear probe'): the encoder is frozen; a small network ensemble on
       [embedding, theta, amplifier] predicts the labels;
     * ``PretrainedCNN``   ('fine-tuning'): the whole network starts from the pre-trained encoder.

C. DEEP KERNEL LEARNING (``DeepKernelGP``)
   A Gaussian process whose kernel works on [encoder(curves), theta, ...] instead of the raw inputs:
       k(i, j) = s^2 Matern52(|z_i - z_j| / l) + c_amp^2 [same amplifier] + noise^2 [i = j],   z = (g_phi(curves), x)
   The encoder's weights phi and the GP's hyperparameters maximize the exact marginal likelihood together
   (one GP per target, one shared encoder).  Starting phi from the pre-trained encoder (D) keeps it from
   over-fitting a few hundred packets.  The GP gives the error bars Bayesian optimization needs.

Curves are standardized per position with the PRE-TRAINING data (they are unlabeled, so this leaks nothing).
"""

from __future__ import annotations

import copy

import numpy as np

from .models import CurveCNN, MLPEnsemble, _device, _Scaler, _torch


# --------------------------------------------------------------------------- #
# encoder                                                                      #
# --------------------------------------------------------------------------- #


def make_encoder(C: int, emb_dim: int = 16, width: int = 32):
    """The CurveCNN's convolution stack followed by a linear map to ``emb_dim`` numbers."""
    torch = _torch()
    nn, w = torch.nn, width
    return nn.Sequential(
        nn.Conv1d(C, w // 2, 7, padding=3), nn.SiLU(), nn.MaxPool1d(2),
        nn.Conv1d(w // 2, w, 7, padding=3), nn.SiLU(), nn.MaxPool1d(2),
        nn.Conv1d(w, w, 5, padding=2), nn.SiLU(), nn.AdaptiveAvgPool1d(8),
        nn.Flatten(), nn.Linear(w * 8, emb_dim))


def _info_nce(z1, z2, temperature: float):
    """NT-Xent: row i of z1 and row i of z2 are a positive pair, every other row a negative."""
    torch = _torch()
    z = torch.nn.functional.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / temperature
    n = len(z1)
    sim = sim - 1e9 * torch.eye(2 * n, device=z.device)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return torch.nn.functional.cross_entropy(sim, target)


class Pretrainer:
    """
    Model D's pre-training.  ``fit(curves, theta)`` with curves (n, C, L) (``data.curve_tensor``) and theta
    (n, p) or None; ``encode(curves)`` gives the (n, emb_dim) embedding.  ``losses`` keeps the curve of
    every task's loss per epoch.  Weights of the three tasks: ``w_rec``, ``w_theta``, ``w_con`` (0 = off).
    """

    def __init__(self, emb_dim: int = 16, width: int = 32, epochs: int = 40, batch: int = 256, lr: float = 2e-3,
                 mask_frac: float = 0.25, block: int = 32, p_whole_curve: float = 0.5, w_rec: float = 1.0, w_theta: float = 1.0, w_con: float = 0.3,
                 temperature: float = 0.2, seed: int = 0):
        self.emb_dim, self.width, self.epochs, self.batch, self.lr = emb_dim, width, epochs, batch, lr
        self.mask_frac, self.block, self.w_rec, self.w_theta, self.w_con = mask_frac, block, w_rec, w_theta, w_con
        self.p_whole_curve = p_whole_curve
        self.temperature, self.seed = temperature, seed
        self.device = _device(prefer_gpu=True)

    def standardize(self, curves):
        return (curves - self.c_mean) / self.c_sd

    def _augment(self, c):
        """A second 'view' of the same curves: noise, a +-5 % scale per channel, a shift of up to 2 samples."""
        torch = _torch()
        c = c * (1 + 0.05 * torch.randn(c.shape[0], c.shape[1], 1, device=c.device))
        c = c + 0.05 * torch.randn_like(c)
        return torch.roll(c, int(torch.randint(-2, 3, (1,))), dims=2)

    def fit(self, curves: np.ndarray, theta: np.ndarray | None = None):
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.c_mean = curves.mean(0, keepdims=True)
        self.c_sd = curves.std(0, keepdims=True)
        self.c_sd[self.c_sd == 0] = 1.0
        n, C, L = curves.shape
        self.C, self.L = C, L
        ct = torch.tensor(self.standardize(curves), dtype=torch.float32, device=self.device)
        self.encoder = make_encoder(C, self.emb_dim, self.width).to(self.device)
        decoder = nn.Sequential(nn.Linear(self.emb_dim, 256), nn.SiLU(), nn.Linear(256, C * L)).to(self.device)
        heads = [decoder]
        if theta is not None and self.w_theta > 0:
            self.st = _Scaler().fit(theta)
            tt = torch.tensor(self.st(theta), dtype=torch.float32, device=self.device)
            theta_head = nn.Linear(self.emb_dim, theta.shape[1]).to(self.device)
            heads.append(theta_head)
        proj = nn.Sequential(nn.Linear(self.emb_dim, 32), nn.SiLU(), nn.Linear(32, 32)).to(self.device)
        heads.append(proj)
        params = list(self.encoder.parameters()) + [p for h in heads for p in h.parameters()]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        n_blocks = L // self.block
        self.losses = {"reconstruction": [], "theta": [], "contrastive": []}
        for epoch in range(self.epochs):
            sums = {k: 0.0 for k in self.losses}
            perm = rng.permutation(n)
            for start in range(0, n, self.batch):
                b = perm[start:start + self.batch]
                x = ct[b]
                # mask whole blocks of positions (the same in every channel)
                hide = torch.rand(len(b), 1, n_blocks, device=self.device) < self.mask_frac
                mask = hide.repeat_interleave(self.block, dim=2)
                if mask.shape[2] < L:
                    mask = torch.cat([mask, torch.zeros(len(b), 1, L - mask.shape[2], dtype=torch.bool, device=self.device)], 2)
                mask = mask.expand(-1, C, -1).float()
                whole = torch.rand(len(b), device=self.device) < self.p_whole_curve         # hide one whole curve instead
                pick = torch.nn.functional.one_hot(torch.randint(0, C, (len(b),), device=self.device), C).float()
                mask = torch.where(whole[:, None, None], pick[:, :, None].expand(-1, -1, L), mask)
                z = self.encoder(x * (1 - mask))
                rec = decoder(z).view(len(b), C, L)
                loss_rec = ((rec - x) ** 2 * mask).sum() / mask.sum().clamp_min(1)
                loss = self.w_rec * loss_rec
                sums["reconstruction"] += float(loss_rec.detach()) * len(b)
                if theta is not None and self.w_theta > 0:
                    loss_t = ((theta_head(self.encoder(x)) - tt[b]) ** 2).mean()
                    loss = loss + self.w_theta * loss_t
                    sums["theta"] += float(loss_t.detach()) * len(b)
                if self.w_con > 0:
                    loss_c = _info_nce(proj(self.encoder(self._augment(x))), proj(self.encoder(self._augment(x))), self.temperature)
                    loss = loss + self.w_con * loss_c
                    sums["contrastive"] += float(loss_c.detach()) * len(b)
                opt.zero_grad()
                loss.backward()
                opt.step()
            for k in self.losses:
                self.losses[k].append(sums[k] / n)
        self.encoder.eval()
        return self

    def encode(self, curves: np.ndarray) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            ct = torch.tensor(self.standardize(curves), dtype=torch.float32, device=self.device)
            return np.concatenate([self.encoder(ct[i:i + 2048]).cpu().numpy() for i in range(0, len(ct), 2048)])


# --------------------------------------------------------------------------- #
# D: using the pre-trained encoder with few labels                             #
# --------------------------------------------------------------------------- #


class PretrainedHead:
    """Linear probe: frozen encoder; an MLP ensemble on [embedding, scalars].  X = {"curves", "scalars"}."""
    name = "D: pre-trained encoder (frozen) + NN ensemble"

    def __init__(self, pretrainer: Pretrainer, **mlp_kw):
        self.pre, self.mlp = pretrainer, MLPEnsemble(**mlp_kw)

    def _inputs(self, X):
        z = self.pre.encode(X["curves"])
        return np.hstack([z, X["scalars"]]) if X.get("scalars") is not None else z

    def fit(self, X, Y, groups=None):
        self.mlp.fit(self._inputs(X), Y, groups=groups)
        return self

    def predict(self, X):
        return self.mlp.predict(self._inputs(X))


class PretrainedCNN(CurveCNN):
    """Fine-tuning: the CurveCNN with its convolutions and embedding started from the pre-trained encoder."""
    name = "D: pre-trained encoder, fine-tuned"

    def __init__(self, pretrainer: Pretrainer, lr: float = 5e-4, **kw):
        super().__init__(lr=lr, **kw)
        self.pre = pretrainer

    def _net(self, C, p, T):
        torch = _torch()
        nn = torch.nn
        encoder = copy.deepcopy(self.pre.encoder)
        emb = self.pre.emb_dim

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = encoder
                s.head = nn.Sequential(nn.Linear(emb + p, 64), nn.SiLU())
                s.mean, s.logvar = nn.Linear(64, T), nn.Linear(64, T)

            def forward(s, c, x=None):
                z = s.enc(c)
                if x is not None:
                    z = torch.cat([z, x], dim=1)
                z = s.head(z)
                return s.mean(z), s.logvar(z).clamp(-12, 6)
        return Net()

    def _prep(self, X, fit=False):
        torch = _torch()
        if fit:
            self.sx = _Scaler().fit(X["scalars"]) if X.get("scalars") is not None else None
        k = self.INPUT_CLIP
        ct = torch.tensor(np.clip(self.pre.standardize(X["curves"]), -k, k), dtype=torch.float32, device=self.device)
        xt = (torch.tensor(np.clip(self.sx(X["scalars"]), -k, k), dtype=torch.float32, device=self.device) if self.sx is not None else None)
        return ct, xt


# --------------------------------------------------------------------------- #
# C: deep kernel learning                                                      #
# --------------------------------------------------------------------------- #


class DeepKernelGP:
    """
    Model C.  X = {"curves": (n, C, L), "x": (n, p) other inputs (theta, state), "amp": (n,) ids}; Y (n, T).
    ``pretrainer``: start the encoder from model D (None: random start, curves standardized on the
    training rows).  ``finetune``: train the encoder with the GP (False: frozen encoder, a GP on the
    embedding).  Adam on the summed negative log marginal likelihood of all targets.
    """
    name = "C: deep kernel GP"

    def __init__(self, pretrainer: Pretrainer | None = None, finetune: bool = True, emb_dim: int = 16, steps: int = 250,
                 lr_encoder: float = 3e-4, lr_gp: float = 0.03, max_points: int = 1200, seed: int = 0, amp_term: bool = True):
        """amp_term=False: no amplifier term in the kernel (no amplifier labels; X["amp"] is not needed)."""
        self.amp_term = amp_term
        self.pre, self.finetune, self.emb_dim, self.steps = pretrainer, finetune, emb_dim, steps
        self.lr_encoder, self.lr_gp, self.max_points, self.seed = lr_encoder, lr_gp, max_points, seed
        self.device = _device(prefer_gpu=True)

    def _standardize(self, curves):
        return self.pre.standardize(curves) if self.pre is not None else (curves - self.c_mean) / self.c_sd

    def _z(self, ct, xt):
        torch = _torch()
        e = self.encoder(ct).cpu().double()
        e = (e - self.e_mean) / self.e_sd
        return torch.cat([e, xt], dim=1)

    @staticmethod
    def _matern(torch, A, B, ls):
        A, B = A / ls, B / ls
        sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
        r = torch.sqrt(sq.clamp_min(0) + 1e-12)
        return (1 + np.sqrt(5.0) * r + 5.0 / 3.0 * r * r) * torch.exp(-np.sqrt(5.0) * r)

    def _K(self, torch, q, za, zb, aa, ab):
        K = q["s"].exp() ** 2 * self._matern(torch, za, zb, q["ls"].exp())
        if self.amp_term:
            K = K + q["c"].exp() ** 2 * (aa[:, None] == ab[None, :]).double()
        return K

    def fit(self, X, Y):
        torch = _torch()
        torch.manual_seed(self.seed)
        Y = np.atleast_2d(np.asarray(Y, float).T).T
        n = len(Y)
        idx = np.arange(n) if n <= self.max_points else np.sort(np.random.default_rng(self.seed).choice(n, self.max_points, replace=False))
        curves = X["curves"][idx]
        if self.pre is None:
            self.c_mean, self.c_sd = curves.mean(0, keepdims=True), curves.std(0, keepdims=True)
            self.c_sd[self.c_sd == 0] = 1.0
            self.encoder = make_encoder(curves.shape[1], self.emb_dim).to(self.device)
        else:
            self.encoder = copy.deepcopy(self.pre.encoder).to(self.device)
            self.emb_dim = self.pre.emb_dim
        self.sx = _Scaler().fit(X["x"][idx])
        ct = torch.tensor(self._standardize(curves), dtype=torch.float32, device=self.device)
        self.xt = torch.tensor(self.sx(X["x"][idx]), dtype=torch.float64)
        self.at = torch.as_tensor(np.asarray(X.get("amp", np.zeros(n, int)))[idx])
        with torch.no_grad():
            e0 = self.encoder(ct).cpu().double()
        self.e_mean, self.e_sd = e0.mean(0), e0.std(0).clamp_min(1e-6)
        self.ym, self.ys = Y[idx].mean(0), Y[idx].std(0) + 1e-12
        yt = torch.tensor((Y[idx] - self.ym) / self.ys, dtype=torch.float64)
        d = self.emb_dim + self.xt.shape[1]
        T = Y.shape[1]
        self.q = [{"ls": torch.full((d,), np.log(np.sqrt(d)), dtype=torch.float64, requires_grad=True),
                   "s": torch.tensor(np.log(0.7), dtype=torch.float64, requires_grad=True),
                   "c": torch.tensor(np.log(0.5), dtype=torch.float64, requires_grad=True),
                   "noise": torch.tensor(np.log(0.1), dtype=torch.float64, requires_grad=True),
                   "mu": torch.tensor(0.0, dtype=torch.float64, requires_grad=True)} for _ in range(T)]
        groups = [{"params": [v for q in self.q for v in q.values()], "lr": self.lr_gp}]
        if self.finetune:
            groups.append({"params": list(self.encoder.parameters()), "lr": self.lr_encoder})
        else:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        opt = torch.optim.Adam(groups)
        eye = torch.eye(len(idx), dtype=torch.float64)
        self.encoder.train(self.finetune)
        for step in range(self.steps):
            opt.zero_grad()
            z = self._z(ct, self.xt)
            loss = 0.0
            for t, q in enumerate(self.q):
                qc = {k: (v.clamp(-7, 4) if k != "mu" else v) for k, v in q.items()}
                K = self._K(torch, qc, z, z, self.at, self.at) + (qc["noise"].exp() ** 2 + 1e-5) * eye
                L = torch.linalg.cholesky(K)
                r = (yt[:, t] - qc["mu"])[:, None]
                loss = loss + 0.5 * (r * torch.cholesky_solve(r, L)).sum() + torch.log(torch.diagonal(L)).sum()
            (loss / len(idx)).backward()
            opt.step()
        self.encoder.eval()
        with torch.no_grad():
            self.z = self._z(ct, self.xt)
            self.fits = []
            for q in self.q:
                qc = {k: (v.detach().clamp(-7, 4) if k != "mu" else v.detach()) for k, v in q.items()}
                K = self._K(torch, qc, self.z, self.z, self.at, self.at) + (qc["noise"].exp() ** 2 + 1e-5) * eye
                L = torch.linalg.cholesky(K)
                t = len(self.fits)
                alpha = torch.cholesky_solve((yt[:, t] - qc["mu"])[:, None], L)
                self.fits.append((qc, L, alpha))
        return self

    def predict(self, X):
        torch = _torch()
        with torch.no_grad():
            ct = torch.tensor(self._standardize(X["curves"]), dtype=torch.float32, device=self.device)
            xs = torch.tensor(self.sx(X["x"]), dtype=torch.float64)
            zs = torch.cat([self._z(ct[i:i + 2048], xs[i:i + 2048]) for i in range(0, len(xs), 2048)])
            amp = torch.as_tensor(np.asarray(X.get("amp", np.zeros(len(X["x"]), int))))
            means, stds = [], []
            for t, (q, L, alpha) in enumerate(self.fits):
                Ks = self._K(torch, q, zs, self.z, amp, self.at)
                m = q["mu"] + (Ks @ alpha)[:, 0]
                v = torch.linalg.solve_triangular(L, Ks.T, upper=False)
                var = q["s"].exp() ** 2 + (q["c"].exp() ** 2 if self.amp_term else 0.0) - (v * v).sum(0) + q["noise"].exp() ** 2
                means.append(m.numpy() * self.ys[t] + self.ym[t])
                stds.append(np.sqrt(var.clamp_min(1e-12).numpy()) * self.ys[t])
        return np.column_stack(means), np.column_stack(stds)

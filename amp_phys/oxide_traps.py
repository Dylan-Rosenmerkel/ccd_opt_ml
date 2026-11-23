"""
oxide_traps.py -- oxide trap density of each output transistor from its OWN 1/f noise, and single-trap (RTS) steps.

1. OWN noise: L1 strips record all channels simultaneously.  Per skip, each channel's value minus beta x the mean of
   the other channels of the board (shared/setup part removed; pixel means removed first).
2. sigma_N: the noise of N-skip averages of the own series (difference of consecutive blocks / sqrt 2), N = 1..240.
3. Fit  sigma_N^2 = S_w * I_w(N) + K_v * I_f(N)  with the EXACT transfer function of the readout:
     |H(f)|^2 = 4 sin^2(pi f dt) sinc^2(f tau)  (CDS: two tau = 10 us windows, centres dt = 13.8 us apart)
              x [sin(pi f N T) / (N sin(pi f T))]^2  (average of N skips, period T = 32.6 us)
   I_w = int |H|^2 df (white), I_f = int |H|^2 / f df (1/f), numerically over 0.1 Hz - 7.5 MHz.  K_v: output-referred
   1/f coefficient (ADU^2), converted to the GATE (= sense node): / G^2 (ADU/uV) / A^2 (follower gain).
4. McWhorter number fluctuation: S_Vg(f) = q^2 k T lambda N_t / (W L C^2 f)  ->  N_t = K_vg W L C^2 / (q^2 k T lambda)
   (lambda = 0.1 nm tunnelling length; W L = 345 um^2 from the layout; C = C_ox of the ON stack 0.454 fF/um^2, or the
   buried-channel coupling 0.125 fF/um^2; A = 0.85 (layout) or 0.44 (kTC route); T = 130 K).  An UPPER limit: any own
   non-amplifier slow noise left after the shared part is removed also counts.
5. RTS: running means of 25 skips; steps = |difference of adjacent means| > 6 x their noise; step amplitudes and the
   spacing of steps (dwell times).  One trapped charge shifts the gate by q / (C W L): compared with the step sizes.

    .venv/bin/python amp_phys/oxide_traps.py --files <..._L1_101_...RJ45ADCDATA0> ... --out amp_phys/results/v3n/oxide_traps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strip_noise import load  # noqa: E402

Q, KB = 1.602176634e-19, 1.380649e-23
T_SKIP, TAU, DT = 32.6e-6, 10e-6, 13.8e-6
NS = (1, 2, 5, 10, 20, 40, 80, 160, 240)


def transfer_integrals(Ns):
    f = np.logspace(-1, np.log10(7.5e6), 400000)
    df = np.gradient(f)
    x = np.pi * f * TAU
    cds = 4 * np.sin(np.pi * f * DT) ** 2 * (np.sin(x) / x) ** 2
    out = {}
    for N in Ns:
        y = np.pi * f * T_SKIP
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = (np.sin(N * y) / (N * np.sin(y))) ** 2
        avg = np.nan_to_num(avg, nan=1.0)
        H = cds * avg
        out[N] = (np.sum(H * df), np.sum(H / f * df))
    return out


def sigma_N(own, N):
    nb = own.shape[1] // N
    if nb < 2:
        return np.nan
    b = own[:, :nb * N].reshape(own.shape[0], nb, N).mean(2)
    return np.std(np.diff(b, axis=1)) / np.sqrt(2)


def steps(own, w=25, k=6.0):
    amps, gaps = [], []
    for row in own:
        c = np.convolve(row, np.ones(w) / w, "valid")
        d = c[w:] - c[:-w]
        s = np.std(row) * np.sqrt(2 / w)
        idx = np.flatnonzero(np.abs(d) > k * s)
        if len(idx) == 0:
            continue
        # keep the local maxima of |d| (one per step)
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > w) + 1)
        pos = [g[np.argmax(np.abs(d[g]))] for g in groups]
        amps += [abs(d[p]) for p in pos]
        gaps += list(np.diff(pos))
    return np.array(amps), np.array(gaps)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--gain-adu-per-uv", type=float, default=0.875)
    ap.add_argument("--wl-um2", type=float, default=345.0)
    ap.add_argument("--t", type=float, default=130.0)
    ap.add_argument("--lam-nm", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    I = transfer_integrals(NS)
    own_all = {}
    for f in a.files:
        V = load(f, 20)
        if V is None:
            continue
        chs = sorted(V)
        res = {c: V[c] - V[c].mean(1, keepdims=True) for c in chs}
        for c in chs:
            ref = np.mean([res[o] for o in chs if o != c], 0)
            beta = (res[c].ravel() @ ref.ravel()) / (ref.ravel() @ ref.ravel())
            own_all.setdefault(c, []).append(res[c] - beta * ref)
    rows, curves = [], {}
    for c, lst in sorted(own_all.items()):
        own = np.concatenate(lst)
        sN = np.array([sigma_N(own, N) for N in NS])
        ok = np.isfinite(sN)
        A = np.array([[I[N][0], I[N][1]] for N in NS])[ok]
        y = sN[ok] ** 2
        w = 1 / y                                               # relative errors
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        Sw, Kv = coef
        curves[c] = (sN, Sw, Kv)
        amps, gaps = steps(own)
        r = {"channel": c, "sigma_1 own (ADU)": sN[0], "sigma_240 own (ADU)": sN[-1],
             "white-only sigma_240 (ADU)": sN[0] / np.sqrt(240),
             "K_v output (ADU^2)": Kv, "1/f share at N=240": Kv * I[240][1] / (sN[-1] ** 2),
             "RTS steps per 1000 skips": 1000 * len(amps) / own.size, "median step (ADU)": np.median(amps) if len(amps) else np.nan,
             "median spacing (skips)": np.median(gaps) if len(gaps) else np.nan}
        for Aname, Av in (("A=0.85", 0.85), ("A=0.44", 0.44)):
            Kvg = max(Kv, 0) / (a.gain_adu_per_uv * 1e6) ** 2 / Av ** 2          # V^2 at the gate
            for Cname, Cff in (("C_ox", 0.454), ("C_eff", 0.125)):
                Ccm = Cff * 1e-15 / 1e-8                                       # F/cm^2
                Nt = Kvg * a.wl_um2 * 1e-8 * Ccm ** 2 / (Q ** 2 * KB * a.t * a.lam_nm * 1e-7) * Q   # per eV per cm^3
                r[f"N_t {Aname} {Cname} (eV^-1 cm^-3)"] = Nt
                if Aname == "A=0.85":
                    r[f"one-charge step {Cname} (ADU)"] = Q / (Cff * 1e-15 * a.wl_um2) * Av * a.gain_adu_per_uv * 1e6
        rows.append(r)
    R = pd.DataFrame(rows).set_index("channel")
    R.to_csv(out / "oxide_traps.csv")
    L = ["# Oxide trap density from each amplifier's own 1/f noise (L1 strips, shared part removed)", "",
         R.T.to_markdown(floatfmt=".3g"), "",
         "N_t is an upper limit (any own slow noise that is not the transistor's 1/f also counts).  Typical thermal-oxide",
         "values: 10^16-10^17 eV^-1 cm^-3 (good) to 10^18 (poor)."]
    (out / "oxide_traps.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, (sN, Sw, Kv) in curves.items():
        ax.loglog(NS, sN, "o", label=f"ch{c} measured")
        model = np.sqrt([Sw * I[N][0] + Kv * I[N][1] for N in NS])
        ax.loglog(NS, model, "-", lw=1)
    ax.loglog(NS, curves[min(curves)][0][0] / np.sqrt(np.array(NS)), "k:", lw=1, label="white only (1/√N)")
    ax.set_xlabel("skips averaged N")
    ax.set_ylabel("own noise of the N-skip average (ADU)")
    ax.set_title("Own (unshared) noise vs averaging, with white + 1/f fits", loc="left", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "oxide_traps.png", dpi=150)


if __name__ == "__main__":
    main()

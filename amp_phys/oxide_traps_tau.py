"""
oxide_traps_tau.py -- the output transistor's 1/f noise (-> oxide trap density) from how each amplifier's OWN
single-skip noise depends on the CDS window length tau (v2 varied delayInt 300-3000 cycles; every v2 packet has an
L1 strip at its own settings, all channels simultaneous).

Why the window: through the CDS and the skip averaging, 1/f noise averages down almost like white noise, so its level
cannot be read from the noise-vs-N curve (oxide_traps.py).  But white noise scales as 1/tau while 1/f is nearly
independent of tau: sigma_1^2(tau) separates them.

Per strip and channel: own series = per-skip value - beta x mean of the other channels (pixel means removed);
sigma_1 = std(consecutive-skip difference) / sqrt 2.  Model, with the exact filters for each tau:
    sigma_1^2 = S_w * I_w(tau) + K * I_f(tau)
    |H|^2 = 4 sin^2(pi f dt) sinc^2(f tau) * 2 sin^2(pi f T)        (CDS x consecutive-skip difference / sqrt 2)
    dt = tau + 3.8 us (centre-to-centre; SW gap 380 cycles), T = 32.6 us + 2 (tau - 10 us) (skip period)
K (ADU^2) -> gate-referred K_vg = K / G^2 / A^2 -> McWhorter N_t = K_vg W L C^2 / (q^2 k T lambda) (per eV per cm^3),
for A = 0.85 / 0.44 and C = C_ox (0.454 fF/um^2) / buried-channel C_eff (0.125); 1/f corner f_c = K / S_w.
An upper limit: own slow noise that is not the transistor's also enters K.

    .venv/bin/python amp_phys/oxide_traps_tau.py --run-dir <v2 raw> --data processed_v2 --out amp_phys/results/v2/oxide_traps_tau
"""

from __future__ import annotations

import argparse
import glob
import re
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
F = np.logspace(0, np.log10(7.5e6), 300000)
DF = np.gradient(F)


def integrals(tau):
    dt = tau + 3.8e-6
    T = 32.6e-6 + 2 * (tau - 10e-6)
    x = np.pi * F * tau
    H = 4 * np.sin(np.pi * F * dt) ** 2 * (np.sin(x) / x) ** 2 * 2 * np.sin(np.pi * F * T) ** 2
    return np.sum(H * DF), np.sum(H / F * DF)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--gain-adu-per-uv", type=float, default=0.875)
    ap.add_argument("--wl-um2", type=float, default=345.0)
    ap.add_argument("--t", type=float, default=130.0)
    ap.add_argument("--lam-nm", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(Path(a.data) / "dataset.csv", low_memory=False)
    di = d[d.board == 101].groupby("packet_id")["theta.delayInt"].first()
    rows = []
    for f in sorted(glob.glob(str(Path(a.run_dir) / "*_l1new_*_L1_101_*.RJ45ADCDATA0"))):
        pid = re.match(r"(.+?)_l1new_", Path(f).name).group(1)
        if pid not in di.index:
            continue
        V = load(f, 20)
        if V is None:
            continue
        chs = sorted(V)
        res = {c: V[c] - V[c].mean(1, keepdims=True) for c in chs}
        for c in chs:
            ref = np.mean([res[o] for o in chs if o != c], 0)
            beta = (res[c].ravel() @ ref.ravel()) / (ref.ravel() @ ref.ravel())
            own = res[c] - beta * ref
            rows.append({"packet": pid, "channel": c, "tau_us": di[pid] * 0.01,
                         "sigma_1 own": np.std(np.diff(own, axis=1)) / np.sqrt(2),
                         "sigma_1 total": np.std(np.diff(V[c], axis=1)) / np.sqrt(2)})
    D = pd.DataFrame(rows)
    D.to_csv(out / "per_strip.csv", index=False)
    taus = np.sort(D.tau_us.unique())
    I = {t: integrals(t * 1e-6) for t in taus}
    res = []
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for c, g in D.groupby("channel"):
        for kind, ax in (("own", axs[0]), ("total", axs[1])):
            y = g[f"sigma_1 {kind}"].to_numpy() ** 2
            X = np.array([I[t] for t in g.tau_us])
            w = 1 / y
            (Sw, K), *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
            r = {"channel": c, "kind": kind, "strips": len(g), "S_w (ADU^2/Hz)": Sw, "K (ADU^2)": K,
                 "1/f corner f_c (kHz)": K / Sw / 1e3 if Sw > 0 else np.nan,
                 "white ASD at input (nV/rtHz)": np.sqrt(max(Sw, 0)) / a.gain_adu_per_uv * 1e3,
                 "1/f share of sigma_1^2 at tau=10us": K * integrals(10e-6)[1] / (Sw * integrals(10e-6)[0] + K * integrals(10e-6)[1])}
            for An, Av in (("A=0.85", 0.85), ("A=0.44", 0.44)):
                Kvg = max(K, 0) / (a.gain_adu_per_uv * 1e6) ** 2 / Av ** 2
                for Cn, Cff in (("C_ox", 0.454), ("C_eff", 0.125)):
                    Ccm = Cff * 1e-15 / 1e-8
                    r[f"N_t {An} {Cn} (eV^-1 cm^-3)"] = Kvg * a.wl_um2 * 1e-8 * Ccm ** 2 / (Q ** 2 * KB * a.t * a.lam_nm * 1e-7) * Q
            res.append(r)
            tt = np.linspace(taus.min(), taus.max(), 50)
            ax.plot(g.tau_us, g[f"sigma_1 {kind}"], "o", ms=3, label=f"ch{c}")
            ax.plot(tt, np.sqrt([Sw * integrals(t * 1e-6)[0] + K * integrals(t * 1e-6)[1] for t in tt]), lw=1)
    for ax, t in zip(axs, ("own (shared part removed)", "total")):
        ax.set_xlabel("CDS window τ (µs)")
        ax.set_ylabel("single-skip noise σ₁ (ADU)")
        ax.set_title(f"σ₁ vs window, {t}: white ∝ 1/√τ, 1/f ≈ constant", loc="left", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "oxide_traps_tau.png", dpi=150)
    R = pd.DataFrame(res)
    R.to_csv(out / "oxide_traps_tau.csv", index=False)
    L = ["# 1/f noise and oxide trap density from the CDS-window dependence (v2 strips)", "",
         R.round(4).to_markdown(index=False), "",
         "N_t: upper limit. Thermal SiO2 typically 1e16-1e17 /eV/cm^3 (good) to 1e18 (poor)."]
    (out / "oxide_traps_tau.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

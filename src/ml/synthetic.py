"""
synthetic.py -- a toy "CCD world" that produces packets in the format of dataset.csv.

There are no labeled packets with varied operating parameters yet.  To build and test the models
before the real scan, this module simulates what a scan would produce:

* operating parameters theta (10 of them, real names and nominal values from the .bcf);
* for every amplifier, the LABELS a WADERs fit of the image would give (read noise sigma_e,
  dark charge lambda, gain), from smooth toy "physics" with an optimum, trade-offs, a forbidden
  region (VDRAIN too close to VDD) and amplifier-to-amplifier differences;
* the TRACE FEATURES (a subset of the 85, with the real names) and four trace CURVES, as
  functions of theta, with the measured repeat scatter;
* a slowly drifting hidden STATE (think: temperature, or the CCD's history) that is not part of
  theta.  It changes the dark charge strongly and the noise a little, and it also shifts a few
  trace features.  A model that sees only theta cannot know it; a model that sees the traces can.
* a HIDDEN EFFECT that is deliberately left out of the hand-made feature list: intermittent pickup
  (an outside source that is on in ~40 % of the packets) that adds a noise floor not averaging
  down with the number of skips (sigma_N^2 = sigma_1^2/N + sigma_f^2).  How strongly the CCD picks
  it up depends on theta (SW_low, delay_H), but WHETHER it is on is not in theta: it is visible only
  in the curves (Allan deviation and CDS spectrum).  This tests whether a network on the curves
  can find an effect the hand-made features miss.

THE NUMBERS ARE TOY VALUES chosen to resemble what was measured (sigma_1 ~ 5.8 e-/skip on three
amplifiers, ~4x worse on one; gain ~1.3 ADU/e-; lambda ~ 0.08 e-/pixel in the 16-row image;
repeat scatter of the features).  The functional forms are invented.  Nothing in this module may
be used as physics; it exists to test the code and to compare model types.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: The operating parameters of the toy world: name -> (nominal, low, high).  The real scan has
#: one value per pin; here groups of pins move together (e.g. all H low rails) to keep d = 10.
PARAMETERS = {
    "VDD":      (-19.0, -21.0, -17.0),     # output drain (VDD1 = VDD2)
    "VR":       (-3.0, -4.0, -2.0),        # reset drain
    "VSUB":     (45.0, 35.0, 55.0),        # substrate
    "VDRAIN":   (-23.0, -24.5, -21.5),     # dump drain
    "H_low":    (6.6, 5.6, 7.2),           # horizontal low rails (H1/H2/H3, A/B/C together)
    "V_low":    (7.8, 6.8, 8.4),           # vertical low rails
    "TG":       (7.4, 6.6, 8.0),           # transfer gate (high = low)
    "RG_high":  (9.0, 7.5, 10.5),          # reset gate high rail
    "SW_low":   (-6.0, -7.5, -4.5),        # summing well low rail
    "delay_H":  (1800.0, 1000.0, 3000.0),  # horizontal clock hold (delay_Ha = Hb = Hc), 10 ns ticks
}
NAMES = list(PARAMETERS)
NOMINAL = np.array([PARAMETERS[n][0] for n in NAMES])
LOW = np.array([PARAMETERS[n][1] for n in NAMES])
HIGH = np.array([PARAMETERS[n][2] for n in NAMES])

N_SKIPS = 500                              # skips of the image (and of the matched traces)
ALLAN_N = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 160])

#: The hand-made features of the toy world (real names; the subset that the toy physics moves,
#: plus three that respond to nothing).  The noise floor's features (noise of 10/50-skip
#: averages, correlated noise floor, Allan slope) are deliberately NOT in this list.
FEATURES = [
    "horizontal: minimum", "horizontal: maximum", "slow: H-transfer peak-to-peak (ADU)",
    "vertical: minimum", "vertical: maximum",
    "fit: settling time after pulse (us)", "fit: dip recovery time (us)",
    "skip: pulse peak height", "skip: dip height", "skip: baseline at skip 1",
    "fit: baseline droop amplitude", "skip: CDS rms (detrended)",
    "noise: reset-level scatter (ADU)", "edge: rise time (ns)",
    "discrete: CDS excess kurtosis", "noise: within-cycle power 1-3 MHz", "slow: drift (ADU/ms)",
]
#: Repeat scatter of one trace set (ADU, µs, ...), roughly as measured on 2026-10-12.
REPEAT_SD = np.array([30, 20, 30, 20, 30, 0.001, 0.002, 30, 20, 30, 20, 0.15, 3.0, 1.0, 0.05, 0.002, 0.3])

#: The curves of the toy world, defined like the real ones (trace_features.py): every waveform curve is
#: relative to its own baseline, so ABSOLUTE LEVELS (e.g. "skip: baseline at skip 1") are in the
#: hand-made features only, as in the real curve set.
CURVES = {"skip: mean shape": 489, "skip: baseline droop curve": 499, "horizontal: mean shape": 473,
          "vertical: mean shape": 491, "edge: variation along the cycle": 489,
          "noise: Allan deviation": len(ALLAN_N), "noise: CDS spectrum": 246}


@dataclass
class Amplifier:
    """The fixed properties of one amplifier in the toy world."""
    name: str
    sigma1: float          # single-skip read noise at its optimum (e-)
    gain: float            # ADU per e-
    vdd_opt: float         # VDD of lowest noise
    offsets: np.ndarray    # its own offset of every feature
    shape_scale: float     # its own scale of the waveform curves


def make_amplifiers(rng: np.random.Generator, n_boards: int = 1) -> list[Amplifier]:
    """Four amplifiers per board, the third about 4x noisier (as CCD_A on module 101)."""
    amps = []
    for b in range(n_boards):
        # a second module differs as a whole: its own gain, noise and VDD optimum (board 0 is the reference)
        g_board = 1.0 if b == 0 else rng.uniform(0.8, 1.2)
        s_board = 1.0 if b == 0 else rng.uniform(0.8, 1.3)
        v_board = 0.0 if b == 0 else rng.normal(0, 0.5)
        for ch, (s1, g) in enumerate([(5.8, 1.34), (5.9, 1.33), (23.0, 2.68), (6.0, 1.27)]):
            amps.append(Amplifier(name=f"b{b}_ch{ch}", sigma1=s1 * s_board * rng.uniform(0.95, 1.05),
                                  gain=g * g_board * rng.uniform(0.97, 1.03), vdd_opt=-19.5 + v_board + rng.normal(0, 0.3),
                                  offsets=rng.normal(0, 1, len(FEATURES)) * REPEAT_SD * 50,
                                  shape_scale=rng.uniform(0.9, 1.1)))
    return amps


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def physics(theta: np.ndarray, amp: Amplifier, state: float, pickup: float = 0.0,
            image_only: bool = False) -> dict[str, float]:
    """
    The toy truth for one amplifier: single-skip noise, noise floor, labels.

    theta: the 10 parameters (order of NAMES).  state: the hidden drift (~N(0,1)).
    pickup: 1 when the intermittent outside source is on during this packet, else 0.
    image_only: add a noise term that only the image sees ("world v3"): low-frequency noise between
    rows (think: the video board or the substrate drifting over the 4.8-min image) that a 0.2-s trace
    of 10 pixels cannot measure.  It depends on VR and delay_H.  Models built on the trace physics
    must learn it from the labels.
    """
    VDD, VR, VSUB, VDRAIN, H_low, V_low, TG, RG_high, SW_low, delay_H = theta
    dH, dV, dTG = H_low - 6.6, V_low - 7.8, TG - 7.4
    dRG, dSW, dDH = RG_high - 9.0, SW_low + 6.0, (delay_H - 1800) / 600
    # single-skip noise: optimum in VDD and VR, reset-gate dependence, charge-transfer loss when the
    # H low rails are too high (small swing), forbidden region when VDRAIN is within 2.5 V of VDD
    sigma1 = amp.sigma1 * (1 + 0.05 * (VDD - amp.vdd_opt) ** 2 + 0.04 * (VR + 3.3) ** 2 + 0.03 * dRG ** 2
                           + 0.6 * max(dH - 0.2, 0) ** 2 + 0.03 * dDH ** 2)
    sigma1 *= 1 + 2.0 * max(VDRAIN - VDD + 2.5, 0) ** 2
    sigma1 *= 1 + 0.03 * state
    # hidden effect: intermittent pickup -> a noise floor (e-) that does not average down with skips;
    # the susceptibility h grows when SW_low is raised and with longer H clocks
    h = (0.3 + _sigmoid(3 * (dSW - 0.3))) * (1 + 0.4 * dDH)
    floor = 0.25 * max(h, 0) * pickup
    lf = 0.10 * (1 + 0.5 * (VR + 3) ** 2) * (1 + 0.3 * dDH) if image_only else 0.0
    sigma_e = np.sqrt(sigma1 ** 2 / N_SKIPS + floor ** 2 + lf ** 2)
    gain = amp.gain * (1 + 0.02 * (VDD + 19) - 0.015 * (VR + 3)) * (1 + 0.005 * state)
    # dark charge per pixel in the 16-row image: dark current (substrate, V low rails, temperature)
    # plus clock-induced charge (larger clock swings: lower H and V low rails, lower TG)
    dark = 0.03 * np.exp(0.04 * (VSUB - 45)) * (1 + 1.5 * _sigmoid(4 * (dV + 0.3))) * np.exp(0.35 * state)
    cic = 0.025 * np.exp(-1.2 * dH) * np.exp(-0.6 * dV) * np.exp(-0.4 * dTG)
    lam = dark + cic
    return {"sigma1": sigma1, "floor": floor, "sigma_e": sigma_e, "gain": gain, "lambda": lam, "h": h,
            "dark": dark, "cic": cic, "lf": lf}


def features(theta: np.ndarray, amp: Amplifier, state: float, phys: dict, rng: np.random.Generator,
             n_sets: int) -> np.ndarray:
    """The hand-made features of one amplifier, averaged over n_sets trace sets (repeat scatter / sqrt(n))."""
    VDD, VR, VSUB, VDRAIN, H_low, V_low, TG, RG_high, SW_low, delay_H = theta
    dVDD, dVR, dH, dV, dTG = VDD + 19, VR + 3, H_low - 6.6, V_low - 7.8, TG - 7.4
    dRG, dSW, dDH = RG_high - 9.0, SW_low + 6.0, (delay_H - 1800) / 600
    true = np.array([
        -100000 + 30000 * dH,                                   # horizontal: minimum
        30000 + 4000 * dH - 2000 * dDH,                         # horizontal: maximum
        130000 - 20000 * dH + 3000 * dDH,                       # H-transfer peak-to-peak
        -23000 + 11000 * dV + 8000 * dTG,                       # vertical: minimum
        15000 - 3000 * dV,                                      # vertical: maximum
        0.63 - 0.006 * dVDD + 0.002 * dVR + 0.001 * state,      # settling time (µs)
        0.32 - 0.007 * dVDD,                                    # dip recovery time (µs)
        46000 + 3000 * dRG - 300 * dVDD,                        # pulse peak height
        -13000 + 1800 * dSW,                                    # dip height
        -19500 + 1500 * dVR + 800 * dVDD + 300 * state + 3000 * dH,   # baseline at skip 1
        -2400 + 3000 * dH + 200 * state,                        # baseline droop amplitude
        phys["sigma1"] * amp.gain,                              # CDS rms per skip (ADU): sigma_1 only
        140 + 2 * (phys["sigma1"] - amp.sigma1) + 5 * state,    # reset-level scatter
        20 - 2 * dRG + 1.5 * dVDD,                              # rise time (ns)
        0.0, 0.0, 0.0,                                          # respond to nothing
    ])
    return true + amp.offsets + rng.normal(0, 1, len(FEATURES)) * REPEAT_SD / np.sqrt(n_sets)


def curves(theta: np.ndarray, amp: Amplifier, state: float, phys: dict, rng: np.random.Generator,
           n_sets: int) -> dict[str, np.ndarray]:
    """
    The trace curves of one amplifier, averaged over n_sets trace sets.  They carry the same
    information as the hand-made features (the features are computed from the trace), except the
    absolute levels, which the real curves remove too.  They also carry the noise floor of the
    intermittent pickup, which no hand-made feature of the toy world describes.
    """
    VDD, VR, VSUB, VDRAIN, H_low, V_low, TG, RG_high, SW_low, delay_H = theta
    dVDD, dVR, dH, dV, dTG = VDD + 19, VR + 3, H_low - 6.6, V_low - 7.8, TG - 7.4
    dRG, dSW, dDH = RG_high - 9.0, SW_low + 6.0, (delay_H - 1800) / 600
    noise = lambda size, sd: rng.normal(0, sd / np.sqrt(n_sets), size)
    k = np.arange(489.0)
    settle_us = 0.63 - 0.006 * dVDD + 0.002 * dVR + 0.001 * state           # as the settling-time feature
    tau = settle_us * 15                                                    # samples
    dip_tau = (0.32 - 0.007 * dVDD) * 15 / 2
    pulse = (46000 + 3000 * dRG - 300 * dVDD) * np.exp(-np.clip(k - 365, 0, None) / tau) * (k >= 365)
    dip = (-13000 + 1800 * dSW) * np.exp(-np.clip(k - 158, 0, None) / dip_tau) * (k >= 158)
    skip_shape = amp.shape_scale * (pulse + dip) + noise(k.size, 30)
    kk = np.arange(1, 500.0)
    droop = (-2400 + 3000 * dH + 200 * state) * (1 - np.exp(-kk / 236)) + noise(kk.size, 10)
    j = np.arange(473.0) / 473
    h_shape = amp.shape_scale * ((-100000 + 30000 * dH) * np.sin(np.pi * j) ** 2 * 0.5
                                 + (30000 + 4000 * dH - 2000 * dDH) * np.sin(3 * np.pi * j) * np.exp(-2 * j))
    h_shape += noise(j.size, 40)
    v = np.arange(491.0) / 491
    v_shape = amp.shape_scale * ((-23000 + 11000 * dV + 8000 * dTG) * np.sin(2 * np.pi * v) * np.exp(-v)
                                 + (15000 - 3000 * dV) * np.sin(4 * np.pi * v) ** 2) + noise(v.size, 30)
    reset_scatter = 140 + 2 * (phys["sigma1"] - amp.sigma1) + 5 * state     # as the reset-level-scatter feature
    s1_adu, fl_adu = phys["sigma1"] * amp.gain, phys["floor"] * amp.gain
    variation = np.full(489, s1_adu * 10.0)
    variation[150:170] += 1000 * np.exp(-np.arange(20) / 4)
    variation[365:400] += 3000 * np.exp(-np.arange(35) / 5)
    variation[400:] = reset_scatter * 1.5
    variation *= np.exp(noise(489, 0.03))
    allan = np.sqrt(s1_adu ** 2 / ALLAN_N + fl_adu ** 2) * (1 + noise(ALLAN_N.size, 0.03))
    f = np.linspace(0, 15e3, 246)                               # Hz
    spectrum = s1_adu ** 2 / 15e3 * (1 + (fl_adu / max(s1_adu, 1e-9)) ** 2 * N_SKIPS * 200 / (1 + (f / 30) ** 2))
    spectrum *= np.exp(noise(f.size, 0.1))
    return {"skip: mean shape": skip_shape, "skip: baseline droop curve": droop, "horizontal: mean shape": h_shape,
            "vertical: mean shape": v_shape, "edge: variation along the cycle": variation,
            "noise: Allan deviation": allan, "noise: CDS spectrum": spectrum}


def allowed(thetas: np.ndarray) -> np.ndarray:
    """
    The toy world's relation between parameters (a stand-in for the real ones, which ccd_limits.py enforces
    from docs/voltage_rules.txt, e.g. VDD > VDRAIN): the dump drain must stay at least 2.5 V more negative
    than VDD.  Beyond that the noise explodes.
    """
    thetas = np.atleast_2d(thetas)
    return thetas[:, NAMES.index("VDRAIN")] - thetas[:, NAMES.index("VDD")] <= -2.5


def oat_thetas(n_values: int = 20) -> np.ndarray:
    """
    The one-at-a-time design of scan_packets.py (``type = "oat"``): every parameter at n_values evenly
    spaced values, the others at nominal.  Points that break the relations are dropped (as the real
    limit checks refuse them).
    """
    out = []
    for i in range(len(NAMES)):
        for v in np.linspace(LOW[i], HIGH[i], n_values):
            t = NOMINAL.copy()
            t[i] = v
            out.append(t)
    out = np.array(out)
    return out[allowed(out)]


def sample_theta(rng: np.random.Generator, n: int, design: str = "random", respect_relations: bool = True) -> np.ndarray:
    """
    Operating points: 'random' (uniform in the ranges, as a random scan design; points that break
    the relations are redrawn, as the real scan's limit checks would refuse them) or 'nominal'.
    """
    if design == "nominal":
        return np.tile(NOMINAL, (n, 1))
    out = np.empty((0, len(NAMES)))
    while len(out) < n:
        cand = LOW + (HIGH - LOW) * rng.random((2 * n, len(NAMES)))
        out = np.vstack([out, cand[allowed(cand)] if respect_relations else cand])
    return out[:n]


#: Rows of the image and the relative scatter of one row's lambda (a WADERs fit of ~1000 pixels with
#: ~0.08 e- each: ~80 events, ~12 %); the mean of the 16 rows then scatters by 3 %, as the total label.
N_ROWS, ROW_LAMBDA_SD = 16, 0.12


def lambda_per_row(dark: float, cic: float, rng: np.random.Generator) -> tuple[float, float, float]:
    """
    Per-row dark charge of the image, and its split into dark current and clock-induced charge.

    With EXPOSURE 0 a pixel of row r collects dark current while the rows below it are read, so the
    dark part grows linearly with r (mean over rows = ``dark``) while CIC is the same in every row:
    lambda_r = cic + dark (r + 1/2) / (N_ROWS / 2).  A straight-line fit of the measured rows gives
    the two parts.  Returns (mean over rows, fitted dark, fitted cic).
    """
    x = (np.arange(N_ROWS) + 0.5) / (N_ROWS / 2)
    rows = (cic + dark * x) * (1 + ROW_LAMBDA_SD * rng.normal(size=N_ROWS))
    slope, intercept = np.polyfit(x, rows, 1)
    return float(rows.mean()), float(slope), float(intercept)


def make_dataset(n_packets: int, seed: int = 0, n_boards: int = 1, n_sets: int = 10, n_ref_sets: int = 5,
                 thetas: np.ndarray | None = None, drift: float = 0.95, label_noise: bool = True,
                 world_seed: int = 1234, image_only_noise: bool = False, per_row_lambda: bool = False,
                 within_packet_pickup: bool = False, damaged_fraction: float = 0.0):
    """
    Simulate a scan: n_packets packets, each with traces at the reference and at theta and an image.

    Returns (df, curves): df has one row per (packet, amplifier) with the columns of dataset.csv
    (theta.*, delta.*, trace.*, ref.*, diff.*, label.*), plus 'truth.*' columns (the toy truth,
    for diagnostics only: never use them as inputs).  curves maps a curve name to an array
    (n_rows, length) aligned with df: the curves at theta (mean over the n_sets trace sets), and
    with the prefix "ref: " the curves of the reference traces.

    world_seed fixes the amplifiers (the "CCDs"); seed fixes the scan (operating points, drift,
    noise).  Two scans with different seeds and the same world_seed are two runs on the same CCDs.

    image_only_noise: add the image-only noise term of ``physics`` (world v3).
    per_row_lambda: measure lambda row by row (``lambda_per_row``) and add the labels
    label.lambda_dark and label.lambda_cic (world v3).  The defaults give world v2 exactly.
    within_packet_pickup, damaged_fraction (world v4, for set models): when the outside source is
    active in a packet it is on only a fraction d ~ U(0.2, 1) of the time; each trace set (0.2 s)
    sees it fully or not at all (probability d, the same sets for every amplifier), while the
    4.8-min image sees its time average (floor^2 * d).  A fraction ``damaged_fraction`` of the trace
    sets is damaged (a glitch that inflates the long-average scatter).  The Allan deviation of every
    trace set is returned as curves["sets: noise: Allan deviation"] (n_rows, n_sets, 13), and the
    usual curve is their mean.
    """
    world = np.random.default_rng(world_seed)
    amps = make_amplifiers(world, n_boards)
    rng = np.random.default_rng(seed)
    thetas = sample_theta(rng, n_packets) if thetas is None else np.asarray(thetas, float)
    state, rows, curve_rows = 0.0, [], {name: [] for name in CURVES}
    for p, theta in enumerate(thetas):
        state = drift * state + np.sqrt(1 - drift ** 2) * rng.normal()   # slow drift between packets
        pickup = float(rng.random() < 0.4)                                # the outside source, on or off
        per_set = within_packet_pickup or damaged_fraction > 0
        if per_set:
            duty = rng.uniform(0.2, 1.0) * pickup if within_packet_pickup else pickup
            on_sets = (rng.random(n_sets) < duty).astype(float) if within_packet_pickup else np.full(n_sets, pickup)
            damaged = rng.random(n_sets) < damaged_fraction
            pickup = float(np.sqrt(duty)) if within_packet_pickup else pickup   # the image sees the time average
        for a, amp in enumerate(amps):
            phys = physics(theta, amp, state, pickup, image_only_noise)
            ref_phys = physics(NOMINAL, amp, state, pickup)
            f_theta = features(theta, amp, state, phys, rng, n_sets)
            f_ref = features(NOMINAL, amp, state, ref_phys, rng, n_ref_sets)
            noise = (lambda rel: 1 + rel * rng.normal()) if label_noise else (lambda rel: 1.0)
            if per_row_lambda:
                lam_label, dark_fit, cic_fit = lambda_per_row(phys["dark"], phys["cic"], rng)
            row = {"packet_id": f"syn_{p:05d}", "board": 100 + a // 4, "channel": a % 4, "amplifier": amp.name,
                   **{f"theta.{n}": v for n, v in zip(NAMES, theta)},
                   **{f"delta.{n}": v - v0 for n, v, v0 in zip(NAMES, theta, NOMINAL)},
                   **{f"trace.{n}": v for n, v in zip(FEATURES, f_theta)},
                   **{f"ref.{n}": v for n, v in zip(FEATURES, f_ref)},
                   **{f"diff.{n}": v - r for n, v, r in zip(FEATURES, f_theta, f_ref)},
                   "label.sigma_e": phys["sigma_e"] * noise(0.015),
                   "label.lambda_e_per_bin": lam_label if per_row_lambda else phys["lambda"] * noise(0.03),
                   "label.gain_adu_per_e": phys["gain"] * noise(0.008),
                   "label.ok": True,
                   "truth.state": state, "truth.pickup": pickup, "truth.floor": phys["floor"], "truth.sigma1": phys["sigma1"],
                   "truth.sigma_e": phys["sigma_e"], "truth.lambda": phys["lambda"], "truth.gain": phys["gain"],
                   "truth.dark": phys["dark"], "truth.cic": phys["cic"], "truth.image_only": phys["lf"]}
            if per_row_lambda:
                row.update({"label.lambda_dark": dark_fit, "label.lambda_cic": cic_fit})
            rows.append(row)
            theta_curves = curves(theta, amp, state, phys, rng, n_sets)
            if per_set:
                on_floor = physics(theta, amp, state, 1.0, image_only_noise)["floor"] * phys["gain"]
                s1 = phys["sigma1"] * phys["gain"]
                sets = np.sqrt(s1 ** 2 / ALLAN_N + on_sets[:, None] * on_floor ** 2)
                sets = sets * (1 + 0.03 * rng.normal(size=sets.shape))
                glitch = 1 + rng.uniform(1, 3, (n_sets, 1)) * np.sqrt(ALLAN_N / ALLAN_N[-1])
                sets = np.where(damaged[:, None], sets * glitch, sets)
                theta_curves["noise: Allan deviation"] = sets.mean(0)
                curve_rows.setdefault("sets: noise: Allan deviation", []).append(sets)
                row["truth.duty"] = duty
                row["truth.n_damaged"] = int(damaged.sum())
            for name, c in theta_curves.items():
                curve_rows[name].append(c)
            for name, c in curves(NOMINAL, amp, state, ref_phys, rng, n_ref_sets).items():
                curve_rows.setdefault("ref: " + name, []).append(c)     # the reference traces' curves
    return pd.DataFrame(rows), {k: np.array(v) for k, v in curve_rows.items()}


def objective(sigma_e: np.ndarray, lam: np.ndarray, weight_lambda: float = 0.5) -> np.ndarray:
    """The quantity the toy optimization minimizes: log sigma_e + w log lambda (both 'smaller is better')."""
    return np.log(sigma_e) + weight_lambda * np.log(lam)


def true_objective(theta: np.ndarray, channels=(0, 1, 3), weight_lambda: float = 0.5, world_seed: int = 1234) -> float:
    """Noise-free objective at theta, averaged over the good amplifiers, at state 0 (for scoring)."""
    amps = make_amplifiers(np.random.default_rng(world_seed))
    vals = []
    for ch in channels:
        ph = physics(np.asarray(theta, float), amps[ch], 0.0)
        vals.append(objective(ph["sigma_e"], ph["lambda"], weight_lambda))
    return float(np.mean(vals))

"""
simulate_manual_run.py -- simulated packets taken by hand, with the real file layout and a KNOWN truth.

The level-1 chain (manual_packet.sh -> make_manual_manifest.py -> process_packet.py with WADERs -> image_rows.py ->
ml/real_inputs.py -> train_lvl1.py) has never seen an image of the planned geometry.  This writes what a series of
hand-taken packets would leave on disk, so the whole chain can be rehearsed and every label checked against the truth:

    <out>/records/<PACKET_ID>.json     the packet records, as manual_packet.sh writes them
    <out>/run/<PACKET_ID>_ref<s>_2x5x500_1x1_ch<c>_101_<date>_<n>.{RJ45ADCDATA<c>,meta}
    <out>/run/<PACKET_ID>_new<s>_...                                     traces (simulate_trace.py), board 101
    <out>/run/img_16x1000x500_1x1_L2_101_<date>_<n>.meta, avg_<same>.fits.bz2
                                       the image of img.toml: 16 rows x 1000 columns, NDCM 500, 1x1, NCOLPRE 5300,
                                       4 amplifiers (one FITS extension each), pedestal ~31 ADU as in real images
    <out>/truth.json                   per packet and channel: sigma_e, lambda (mean of the rows), gain, dark current,
                                       CIC, lambda of every row

The TRUTH MODEL (invented, chosen to resemble module 101):
    gain g (ADU/e-)          1.310, 1.300, 1.280, 1.250 for ch0..3
    single-skip noise s1     5.8, 5.8, 23 (ch2 ~4x noisier, as measured), 5.9 e- at delayInt = 1000, scaling as
                             sqrt(1000 / delayInt); a floor 0.08 e- x (delayInt / 1000)^0.3
    sigma_e                  sqrt(s1^2 / 500 + floor^2)
    dark current DC          0.010 e-/pixel x exp((VSUB - 45) / 25)
    CIC                      0.008 e-/pixel x exp(-2 x (H low rails - nominal))
    row r of 16              lambda_r = CIC + DC (r + 1/2) / 8;  columns 0..851 active, 852..999 overscan (no charge)
    a few large events       0.03 % of the active pixels get 50..3000 e- (muon-like), as WADERs must cope with
The traces' white noise per sample is 53 ADU x s1 / 5.8 (so the trace features follow the image's noise).  The traces'
waveform is a rough imitation (simulate_trace.py): the rehearsal checks the PLUMBING and the label extraction, not
whether real traces predict real images (that is what level 1 measures on real packets).

Each packet varies delayInt, VSUB and the H low rails (inside configs/limits.toml's windows; every configuration is
checked with ccd_limits).  Reference traces are at the nominal configuration.

    python lvl1/simulate_manual_run.py --out <dir> [--packets 6 --ref-sets 2 --sets 3 --seed 0]
"""

from __future__ import annotations

import argparse
import bz2
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ccd_limits import Limits  # noqa: E402
from simulate_trace import simulate_trace  # noqa: E402

GAIN = np.array([1.310, 1.300, 1.280, 1.250])
S1 = np.array([5.8, 5.8, 23.0, 5.9])
H_LOWS = ("H1_A_L", "H1_B_L", "H2_C_L", "H3_A_L", "H3_B_L")
NROW, NCOL, NDCM, NCOLPRE, PRESCAN, PHYSCOL, PHYSROW = 16, 1000, 500, 5300, 8, 6144, 1536
N_ACTIVE = PHYSCOL + PRESCAN - NCOLPRE                          # 852 columns on the CCD, the rest overscan
TRACE_GEOMETRY = dict(NROW=2, NCOL=5, NDCM=500, NPBIN=1, NSBIN=1, EXPOSURE=0, NCOLPRE=100)


def truth_for(theta: dict, nominal: dict) -> list[dict]:
    """sigma_e, gain, dark current, CIC and per-row lambda of the 4 channels at the settings theta."""
    d_int = float(theta.get("delayInt", 1000))
    dh = np.mean([theta[k] - nominal[k] for k in H_LOWS])
    dc = 0.010 * np.exp((theta["VSUB"] - 45.0) / 25.0)
    cic = 0.008 * np.exp(-2.0 * dh)
    xr = (np.arange(NROW) + 0.5) / (NROW / 2)
    out = []
    for ch in range(4):
        s1 = S1[ch] * np.sqrt(1000.0 / d_int)
        floor = 0.08 * (d_int / 1000.0) ** 0.3
        lam_rows = cic + dc * xr
        out.append({"channel": ch, "gain": float(GAIN[ch]), "sigma1": float(s1), "floor": float(floor),
                    "sigma_e": float(np.sqrt(s1 ** 2 / NDCM + floor ** 2)), "dark": float(dc), "cic": float(cic),
                    "lambda": float(lam_rows.mean()), "lambda_rows": lam_rows.tolist()})
    return out


def write_image(stem: Path, truth: list[dict], header_keys: dict, rng) -> Path:
    """avg_<stem>.fits.bz2 (4 extensions, NROW x NCOL float32, positive charge on a ~31 ADU pedestal) and <stem>.meta."""
    from astropy.io import fits
    hdus = [fits.PrimaryHDU()]
    for k, v in header_keys.items():
        hdus[0].header[k if len(k) <= 8 else f"HIERARCH {k}"] = v
    for t in truth:
        ped = 31.0 + 0.05 * np.arange(NROW)[:, None] + rng.normal(0, 0.02, (NROW, 1))
        n = np.zeros((NROW, NCOL))
        n[:, :N_ACTIVE] = rng.poisson(np.array(t["lambda_rows"])[:, None], (NROW, N_ACTIVE))
        big = rng.random((NROW, N_ACTIVE)) < 3e-4
        n[:, :N_ACTIVE][big] += rng.uniform(50, 3000, big.sum())
        data = ped + t["gain"] * n + rng.normal(0, t["sigma_e"] * t["gain"], (NROW, NCOL))
        hdus.append(fits.ImageHDU(data.astype(np.float32), name=str(t["channel"])))
    path = stem.parent / f"avg_{stem.name}.fits.bz2"
    raw = stem.parent / f"avg_{stem.name}.fits"
    fits.HDUList(hdus).writeto(raw, overwrite=True)
    path.write_bytes(bz2.compress(raw.read_bytes()))
    raw.unlink()
    stem.with_suffix(".meta").write_text(json.dumps(header_keys, indent=1))
    return path


def simulate(out: Path, n_packets: int = 6, n_ref_sets: int = 2, n_sets: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    limits = Limits.load(ROOT / "configs" / "limits.toml")
    nominal = {**dict(limits.nominal), "delayInt": 1000}
    run, records = out / "run", out / "records"
    run.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)
    clock = datetime(2026, 10, 20, 9, 0, 0)
    count, truth_all = 0, {}

    def stamp():
        nonlocal clock, count
        clock += timedelta(seconds=40)
        count += 1
        return clock.strftime("%Y%m%d_%H%M%S"), count

    for p in range(n_packets):
        pid = f"m{p + 1:04d}"
        # settings: delayInt, VSUB and the H low rails; redrawn until every rule and window holds
        while True:
            dint = float(rng.choice([500, 700, 1000, 1500, 2000, 3000]))
            vsub = float(rng.uniform(36, 69))
            dh = float(rng.uniform(-0.5, 0.3))
            changes = {"delayInt": dint, "VSUB": round(vsub, 2), **{k: round(nominal[k] + dh, 2) for k in H_LOWS}}
            theta = {**nominal, **changes}
            if not limits.violations(theta):
                break
        record = {"record_version": 1, "packet_id": pid, "scan_name": "simulated rehearsal", "queued_at": clock.isoformat(),
                  "host": "simulated", "script": "simulate_manual_run.py", "changes": [f"{k}={v:g}" for k, v in changes.items()],
                  "bcf": ["acm101=module_PA03.bcf"], "trace_prefix_ref": f"{pid}_ref", "trace_prefix_new": f"{pid}_new",
                  "n_ref_sets": n_ref_sets, "n_sets": n_sets, "trace_channels": ["0", "1", "2", "3"],
                  "trace_args": "subroutine=AcquireImage, EXPOSURE=0, NROW=2, NCOL=5, NPBIN=1, NSBIN=1, NDCM=500, NCOLPRE=100",
                  "flush_between": "set", "image_toml": "image_params/tc1/img.toml", "leader": 101}
        (records / f"{pid}.json").write_text(json.dumps(record, indent=1))
        truth = truth_for(theta, nominal)
        truth_ref = truth_for(nominal, nominal)
        for kind, n, th, tr in (("ref", n_ref_sets, nominal, truth_ref), ("new", n_sets, theta, truth)):
            for s in range(1, n + 1):
                for ch in range(4):
                    date, c = stamp()
                    stem = run / f"{pid}_{kind}{s}_2x5x500_1x1_ch{ch}_101_{date}_{c}"
                    start = clock
                    simulate_trace(stem, seed=int(rng.integers(1 << 30)), channel=ch, charge_per_pixel=0.05,
                                   noise_sd=53.0 * tr[ch]["sigma1"] / 5.8, **TRACE_GEOMETRY,
                                   ACMID=101, DATEini=start.isoformat(), DATEend=(start + timedelta(seconds=1)).isoformat(),
                                   **{k: v for k, v in th.items() if k not in TRACE_GEOMETRY})
        date, c = stamp()
        clock += timedelta(minutes=1)
        stem = run / f"img_{NROW}x{NCOL}x{NDCM}_1x1_L2_101_{date}_{c}"
        header = {"ACMID": 101, "BINNAME": stem.name, "PRESCAN": PRESCAN, "PHYSROW": PHYSROW, "PHYSCOL": PHYSCOL,
                  "PHYSCOl": PHYSCOL, "NROW": NROW, "NCOL": NCOL, "NDCM": NDCM, "NPBIN": 1, "NSBIN": 1, "EXPOSURE": 0,
                  "NCOLPRE": NCOLPRE, "NROWPRE": 0, "DATEini": clock.isoformat(),
                  "DATEend": (clock + timedelta(minutes=5)).isoformat(),
                  **{f"ch{i}": f"CCD_{'DCAB'[i]}" for i in range(4)}, **theta}
        write_image(stem, truth, header, rng)
        clock += timedelta(minutes=8)
        truth_all[pid] = {"theta": theta, "channels": truth}
        print(f"{pid}: delayInt {theta['delayInt']:.0f}, VSUB {theta['VSUB']:.1f}, ΔH {dh:+.2f} -> "
              + ", ".join(f"ch{t['channel']} σe {t['sigma_e']:.3f} λ {t['lambda']:.4f}" for t in truth), flush=True)
    (out / "truth.json").write_text(json.dumps(truth_all, indent=1))
    return truth_all


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--packets", type=int, default=6)
    parser.add_argument("--ref-sets", type=int, default=2)
    parser.add_argument("--sets", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    simulate(Path(args.out), args.packets, args.ref_sets, args.sets, args.seed)


if __name__ == "__main__":
    main()

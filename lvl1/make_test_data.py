"""
make_test_data.py -- synthetic packets in exactly the format process_packet.py writes (model_inputs/*.npz).

Real level-1 data do not exist yet.  This writes a synthetic scan (the toy CCD world of src/ml/synthetic.py: world
v4 with the image-only noise, per-row lambda, intermittent pickup and damaged trace sets) as the same five files, so
that train_lvl1.py and the walkthrough notebook can be run end to end, and every model checked against a known
truth.  The numbers say nothing about the real CCD.

    python lvl1/make_test_data.py --out lvl1/test_data --packets 100 [--unlabeled 300]

Labels get the errors WADERs would report (1.5 % sigma_e, 3 % lambda, 0.8 % gain); dark current and CIC come with
the covariance of the straight-line fit through the 16 rows.  Every row is put on board 101.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ml import data, real_inputs, structured as st, synthetic  # noqa: E402

LABEL_ERR = (0.015, 0.03, 0.008)


def write(out: Path, n_packets: int, n_unlabeled: int, seed: int) -> list[Path]:
    kw = dict(n_boards=1, image_only_noise=True, per_row_lambda=True, within_packet_pickup=True, damaged_fraction=0.1)
    df, curves = synthetic.make_dataset(n_packets, seed=seed, **kw)
    df = df.assign(board=101, fake=False)
    Y = data.make_targets(df)[0]
    n = len(df)
    base = {"packet_id": df["packet_id"].to_numpy(str), "board": df["board"].to_numpy(int), "channel": df["channel"].to_numpy(int),
            "amplifier": np.array([f"toy_{c}" for c in df["channel"]]), "Y": Y, "Y_err": np.tile(LABEL_ERR, (n, 1)),
            "targets": np.array(real_inputs.TARGETS), "label_ok": np.ones(n, bool), "fake": np.zeros(n, bool),
            "theta": df[data.columns(df, "theta")].to_numpy(float), "theta_names": np.array(data.columns(df, "theta"))}
    own = real_inputs._own_features(df)
    names = sorted(k for k in curves if not k.startswith(("ref: ", "sets: ")))
    tensor = lambda c, nm: data.curve_tensor(c, nm, length=real_inputs.CURVE_LENGTH, log_names=real_inputs._log_names(nm),
                                             log_frequency_spectra=True)
    ref_curves = {k: curves["ref: " + k] for k in names}
    parts = df[["label.lambda_dark", "label.lambda_cic"]].to_numpy(float)
    cov = st.line_fit_covariance(parts[:, 0], parts[:, 1])
    allan_ns = np.asarray(synthetic.ALLAN_N, float)
    dest = out / "model_inputs"
    dest.mkdir(parents=True, exist_ok=True)
    files = {
        "features": {**base, **own},
        "curves": {**base, "curves_theta": tensor(curves, names), "curves_ref": tensor(ref_curves, names), "curve_names": np.array(names)},
        "physics": {**base, **own, "n_skips": np.full(n, float(synthetic.N_SKIPS)), "allan_ns": allan_ns,
                    # the median over the trace sets, as process_packet combines the curves of real traces
                    "allan_curve_name": np.array("noise: Allan deviation"), "allan": np.median(curves["sets: noise: Allan deviation"], axis=1),
                    "lambda_parts": parts, "lambda_cov": cov},
        "sets": {**base, "sets": curves["sets: noise: Allan deviation"], "n_sets": np.full(n, curves["sets: noise: Allan deviation"].shape[1]),
                 "allan_ns": allan_ns, "allan_curve_name": np.array("noise: Allan deviation")},
    }
    # unlabeled traces for pre-training: the labeled packets' traces plus extra trace-only packets
    du, cu = synthetic.make_dataset(n_unlabeled, seed=seed + 1000, **kw) if n_unlabeled else (df.iloc[:0], {k: v[:0] for k, v in curves.items()})
    stack = lambda c: np.concatenate([tensor(c, names), tensor({k: c["ref: " + k] for k in names}, names)])
    pre_c = np.concatenate([stack(curves), stack(cu)]) if n_unlabeled else stack(curves)
    th = lambda d: d[data.columns(df, "theta")].to_numpy(float)
    pre_theta = np.concatenate([th(df), th(df), th(du), th(du)]) if n_unlabeled else np.concatenate([th(df), th(df)])
    m = len(pre_c)
    files["pretrain"] = {"curves": pre_c, "curve_names": np.array(names), "packet_id": np.array([f"u{i}" for i in range(m)]),
                         "kind": np.array(["theta"] * m), "board": np.full(m, 101), "channel": np.zeros(m, int),
                         "theta": pre_theta, "theta_names": np.array(data.columns(df, "theta"))}
    written = []
    for name, arrays in files.items():
        path = dest / f"{name}.npz"
        np.savez_compressed(path, family=np.array(name), description=np.array(real_inputs.FAMILIES[name] + " (SYNTHETIC test data)"), **arrays)
        written.append(path)
    df.to_csv(out / "dataset.csv", index=False)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(ROOT / "lvl1" / "test_data"))
    parser.add_argument("--packets", type=int, default=100)
    parser.add_argument("--unlabeled", type=int, default=300, help="extra trace-only packets for pre-training")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    for p in write(Path(args.out), args.packets, args.unlabeled, args.seed):
        print("wrote", p)


if __name__ == "__main__":
    main()

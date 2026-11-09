"""
learning_curve.py -- does more training data help?  Every model is trained on random subsets of the A+B packets
(board 101) and scored on session C, at the feature set its A+B cross-validation chose (lvl1/explain_models.py).

    .venv/bin/python lvl1/learning_curve.py --sizes 8 16 24 32 --repeats 5 [--exclude-amps 101:2]
        [--exclude-features lvl1/feature_study/real_AB/amp_fingerprint_eta90.txt] [--models ...] --out FILE.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lvl1"))
sys.path.insert(0, str(ROOT / "src"))

import explain_models as em  # noqa: E402
import train_lvl1 as tl  # noqa: E402
from ml import real_inputs  # noqa: E402

TARGETS = tl.TARGETS


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", default=str(ROOT / "processed" / "model_inputs"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--models", nargs="*", default=["mean", "settings_amp_trees", "trees", "nn", "cnn", "physics_white"])
    parser.add_argument("--exclude-amps", nargs="*", default=[])
    parser.add_argument("--exclude-features")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude-fractional", action="store_true", help="drop fractional-window rows")
    em.add_config_args(parser)
    args = parser.parse_args()
    em.configure(args.tier_file, args.plan, args.holdout_session, args.ablation_report)

    files = {p.stem: real_inputs.load(p) for p in sorted(Path(args.inputs).glob("*.npz"))}
    base = files["physics"]
    ok = (np.isin(base["board"], [101]) & np.isfinite(base["Y"]).all(1) & base["label_ok"] & ~base["fake"]
          & base["trace_ok"].astype(bool))
    for bc in args.exclude_amps:
        b, c = (int(x) for x in bc.split(":"))
        ok &= ~((base["board"] == b) & (base["channel"] == c))
    if args.exclude_fractional:
        ok &= ~base["fractional_window"].astype(bool)
    rows = np.flatnonzero(ok)
    packets = base["packet_id"][rows].astype(str)
    held = np.array([tl.matches_packet(p, tl.session_packets(em.PLAN, em.HOLDOUT_SESSION)) for p in packets])
    test = np.flatnonzero(held)
    ab_packets = np.unique(packets[~held])
    runner = tl.Runner(files, rows, quick=False, tiers=tl.load_tiers(em.TIER_FILE))
    if args.exclude_features:
        runner.exclude = {ln.strip() for ln in Path(args.exclude_features).read_text().splitlines()
                          if ln.strip() and not ln.startswith("#")}
    sets = em.chosen_sets()
    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results["_info"] = vars(args) | {"n_ab_packets": len(ab_packets), "n_test_rows": len(test)}
    Y = runner.Y[test]
    for n in args.sizes:
        for r in range(args.repeats if n < len(ab_packets) else 1):
            pick = rng.choice(ab_packets, size=min(n, len(ab_packets)), replace=False)
            train = np.flatnonzero(np.isin(packets, pick) & ~held)
            for model in args.models:
                key = f"{model}|{n}|{r}"
                if key in results:
                    continue
                err = {}
                for spec in dict.fromkeys(sets[model][t] for t in TARGETS):
                    runner.feature_spec = spec
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            m, _ = getattr(runner, model)(train, test)
                    except Exception as exc:                 # noqa: BLE001 - e.g. too few rows for a model
                        print(f"{key} [{spec}] failed: {type(exc).__name__}: {exc}", flush=True)
                        continue
                    for i, t in enumerate(TARGETS):
                        if sets[model][t] == spec:
                            err[t] = float(100 * np.sqrt(np.mean((m[:, i] - Y[:, i]) ** 2)))
                results[key] = {"model": model, "n_packets": int(n), "repeat": r, "errors": err}
                print(f"{key}: " + "  ".join(f"{t} {v:.1f}" for t, v in err.items()), flush=True)
                out_path.write_text(json.dumps(results, indent=1))
    print("written", out_path)


if __name__ == "__main__":
    main()

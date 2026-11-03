"""
test_new_module.py -- do the level-1 models, trained on board 101 only, predict another module's amplifiers?

Every model is fitted once on the board-101 rows of sessions A+B (exactly the session C fit of
``train_lvl1.py --holdout-session C``, at the feature set the A+B cross-validation chose for each target;
lvl1/explain_models.py) and predicts the board-102 rows of every packet (labels OK, a usable trace).  Board 102 is
never trained on; its traces are inputs and its WADERs labels are the truth.

The board-102 rows get packet ids of their own (``<id>@102``), so the packet state of a board-101 row is computed from
board 101 alone and the board-101 fit is the same as without board 102; the pre-trained encoder is also trained on
board-101 traces only.

Scores per model and target (RMS of log(predicted / true), %), on the board-102 rows:
* error: as predicted;
* after one image: each amplifier's offset (mean log ratio) taken from its FIRST nominal packet with a usable label and
  removed, scored on its other rows -- what one imaged packet of a new module would fix;
* per amplifier, and by session (A+B: settings board 101 was trained on; C: new joint settings).

    .venv/bin/python lvl1/test_new_module.py [--out processed/lvl1_results/2026-11-04_board102]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lvl1"))
sys.path.insert(0, str(ROOT / "src"))

import explain_models as em  # noqa: E402
import train_lvl1 as tl  # noqa: E402
from ml import real_inputs  # noqa: E402

NOMINAL = ("L1_01", "L1_07", "L1_14", "L1_32", "L1_43")
TARGETS = tl.TARGETS


def rms(a) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(100 * np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")


def scores(pred: np.ndarray, Y: np.ndarray, amp: np.ndarray, pid: np.ndarray, session: np.ndarray) -> dict:
    out = {}
    for i, t in enumerate(TARGETS):
        r = pred[:, i] - Y[:, i]
        cal = np.full(len(r), np.nan)                      # after one image: each amplifier's first nominal offset removed
        used = np.zeros(len(r), bool)
        for a in np.unique(amp):
            rows = np.flatnonzero(amp == a)
            first = next((j for p in NOMINAL for j in rows if pid[j] == p and np.isfinite(r[j])), None)
            if first is None:
                continue
            used[first] = True
            cal[rows] = r[rows] - r[first]
        cal[used] = np.nan
        out[t] = {"error": rms(r), "bias": float(100 * np.nanmean(r)), "after_one_image": rms(cal),
                  "per_amp": {a: {"error": rms(r[amp == a]), "bias": float(100 * np.nanmean(r[amp == a])),
                                  "n": int((amp == a).sum())} for a in np.unique(amp)},
                  "per_session": {s: rms(r[np.isin(session, list(s))]) for s in ("AB", "C")}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", default=str(ROOT / "processed" / "model_inputs"))
    parser.add_argument("--out", default=str(ROOT / "processed" / "lvl1_results" / "2026-11-04_board102"))
    parser.add_argument("--models", nargs="*", default=list(em.BASELINES + em.TRACE_MODELS))
    args = parser.parse_args()

    files = {p.stem: real_inputs.load(p) for p in sorted(Path(args.inputs).glob("*.npz"))}
    for d in files.values():                               # board 102: packets of its own (packet state per board)
        if "board" in d and "packet_id" in d and len(d["board"]) == len(d["packet_id"]):
            pid = np.asarray(d["packet_id"]).astype(object)
            b102 = np.asarray(d["board"]) == 102
            pid[b102] = [f"{p}@102" for p in pid[b102]]
            d["packet_id"] = pid.astype(str)
    base = files["physics"]
    usable = np.isfinite(base["Y"]).all(1) & base["label_ok"] & ~base["fake"] & base["trace_ok"].astype(bool)
    rows = np.flatnonzero(usable & np.isin(base["board"], [101, 102]))
    board = base["board"][rows]
    pid = np.array([p.split("@")[0] for p in base["packet_id"][rows].astype(str)])
    plan = json.loads(em.PLAN.read_text())
    ses_of = {p["packet_id"]: str(p.get("session")) for p in plan["packets"]}
    session = np.array([ses_of.get(p, "?") for p in pid])
    train = np.flatnonzero((board == 101) & (session != "C"))
    test = np.flatnonzero(board == 102)
    amp = np.array([f"ch{c}" for c in base["channel"][rows]])[test]
    runner = tl.Runner(files, rows, quick=False, tiers=tl.load_tiers(em.TIER_FILE))
    # the encoder pre-trained on board-101 traces only (the same as in the session C run)
    keep_rows, runner.rows = runner.rows, rows[board == 101]
    if "pretrain" in files:
        runner._pretrained()
    runner.rows = keep_rows
    print(f"training: {len(train)} board-101 rows (A+B); test: {len(test)} board-102 rows "
          f"({', '.join(f'{a} {int((amp == a).sum())}' for a in np.unique(amp))})")

    sets = em.chosen_sets()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"_chosen_sets": sets, "_test": {"packet_id": pid[test].tolist(), "amplifier": amp.tolist(),
                                               "session": session[test].tolist(), "Y": runner.Y[test].tolist()}}
    for model in args.models:
        pred = np.full((len(test), 3), np.nan)
        for spec in dict.fromkeys(sets[model][t] for t in TARGETS):
            runner.feature_spec = spec
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m, _ = getattr(runner, model)(train, test)
            except tl.SkipModel as exc:
                print(f"{model} [{spec}] skipped: {exc}")
                continue
            for i, t in enumerate(TARGETS):
                if sets[model][t] == spec:
                    pred[:, i] = m[:, i]
            print(f"{model} [{spec}] {time.time() - t0:.0f} s", flush=True)
        sc = scores(pred, runner.Y[test], amp, pid[test], session[test])
        results[model] = {"sets": sets[model], "scores": sc, "pred": pred.tolist()}
        print(f"  {model:18s} " + "  ".join(f"{t} {sc[t]['error']:5.1f} (bias {sc[t]['bias']:+5.1f}, one image "
                                            f"{sc[t]['after_one_image']:5.1f})" for t in TARGETS), flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=1))
    print("written", out / "results.json")


if __name__ == "__main__":
    main()

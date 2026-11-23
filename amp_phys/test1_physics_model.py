"""
test1_physics_model.py -- test 1 of the ML prognosis: can ROBUST trace physics predict the image noise sigma_e of an
amplifier the model has never seen, better than the current models?  Train on session G, score once on sealed session H
(board 101 ch0/1/3, valid rows only).

Models (log sigma_e, electrons):
  settings: trees           voltages only (reference; the scorecard's settings models give 17.6-18.8 %)
  physics, no ML            sigma_e = c * sigma_500(trace) / gain(settings): the trace's white + 1/f levels through the
                            exact 500-skip filter (physics_features.py), the gain from a settings model, one constant c
                            fitted on the training rows
  physics features: trees   trees on the physics features only (log sigma_500, log S_w, log K, sigma_1 at the windows)
  physics + settings: trees trees on both
Cells: "new amplifier" (leave one amplifier out of training) and "trained" (all three amplifiers in training).
Scores on H: typical error = RMS of the log error (%); changes: all pairs of H rows of one amplifier, Spearman rho of
predicted vs true d log sigma_e and the size error (RMS, %).  Scorecard v3 reference on H (new amplifier): trace
physics 13.5 %, trace trees 12.7 %, settings gp 18.8 %; change rho: settings + trace ensemble 0.73, settings trees 0.64.

    .venv/bin/python amp_phys/test1_physics_model.py --data processed_v3n --features <physics_features_v3.csv> --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor


def load(a):
    d = pd.read_csv(Path(a.data) / "dataset.csv", low_memory=False)
    import json
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lvl1"))
    from fleet_tests import planned_id
    packets = json.loads(Path(a.plan).read_text())["packets"]
    session = {q["packet_id"]: str(q.get("session", "?")) for q in packets}
    d["session"] = [session.get(planned_id(str(x), session), "?") for x in d.packet_id]
    t = lambda c: d[c].astype(str).str.lower().eq("true")  # noqa: E731
    ok = d.board.eq(101) & d.channel.isin([0, 1, 3]) & d.session.isin(["G", "H"]) & t("label.ok")
    if "label.fractional_window" in d:
        ok &= ~t("label.fractional_window")
    d = d[ok].copy()
    knobs = [c for c in d.columns if c.startswith("theta.") and pd.to_numeric(d[c], errors="coerce").std() > 1e-6]
    F = pd.read_csv(a.features)
    d = d.merge(F, on=["packet_id", "board", "channel"], how="inner")
    d = d[(d["S_w"] > 0) & np.isfinite(d["sigma_500 pred (ADU)"])].reset_index(drop=True)
    return d, knobs


def trees(seed=0):
    return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.5, random_state=seed, n_jobs=-1)


def phys_cols(d):
    X = pd.DataFrame({"log s500": np.log(d["sigma_500 pred (ADU)"]), "log Sw": np.log(d["S_w"].clip(1e-12)),
                      "log K": np.log(d["K"].clip(1e-3))})
    for c in [c for c in d.columns if c.startswith("sigma_1 tau=")]:
        X[c] = np.log(d[c])
    return X


def run_cell(d, knobs, tr, te):
    y = np.log(pd.to_numeric(d["label.sigma_e"]))
    g = np.log(pd.to_numeric(d["label.gain_adu_per_e"]))
    S = d[knobs].apply(pd.to_numeric, errors="coerce").fillna(0)
    P = phys_cols(d)
    pred = {}
    m = trees().fit(S[tr], y[tr]); pred["settings: trees"] = m.predict(S[te])
    gm = trees().fit(S[tr], g[tr])
    ghat_tr, ghat_te = gm.predict(S[tr]), gm.predict(S[te])
    base_tr = P["log s500"][tr] - ghat_tr
    c = np.median(y[tr] - base_tr)
    pred["physics, no ML"] = P["log s500"][te] - ghat_te + c
    m = trees().fit(P[tr], y[tr]); pred["physics features: trees"] = m.predict(P[te])
    PS = pd.concat([P, S], axis=1)
    PS["log s500 - log gain(settings)"] = np.nan
    PS.loc[tr, "log s500 - log gain(settings)"] = base_tr
    PS.loc[te, "log s500 - log gain(settings)"] = P["log s500"][te] - ghat_te
    m = trees().fit(PS[tr], y[tr]); pred["physics + settings: trees"] = m.predict(PS[te])
    return y[te].to_numpy(), pred


def score(rows):
    out = []
    for name in rows[0][1]:
        e, rho, size = [], [], []
        for (yt, pr, amp) in [(r[0], r[1][name], r[2]) for r in rows]:
            pr = np.asarray(pr)
            e.append(pr - yt)
            i, j = np.triu_indices(len(yt), 1)
            dt, dp = yt[j] - yt[i], pr[j] - pr[i]
            rho.append(spearmanr(dp, dt)[0])
            size.append(np.sqrt(np.mean((dp - dt) ** 2)))
        e = np.concatenate(e)
        out.append({"model": name, "typical error % (RMS log)": 100 * np.sqrt(np.mean(e ** 2)),
                    "median |error| %": 100 * np.median(np.abs(e)), "change rho (all H pairs)": np.mean(rho),
                    "change size error %": 100 * np.mean(size)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--plan", default="lvl1/packet_plan_v3n/plan.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d, knobs = load(a)
    print(f"{len(d)} rows ({(d.session == 'G').sum()} G, {(d.session == 'H').sum()} H), {len(knobs)} knobs")
    res = {}
    for cell in ("new amplifier", "trained"):
        rows = []
        for amp in (0, 1, 3):
            te = (d.session.eq("H") & d.channel.eq(amp)).to_numpy()
            tr = d.session.eq("G").to_numpy() & (~d.channel.eq(amp).to_numpy() if cell == "new amplifier" else True)
            yt, pr = run_cell(d, knobs, tr, te)
            rows.append((yt, pr, amp))
        res[cell] = score(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = ["# Test 1: robust trace physics for sigma_e (train G, sealed H, board 101 ch0/1/3)", ""]
    for cell, R in res.items():
        R.to_csv(out / f"test1_{cell.replace(' ', '_')}.csv", index=False)
        L += [f"## {cell}", "", R.round(3).to_markdown(index=False), ""]
    L += ["Scorecard v3 on H, new amplifier: trace physics 13.5 %, trace trees 12.7 %, settings gp 18.8 %;",
          "change rho: ensemble settings + trace 0.73, settings trees 0.64, physics difference + correction 0.65.",
          "Trained: trace physics 7.6 %, settings + amplifier trees 7.7 %; change rho ensemble settings + trace 0.82."]
    (out / "test1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

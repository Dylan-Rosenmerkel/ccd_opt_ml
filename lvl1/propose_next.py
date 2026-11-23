"""
propose_next.py -- the LIVE optimisation step: fleet Bayesian optimisation with a validity gate, for one board being
tuned.  Each call reads everything measured so far and writes the next packet plan (run with daq_scripts/run_plan.sh).

    stage screen   which settings are worth a TRACE (seconds each)?
                   1. draw --n-draw rule-safe candidate settings (make_packet_plan_v3split.Design: the voltage windows of
                      configs/limits.toml, the relationship rules with margins, clock swings, reset-safe, rampable),
                      15 voltage groups, timing nominal;
                   2. score each with the SETTINGS-ONLY models trained on the fleet (every labelled amplifier) and on
                      this board's images so far: expected improvement of the board objective
                         J = w log sigma_e + (1 - w) log lambda,  averaged over the board's amplifiers
                      (multi-task GP: a shared response + each amplifier's offset), times P(usable image) from a tree
                      classifier trained on every usable / unusable outcome (fit failed, lambda < 0.005 e/pix);
                   3. hard gate: P(usable) < --gate is never proposed; keep the best --n-screen, spread apart;
                   4. write them as TRACE-ONLY packets (screen: true) + proposals.csv (predictions, EI, P).
    stage image    which screened setting is worth an IMAGE (~10 min)?  After the screen packets are taken and processed
                   (and amp_phys/physics_features.py + gain_features.py run on them): re-score the screened settings with
                   the validity classifier that also sees their TRACES (physics features: 18 of 20 unusable settings
                   caught on sealed v3 data, against ~10 from the settings alone), gate, and write the best --n-image as
                   IMAGE packets.
    stage status   the stopping rule: the largest expected improvement left (in %) against the measurement
                   repeatability (sigma_e ~1.5-2 %, lambda ~4 %): stop when it is below --stop.

Replays of this procedure on v3 (lvl1/fleet_tests.py bo --include-invalid --gate 0.5, amp_phys/results/fleet_bo/)
reach the best usable setting within 2 % in ~2-3.5 images with no unusable image, with or without a nominal image.

    .venv/bin/python lvl1/propose_next.py screen --data processed_v3n --board 101 --channels 0 1 3 --w 0.5 \\
        --out lvl1/live_plan_round1
    .venv/bin/python lvl1/propose_next.py image  --data processed_v3n processed_live --board 101 --channels 0 1 3 \\
        --screened lvl1/live_plan_round1/proposals.csv --phys <physics_features.csv> <gain_features.csv> --out lvl1/live_plan_round1_img
    .venv/bin/python lvl1/propose_next.py status --data ... --board 101 --channels 0 1 3 --w 0.5
Options: --new-board (ignore this board's own images: a board seen for the first time), --exclude-amps (leave
amplifiers out of the fleet), --seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

from fleet_tests import LAMBDA_FLOOR, PHYS_COLS, MultiTaskGP  # noqa: E402
from make_packet_plan_v3 import GROUPS, SWING  # noqa: E402
from make_packet_plan_v3split import Design, write_plan  # noqa: E402

LAB = {"sigma_e": "label.sigma_e", "lambda": "label.lambda_e_per_bin"}
REPEAT = {"sigma_e": 1.7, "lambda": 4.0}          # % label repeatability at nominal (v2/v3)
TIMING_PINS = ("delayInt", "delayRGH", "delay_Ha", "delay_Hb", "delay_Hc")


# ---------------------------------------------------------------------------------------------------------- the data
def load(args, d0: Design):
    D = pd.concat([pd.read_csv(Path(p) / "dataset.csv", low_memory=False) for p in args.data], ignore_index=True)
    t = lambda c: D[c].astype(str).str.lower().eq("true") if c in D else pd.Series(False, index=D.index)  # noqa: E731
    D = D[~t("label.fractional_window") & ~t("fake")].copy()
    # nominal timing only: the search moves the voltages
    for p in TIMING_PINS:
        if f"theta.{p}" in D:
            D = D[np.isclose(pd.to_numeric(D[f"theta.{p}"], errors="coerce"), d0.nominal[p]) | D[f"theta.{p}"].isna()]
    D["amp"] = D.board.astype(str) + ":" + D.channel.astype(str)
    for k, c in LAB.items():
        x = pd.to_numeric(D[c], errors="coerce")
        D[f"log_{k}"] = np.log(x.where(x > 0))
    fin = np.isfinite(D[[f"log_{k}" for k in LAB]]).all(1)
    D["valid"] = t("label.ok") & fin & (D["log_lambda"] > np.log(LAMBDA_FLOOR))
    for k in LAB:
        D.loc[~D["valid"], f"log_{k}"] = np.nan
    # the board's OTHER channels stay in the fleet (they are knowledge); only --channels enter the objective
    # an amplifier that never gave a usable image (e.g. no charge reaches it) says nothing about the SETTINGS: left out
    dead = [a for a, g in D.groupby("amp") if not g.valid.any()]
    if dead:
        print(f"left out (no usable image at all): {', '.join(dead)}")
        D = D[~D.amp.isin(dead)]
    if args.exclude_amps:
        D = D[~D.amp.isin(args.exclude_amps)]
    if args.new_board:
        D = D[~D.board.eq(args.board)]
    return D.reset_index(drop=True)


def group_values(cfg: dict, d0: Design) -> np.ndarray:
    """A setting as its 15 group deltas from nominal (a swing group: high - low change)."""
    out = []
    for g, pins in GROUPS.items():
        if g in SWING:
            hi, lo = next(iter(SWING[g].items()))
            out.append((cfg[hi] - cfg[lo]) - (d0.nominal[hi] - d0.nominal[lo]))
        else:
            out.append(cfg[pins[0]] - d0.nominal[pins[0]])
    return np.array(out, float)


def data_groups(D: pd.DataFrame, d0: Design) -> np.ndarray:
    rows = []
    for _, r in D.iterrows():
        cfg = {p: float(r[f"theta.{p}"]) if f"theta.{p}" in r and pd.notna(r[f"theta.{p}"]) else d0.nominal[p]
               for g in GROUPS.values() for p in g}
        rows.append(group_values(cfg, d0))
    return np.array(rows)


# -------------------------------------------------------------------------------------------------------- the models
class Models:
    def __init__(self, D, G, w, phys=None):
        self.mu, self.sd = G.mean(0), G.std(0) + 1e-9
        self.D, self.w = D, w
        self.X = (G - self.mu) / self.sd
        J = w * D.log_sigma_e.to_numpy() + (1 - w) * D.log_lambda.to_numpy()
        self.J, self.V, self.amp = J, D.valid.to_numpy(bool), D.amp.to_numpy()
        v = self.V
        self.gp = MultiTaskGP(self.X[v], J[v], self.amp[v])
        from sklearn.ensemble import ExtraTreesClassifier
        self.clf = None
        if (~self.V).sum() >= 2:
            self.clf = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                            random_state=0, n_jobs=-1).fit(self.X, self.V)
        self.phys = phys

    def predict_board(self, Gc, amps):
        """Mean and sd of the board objective (mean over amplifiers) for candidate settings Gc."""
        Xc = (Gc - self.mu) / self.sd
        v = self.V
        ms, ss = [], []
        for a in amps:
            m, s = self.gp.predict(self.X[v], self.J[v], self.amp[v], Xc, [a] * len(Xc))
            ms.append(m)
            ss.append(s)
        return np.mean(ms, 0), np.sqrt(np.mean(np.square(ss), 0) / len(amps))

    def best_board(self, amps):
        """The best usable board objective measured so far (mean over the board's amplifiers, per packet)."""
        B = self.D[self.D.amp.isin(amps) & self.D.valid]
        if B.empty:
            return None
        J = self.w * B.log_sigma_e + (1 - self.w) * B.log_lambda
        per = J.groupby(B.packet_id).agg(["mean", "count"])
        per = per[per["count"] == len(amps)]
        return float(per["mean"].min()) if len(per) else None

    def p_usable(self, Gc):
        if self.clf is None:
            return np.ones(len(Gc))
        p = self.clf.predict_proba((Gc - self.mu) / self.sd)
        return p[:, list(self.clf.classes_).index(True)]


def ei(m, s, best):
    z = (best - m) / s
    return (best - m) * norm.cdf(z) + s * norm.pdf(z)


def spread(Gc, order, k, min_dist):
    """The first k of ``order`` that are at least min_dist apart (standardised group space)."""
    keep = []
    for i in order:
        if all(np.linalg.norm(Gc[i] - Gc[j]) >= min_dist for j in keep):
            keep.append(i)
        if len(keep) == k:
            break
    return keep


# ---------------------------------------------------------------------------------------------------------- stages
def stage_screen(args):
    d0 = Design()
    D = load(args, d0)
    G = data_groups(D, d0)
    amps = [f"{args.board}:{c}" for c in args.channels]
    M = Models(D, G, args.w)
    print(f"fleet: {len(D)} rows, {D.amp.nunique()} amplifiers, {int((~D.valid).sum())} unusable; board {args.board} "
          f"images so far: {int(D.board.eq(args.board).sum())}")
    cands, rej = d0.draw(list(GROUPS), args.n_draw, args.seed)
    Gc = np.array([group_values({**d0.nominal, **c}, d0) for c in cands])
    m, s = M.predict_board(Gc, amps)
    best = M.best_board(amps)
    b = best if best is not None else float(m.min())
    e = ei(m, s, b)
    p = M.p_usable(Gc)
    score = np.where(p >= args.gate, e * p, -np.inf)
    order = [i for i in np.argsort(-score) if np.isfinite(score[i])]
    pick = spread((Gc - M.mu) / M.sd, order, args.n_screen, args.min_dist)
    rows = []
    for rank, i in enumerate(pick, 1):
        rows.append({"rank": rank, "pred J (board mean, log)": m[i], "pred sd": s[i], "expected improvement %": 100 * e[i],
                     "P(usable)": p[i], "score": score[i],
                     **{f"group {g}": round(x, 3) for g, x in zip(GROUPS, Gc[i])}, "changes": json.dumps(cands[i])})
    P = pd.DataFrame(rows)
    out = Path(args.out)
    if P.empty:
        print(f"no candidate passed the validity gate ({args.gate}); P(usable) ranged {p.min():.2f}-{p.max():.2f}: "
              "lower --gate or draw more candidates")
        return
    planned = [{"session": args.session, "purpose": f"live screen {r['rank']} (trace only): EI {r['expected improvement %']:.2f} %, "
                f"P(usable) {r['P(usable)']:.2f}", "changes": json.loads(r["changes"]), "block": "screen", "screen": True}
               for r in rows]
    write_plan(d0, out, args.prefix, planned, run=dict(RUN), env={}, boards=[f"acm{args.board}"],
               title=f"Live optimisation, board {args.board}: screens (traces only)",
               intro=[f"Proposed by `lvl1/propose_next.py screen` (w = {args.w}, gate {args.gate}): the {len(rows)} best of "
                      f"{len(cands)} rule-safe candidates by expected improvement x P(usable). After the run: process the "
                      "packets, run amp_phys/physics_features.py and gain_features.py, then `propose_next.py image`."])
    P.insert(0, "packet_id", [q["packet_id"] for q in planned])
    P.to_csv(out / "proposals.csv", index=False)
    print(P[["packet_id", "rank", "expected improvement %", "P(usable)", "pred J (board mean, log)"]].round(3).to_string(index=False))
    print(f"best board objective so far: {best}; candidates drawn {len(cands)} (rule failures redrawn: {rej}); "
          f"gated out: {int((p < args.gate).sum())}")


def stage_image(args):
    d0 = Design()
    D = load(args, d0)
    G = data_groups(D, d0)
    amps = [f"{args.board}:{c}" for c in args.channels]
    M = Models(D, G, args.w)
    S = pd.read_csv(args.screened)
    cands = [json.loads(c) for c in S["changes"]]
    Gc = np.array([group_values({**d0.nominal, **c}, d0) for c in cands])
    m, s = M.predict_board(Gc, amps)
    best = M.best_board(amps)
    b = best if best is not None else float(m.min())
    e = ei(m, s, b)
    p = M.p_usable(Gc)
    # the traces of the screened settings: a validity classifier that also sees the physics features
    if args.phys:
        F = pd.concat([pd.read_csv(f) for f in args.phys]).groupby(["packet_id", "board", "channel"], as_index=False).first()
        cols = [c for c in PHYS_COLS if c in F]
        DF = D[["packet_id", "board", "channel"]].merge(F, on=["packet_id", "board", "channel"], how="left")
        tr = DF[cols].notna().all(1).to_numpy()
        from sklearn.ensemble import ExtraTreesClassifier
        lx = lambda T: np.column_stack([np.log(np.abs(T[c].to_numpy(float)).clip(1e-9)) if c != "CDS offset (ADU)"  # noqa: E731
                                        else T[c].to_numpy(float) for c in cols])
        Xf = np.hstack([M.X[tr], lx(DF[tr])])
        if (~M.V[tr]).sum() >= 2:
            clf = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0,
                                       n_jobs=-1).fit(Xf, M.V[tr])
            SF = F[F.board.eq(args.board) & F.channel.isin(args.channels)]
            pt = []
            for pid in S["packet_id"]:
                q = SF[SF.packet_id.str.startswith(str(pid))]
                if q.empty:
                    pt.append(np.nan)
                    continue
                Xq = np.hstack([np.repeat(((Gc[len(pt)] - M.mu) / M.sd)[None], len(q), 0), lx(q)])
                pt.append(float(clf.predict_proba(Xq)[:, list(clf.classes_).index(True)].min()))   # the worst amplifier
            pt = np.array(pt)
            p = np.where(np.isfinite(pt), pt, p)
            print(f"validity from the traces for {int(np.isfinite(pt).sum())} of {len(S)} screened settings")
    score = np.where(p >= args.gate, e * p, -np.inf)
    order = [i for i in np.argsort(-score) if np.isfinite(score[i])][:args.n_image]
    planned = [{"session": args.session, "purpose": f"live image (from screen {S['packet_id'][i]}): EI {100 * e[i]:.2f} %, "
                f"P(usable) {p[i]:.2f}", "changes": cands[i], "block": "image"} for i in order]
    out = Path(args.out)
    write_plan(d0, out, args.prefix, planned, run=dict(RUN), env={}, boards=[f"acm{args.board}"],
               title=f"Live optimisation, board {args.board}: image", intro=["Proposed by `lvl1/propose_next.py image`."])
    R = S.assign(**{"EI % (now)": 100 * e, "P(usable) with traces": p, "chosen": [i in order for i in range(len(S))]})
    R.to_csv(out / "image_choice.csv", index=False)
    print(R[["packet_id", "EI % (now)", "P(usable) with traces", "chosen"]].round(3).to_string(index=False))


def stage_status(args):
    d0 = Design()
    D = load(args, d0)
    G = data_groups(D, d0)
    amps = [f"{args.board}:{c}" for c in args.channels]
    M = Models(D, G, args.w)
    cands, _ = d0.draw(list(GROUPS), args.n_draw, args.seed)
    Gc = np.array([group_values({**d0.nominal, **c}, d0) for c in cands])
    m, s = M.predict_board(Gc, amps)
    best = M.best_board(amps)
    if best is None:
        print("no usable image of this board yet: continue")
        return
    p = M.p_usable(Gc)
    e = np.where(p >= args.gate, ei(m, s, best), 0.0)
    left = 100 * float(e.max())
    rep = args.w * REPEAT["sigma_e"] + (1 - args.w) * REPEAT["lambda"]
    print(f"best usable board objective so far {best:.4f} (log); largest expected improvement left {left:.2f} % "
          f"(stop at {args.stop:g} %; label repeatability for w = {args.w}: ~{rep:.1f} %)")
    print("STOP: further images are unlikely to beat the best by more than the threshold" if left < args.stop
          else "CONTINUE: run the screen stage again")


RUN = {"N_REF_SETS": 1, "N_SETS": 3, "N_POST_SETS": 1, "SCREEN_REF_SETS": 0, "SCREEN_SETS": 2,
       "IMAGE_TOML": "image_params/tc1/img32.toml"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["screen", "image", "status"])
    ap.add_argument("--data", nargs="+", required=True, help="processed folders (dataset.csv) with everything measured")
    ap.add_argument("--board", type=int, required=True)
    ap.add_argument("--channels", type=int, nargs="+", default=[0, 1, 3])
    ap.add_argument("--w", type=float, default=0.5, help="J = w log sigma_e + (1 - w) log lambda")
    ap.add_argument("--gate", type=float, default=0.5)
    ap.add_argument("--n-draw", type=int, default=600)
    ap.add_argument("--n-screen", type=int, default=12)
    ap.add_argument("--n-image", type=int, default=1)
    ap.add_argument("--min-dist", type=float, default=0.75, help="screens at least this far apart (standardised)")
    ap.add_argument("--screened", help="image: proposals.csv of the screen stage")
    ap.add_argument("--phys", nargs="*", default=[], help="image: physics/gain feature csv files covering the screens")
    ap.add_argument("--stop", type=float, default=1.0, help="status: stop below this expected improvement (%%)")
    ap.add_argument("--new-board", action="store_true", help="ignore the board's own images (first contact)")
    ap.add_argument("--exclude-amps", nargs="*", default=[])
    ap.add_argument("--session", default="L")
    ap.add_argument("--prefix", default="LIVE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "lvl1" / "live_plan"))
    a = ap.parse_args()
    {"screen": stage_screen, "image": stage_image, "status": stage_status}[a.stage](a)


if __name__ == "__main__":
    main()

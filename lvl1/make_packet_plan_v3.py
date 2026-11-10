"""
make_packet_plan_v3.py -- the third level-1 packet plan (2026-11-10): a VOLTAGE-focused training set.  One changes
file per packet, plan.json and PLAN.md, in ``lvl1/packet_plan_v3/``.  v1 and v2 are kept as they were taken.

What v2 showed (lvl1/lambda_settings_figures.py, lvl1/voltage_dark_current.py, 2026-11-10):
* the dark charge lambda is driven by the timing: lambda ~ delayInt^0.5 on every amplifier, plus delayRGH.  That part
  is simple and transfers between amplifiers; it hides the voltages;
* with the timing removed, lambda still varies by ~9 % RMS (label noise ~2.5 %): real voltage effects.  Almost all
  of it is VSUB, and VSUB's effect differs 8x between amplifiers (ch1/ch3 strong, ch0/ch2 flat);
* every other voltage moved by at most one max_step (0.5 V for the clocks), in LOCKED groups (V1/V2/V3 lows together,
  H1/H2/H3 lows together, the highs never), and always together with a timing change (13 of 52 packets at nominal
  timing): their effects (< 1.5 % per SD) cannot be resolved or told apart;
* sigma_e direction falls from ~90 % to ~74 % when delayInt does not change: the subtler effects need training data;
* L2_72 (delayRGH and VSUB both at their maximum, long delayInt) had lambda 80 % above the settings model: an
  untested corner, an interaction;
* delayInt not a multiple of 20 cycles gives fractional CDS windows (N / N+1 samples) and biased image labels.

What v3 does:
* session G, training (most of it at NOMINAL TIMING, so every change is a voltage change):
  - VOLTAGE SOBOL: a joint space-filling design over 15 voltage groups that now move INDEPENDENTLY (V1, V2, V3 lows
    apart; H1, H2, H3 lows apart; the V and H clock HIGHS added, so a low-level effect can be told from a swing
    effect), each over its whole voltage window (cut by the group's rules);
  - ONE-AT-A-TIME sweeps: every group to its lowest and highest allowed value with all else nominal, and VSUB at
    5 levels: the single effects, easy to read and to check the joint fits against;
  - VOLTAGE x TIMING: Sobol voltage points at a short and a long delayInt (the voltage effects may scale with the
    readout time);
  - the CORNER: delayRGH and VSUB high together at three delayInt values (the L2_72 interaction, now measured);
  - nominal every NOMINAL_EVERY packets (drift), replicates of earlier points (label noise at non-nominal settings),
    and references at the previous packet's settings every second packet (changes in both directions), as v2;
* session H, the prospective test: a separate voltage Sobol sequence (nominal timing) and a few voltage x timing
  points, scored once after the choice is frozen;
* every delayInt is a multiple of 20 cycles (whole ADC samples: no fractional windows).

Rules (2026-11-10, the user: the voltage WINDOWS and the RELATIONSHIPS between the voltages are what must hold; the
v1/v2 restriction "every voltage within one max_step of nominal" is dropped): every configuration passes ccd_limits
(the windows of configs/limits.toml, the ACM manual's hardware ranges, the group's voltage rules: V lows above every
TG level, H lows below every TG level, SW_L at or below OG_L, H3 lows above SW_H, ...); points that fail are redrawn.
manual_packet.sh applies a changes file one line at a time from nominal, so the lines of every changes file are
ORDERED so that every intermediate state passes the same rules too (a point with no such order is redrawn).  Changes
on board 101 only; both readout stages / both CCD halves (A, B) together.  The order of the packets is shuffled.

    python lvl1/make_packet_plan_v3.py [--out lvl1/packet_plan_v3] [--sobol 48] [--test 16] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lvl1"))

from ccd_limits import Limits  # noqa: E402
from make_packet_plan import TIMING  # noqa: E402
from make_packet_plan_v2 import packet_minutes  # noqa: E402

#: voltage groups (pins that move together: both readout stages, both CCD halves A/B); each a DELTA from nominal (V)
GROUPS = {
    "VSUB": ["VSUB"], "VDD": ["VDD1", "VDD2"], "VR": ["VR1", "VR2"],
    "RG_H": ["RG_1_H", "RG_2_H"], "RG_L": ["RG_1_L", "RG_2_L"], "SW_L": ["SW_1_L", "SW_2_L"],
    "V1_L": ["V1_A_L", "V1_B_L"], "V2_L": ["V2_C_L"], "V3_L": ["V3_A_L", "V3_B_L"],
    "V_swing": ["V1_A_H", "V1_B_H", "V2_C_H", "V3_A_H", "V3_B_H"],
    "H1_L": ["H1_A_L", "H1_B_L"], "H2_L": ["H2_C_L"], "H3_L": ["H3_A_L", "H3_B_L"],
    "H_swing": ["H1_A_H", "H1_B_H", "H2_C_H", "H3_A_H", "H3_B_H"],
    "TG": ["TG_A_H", "TG_A_L", "TG_B_H", "TG_B_L"],
}
WHY = {
    "VSUB": "substrate: depletion, dark current (the strongest voltage effect on lambda in v2, amplifier-dependent)",
    "VDD": "output drain: amplifier operating point", "VR": "reset drain", "RG_H": "reset-gate high", "RG_L": "reset-gate low",
    "SW_L": "summing-well low", "V1_L": "vertical phase 1 low: surface dark current, clock-induced charge",
    "V2_L": "vertical phase 2 low", "V3_L": "vertical phase 3 low",
    "V_swing": "vertical clock swing (high - low; the highs were never varied before): with the lows, separates level from swing",
    "H1_L": "horizontal phase 1 low: serial clock-induced charge", "H2_L": "horizontal phase 2 low",
    "H3_L": "horizontal phase 3 low", "H_swing": "horizontal clock swing (high - low; never varied before)",
    "TG": "transfer gate (high = low); bounds the V lows from below and the H lows from above",
}
#: the clock highs are sampled as a SWING: high = nominal high + (its own low's change) + the swing delta, so a
#: high follows its low and the swing (high - low) is a knob of its own (review 2026-11-10: sampled independently,
#: V_H - V_L fell below 1 V in most packets, which breaks the parallel transfer)
SWING = {"V_swing": {"V1_A_H": "V1_A_L", "V1_B_H": "V1_B_L", "V2_C_H": "V2_C_L", "V3_A_H": "V3_A_L", "V3_B_H": "V3_B_L"},
         "H_swing": {"H1_A_H": "H1_A_L", "H1_B_H": "H1_B_L", "H2_C_H": "H2_C_L", "H3_A_H": "H3_A_L", "H3_B_H": "H3_B_L"}}
MIN_SWING = 1.5            # V: every V and H clock keeps high - low >= this
MARGIN = 0.5               # V: every relationship rule a packet touches keeps at least this margin (the user, 2026-11-10)
RAMP_VSUB = 5.0            # V: VSUB ramped in steps of at most this; every other voltage in steps of its max_step
VSUB_LEVELS = (-10.0, -5.0, 5.0, 12.5, 25.0)   # one-at-a-time VSUB sweep (35, 40, 50, 57.5, 70 V)
VSUB_REPEAT = 1
MIDPOINTS = ("V1_L", "H1_L", "V_swing", "H_swing", "TG")   # one-at-a-time midpoints too
MIN_OAT = 0.2              # V: one-at-a-time points closer to nominal than this are left out
DELAYINT_LEVELS = (500, 2500)               # voltage x timing blocks (multiples of 20)
#: the corner as a factorial at three delayInt: delayRGH high alone, VSUB high alone, both (the L2_72 interaction)
CORNER = [{**rgh, **vs, **({"delayInt": t} if t != 1000 else {})}
          for t in (500, 1000, 2500) for rgh, vs in (({"delayRGH": 1500}, {}), ({}, {"VSUB": 20.0}),
                                                      ({"delayRGH": 1500}, {"VSUB": 20.0}))]
NOMINAL_EVERY = 7
REPLICATES = 4
REF_PREVIOUS_EVERY = 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "lvl1" / "packet_plan_v3"))
    ap.add_argument("--sobol", type=int, default=36, help="voltage Sobol points at nominal timing (session G)")
    ap.add_argument("--per-timing", type=int, default=12, help="voltage Sobol points at each delayInt of DELAYINT_LEVELS")
    ap.add_argument("--test", type=int, default=20, help="voltage Sobol points of the prospective test (session H)")
    ap.add_argument("--test-timing", type=int, default=4, help="voltage x timing points of the test")
    ap.add_argument("--no-oat", action="store_true", help="leave out the one-at-a-time sweeps")
    ap.add_argument("--n-ref", type=int, default=3)
    ap.add_argument("--n-new", type=int, default=3)
    ap.add_argument("--n-post", type=int, default=2)
    ap.add_argument("--image-rows", type=int, default=32)
    ap.add_argument("--image-toml", default="image_params/tc1/img32.toml")
    ap.add_argument("--prefix", default="L3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    limits = Limits.load(ROOT / "configs" / "limits.toml")
    nominal = {**dict(limits.nominal), **TIMING}
    step = {p: (limits.window_for(p).max_step if limits.window_for(p) else None) for p in nominal}
    rng = np.random.default_rng(args.seed)
    names = list(GROUPS)

    ramp_step = {p: (RAMP_VSUB if p == "VSUB" else (step.get(p) or 0.5)) for p in nominal}

    def margins(cfg: dict):
        """(rule, a - b) for every relationship rule of src/ccd_limits.py (a must be above b).  TG high = TG low
        (equality allowed, they move together) is left out."""
        for pin, v in cfg.items():
            if pin.endswith("_H") and not pin.startswith(("TG", "delay")) and pin[:-2] + "_L" in cfg:
                yield f"{pin} > {pin[:-2]}_L", v - cfg[pin[:-2] + "_L"]
        for i in "12":
            for a, b in ((f"VDD{i}", f"VDRAIN{i}"), (f"SW_{i}_H", f"OG_{i}_H"), (f"DG_{i}_H", f"VR{i}"),
                         (f"VR{i}", f"DG_{i}_L"), (f"VR{i}", f"OG_{i}_L"), (f"OG_{i}_L", f"SW_{i}_L"),
                         (f"DG_{i}_L", f"VDRAIN{i}"), (f"VR{i}", f"VDD{i}")):
                if a in cfg and b in cfg:
                    yield f"{a} > {b}", cfg[a] - cfg[b]
        tg = [cfg[p] for p in cfg if p.startswith("TG_")]
        for p in cfg:
            if p[0] == "V" and p[1] in "123" and p.endswith("_L"):
                yield f"{p} > TG", cfg[p] - max(tg)
            if p[0] == "H" and p[1] in "123" and p.endswith("_L"):
                yield f"TG > {p}", min(tg) - cfg[p]
        for p in ("H3_A_L", "H3_B_L"):
            for q in ("SW_1_H", "SW_2_H"):
                yield f"{p} > {q}", cfg[p] - cfg[q]

    nom_margin = dict(margins(nominal))

    def margin_problems(changes: dict) -> list[str]:
        cfg = {**nominal, **changes}
        out = []
        for rule, m in margins(cfg):
            need = MIN_SWING if (" > " in rule and rule.split(" > ")[0][:1] in "VH" and rule.split(" > ")[0].endswith("_H")
                                 and rule.split(" > ")[0][1] in "123") else MARGIN   # (a relation left exactly as at nominal is fine)
            if m < need - 1e-9 and abs(m - nom_margin.get(rule, m)) > 1e-9:
                out.append(f"{rule}: margin {m:.2f} < {need:g}")
        return out

    def ramp(start: dict, target: dict) -> list[tuple[str, float]] | None:
        """The lines that take the board from ``start`` to ``target`` (full configurations): every voltage in steps of
        at most its ramp step, all moving together; within a step the pins are applied one at a time in an order that
        keeps every intermediate configuration inside the windows and the rules.  None if there is no such order."""
        pins = [p for p in target if not p.startswith("delay") and abs(target[p] - start[p]) > 1e-9]
        k = max([int(np.ceil(abs(target[p] - start[p]) / ramp_step[p] - 1e-9)) for p in pins] + [1])
        state, seq = dict(start), []
        for j in range(1, k + 1):
            todo = {p: (target[p] if j == k else round(start[p] + j / k * (target[p] - start[p]), 3)) for p in pins}
            while todo:
                for p in sorted(todo, key=lambda q: abs(todo[q] - state[q])):
                    if not limits.violations({**state, p: todo[p]}):
                        state[p] = todo.pop(p)
                        seq.append((p, state[p]))
                        break
                else:
                    return None
        seq += [(p, target[p]) for p in target if p.startswith("delay") and target[p] != start[p]]
        return seq

    def problems(changes: dict) -> list[str]:
        v = limits.violations({**nominal, **changes}) + margin_problems(changes)
        if v:
            return v
        cfg = {**nominal, **changes}
        if ramp(nominal, cfg) is None or ramp(cfg, nominal) is None:
            return ["no rule-safe ramp"]
        return []

    def expand(gv: dict, timing=None) -> dict:
        """{group: delta} -> {pin: value}; a swing group's highs follow their lows."""
        ch = {}
        for g, x in gv.items():
            if g not in SWING:
                for p in GROUPS[g]:
                    ch[p] = round(nominal[p] + x, 2)
        for g, x in gv.items():
            if g in SWING:
                for hi, lo in SWING[g].items():
                    ch[hi] = round(nominal[hi] + (ch.get(lo, nominal[lo]) - nominal[lo]) + x, 2)
        return {**ch, **(timing or {})}

    # the allowed range of every group ALONE (all else nominal): its window, cut by the rules and the margins
    box = {}
    for g in names:
        wins = [limits.window_for(p) for p in GROUPS[g]]
        lo = max(w.minimum - nominal[p] for w, p in zip(wins, GROUPS[g]))
        hi = min(w.maximum - nominal[p] for w, p in zip(wins, GROUPS[g]))
        grid = np.round(np.arange(np.floor(lo * 20) / 20, hi + 1e-9, 0.05), 3)
        ok = [x for x in grid if not problems(expand({g: x}))]
        box[g] = (float(min(ok)), float(max(ok))) if ok else (0.0, 0.0)

    def to_changes(u, timing=None) -> dict:
        return expand({g: box[g][0] + x * (box[g][1] - box[g][0]) for g, x in zip(names, u)}, timing)

    def draw(n, seed, timing=None):
        sob = qmc.Sobol(len(names), scramble=True, seed=seed)
        pts, rej = [], 0
        while len(pts) < n:
            for u in sob.random(16):
                c = to_changes(u, timing)
                if problems(c):
                    rej += 1
                elif len(pts) < n:
                    pts.append(c)
        return pts, rej

    def group_change(g, delta, extra=None):
        gv = {g: delta}
        ex = dict(extra or {})
        if "VSUB" in ex:
            gv["VSUB"] = ex.pop("VSUB")
        return expand(gv, ex)

    G, rejected = [], {}
    pts, rejected["voltage Sobol"] = draw(args.sobol, args.seed)
    G += [{"purpose": "voltage Sobol (nominal timing)", "changes": c, "block": "voltage"} for c in pts]
    if not args.no_oat:
        for g in names:
            if g == "VSUB":
                continue
            lo, hi = box[g]
            levels = {lo, hi} | ({round(lo / 2, 2), round(hi / 2, 2)} if g in MIDPOINTS else set())
            for d_ in sorted(x for x in levels if abs(x) >= MIN_OAT):
                ch = group_change(g, d_)
                if not problems(ch):
                    G.append({"purpose": f"one-at-a-time: {g} {d_:+g} V", "changes": ch, "block": "oat"})
        for d_ in VSUB_LEVELS:
            ch = group_change("VSUB", d_)
            if box["VSUB"][0] - 1e-9 <= d_ <= box["VSUB"][1] + 1e-9 and not problems(ch):
                for r in range(VSUB_REPEAT):
                    G.append({"purpose": f"one-at-a-time: VSUB {d_:+g} V" + (" (repeat)" if r else ""), "changes": ch,
                              "block": "oat"})
    for k, t in enumerate(DELAYINT_LEVELS):
        pts, rejected[f"voltage x delayInt {t}"] = draw(args.per_timing, args.seed + 10 + k, {"delayInt": t})
        G += [{"purpose": f"voltage Sobol at delayInt {t}", "changes": c, "block": "voltage x timing"} for c in pts]
    for c in CORNER:
        c = dict(c)
        ch = expand({"VSUB": c.pop("VSUB")} if "VSUB" in c else {}, c)
        what = " + ".join(x for x, on in (("delayRGH 1500", "delayRGH" in ch), ("VSUB 65 V", "VSUB" in ch)) if on)
        if not problems(ch):
            G.append({"purpose": f"corner factorial: {what} at delayInt {ch.get('delayInt', 1000)}", "changes": ch,
                      "block": "corner"})
    for p in G:
        assert not problems(p["changes"]), p
        assert p["changes"].get("delayInt", 1000) % 20 == 0
    H = []
    pts, rejected["test voltage Sobol"] = draw(args.test, args.seed + 1000)
    H += [{"purpose": "test: voltage Sobol (nominal timing)", "changes": c, "block": "voltage"} for c in pts]
    for k in range(args.test_timing):
        t = DELAYINT_LEVELS[k % len(DELAYINT_LEVELS)]
        c, r = draw(1, args.seed + 2000 + k, {"delayInt": t})
        H.append({"purpose": f"test: voltage Sobol at delayInt {t}", "changes": c[0], "block": "voltage x timing"})

    def session(points, name, replicates, end_nominal=True):
        order = [{**p, "session": name} for p in points]
        rng.shuffle(order)
        if replicates:
            idx = rng.choice(len(order) // 2, size=replicates, replace=False)
            for k, i in enumerate(sorted(idx)):
                rep = {**order[i], "purpose": f"REPLICATE of {order[i]['purpose']} (packet {i + 1} of this session)"}
                pos = len(order) // 2 + (k + 1) * (len(order) // 2) // (replicates + 1)
                order.insert(pos, rep)
        res = []
        for i, p in enumerate(order):
            if i % (NOMINAL_EVERY - 1) == 0:
                res.append({"session": name, "purpose": "nominal (drift, label noise)", "changes": {}, "block": "nominal"})
            res.append(p)
        if end_nominal:
            res.append({"session": name, "purpose": "nominal (end of the session)", "changes": {}, "block": "nominal"})
        return res

    # G ends straight into H's first nominal (no two nominals in a row)
    planned = session(G, "G", REPLICATES, end_nominal=False) + session(H, "H", 0)
    eligible = [i for i in range(1, len(planned)) if planned[i]["session"] == "G" and planned[i]["changes"]
                and planned[i - 1]["changes"]]
    for k, i in enumerate(eligible):
        if k % REF_PREVIOUS_EVERY == REF_PREVIOUS_EVERY - 1:
            ref = planned[i - 1]["changes"]
            if not problems(ref):
                planned[i]["reference_changes"] = ref

    def lines_of(seq):
        return [f"acm101:{k}={v}" if isinstance(v, int) else f"acm101:{k}={v:g}" for k, v in seq]

    for old in list(out.glob(f"{args.prefix}_*.txt")):
        old.unlink()
    n_lines = []
    for i, p in enumerate(planned, start=1):
        p["packet_id"] = f"{args.prefix}_{i:03d}"
        timing = {k: p["changes"].get(k, v) for k, v in TIMING.items()}
        p["minutes"] = round(packet_minutes(timing, args.n_ref, args.n_new + args.n_post, args.image_rows), 1)
        head = [f"# {p['packet_id']} (session {p['session']}): {p['purpose']}",
                f"# estimated {p['minutes']:.0f} min; generated by lvl1/make_packet_plan_v3.py (seed {args.seed})",
                "# lines in a rule-safe order, large changes ramped (a pin may appear several times: the last value holds)"]
        for kind, ch in (("", p["changes"]), ("_ref", p.get("reference_changes"))):
            if kind and not ch:
                continue
            cfg = {**nominal, **ch}
            fwd, back = ramp(nominal, cfg), ramp(cfg, nominal)
            assert fwd is not None and back is not None, p["packet_id"]
            body = lines_of(fwd) if ch else ["# no changes: the nominal configuration (manual_packet.sh needs one line; "
                                             "this re-sends a nominal value)", "acm101:delayInt=1000"]
            n_lines.append(len(fwd))
            title = head if not kind else [f"# {p['packet_id']}: the reference settings (those of the previous packet)",
                                           head[2]]
            (out / f"{p['packet_id']}{kind}.txt").write_text("\n".join(title + body) + "\n")
            if ch:
                # the way back to nominal, in a rule-safe order, ramped: for manual_packet.sh to send before its reset
                # (the .bcf reload restores the pins in an order we do not control)
                (out / f"{p['packet_id']}{kind}_undo.txt").write_text("\n".join(
                    [f"# {p['packet_id']}{kind}: back to nominal, rule-safe order (send before the .bcf reset)"]
                    + lines_of(back)) + "\n")

    # ---- PLAN.md
    blocks = {}
    for p in planned:
        blocks.setdefault((p["session"], p["block"]), []).append(p)
    hrs = lambda ps: sum(q["minutes"] for q in ps) / 60
    env = f"N_REF_SETS={args.n_ref} N_SETS={args.n_new} N_POST_SETS={args.n_post} IMAGE_TOML={args.image_toml} "
    doc = ["# Level-1 packet plan, version 3: the voltages", "",
           "Generated by `lvl1/make_packet_plan_v3.py` (reasons and rules in its docstring). Take the packets in order with", "",
           "```bash", f"PLAN_DIR=lvl1/packet_plan_v3 bash daq_scripts/run_plan.sh      # or one at a time:",
           f"{env}PACKET_ID={args.prefix}_NNN CHANGES_FILE=lvl1/packet_plan_v3/{args.prefix}_NNN.txt bash daq_scripts/manual_packet.sh",
           "```", "", "(`DRY_RUN=1` first.)", "",
           f"**Voltage windows:** configs/limits.toml (`reviewed = {str(limits.reviewed).lower()}`). Every voltage may use "
           "its whole window (no max_step restriction); every configuration, and every intermediate state of the "
           "one-line-at-a-time application, passes the windows and the group's relationship rules (src/ccd_limits.py).", "",
           "| session | block | packets | time | use |", "|---|---|---|---|---|"]
    use = {"voltage": "voltages only, joint Sobol, nominal timing", "oat": "one voltage group at a time: lowest / highest allowed (and midpoints for V1_L, H1_L, the swings, TG); VSUB at 35, 40, 50, 57.5, 70 V",
           "voltage x timing": f"voltage Sobol at delayInt {', '.join(map(str, DELAYINT_LEVELS))}",
           "corner": "factorial: delayRGH 1500 alone, VSUB 65 V alone, both, at delayInt 500 / 1000 / 2500 (the L2_72 interaction)", "nominal": "drift, label noise"}
    for (s, b), ps in blocks.items():
        doc.append(f"| {s} | {b} | {len(ps)} | {hrs(ps):.1f} h | {use.get(b, b)}{' (prospective test, scored once)' if s == 'H' else ''} |")
    G_ps = [p for p in planned if p["session"] == "G"]
    H_ps = [p for p in planned if p["session"] == "H"]
    vonly = sum(1 for p in G_ps if p["changes"] and not any(k.startswith("delay") for k in p["changes"]))
    doc += [f"| G | all | {len(G_ps)} | {hrs(G_ps):.1f} h | training |", f"| H | all | {len(H_ps)} | {hrs(H_ps):.1f} h | test |",
            f"| | **total** | **{len(planned)}** | **{hrs(planned):.1f} h** | |", "",
            f"**{vonly} of {len(G_ps)}** session-G packets change voltages only (v2: 13 of 52 had the timing at nominal). "
            f"{sum(bool(p.get('reference_changes')) for p in planned)} packets take their reference at the previous "
            "packet's settings, so a within-packet change is often voltage -> voltage at fixed timing.", "",
            "## The voltage box (each group alone; joint points must also pass the rules, rejected points are redrawn)", "",
            "| group | pins | nominal | range (delta) | v2 |", "|---|---|---|---|---|"]
    v2 = {"VSUB": "+-5", "VDD": "+-1", "VR": "+-0.5", "RG_H": "+-0.5", "RG_L": "+-0.5", "SW_L": "-0.5..0",
          "V1_L": "0..+0.5, locked with V2, V3", "V2_L": "locked with V1, V3", "V3_L": "locked with V1, V2",
          "V_swing": "never varied (highs fixed)", "H1_L": "+-0.5, locked with H2, H3", "H2_L": "locked", "H3_L": "locked",
          "H_swing": "never varied (highs fixed)", "TG": "+-0.25"}
    for g in names:
        doc.append(f"| {g} | {', '.join(GROUPS[g])} | {nominal[GROUPS[g][0]]:g} | {box[g][0]:+g} to {box[g][1]:+g} V | {v2[g]} |")
    doc += ["", "Rejected joint points (broke a rule): " + ", ".join(f"{k} {v}" for k, v in rejected.items()) + ".", "",
            "Why these groups: " + "; ".join(f"**{k}** {v}" for k, v in WHY.items()) + ".", "",
            "## The order of the changes, ramps, and the way back", "",
            "manual_packet.sh sends a changes file line by line. Every file is written so that every intermediate "
            "configuration passes the windows and the relationship rules: large changes are ramped (VSUB in steps of at "
            f"most {RAMP_VSUB:g} V, every other voltage in steps of its max_step), all pins moving together, and within a "
            "step the pins are sent in a rule-safe order (a pin may appear several times; the last value holds). Every "
            f"V and H clock keeps a swing (high - low) of at least {MIN_SWING:g} V, and every relationship rule a packet "
            f"touches keeps a margin of at least {MARGIN:g} V (the rail readback tolerance is 0.05 V).", "",
            "**The way back.** Each `<id>.txt` (and `<id>_ref.txt`) has an `<id>_undo.txt` (`<id>_ref_undo.txt`): the way "
            "back to nominal, ramped and in a rule-safe order. manual_packet.sh's reset reloads the module's .bcf, which "
            "restores the pins in an order we do not control (it can put a TG high briefly below its TG low after a TG "
            "change); sending the undo file just before the reset makes the reset a no-op. manual_packet.sh does not do "
            "this yet.", "",
            "## Before taking it", "",
            f"* `{args.image_toml}` on the DAQ machine (as v2);",
            "* record the CCD temperature per packet if a readout exists (lambda drifts);",
            "* the voltage windows (`reviewed = false`).", "",
            "## The packets", "", "| packet | session | purpose | changes (board 101) | min |", "|---|---|---|---|---|"]
    for p in planned:
        ch = ", ".join(f"{k}={v}" if isinstance(v, int) else f"{k}={v:g}" for k, v in p["changes"].items()) or "–"
        pur = p["purpose"] + ("; reference at the previous packet's settings" if p.get("reference_changes") else "")
        doc.append(f"| {p['packet_id']} | {p['session']} | {pur} | {ch} | {p['minutes']:.0f} |")
    doc += ["", "## After the packets", "", "```bash",
            "python src/make_manual_manifest.py ~/packet_records/L3_*.json --run-dir <raw-file directory> --out manifests",
            "cd src && python process_packet.py ../manifests/L3_*.json && cd ..", "```", ""]
    (out / "PLAN.md").write_text("\n".join(doc))
    (out / "plan.json").write_text(json.dumps({"reviewed_limits": limits.reviewed, "groups": GROUPS, "box": box,
                                               "seed": args.seed,
                                               "run": {"N_REF_SETS": args.n_ref, "N_SETS": args.n_new, "N_POST_SETS": args.n_post,
                                                       "IMAGE_TOML": args.image_toml},
                                               "packets": planned}, indent=1, default=str))
    print("\n".join(doc[:40]))
    print(f"... {len(planned)} packets, {hrs(planned):.1f} h; wrote {out}")


if __name__ == "__main__":
    main()

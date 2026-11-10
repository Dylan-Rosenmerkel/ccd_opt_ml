"""
make_packet_plan_v3split.py -- v3 split into three plans (2026-11-10, after the read-through of v2 and the v3 draft),
each in the format of manual_packet.sh / run_plan.sh (changes files, <id>_undo.txt, plan.json, PLAN.md).
lvl1/packet_plan_v3 (make_packet_plan_v3.py) is kept as it was drafted.

Why split.  v3 asks two questions that want different data:
* the NOISE: the traces carry it (sigma_e on a new amplifier 9.7 % against 18.1 % from the settings) and the change
  models are weakest at fixed timing (direction 74 %).  -> many voltage-only packets, traces both sides, the image;
* the DARK CHARGE: the traces add nothing to the settings for lambda in v2, so its voltage map is a settings-only
  design that needs precise labels (32 rows, replicates) and almost no traces.  Most of lambda is charge made during
  the readout (the overscan held 70-85 % of it on the first night), so the dark-current RATE needs an exposure contrast.
Before either, two things nobody has measured decide whether the labels mean what we think:
* SETTLING: every packet images a few minutes after its voltage change.  If lambda (or the noise) is still relaxing
  then, every voltage effect is partly a transient (L2_72 may be one);
* EXPOSURE: how much of lambda is a rate (grows with the exposure) and how much is made by the readout.

The plans (python lvl1/make_packet_plan_v3split.py [--plan pre noise dark]):
* ``packet_plan_pre``   (PT_NN): the pre-tests.  The same setting imaged after a short and a long HOLD (the hold is
  made with PAUSE_S, the wait after every trace: the new traces are then spread over the hold and show the trace
  settling too), nominal with the same long hold as the control; nominal and VSUB + 15 V at EXPOSURE 0 / 300 / 900 s.
  (An 8-row image for the noise set was dropped: the user, 2026-11-10, keeps the full image so the noise packets
  also train the dark-charge models.)
* ``packet_plan_v3n``   (N3_NNN): the noise set.  Voltage-only joint Sobol over all 15 groups of v3 at nominal timing,
  the 32-row image (so it is the joint design for lambda as well), 1 nominal reference set + 3 new + 1 post, strips;
  sessions G (training) and H (prospective test).
* ``packet_plan_v3d``   (D3_NNN): the dark-charge add-on.  Settings-only: 8 groups with a real range (VSUB, VDD, VR, SW_L,
  H1/H2/H3 lows, H swing; the V lows and TG can move 0.5 V at most and are left out), one-at-a-time ends
  + VSUB levels, 32 rows, one trace set each side, no strips, 4 replicates, and an EXPOSURE pair (the same setting
  again with EXPOSURE 600 s, right after its EXPOSURE 0 packet) for a subset: rate against readout charge directly.

What is new in the format (backward compatible; plans without it run as before):
* plan.json "env" (plan level) and a packet's "env": environment variables for manual_packet.sh (IMAGE_TOML, PAUSE_S,
  N_REF_SETS, L1_STRIP, PHASE_TIMEOUT_S ...), passed by run_plan.sh; a packet's env wins;
* run_plan.sh passes <id>_undo.txt / <id>_ref_undo.txt as UNDO_FILE / REF_UNDO_FILE and manual_packet.sh sends them
  before every reset from those settings (the rule-safe, ramped way back: the open item of the v3 plan).
Safety as v3 (windows, relationship rules with a 0.5 V margin, swing >= 1.5 V, ramps, rule-safe line order, every
file replayed), and one more check: RESET-SAFE.  If a packet is aborted the next reset reloads the .bcf in an order we
do not control, so every point must also pass the rules with any ONE or TWO of its pins back at nominal (the rules
are pairwise, so this covers every partial reset).  Points that fail are redrawn.

    python lvl1/make_packet_plan_v3split.py                       # all three
    python lvl1/make_packet_plan_v3split.py --plan noise --train 80 --test 24 --boards acm101 acm102
"""

from __future__ import annotations

import argparse
import itertools
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
from make_packet_plan_v3 import GROUPS, MARGIN, MIN_SWING, RAMP_VSUB, SWING, WHY  # noqa: E402

IMG = {8: "image_params/tc1/img8.toml", 32: "image_params/tc1/img32.toml"}       # on the DAQ machine (configs/image_params_*)
IMG_EXP = "image_params/tc1/img32_exp{:d}.toml"                                   # 32 rows with EXPOSURE (s)
DARK_GROUPS = ("VSUB", "VDD", "VR", "SW_L", "H1_L", "H2_L", "H3_L", "H_swing")
VSUB_LEVELS = (-10.0, -5.0, 5.0, 12.5, 25.0)
NOMINAL_EVERY = 7
PAUSE_DEFAULT = 2.0


class Design:
    """The rules, ramps and sampling of make_packet_plan_v3.py (same code, as methods), plus the reset-safe check."""

    def __init__(self):
        self.limits = Limits.load(ROOT / "configs" / "limits.toml")
        self.nominal = {**dict(self.limits.nominal), **TIMING}
        step = {p: (self.limits.window_for(p).max_step if self.limits.window_for(p) else None) for p in self.nominal}
        self.ramp_step = {p: (RAMP_VSUB if p == "VSUB" else (step.get(p) or 0.5)) for p in self.nominal}
        self.nom_margin = dict(self.margins(self.nominal))
        self._box = {}

    def margins(self, cfg):
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

    def margin_problems(self, changes):
        out = []
        for rule, m in self.margins({**self.nominal, **changes}):
            a = rule.split(" > ")[0]
            need = MIN_SWING if (a[:1] in "VH" and a.endswith("_H") and a[1] in "123") else MARGIN
            if m < need - 1e-9 and abs(m - self.nom_margin.get(rule, m)) > 1e-9:
                out.append(f"{rule}: margin {m:.2f} < {need:g}")
        return out

    def ramp(self, start, target):
        """Lines from ``start`` to ``target``: ramped, every intermediate configuration inside the windows and rules."""
        pins = [p for p in target if not p.startswith("delay") and abs(target[p] - start[p]) > 1e-9]
        k = max([int(np.ceil(abs(target[p] - start[p]) / self.ramp_step[p] - 1e-9)) for p in pins] + [1])
        state, seq = dict(start), []
        for j in range(1, k + 1):
            todo = {p: (target[p] if j == k else round(start[p] + j / k * (target[p] - start[p]), 3)) for p in pins}
            while todo:
                for p in sorted(todo, key=lambda q: abs(todo[q] - state[q])):
                    if not self.limits.violations({**state, p: todo[p]}):
                        state[p] = todo.pop(p)
                        seq.append((p, state[p]))
                        break
                else:
                    return None
        seq += [(p, target[p]) for p in target if p.startswith("delay") and target[p] != start[p]]
        return seq

    def reset_unsafe(self, changes):
        """Rules broken with any one or two of the changed pins back at nominal (an uncontrolled .bcf reset)."""
        pins = [p for p in changes if not p.startswith("delay") and abs(changes[p] - self.nominal[p]) > 1e-9]
        cfg = {**self.nominal, **changes}
        for back in itertools.chain(((p,) for p in pins), itertools.combinations(pins, 2)):
            v = self.limits.violations({**cfg, **{p: self.nominal[p] for p in back}})
            if v:
                return [f"reset-unsafe with {'+'.join(back)} at nominal: {v[0]}"]
        return []

    def problems(self, changes):
        v = self.limits.violations({**self.nominal, **changes}) + self.margin_problems(changes)
        if v:
            return v
        cfg = {**self.nominal, **changes}
        if self.ramp(self.nominal, cfg) is None or self.ramp(cfg, self.nominal) is None:
            return ["no rule-safe ramp"]
        return self.reset_unsafe(changes)

    def expand(self, gv, extra=None):
        """{group: delta} -> {pin: value}; a swing group's highs follow their lows."""
        ch = {}
        for g, x in gv.items():
            if g not in SWING:
                for p in GROUPS[g]:
                    ch[p] = round(self.nominal[p] + x, 2)
        for g, x in gv.items():
            if g in SWING:
                for hi, lo in SWING[g].items():
                    ch[hi] = round(self.nominal[hi] + (ch.get(lo, self.nominal[lo]) - self.nominal[lo]) + x, 2)
        return {**ch, **(extra or {})}

    def box(self, g):
        """The allowed range of group ``g`` alone (all else nominal)."""
        if g not in self._box:
            wins = [self.limits.window_for(p) for p in GROUPS[g]]
            lo = max(w.minimum - self.nominal[p] for w, p in zip(wins, GROUPS[g]))
            hi = min(w.maximum - self.nominal[p] for w, p in zip(wins, GROUPS[g]))
            grid = np.round(np.arange(np.floor(lo * 20) / 20, hi + 1e-9, 0.05), 3)
            ok = [x for x in grid if not self.problems(self.expand({g: x}))]
            self._box[g] = (float(min(ok)), float(max(ok))) if ok else (0.0, 0.0)
        return self._box[g]

    def draw(self, names, n, seed):
        """n joint Sobol points over the groups ``names`` (each over its box); points that break a rule are redrawn."""
        sob = qmc.Sobol(len(names), scramble=True, seed=seed)
        pts, rej = [], 0
        while len(pts) < n:
            for u in sob.random(16):
                c = self.expand({g: self.box(g)[0] + x * (self.box(g)[1] - self.box(g)[0]) for g, x in zip(names, u)})
                if self.problems(c):
                    rej += 1
                elif len(pts) < n:
                    pts.append(c)
        return pts, rej


def with_nominals(points, session, rng, replicates=0, shuffle=True, end_nominal=True, units=None):
    """Shuffle (``units``: lists of packets that stay together, in order), add replicates of first-half points in the
    second half, a nominal packet every NOMINAL_EVERY packets and at the end."""
    units = units if units is not None else [[p] for p in points]
    units = [[{**p, "session": session} for p in u] for u in units]
    if shuffle:
        rng.shuffle(units)
    if replicates:
        singles = [i for i in range(len(units) // 2) if len(units[i]) == 1 and not units[i][0].get("env")]
        for k, i in enumerate(sorted(rng.choice(singles, size=min(replicates, len(singles)), replace=False))):
            rep = {**units[i][0], "purpose": f"REPLICATE of {units[i][0]['purpose']}", "block": "replicate"}
            units.insert(len(units) // 2 + (k + 1) * (len(units) // 2) // (replicates + 1), [rep])
    res, since = [], NOMINAL_EVERY
    for u in units:
        if since + len(u) > NOMINAL_EVERY - 1:
            res.append({"session": session, "purpose": "nominal (drift, label noise)", "changes": {}, "block": "nominal"})
            since = 0
        res += u
        since += len(u)
    if end_nominal:
        res.append({"session": session, "purpose": "nominal (end of the session)", "changes": {}, "block": "nominal"})
    return res


def write_plan(d: Design, out: Path, prefix: str, planned: list, run: dict, env: dict, boards: list, title: str,
               intro: list, digits: int = 3):
    """Changes files (+ undo), plan.json and PLAN.md in the format of run_plan.sh.  Returns the replay summary."""
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob(f"{prefix}_*.txt")):
        old.unlink()
    plan_env = {**run, **env}

    def lines_of(seq):
        return [f"{b}:{k}={v}" if isinstance(v, int) else f"{b}:{k}={v:g}" for k, v in seq for b in boards]

    def minutes(p):
        e = {**plan_env, **p.get("env", {})}
        timing = {k: p["changes"].get(k, v) for k, v in TIMING.items()}
        rows = next((r for r, f in IMG.items() if e["IMAGE_TOML"] == f), 32)
        m = packet_minutes(timing, int(e["N_REF_SETS"]), int(e["N_SETS"]) + int(e["N_POST_SETS"]), rows)
        pauses = 4 * (int(e["N_REF_SETS"]) + int(e["N_SETS"]) + int(e["N_POST_SETS"])) + 2 + 1
        m += pauses * (float(e.get("PAUSE_S", PAUSE_DEFAULT)) - PAUSE_DEFAULT) / 60 + p.get("exposure_s", 0) / 60
        return round(m + (1.2 if str(e.get("L1_STRIP", 1)) == "1" else 0.0), 1)       # two 2-row strips

    worst_step, n_files = 0.0, 0
    for i, p in enumerate(planned, start=1):
        p["packet_id"] = f"{prefix}_{i:0{digits}d}"
        p["minutes"] = minutes(p)
        assert not d.problems(p["changes"]), (p["packet_id"], d.problems(p["changes"]))
        cfg = {**d.nominal, **p["changes"]}
        fwd, back = d.ramp(d.nominal, cfg), d.ramp(cfg, d.nominal)
        # replay both files line by line: every intermediate state inside the windows and the rules
        for seq, start in ((fwd, d.nominal), (back, cfg)):
            state = dict(start)
            for k, v in seq:
                worst_step = max(worst_step, abs(v - state[k]) if not k.startswith("delay") and k != "VSUB" else 0.0)
                state[k] = v
                assert not d.limits.violations(state), (p["packet_id"], k, v)
        head = [f"# {p['packet_id']} (session {p['session']}): {p['purpose']}",
                f"# estimated {p['minutes']:.0f} min; generated by lvl1/make_packet_plan_v3split.py",
                "# lines in a rule-safe order, large changes ramped (a pin may appear several times: the last value holds)"]
        body = lines_of(fwd) if p["changes"] else ["# no changes: the nominal configuration (manual_packet.sh needs one "
                                                   "line; this re-sends a nominal value)", f"{boards[0]}:delayInt=1000"]
        (out / f"{p['packet_id']}.txt").write_text("\n".join(head + body) + "\n")
        n_files += 1
        if p["changes"]:
            (out / f"{p['packet_id']}_undo.txt").write_text("\n".join(
                [f"# {p['packet_id']}: back to nominal, ramped, rule-safe order (run_plan.sh passes it as UNDO_FILE)"]
                + lines_of(back)) + "\n")
            n_files += 1

    blocks = {}
    for p in planned:
        blocks.setdefault((p["session"], p["block"]), []).append(p)
    hrs = lambda ps: sum(q["minutes"] for q in ps) / 60
    doc = [f"# {title}", "", "Generated by `lvl1/make_packet_plan_v3split.py` (reasons in its docstring). Take the packets in order with", "",
           "```bash", f"DRY_RUN=1 PLAN_DIR=lvl1/{out.name} bash daq_scripts/run_plan.sh", f"PLAN_DIR=lvl1/{out.name} bash daq_scripts/run_plan.sh", "```", "",
           "Plan settings (plan.json `run` / `env`; a packet's own `env` wins): " + ", ".join(f"`{k}={v}`" for k, v in plan_env.items())
           + f". Changes go to: {', '.join(boards)}.", ""] + intro + ["",
           "| session | block | packets | time |", "|---|---|---|---|"]
    for (s, b), ps in blocks.items():
        doc.append(f"| {s} | {b} | {len(ps)} | {hrs(ps):.1f} h |")
    doc += [f"| | **total** | **{len(planned)}** | **{hrs(planned):.1f} h** |", "",
            f"Checks: every configuration passes the windows of configs/limits.toml (`reviewed = {str(d.limits.reviewed).lower()}`), "
            f"the relationship rules with a {MARGIN:g} V margin, clock swing >= {MIN_SWING:g} V, and is reset-safe (any one or two "
            f"pins back at nominal). All {n_files} files replayed line by line; largest single step {worst_step:g} V "
            f"(VSUB {RAMP_VSUB:g} V).", "",
            "## The packets", "", "| packet | session | purpose | env | changes | min |", "|---|---|---|---|---|---|"]
    for p in planned:
        ch = ", ".join(f"{k}={v}" if isinstance(v, int) else f"{k}={v:g}" for k, v in p["changes"].items()) or "–"
        ev = " ".join(f"{k}={v}" for k, v in p.get("env", {}).items()) or "–"
        doc.append(f"| {p['packet_id']} | {p['session']} | {p['purpose']} | {ev} | {ch} | {p['minutes']:.0f} |")
    doc += ["", "## After the packets", "", "```bash",
            f"python src/make_manual_manifest.py ~/packet_records/{prefix}_*.json --run-dir <raw-file directory> --out manifests",
            f"cd src && python process_packet.py ../manifests/{prefix}_*.json && cd ..", "```", ""]
    (out / "PLAN.md").write_text("\n".join(doc))
    (out / "plan.json").write_text(json.dumps({"reviewed_limits": d.limits.reviewed, "groups": GROUPS,
                                               "box": {g: d.box(g) for g in GROUPS}, "boards": boards,
                                               "run": run, "env": env, "packets": planned}, indent=1, default=str))
    print(f"{out.name}: {len(planned)} packets, {hrs(planned):.1f} h, {n_files} files, largest step {worst_step:g} V")
    for (s, b), ps in blocks.items():
        print(f"   {s} {b:<22} {len(ps):4d}  {hrs(ps):5.1f} h")


def hold_env(minutes_hold: float, n_sets: int = 3) -> dict:
    """Environment for a packet whose image comes ``minutes_hold`` after the change: the wait after every trace
    (PAUSE_S) is stretched over the 2 settling traces and the new sets; no post-image sets (they would wait too)."""
    pause = round(minutes_hold * 60 / (2 + 4 * n_sets), 1)
    return {"PAUSE_S": pause, "N_POST_SETS": 0, "PHASE_TIMEOUT_S": int(max(900, 3 * minutes_hold * 60))}


def plan_pre(d: Design, args):
    vsub = d.expand({"VSUB": 15.0})
    hlow = d.expand({"H1_L": -1.0, "H2_L": -1.0, "H3_L": -1.0})
    vdd = d.expand({"VDD": -2.0})
    nom = lambda why="nominal (drift, label noise)", **kw: {"purpose": why, "changes": {}, "block": "nominal", **kw}
    exp = lambda s: {"env": {"IMAGE_TOML": IMG_EXP.format(s)}, "exposure_s": s}
    P = [nom(),
         {"purpose": "settling: VSUB +15 V, image right after the change", "changes": vsub, "block": "settling"},
         nom(),
         {"purpose": "settling: VSUB +15 V, image 10 min after the change", "changes": vsub, "block": "settling", "env": hold_env(10)},
         nom(),
         {"purpose": "settling: VSUB +15 V, image 30 min after the change", "changes": vsub, "block": "settling", "env": hold_env(30)},
         nom("settling control: nominal with the 30 min hold (does waiting alone move the labels)", env=hold_env(30)),
         {"purpose": "settling: H lows -1 V, image right after the change", "changes": hlow, "block": "settling"},
         {"purpose": "settling: H lows -1 V, image 20 min after the change", "changes": hlow, "block": "settling", "env": hold_env(20)},
         nom(),
         {**nom("exposure: nominal, EXPOSURE 300 s"), "block": "exposure", **exp(300)},
         {**nom("exposure: nominal, EXPOSURE 900 s"), "block": "exposure", **exp(900)},
         {"purpose": "exposure: VSUB +15 V, EXPOSURE 300 s", "changes": vsub, "block": "exposure", **exp(300)},
         {"purpose": "exposure: VSUB +15 V, EXPOSURE 900 s", "changes": vsub, "block": "exposure", **exp(900)},
         nom("nominal (end)")]
    P = [{**p, "session": "P"} for p in P]
    write_plan(d, ROOT / "lvl1" / "packet_plan_pre", "PT", P, {"N_REF_SETS": 1, "N_SETS": 3, "N_POST_SETS": 0, "IMAGE_TOML": IMG[32]},
               {}, args.boards, "Pre-tests before v3: settling and exposure", [
        "**Order matters: not shuffled.** Two questions, each read off a handful of packets:",
        "* **Settling.** The same setting imaged right after the change and after a 10 / 20 / 30 min hold (the hold is "
        "`PAUSE_S`, the wait after every trace, so the 12 new traces are spread over it). If lambda or sigma_e at the long "
        "hold differs from the short hold by more than the nominal scatter, the packets image a transient and every later "
        "plan needs a hold. The nominal packet with the same hold is the control.",
        "* **Exposure.** lambda against EXPOSURE (0, 300, 900 s; the 32-row readout itself is 573 s) at nominal and at "
        "VSUB +15 V: the slope is the dark-current rate, the intercept the charge made by the readout. If the slope is "
        "small the exposure pairs of the dark-charge map can be dropped.",
        "Needed on the DAQ machine: `image_params/tc1/img32.toml`, `img32_exp300.toml`, `img32_exp900.toml` "
        "(configs/image_params_*.toml).",
        "Not checked: `process_packet.py` on images with an exposure; a `time.sleep` of ~2 min in "
        "the cdaq queue (the holds)."], digits=2)


def plan_noise(d: Design, args):
    names = list(GROUPS)
    rng = np.random.default_rng(args.seed + 1)
    g, rej_g = d.draw(names, args.train, args.seed + 100)
    h, rej_h = d.draw(names, args.test, args.seed + 1100)
    G = [{"purpose": "voltage Sobol (nominal timing)", "changes": c, "block": "voltage"} for c in g]
    H = [{"purpose": "test: voltage Sobol (nominal timing)", "changes": c, "block": "voltage"} for c in h]
    planned = with_nominals(G, "G", rng, replicates=args.replicates, end_nominal=False) + with_nominals(H, "H", rng)
    write_plan(d, ROOT / "lvl1" / "packet_plan_v3n", "N3", planned,
               {"N_REF_SETS": 1, "N_SETS": 3, "N_POST_SETS": 1, "IMAGE_TOML": IMG[args.noise_rows]}, {}, args.boards,
               "v3 noise set: voltage-only changes at nominal timing", [
        "For the sigma_e change models at fixed timing (v2: direction 74 %, settings 60 %). Every packet changes voltages "
        "only, the groups of v3 jointly over their whole allowed ranges (TG stays at nominal: a TG change is not reset-safe, "
        "one TG pin reset before the other puts its high below its low). The image has "
        f"{args.noise_rows} rows (32: the full image, so every packet is also a dark-charge label and the set trains the "
        "lambda models too; `--noise-rows 8` is a 2.4 min image for sigma_e and gain only). One nominal reference set per packet keeps the "
        "within-packet pair; the other pairs are packet against packet (v1: no worse than with references). Session G is "
        "training, H the prospective test (scored once after the freeze).",
        f"Rejected joint points (a rule, a margin, or not reset-safe): G {rej_g}, H {rej_h}.",
        "Take it only after the pre-tests: if the labels need a hold, add `PAUSE_S` to plan.json `env`.",
        "Board 102: `--boards acm101 acm102` writes every line for both boards (only if both .bcf files hold the same "
        "nominal voltages; then also `CHECK_BOARDS=\"101 102\"`)."])


def plan_dark(d: Design, args):
    names = list(DARK_GROUPS)
    rng = np.random.default_rng(args.seed + 2)
    pts, rej = d.draw(names, args.dark, args.seed + 200)
    units = []
    for i, c in enumerate(pts):
        p = {"purpose": "dark Sobol (8 groups, nominal timing)", "changes": c, "block": "voltage"}
        if i < args.exposure_pairs:
            units.append([p, {**p, "purpose": f"the previous setting again with EXPOSURE {args.exposure} s", "block": "exposure pair",
                              "env": {"IMAGE_TOML": IMG_EXP.format(args.exposure)}, "exposure_s": args.exposure}])
        else:
            units.append([p])
    for g in names:
        if g == "VSUB":
            continue
        for x in d.box(g):
            if abs(x) >= 0.2:
                units.append([{"purpose": f"one-at-a-time: {g} {x:+g} V", "changes": d.expand({g: x}), "block": "oat"}])
    for x in VSUB_LEVELS:
        if d.box("VSUB")[0] - 1e-9 <= x <= d.box("VSUB")[1] + 1e-9:
            units.append([{"purpose": f"one-at-a-time: VSUB {x:+g} V", "changes": d.expand({"VSUB": x}), "block": "oat"}])
    planned = with_nominals(None, "D", rng, replicates=args.replicates, units=units)
    # nominal with the exposure too: the rate at nominal, at the start and the end
    nomexp = {"session": "D", "purpose": f"nominal with EXPOSURE {args.exposure} s", "changes": {}, "block": "exposure pair",
              "env": {"IMAGE_TOML": IMG_EXP.format(args.exposure)}, "exposure_s": args.exposure}
    planned = planned[:1] + [dict(nomexp)] + planned[1:] + [dict(nomexp)]
    write_plan(d, ROOT / "lvl1" / "packet_plan_v3d", "D3", planned,
               {"N_REF_SETS": 1, "N_SETS": 1, "N_POST_SETS": 0, "IMAGE_TOML": IMG[32]}, {"L1_STRIP": 0}, args.boards,
               "v3 dark-charge add-on: one-at-a-time sweeps and exposure pairs", [
        "What the noise set (joint Sobol, 32-row image) cannot give for lambda: single effects and the rate. A settings-only measurement (the traces added nothing to the settings for lambda "
        "in v2), so one trace set each side, no strips, the 32-row image. Groups with a real range only: "
        + ", ".join(f"{g} {d.box(g)[0]:+g}..{d.box(g)[1]:+g} V" for g in names) + ". The V lows, V swing and TG are left "
        "out: the rules leave them 0.5 V or less, below what the label noise resolves.",
        f"**Exposure pairs** ({args.exposure_pairs} settings + nominal twice): the setting again with EXPOSURE "
        f"{args.exposure} s right after its EXPOSURE 0 packet. The difference is the dark-current rate at that setting; "
        "what is left is charge made by the readout. Drop them (`--exposure-pairs 0`) if the pre-tests show no rate.",
        f"Rejected joint points: {rej}. Needed on the DAQ machine: `{IMG[32]}`, `{IMG_EXP.format(args.exposure)}`.",
        "Take it only after the pre-tests: if lambda settles slowly, add a hold (`PAUSE_S` in plan.json `env`)."])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", nargs="+", default=["pre", "noise", "dark"], choices=["pre", "noise", "dark"])
    ap.add_argument("--boards", nargs="+", default=["acm101"], help="boards the changes are sent to")
    ap.add_argument("--train", type=int, default=80, help="noise set: voltage Sobol points of session G")
    ap.add_argument("--test", type=int, default=24, help="noise set: voltage Sobol points of session H")
    ap.add_argument("--noise-rows", type=int, default=32, choices=sorted(IMG),
                    help="noise set image rows (32: the image also gives the dark-charge label; the user, 2026-11-10)")
    ap.add_argument("--dark", type=int, default=10, help="dark map: joint Sobol points (the noise set, with its 32-row "
                    "image, is the main joint design for lambda too)")
    ap.add_argument("--exposure-pairs", type=int, default=10)
    ap.add_argument("--exposure", type=int, default=600, help="s")
    ap.add_argument("--replicates", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = Design()
    for name, fn in (("pre", plan_pre), ("noise", plan_noise), ("dark", plan_dark)):
        if name in args.plan:
            fn(d, args)


if __name__ == "__main__":
    main()

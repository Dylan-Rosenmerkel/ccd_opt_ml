"""
make_packet_plan_v2.py -- the second level-1 packet plan (after the first data, 2026-11-02/03): one changes file per
packet and PLAN.md, in ``lvl1/packet_plan_v2/``.  The first plan (lvl1/make_packet_plan.py, lvl1/packet_plan/) is kept
as it was taken.

What the first data showed (docs/context/2026-11-03_session_context.md) and what v2 changes:
* most knobs were changed once, by one step, one at a time: a model sees each setting about once and must interpolate
  from single points.  v2 is a JOINT space-filling design (scrambled Sobol points) over all 12 knob groups, each knob
  taking many levels across its whole allowed range;
* lambda's label drifted over the night (nominal ch0 0.31 -> 0.24 e/pix): a nominal packet every NOMINAL_EVERY packets
  (instead of 4 in 32) tracks the drift and measures the label noise;
* replicates away from nominal: REPLICATES earlier design points are taken again later in the session;
* the prospective test is kept: a separate block of joint points (session E), never used to choose a model.

Rules (as v1): every configuration passes ccd_limits (windows, the ACM manual's hardware ranges, the group's voltage
rules; points that fail are redrawn); every voltage within ONE max_step of nominal (manual_packet.sh applies a change
in one command and resets at the end: no ramps); changes on board 101 only; both readout stages / bias sets together.
The order is shuffled, so no setting is confounded with time.

Each packet: ``PACKET_ID=L2_NN CHANGES_FILE=lvl1/packet_plan_v2/L2_NN.txt bash daq_scripts/manual_packet.sh``
(the 2026-11-03 flush scheme: flush before and after the reference traces, settling traces after the change, flush
before and after the image, a 2 s pause after every trace and the image).

    python lvl1/make_packet_plan_v2.py [--out lvl1/packet_plan_v2] [--train 48] [--test 12] [--seed 0]
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
from make_packet_plan import GROUPS, TIMING, WHY, seconds_per_skip  # noqa: E402

#: the design box: timing as values (10 ns cycles, sampled on a log scale), voltages as deltas from nominal (V).
#: The voltage ranges are one max_step either way except where the group's rules cut them (found in v1: SW_L > 0 puts
#: SW_L above OG_L; V_low < 0 puts the V lows below the TG; the TG only within +-0.25 V of 7.4).
BOX = {
    "delayInt": (300, 3000), "delay_H": (900, 3600), "delayRGH": (400, 1600),
    "VDD": (-1.0, 1.0), "VR": (-0.5, 0.5), "VSUB": (-5.0, 5.0),
    "RG_H": (-0.5, 0.5), "RG_L": (-0.5, 0.5), "SW_L": (-0.5, 0.0),
    "H_low": (-0.5, 0.5), "V_low": (0.0, 0.5), "TG": (-0.25, 0.25),
}
LOG_SCALE = {"delayInt", "delay_H", "delayRGH"}
NOMINAL_EVERY = 7          # a nominal packet at positions 1, 8, 15, ... of each session
REPLICATES = 3             # earlier training points repeated later


def packet_minutes(timing: dict, n_ref: int, n_new: int, image_rows: int = 16) -> float:
    """Rough packet time with the 2026-11-03 scheme, from the 2026-11-02 run (one packet every ~9 min with 5 + 10
    sets, no flushes, 2 s pauses per set): 4 flushes (~0.5 min each), a 2 s pause after every trace (4 per set)
    instead of per set, traces scaled with the number of sets, the image (4.8 min at the reference timing) and the
    traces scaled with the skip cycle."""
    scale = seconds_per_skip(timing) / seconds_per_skip(TIMING)
    sets = n_ref + n_new
    traces = sets / 15 * 3.0 + sets * 4 * 2.0 / 60          # ~3 min of traces for 15 sets, plus the pauses
    image = 4.8 * image_rows / 16                           # 16 rows: 4.8 min (17.9 s per row)
    return 2.0 + traces * scale + image * scale + 1.5       # flushes, traces, image, resets and queue overhead


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(ROOT / "lvl1" / "packet_plan_v2"))
    parser.add_argument("--train", type=int, default=48, help="joint design points for training (session D)")
    parser.add_argument("--test", type=int, default=12, help="joint design points of the prospective test (session E)")
    parser.add_argument("--n-ref", type=int, default=3, help="reference trace sets per packet (N_REF_SETS)")
    parser.add_argument("--n-new", type=int, default=5, help="trace sets at the new settings (N_SETS)")
    parser.add_argument("--image-rows", type=int, default=32, help="rows of the image (32: configs/image_params_32rows.toml)")
    parser.add_argument("--image-toml", default="image_params/tc1/img32.toml",
                        help="the image TOML on the DAQ machine (relative to CDAQ_DIR)")
    parser.add_argument("--prefix", default="L2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    limits = Limits.load(ROOT / "configs" / "limits.toml")
    nominal = {**dict(limits.nominal), **TIMING}
    step = {name: (limits.window_for(name).max_step if limits.window_for(name) else None) for name in nominal}
    rng = np.random.default_rng(args.seed)
    names = list(BOX)

    def to_changes(u: np.ndarray) -> dict:
        """A point of the unit cube -> {pin: value} (timing rounded to 10 cycles, voltages to 10 mV)."""
        changes = {}
        for g, x in zip(names, u):
            lo, hi = BOX[g]
            v = np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))) if g in LOG_SCALE else lo + x * (hi - lo)
            for pin in GROUPS.get(g, [g]):
                changes[pin] = int(round(v, -1)) if pin.startswith("delay") else round(nominal[pin] + v, 2)
        return changes

    def problems(changes: dict) -> list[str]:
        too_big = [p for p, v in changes.items() if step.get(p) and abs(v - nominal[p]) > step[p] + 1e-9]
        return limits.violations({**nominal, **changes}) + [f"{p}: more than one max_step" for p in too_big]

    def draw(n: int, sampler) -> tuple[list[dict], int]:
        """n valid joint points from the Sobol sequence (invalid points are skipped, and counted)."""
        points, rejected = [], 0
        while len(points) < n:
            for u in sampler.random(16):
                ch = to_changes(u)
                if problems(ch):
                    rejected += 1
                elif len(points) < n:
                    points.append(ch)
        return points, rejected

    train, rej_train = draw(args.train, qmc.Sobol(len(names), scramble=True, seed=args.seed))
    test, rej_test = draw(args.test, qmc.Sobol(len(names), scramble=True, seed=args.seed + 1000))

    def session(points: list[dict], name: str, replicates: int) -> list[dict]:
        order = [{"session": name, "purpose": "joint change (Sobol design)", "changes": c} for c in points]
        rng.shuffle(order)
        # replicates: points from the first half of the session, repeated in the second half
        if replicates:
            idx = rng.choice(len(order) // 2, size=replicates, replace=False)
            for k, i in enumerate(sorted(idx)):
                rep = {**order[i], "purpose": f"REPLICATE of design point {i + 1} of this session (pure error, drift)"}
                pos = len(order) // 2 + (k + 1) * (len(order) // 2) // (replicates + 1)
                order.insert(pos, rep)
        # a nominal packet every NOMINAL_EVERY packets, and one at the end
        with_nominal = []
        for i, p in enumerate(order):
            if i % (NOMINAL_EVERY - 1) == 0:
                with_nominal.append({"session": name, "purpose": "nominal (drift, label noise)", "changes": {}})
            with_nominal.append(p)
        with_nominal.append({"session": name, "purpose": "nominal (end of the session)", "changes": {}})
        return with_nominal

    planned = session(train, "D", REPLICATES) + session(test, "E", 0)
    for i, p in enumerate(planned, start=1):
        p["packet_id"] = f"{args.prefix}_{i:02d}"
        timing = {k: p["changes"].get(k, v) for k, v in TIMING.items()}
        p["minutes"] = round(packet_minutes(timing, args.n_ref, args.n_new, args.image_rows), 1)
        lines = [f"# {p['packet_id']} (session {p['session']}): {p['purpose']}",
                 f"# estimated {p['minutes']:.0f} min; generated by lvl1/make_packet_plan_v2.py (seed {args.seed})"]
        if not p["changes"]:
            lines.append("# no changes: the nominal configuration (manual_packet.sh needs one line; this re-sends a nominal value)")
            lines.append("acm101:delayInt=1000")
        lines += [f"acm101:{k}={v}" if isinstance(v, int) else f"acm101:{k}={v:g}" for k, v in p["changes"].items()]
        (out / f"{p['packet_id']}.txt").write_text("\n".join(lines) + "\n")

    # write the plan
    sessions = {s: [p for p in planned if p["session"] == s] for s in "DE"}
    levels = {g: len({round(c[GROUPS.get(g, [g])[0]], 3) for c in train}) for g in names}
    env = f"N_REF_SETS={args.n_ref} N_SETS={args.n_new} IMAGE_TOML={args.image_toml} "
    doc = ["# Level-1 packet plan, version 2", "",
           "Generated by `lvl1/make_packet_plan_v2.py` (rules and reasons in its docstring). Take each packet with", "",
           "```bash", f"{env}PACKET_ID={args.prefix}_NN CHANGES_FILE=lvl1/packet_plan_v2/{args.prefix}_NN.txt bash daq_scripts/manual_packet.sh",
           "```", "", "(with `DRY_RUN=1` first). **Take the packets in this order.**", "",
           f"**Voltage windows:** configs/limits.toml has `reviewed = {str(limits.reviewed).lower()}`"
           + (": the voltage changes use its PLACEHOLDER windows and are provisional." if not limits.reviewed else "."), "",
           "| session | packets | of which nominal | estimated time | use |", "|---|---|---|---|---|"]
    use = {"D": "training and model choice (setting-grouped CV)", "E": "prospective test: scored once, after the choice is frozen"}
    for s, ps in sessions.items():
        doc.append(f"| {s} | {len(ps)} | {sum(not p['changes'] for p in ps)} | {sum(p['minutes'] for p in ps) / 60:.1f} h | {use[s]} |")
    doc += [f"| all | {len(planned)} | | {sum(p['minutes'] for p in planned) / 60:.1f} h | |", "",
            "## What changed from v1, and why", "",
            "* **Joint design.** Every training packet moves all 12 knob groups at once, to a point of a scrambled Sobol "
            "sequence in the box below (timing on a log scale). v1 changed one knob at a time to one value, so each "
            f"setting was seen about once. Distinct levels per knob in session D: "
            + ", ".join(f"{g} {n}" for g, n in levels.items()) + ".",
            f"* **Nominal every {NOMINAL_EVERY} packets** (v1: 4 in 32), to track the CCD's drift (lambda fell ~20 % over "
            "the 2026-11-02 night) and measure the label noise.",
            f"* **{REPLICATES} replicates** of design points from the first half of session D, in its second half.",
            "* **Session E** is a separate Sobol sequence: the prospective test, as v1's session C.",
            f"* Points that break a rule are redrawn (D: {rej_train} rejected, E: {rej_test}).", "",
            "## The box", "", "| knob | range | pins |", "|---|---|---|"]
    for g, (lo, hi) in BOX.items():
        rng_txt = f"{lo}-{hi} cycles (log)" if g in LOG_SCALE else f"{lo:+g} to {hi:+g} V"
        doc.append(f"| {g} | {rng_txt} | {', '.join(GROUPS.get(g, [g]))} |")
    doc += ["", "Why these parameters: " + "; ".join(f"**{k}** {v}" for k, v in WHY.items()) + ".", "",
            "## Packet procedure (daq_scripts/manual_packet.sh, 2026-11-03)", "",
            "1. reset to nominal and read back; flush;",
            f"2. {args.n_ref} reference trace sets (4 channels, each board separately, 2 s pause after every trace); flush;",
            "3. the changes; 2 settling traces (not used);",
            f"4. {args.n_new} trace sets at the new settings; flush;",
            f"5. the image ({args.image_rows} rows, `{args.image_toml}`), 2 s pause; standalone; flush; reset to nominal "
            "and read back.", "",
            "**Safeguards** (manual_packet.sh, CHECK_TRACES=1): after the reference and after the new traces the script "
            "waits for both boards, checks every board-101 trace (src/check_trace_files.py) and retakes damaged ones; "
            "more than 5 damaged traces in a packet restart it from the reset as `<id>_r1` (the stopped attempt's record "
            "says \"restarted\" and is skipped by make_manual_manifest.py). On the 2026-11-02 data (~8 % of the traces "
            "damaged, mostly a few per packet; L1_37/40/41 lost two thirds) about 1 packet in 10 would restart.", "",
            "## Before taking it", "",
            f"* copy `configs/image_params_32rows.toml` to `<CDAQ_DIR>/{args.image_toml}` on the DAQ machine "
            "(32 rows: the dark-current part of lambda ~3x better than with 16);",
            "* record the CCD temperature per packet (the drift of lambda), if a readout exists;",
            "* the voltage windows (`reviewed = false`).", "",
            "## The packets", "", "| packet | session | purpose | changes (board 101) | min |", "|---|---|---|---|---|"]
    for p in planned:
        ch = ", ".join(f"{k}={v}" if isinstance(v, int) else f"{k}={v:g}" for k, v in p["changes"].items()) or "–"
        doc.append(f"| {p['packet_id']} | {p['session']} | {p['purpose']} | {ch} | {p['minutes']:.0f} |")
    doc += ["", "## After the packets", "", "```bash",
            "python src/make_manual_manifest.py ~/packet_records/L2_*.json --run-dir <raw-file directory> --out manifests",
            "cd src && python process_packet.py ../manifests/L2_*.json && cd ..",
            "```", ""]
    (out / "PLAN.md").write_text("\n".join(doc))
    (out / "plan.json").write_text(json.dumps({"reviewed_limits": limits.reviewed, "box": BOX, "seed": args.seed,
                                               "packets": planned}, indent=1, default=str))
    print("\n".join(doc[:28]))
    print(f"... {len(planned)} packets; wrote {out}")


if __name__ == "__main__":
    main()

"""
make_packet_plan.py -- the planned level-1 packets: one ready-to-use changes file per packet, and PLAN.md.

Level 1 needs packets whose traces and images span a range of noise conditions (sigma_e, lambda, gain), so the
models have something to learn from.  This writes ``lvl1/packet_plan/``:

    PLAN.md          the plan: sessions, the purpose of every packet, its changes, the time estimate, the command
    plan.json        the same as data
    L1_NN.txt        the changes of packet NN for manual_packet.sh (CHANGES_FILE=...)

Each packet is taken with
    PACKET_ID=L1_NN CHANGES_FILE=lvl1/packet_plan/L1_NN.txt bash daq_scripts/manual_packet.sh

RULES the plan follows:
* every configuration (nominal + changes) passes ccd_limits (configs/limits.toml windows, the ACM manual's hardware
  ranges, the group's voltage rules) -- candidates that fail are listed in PLAN.md as refused, never planned;
* every voltage stays within ONE max_step of nominal: manual_packet.sh sets a change in one command and returns to
  nominal at the end by re-reading the .bcf, so no ramp is ever needed;
* changes go to board 101 only (acm101:...): the module that is analysed; board 102 stays at nominal;
* pins of both readout stages / bias sets move together (e.g. VDD1 and VDD2), so all four amplifiers see each change
  (which amplifier sits on which stage is still to be confirmed);
* the timing session needs no voltage change and can be taken now; the voltage sessions use the PLACEHOLDER windows of
  limits.toml (reviewed = false) and are marked provisional until the group's windows are confirmed.

Sessions (in this order):
    A  timing only, one parameter at a time, nominal packets at the start, middle and end (drift, label noise)
    B  one voltage group at a time, one max_step either way
    C  joint changes (a Latin hypercube in the same box): the earlier tests showed a scan with joint changes learns lambda
       5-10 times better than one-at-a-time alone

    python lvl1/make_packet_plan.py [--out lvl1/packet_plan] [--joint 10] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ccd_limits import Limits  # noqa: E402

#: the timing of reference_v5_pumping.seq (10 ns cycles), as in manual_packet.sh
TIMING = {"delay_Va": 6000, "delay_Vb": 6000, "delay_Ha": 1800, "delay_Hb": 1800, "delay_Hc": 1800, "delayInt": 1000,
          "delayRGL": 27, "delayRGH": 800, "delaySWH": 30, "delaySWL": 350, "delayOGL": 48, "delayOGH": 3,
          "delayDGL": 27, "delayDGH": 47}
#: groups of pins that move together
GROUPS = {
    "VDD": ["VDD1", "VDD2"], "VR": ["VR1", "VR2"], "VSUB": ["VSUB"],
    "RG_H": ["RG_1_H", "RG_2_H"], "RG_L": ["RG_1_L", "RG_2_L"], "SW_L": ["SW_1_L", "SW_2_L"],
    "H_low": ["H1_A_L", "H1_B_L", "H2_C_L", "H3_A_L", "H3_B_L"],
    "V_low": ["V1_A_L", "V1_B_L", "V2_C_L", "V3_A_L", "V3_B_L"],
    "TG": ["TG_A_H", "TG_A_L", "TG_B_H", "TG_B_L"],
    "delay_H": ["delay_Ha", "delay_Hb", "delay_Hc"],
}
WHY = {
    "delayInt": "CDS integration window: white noise falls with a longer window, slow noise grows; the strongest noise knob",
    "delay_H": "horizontal clock hold: charge transfer and horizontal clock-induced charge",
    "delayRGH": "reset-gate hold before the pedestal window: settling of the reset",
    "VDD": "output drain: the amplifier's operating point (noise, gain)",
    "VR": "reset drain: reset level and its noise",
    "RG_H": "reset-gate high rail", "RG_L": "reset-gate low rail",
    "SW_L": "summing-well low rail",
    "H_low": "horizontal clock low rails: clock-induced charge",
    "V_low": "vertical clock low rails: clock-induced charge, dark current at the surface",
    "TG": "transfer gate (high = low)",
    "VSUB": "substrate voltage: depletion, dark current",
}


def seconds_per_skip(timing: dict) -> float:
    """One skip cycle: front (OG, reset) + pedestal window + summing well + signal window (10 ns cycles)."""
    front = timing["delayOGH"] + timing["delayOGL"] + timing["delayRGL"] + timing["delayRGH"]
    return (front + 2 * timing["delayInt"] + timing["delaySWH"] + timing["delaySWL"]) * 1e-8


def packet_minutes(timing: dict) -> float:
    """Rough packet time: the default manual packet (~16 min at the reference timing), its image and traces scaled
    with the skip cycle (the 16 x 1000 x 500 image takes 4.8 min at the reference)."""
    scale = seconds_per_skip(timing) / seconds_per_skip(TIMING)
    return 16.0 + 4.8 * (scale - 1) + 2.0 * (scale - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(ROOT / "lvl1" / "packet_plan"))
    parser.add_argument("--joint", type=int, default=10, help="packets with joint changes (session C)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    limits = Limits.load(ROOT / "configs" / "limits.toml")
    nominal = {**dict(limits.nominal), **TIMING}
    step = {name: (limits.window_for(name).max_step if limits.window_for(name) else None) for name in nominal}

    def config(changes: dict) -> dict:
        return {**nominal, **changes}

    def expand(group_values: dict) -> dict:
        """{group or pin: value or delta} -> {pin: value}.  Voltage groups take a DELTA from nominal, timing a value."""
        changes = {}
        for g, v in group_values.items():
            for pin in GROUPS.get(g, [g]):
                changes[pin] = int(v) if pin.startswith("delay") else round(nominal[pin] + v, 3)
        return changes

    planned, refused = [], []

    def add(session, purpose, group_values):
        changes = expand(group_values)
        problems = limits.violations(config(changes))
        too_big = [p for p, v in changes.items() if step.get(p) and abs(v - nominal[p]) > step[p] + 1e-9]
        if problems or too_big:
            refused.append({"session": session, "purpose": purpose, "changes": changes,
                            "why": problems + [f"{p}: more than one max_step from nominal" for p in too_big]})
            return
        timing = {k: changes.get(k, v) for k, v in TIMING.items()}
        planned.append({"session": session, "purpose": purpose, "changes": changes, "minutes": round(packet_minutes(timing), 1)})

    # --- A: timing only
    add("A", "nominal (reference packet: repeatability and label noise)", {})
    for v in (300, 500, 700, 1500, 2000, 3000):
        add("A", f"delayInt = {v}", {"delayInt": v})
    add("A", "nominal again (drift since the first)", {})
    for v in (900, 3600):
        add("A", f"delay_Ha/Hb/Hc = {v}", {"delay_H": v})
    for v in (400, 1600):
        add("A", f"delayRGH = {v}", {"delayRGH": v})
    add("A", "nominal again (end of session A)", {})
    # --- B: one voltage group at a time, one max_step either way
    for g in ("VDD", "VR", "RG_H", "RG_L", "SW_L", "H_low", "V_low", "TG", "VSUB"):
        pin = GROUPS[g][0]
        d = step[pin] if g != "TG" else 0.25
        for sign in (-1, +1):
            add("B", f"{g} {'-' if sign < 0 else '+'}{d:g} V", {g: sign * d})
    add("B", "nominal again (end of session B)", {})
    # --- C: joint changes, Latin hypercube in the same box (voltages within one max_step; timing in session A's range)
    rng = np.random.default_rng(args.seed)
    box = {"delayInt": (500, 2000), "delay_H": (900, 3600), "VDD": (-1, 1), "VR": (-0.5, 0.5), "H_low": (-0.5, 0.3),
           "VSUB": (-5, 5), "RG_H": (-0.5, 0.5)}
    tries = 0
    while sum(p["session"] == "C" for p in planned) < args.joint and tries < 50:
        tries += 1
        n = args.joint
        u = (np.argsort(rng.random((len(box), n)), axis=1) + rng.random((len(box), n))) / n      # Latin hypercube
        for k in range(n):
            if sum(p["session"] == "C" for p in planned) >= args.joint:
                break
            gv = {}
            for i, (g, (lo, hi)) in enumerate(box.items()):
                v = lo + (hi - lo) * u[i, k]
                gv[g] = int(round(v, -1)) if g.startswith("delay") else round(v, 2)
            add("C", "joint change (Latin hypercube)", gv)
    add("C", "nominal again (end of the plan)", {})

    # write
    for i, p in enumerate(planned, start=1):
        p["packet_id"] = f"L1_{i:02d}"
        lines = [f"# {p['packet_id']} (session {p['session']}): {p['purpose']}",
                 f"# estimated {p['minutes']:.0f} min; generated by lvl1/make_packet_plan.py"]
        if not p["changes"]:
            lines.append("# no changes: the nominal configuration (manual_packet.sh still needs one line; this re-sends a nominal value)")
            lines.append("acm101:delayInt=1000")
        lines += [f"acm101:{k}={v:g}" if not isinstance(v, int) else f"acm101:{k}={v}" for k, v in p["changes"].items()]
        (out / f"{p['packet_id']}.txt").write_text("\n".join(lines) + "\n")
    total = {s: (sum(1 for p in planned if p["session"] == s), sum(p["minutes"] for p in planned if p["session"] == s)) for s in "ABC"}
    doc = ["# Level-1 packet plan", "",
           "Generated by `lvl1/make_packet_plan.py` (rules and reasons in its docstring). Take each packet with", "",
           "```bash", "PACKET_ID=L1_NN CHANGES_FILE=lvl1/packet_plan/L1_NN.txt bash daq_scripts/manual_packet.sh", "```", "",
           "(run it with `DRY_RUN=1` first). `manual_packet.sh` checks every change again before anything is queued.", "",
           f"**Voltage windows:** configs/limits.toml has `reviewed = {str(limits.reviewed).lower()}`. "
           + ("Sessions B and C use its PLACEHOLDER windows: they are provisional until the group's windows are confirmed. "
              "Session A changes only timing and can be taken now." if not limits.reviewed else "The windows are reviewed."), "",
           "| session | packets | estimated time |", "|---|---|---|"]
    doc += [f"| {s} | {n} | {m / 60:.1f} h |" for s, (n, m) in total.items()]
    doc += [f"| all | {len(planned)} | {sum(p['minutes'] for p in planned) / 60:.1f} h |", "",
            "A first look at level 1 is possible after session A alone "
            f"({total['A'][0]} packets, {total['A'][0] * 4} examples with 4 amplifiers).", "",
            "## Why these parameters", ""] + [f"* **{k}**: {v}" for k, v in WHY.items()] + ["",
            "## The packets", "",
            "| packet | session | purpose | changes (board 101) | min | done |", "|---|---|---|---|---|---|"]
    for p in planned:
        ch = ", ".join(f"{k}={v:g}" if not isinstance(v, int) else f"{k}={v}" for k, v in p["changes"].items()) or "–"
        doc.append(f"| {p['packet_id']} | {p['session']} | {p['purpose']} | {ch} | {p['minutes']:.0f} | |")
    if refused:
        doc += ["", "## Candidates refused by the checks (never planned)", ""]
        doc += [f"* session {r['session']}, {r['purpose']}: {'; '.join(r['why'])}" for r in refused]
    doc += ["", "## After each packet", "",
            "```bash",
            "python src/make_manual_manifest.py ~/packet_records/L1_NN.json --run-dir <raw-file directory> --out manifests",
            "cd src && python process_packet.py ../manifests/L1_NN.json && cd ..",
            "```", "", "and after a session: `.venv/bin/python lvl1/train_lvl1.py --inputs processed/model_inputs`.", ""]
    (out / "PLAN.md").write_text("\n".join(doc))
    (out / "plan.json").write_text(json.dumps({"reviewed_limits": limits.reviewed, "packets": planned, "refused": refused}, indent=1, default=str))
    print("\n".join(doc[:30]))
    print(f"... {len(planned)} packets planned, {len(refused)} candidates refused; wrote {out}")


if __name__ == "__main__":
    main()

"""
reconstruct_records.py -- rebuild the packet RECORDS (what daq_scripts/manual_packet.sh writes to ~/packet_records) for
a plan whose records were not copied off the DAQ machine, from the plan (plan.json, the changes files) and the raw files
in the run directory.  make_manual_manifest.py then reads them as usual; it checks every requested change against what
the board reported in its .meta files, so a record that does not match what was taken shows up as a warning there.

Written fields follow manual_packet.sh's write_record (record_version 2); the trace / strip / image settings that the
plan did not change are copied from a template record of the previous run (--template).  "reconstructed": true marks
every rebuilt record.  Packets with no raw files are skipped; restarted packets (<id>_r1 ...) are not handled.

    .venv/bin/python lvl1/reconstruct_records.py --plan lvl1/packet_plan_v3n --run-dir <raw dir> \\
        --template <v2 run>/packet_records/L2_10.json --out records_v3n
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def changes_of(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.split("#", 1)[0].strip() for ln in path.read_text().splitlines() if ln.split("#", 1)[0].strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", required=True, help="the plan directory (plan.json, <id>.txt, <id>_ref.txt)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--template", required=True, help="a record of the previous run (trace / strip / image settings)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plan_dir, out = Path(args.plan), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((plan_dir / "plan.json").read_text())
    tpl = json.loads(Path(args.template).read_text())
    run = plan.get("run", {})
    names = os.listdir(args.run_dir)
    prefixes = {}
    for n in names:
        pid = "_".join(n.split("_")[:2])
        prefixes.setdefault(pid, set()).add(n.split("_")[2] if n.count("_") >= 2 else "")
    done, missing = [], []
    for p in plan["packets"]:
        pid = p["packet_id"]
        if pid not in prefixes:
            missing.append(pid)
            continue
        env = {**plan.get("env", {}), **p.get("env", {})}
        kinds = prefixes[pid]
        rec = {
            "record_version": 2, "status": "complete", "reconstructed": True,
            "reconstructed_from": f"{plan_dir}/plan.json + the raw files of {args.run_dir} (records not copied off the DAQ)",
            "packet_id": pid, "scan_name": tpl.get("scan_name", "manual"), "host": tpl.get("host"),
            "script": "manual_packet.sh",
            "changes": changes_of(plan_dir / f"{pid}.txt"),
            "changes_file": str(plan_dir / f"{pid}.txt"),
            "ref_changes": changes_of(plan_dir / f"{pid}_ref.txt"),
            "ref_changes_file": str(plan_dir / f"{pid}_ref.txt") if (plan_dir / f"{pid}_ref.txt").exists() else "",
            "screen": bool(p.get("screen", False)), "bcf": tpl["bcf"],
            "trace_prefix_ref": f"{pid}_ref", "trace_prefix_new": f"{pid}_new", "trace_prefix_post": f"{pid}_post",
            "n_post_sets": int(env.get("N_POST_SETS", run.get("N_POST_SETS", tpl["n_post_sets"]))),
            "n_ref_sets": int(env.get("N_REF_SETS", run.get("N_REF_SETS", tpl["n_ref_sets"]))),
            "n_sets": int(env.get("N_SETS", run.get("N_SETS", tpl["n_sets"]))),
            "trace_channels": tpl["trace_channels"], "trace_args": tpl["trace_args"],
            "flush_between": tpl["flush_between"], "flushes": tpl["flushes"],
            "n_settle_traces": tpl["n_settle_traces"], "pause_s": float(env.get("PAUSE_S", tpl["pause_s"])),
            "trace_boards": tpl["trace_boards"],
        }
        if any(k.startswith("l1") for k in kinds):
            rec.update({"l1_prefix_ref": f"{pid}_l1ref", "l1_prefix_new": f"{pid}_l1new", "l1_args": tpl["l1_args"]})
        rec["image_toml"] = env.get("IMAGE_TOML", run.get("IMAGE_TOML", tpl["image_toml"]))
        rec["leader"] = tpl["leader"]
        (out / f"{pid}.json").write_text(json.dumps(rec, indent=2) + "\n")
        done.append(pid)
    print(f"{len(done)} records written to {out}; {len(missing)} planned packets without raw files: {missing[:10]}")


if __name__ == "__main__":
    main()

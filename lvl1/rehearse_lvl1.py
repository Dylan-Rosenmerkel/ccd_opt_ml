"""
rehearse_lvl1.py -- the level-1 dress rehearsal: the whole chain on simulated packets of the planned geometry.

    1. simulate hand-taken packets (simulate_manual_run.py): packet records, traces, 16 x 1000 x 500 images, truth;
    2. make_manual_manifest.py: a manifest per packet from its record and the .meta files;
    3. process_packet.py with each WADERs route to test:
         run_WADERs  comm_scripts' run_WADERs (geometry from the header, per-CCD configuration, Poisson-like peaks only),
                     with --keep-dead (a 16-row image with almost no exposure has too few charged pixels for its
                     dead-channel test; all four channels of module 101 are connected);
         json        panaSKImg with the configured JSON (img_9.json) and the geometry of the planned image
                     (ccd_rows 16, ccd_cols 852 = PHYSCOL - NCOLPRE + PRESCAN, id_row_start 0);
    4. check_rehearsal.py: every label against the truth (sigma_e, lambda, gain, dark current, CIC; pulls);
    5. train_lvl1.py --quick on the result: the training step runs on what processing wrote.

Writes <out>/REHEARSAL.md (the summary of every step) next to the per-route outputs.

    python lvl1/rehearse_lvl1.py --out <dir> [--packets 6 --routes run_WADERs json]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
RUN_WADERS = Path.home() / "Documents" / "Software" / "comm_scripts" / "bin" / "run_WADERs"


def sh(args, cwd=ROOT, log=None) -> tuple[int, str]:
    done = subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, text=True)
    text = "\n".join(l for l in (done.stdout + done.stderr).splitlines() if "RuntimeWarning" not in l)
    if log:
        Path(log).write_text(text)
    return done.returncode, text


def config_for(route: str, out: Path) -> Path:
    """A processing.toml for one route (the repository's file with the output directory and the WADERs route changed)."""
    text = (ROOT / "configs" / "processing.toml").read_text()
    text = re.sub(r'(?m)^dir = .*$', f'dir = "{out / ("processed_" + route)}"', text, count=1)
    text = re.sub(r'(?m)^runner = .*$', '', text)
    text = re.sub(r'(?m)^run_waders_command = .*$', '', text)
    text = re.sub(r'(?m)^run_waders_args = .*$', '', text)
    if route == "run_WADERs":
        extra = (f'runner = "run_WADERs"\nrun_waders_command = ["{RUN_WADERS}"]\n'
                 'run_waders_args = ["--no-clusters", "--autoconfig-args=--keep-dead"]\n')
    else:
        extra = 'runner = "panaSKImg"\n'
    text = text.replace("[waders]\n", "[waders]\n" + extra, 1)
    text = re.sub(r"(?ms)^\[waders\.image_overrides\]\n(.*?)(?=^\[|\Z)", "[waders.image_overrides]\n" +
                  ("ccd_rows = 16\nccd_cols = 852\nid_row_start = 0\n\n" if route == "json" else "\n"), text)
    path = out / f"processing_{route}.toml"
    path.write_text(text)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--packets", type=int, default=6)
    parser.add_argument("--ref-sets", type=int, default=2)
    parser.add_argument("--sets", type=int, default=3)
    parser.add_argument("--routes", nargs="*", default=["run_WADERs", "json"], choices=["run_WADERs", "json"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = [f"# Level-1 rehearsal ({time.strftime('%Y-%m-%d %H:%M')})", "",
              f"{args.packets} simulated hand-taken packets ({args.ref_sets} reference + {args.sets} trace sets x 4 channels, "
              "board 101; image 16 x 1000 x 500, 1x1, NCOLPRE 5300); truth model in lvl1/simulate_manual_run.py.", ""]

    t0 = time.time()
    rc, text = sh([PY, "-W", "ignore", "lvl1/simulate_manual_run.py", "--out", out, "--packets", args.packets,
                   "--ref-sets", args.ref_sets, "--sets", args.sets, "--seed", args.seed], log=out / "1_simulate.log")
    report += ["## 1. Simulation", "", "```", text.strip(), "```", ""]
    rc, text = sh([PY, "-W", "ignore", "src/make_manual_manifest.py", *sorted((out / "records").glob("*.json")),
                   "--run-dir", out / "run", "--out", out / "manifests"], log=out / "2_manifests.log")
    report += ["## 2. Manifests", "", "```", text.replace(str(out) + "/", "").strip(), "```", ""]
    manifests = sorted((out / "manifests").glob("*.json"))

    for route in args.routes:
        cfg = config_for(route, out)
        t1 = time.time()
        rc, text = sh([PY, "-W", "ignore", "process_packet.py", *manifests, "--config", cfg], cwd=ROOT / "src",
                      log=out / f"3_process_{route}.log")
        report += [f"## 3–5. Route `{route}`", "", f"Processing ({time.time() - t1:.0f} s, return code {rc}):", "", "```",
                   "\n".join(l for l in text.splitlines() if not l.startswith("wrote")).replace(str(out) + "/", "").strip(), "```", ""]
        processed = out / f"processed_{route}"
        rc, text = sh([PY, "-W", "ignore", "lvl1/check_rehearsal.py", processed, out / "truth.json",
                       "--out", out / f"4_check_{route}.md"], log=out / f"4_check_{route}.log")
        summary = text[text.find("labels ("):] if "labels (" in text else text
        report += ["Labels against the truth:", "", summary.split("\n", 1)[0], "", summary.split("\n", 1)[1].strip() if "\n" in summary else "", ""]
        rc, text = sh([PY, "-W", "ignore", "lvl1/train_lvl1.py", "--inputs", processed / "model_inputs", "--quick",
                       "--folds", "3", "--models", "trace_physics", "physics", "physics_joint", "ridge", "gp",
                       "--out", out / f"5_train_{route}"], log=out / f"5_train_{route}.log")
        report += [f"Training (`train_lvl1.py --quick`, return code {rc}; with {args.packets} packets the scores only show "
                   "that the step runs):", "", "```", text.replace(str(out) + "/", "").strip(), "```", ""]
    report.append(f"Total time {time.time() - t0:.0f} s.")
    (out / "REHEARSAL.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()

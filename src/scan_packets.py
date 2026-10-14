"""
scan_packets.py -- take a series of data packets for a list of parameter configurations.

This is the driver around ``take_packet.take_packet``.  It

    1. reads a scan definition (``configs/scan_example.toml``): reference parameters per board,
       the image and trace parameters, the packet options, and a design that changes a few
       parameters around the reference,
    2. builds the list of operating points (reference first, design points in a seeded
       random order, the reference repeated every ``reference_every`` points so the label
       scatter and drift can be measured),
    3. plans EVERY packet (limits, geometry, ramps) before the first command is sent,
    4. takes the packets one after another, stopping at the first failure,
    5. can be restarted: packets whose manifest already says ``ok`` are skipped.

Every packet applies the COMPLETE parameter set of every board and takes its reference traces
at the scan's reference first (``theta_ref``), so each packet carries its own reference.

Design types (``[design] type = ...``)
    ``oat``     one-at-a-time: every listed value of every parameter, others at reference
    ``random``  ``n_points`` random points between ``min`` and ``max`` per parameter
                (``sampling = "uniform"``, or ``"lhs"``: a Latin hypercube, every parameter's range
                cut into n_points strata with one point in each)
    ``mixed``   one-at-a-time sweeps (``values``, or ``n_values`` evenly spaced between ``min`` and
                ``max``) PLUS ``n_random`` Latin-hypercube points in which all parameters change
                together.  Recommended for the models (docs/overleaf_ml_models/ml_models.tex, "Scan
                design"): with the same number of packets, a one-at-a-time scan left the dark charge
                5-10x worse predicted than a half-random one, because it never changes two
                parameters at once; the sweeps stay for reading the effects directly.
    ``list``    explicit ``[[design.points]]`` tables, each a dict of changes to the reference

Random points that break the limits or the relations between parameters (ccd_limits.py) are
redrawn when the plan is built, so a random design never plans a packet the checks would refuse.

A design parameter name changes that parameter on EVERY board in ``theta_boards`` ("VSUB"),
or on one board only when prefixed with the client name ("acm102.VSUB").

Usage
-----
    python scan_packets.py --scan ../configs/scan_example.toml --dry-run
    python scan_packets.py --scan ../configs/scan_example.toml --backend fake --work-dir /tmp/rehearsal
    python scan_packets.py --scan ../configs/scan_example.toml --backend cdaq        # real DAQ
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import tomllib
from pathlib import Path
from typing import Any

from ccd_limits import Limits, LimitsError
from daq_control import StepError, repeat_queue_suspended
from take_packet import (CONFIG_DIR, DEFAULT_SEQUENCER_FILE, DaqSetup, PacketError, PacketRequest,
                         board_thetas, estimate_packet_seconds, make_controller, plan_packet,
                         repeat_queue_backup_file, take_packet)

log = logging.getLogger("scan_packets")

#: Packet options that may appear in [options] of the scan file (see take_packet.PacketRequest).
_COUNT_OPTIONS = ("n_initial_flushes", "n_reference_traces", "n_traces", "flushes_between_traces",
                  "flushes_after_change", "n_final_flushes", "retake_damaged_traces")


# --------------------------------------------------------------------------- #
# Building the list of operating points                                        #
# --------------------------------------------------------------------------- #

def _value(p: dict[str, Any], u: float) -> float:
    """The parameter value at fraction u (0..1) of its [min, max] range (rounded; integers for sequencer constants)."""
    x = p["min"] + u * (p["max"] - p["min"])
    return int(round(x)) if p.get("integer", False) else round(x, 3)


def _oat_values(p: dict[str, Any]) -> list[float]:
    if "values" in p:
        return list(p["values"])
    n = int(p["n_values"])
    return [_value(p, k / (n - 1)) for k in range(n)] if n > 1 else [_value(p, 0.5)]


def _random_points(parameters: list[dict], n: int, rng: random.Random, sampling: str,
                   accept=None, max_tries: int = 200) -> list[dict[str, float]]:
    """
    n random points.  ``sampling``: "uniform", or "lhs" (Latin hypercube: each parameter's range cut into n
    strata, one point per stratum, strata paired at random).  A point that ``accept`` refuses is redrawn
    uniformly (up to ``max_tries`` times; then it is dropped with a warning).
    """
    if sampling == "lhs":
        columns = []
        for _ in parameters:
            strata = list(range(n))
            rng.shuffle(strata)
            columns.append([(k + rng.random()) / n for k in strata])
        fractions = [[col[i] for col in columns] for i in range(n)]
    elif sampling == "uniform":
        fractions = [[rng.random() for _ in parameters] for _ in range(n)]
    else:
        raise ValueError(f"unknown sampling {sampling!r} (uniform or lhs)")
    points = []
    for u in fractions:
        point = {p["name"]: _value(p, x) for p, x in zip(parameters, u)}
        tries = 0
        while accept is not None and not accept(point) and tries < max_tries:
            point = {p["name"]: _value(p, rng.random()) for p in parameters}
            tries += 1
        if accept is not None and not accept(point):
            log.warning("design: dropped a random point that the limits refuse after %d tries", max_tries)
            continue
        points.append(point)
    return points


def design_changes(design: dict[str, Any], seed: int, accept=None) -> list[dict[str, float]]:
    """
    Return the design as a list of "changes to the reference" dictionaries.  ``accept(change) -> bool``
    (from ``build_plan``: the limits and relations) is used to redraw refused random points.
    """
    kind = design["type"]
    if kind == "oat":
        return [{p["name"]: value} for p in design["parameters"] for value in _oat_values(p)]
    if kind == "random":
        return _random_points(design["parameters"], int(design["n_points"]), random.Random(seed),
                              design.get("sampling", "uniform"), accept)
    if kind == "mixed":
        sweeps = [{p["name"]: value} for p in design["parameters"] for value in _oat_values(p)]
        joint = _random_points(design["parameters"], int(design["n_random"]), random.Random(seed),
                               design.get("sampling", "lhs"), accept)
        return sweeps + joint
    if kind == "list":
        return [dict(point) for point in design["points"]]
    raise ValueError(f"unknown design type {kind!r}")


def apply_change(reference: dict[int, dict], change: dict[str, float], setup: DaqSetup) -> dict[int, dict]:
    """
    The reference with a design point's changes on top.

    "VSUB" changes every board in theta_boards; "acm102.VSUB" changes board 102 only.
    """
    theta = copy.deepcopy(reference)
    by_client = {setup.client_name(b): b for b in setup.theta_boards}
    for name, value in change.items():
        if "." in name:
            client, key = name.split(".", 1)
            if client not in by_client:
                raise ValueError(f"design parameter {name!r}: {client} is not in theta_boards")
            theta[by_client[client]][key] = value
        else:
            for board in theta:
                theta[board][name] = value
    return theta


def build_plan(config: dict[str, Any], setup: DaqSetup, limits: Limits | None = None) -> list[tuple[str, dict[int, dict]]]:
    """
    Turn the scan definition into an ordered list of ``(packet_id, theta per board)``.

    Order: reference, then the (shuffled) design points with the reference re-inserted
    after every ``reference_every`` of them.  With ``limits``, random design points that any board's
    limits or relations refuse are redrawn.
    """
    scan = config["scan"]
    reference = board_thetas(config["reference"], config.get("reference_board", {}), setup.theta_boards)
    seed = int(scan.get("seed", 0))
    accept = None
    if limits is not None:
        def accept(change):
            return all(not limits.violations(t) for t in apply_change(reference, change, setup).values())
    changes = design_changes(config["design"], seed, accept)
    random.Random(seed).shuffle(changes)

    every = int(scan.get("reference_every", 0))
    thetas = [copy.deepcopy(reference)]
    for index, change in enumerate(changes, start=1):
        thetas.append(apply_change(reference, change, setup))
        if every and index % every == 0:
            thetas.append(copy.deepcopy(reference))
    return [(f"{scan['name']}_{i:05d}", theta) for i, theta in enumerate(thetas)]


def make_request(packet_id: str, theta: dict[int, dict], config: dict[str, Any], setup: DaqSetup) -> PacketRequest:
    """The PacketRequest of one plan point: theta_ref = the scan's reference."""
    scan, options = config["scan"], config.get("options", {})
    return PacketRequest(
        packet_id=packet_id, theta=theta,
        theta_ref=board_thetas(config["reference"], config.get("reference_board", {}), setup.theta_boards),
        image=dict(config["image"]), trace=dict(config.get("trace", {})),
        sequencer=scan.get("sequencer", DEFAULT_SEQUENCER_FILE),
        erase_policy=options.get("erase_policy", "never"),
        reference_traces=options.get("reference_traces", "always"),
        scan_name=scan["name"],
        **{k: int(options[k]) for k in _COUNT_OPTIONS if k in options})


# --------------------------------------------------------------------------- #
# Resume support                                                               #
# --------------------------------------------------------------------------- #

def load_existing_manifests(work_dir: Path) -> dict[str, dict]:
    """Return ``{packet_id: manifest}`` for the manifests already in ``work_dir/packets``."""
    found = {}
    for path in sorted((work_dir / "packets").glob("*.json")):
        try:
            found[path.stem] = json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning("ignoring unreadable manifest %s", path)
    return found


# --------------------------------------------------------------------------- #
# Dry run: plan every packet                                                   #
# --------------------------------------------------------------------------- #

def validate_plan(plan, limits: Limits, config: dict[str, Any], setup: DaqSetup) -> bool:
    """Plan every packet (all checks of a real run); print one line per packet and the total time."""
    ok, total_s, prev = True, 0.0, None
    for packet_id, theta in plan:
        try:
            steps = plan_packet(make_request(packet_id, theta, config, setup), setup, limits, prev)
        except (LimitsError, KeyError, ValueError) as err:
            print(f"BAD {packet_id}\n      {err}")
            ok = False
            prev = theta
            continue
        seconds = estimate_packet_seconds(steps)
        total_s += seconds
        reference = board_thetas(config["reference"], config.get("reference_board", {}), setup.theta_boards)
        change = {f"acm{b}": {k: round(v - reference[b][k], 6) for k, v in t.items() if v != reference[b][k]}
                  for b, t in theta.items()}
        print(f"OK  {packet_id}  ~{seconds / 60:5.1f} min  {len(steps)} steps  change {change}")
        prev = theta
    print(f"\n{len(plan)} packets, estimated total {total_s / 3600:.1f} h "
          f"(without FITS waits) ({'valid' if ok else 'INVALID'})")
    return ok


# --------------------------------------------------------------------------- #
# Running the scan                                                             #
# --------------------------------------------------------------------------- #

def run_scan(controller, plan, limits: Limits, config: dict[str, Any], setup: DaqSetup,
             allow_unreviewed: bool) -> None:
    """
    Take all packets of ``plan`` that are not yet ``ok``; stop at the first failure.

    The cdaq repeat queue (the group's idle ``daq.flush_ccd()``) is saved and cleared ONCE for
    the whole scan and put back at the end (``[daq] repeat_queue = "suspend"``), so no flush
    the packets did not ask for runs between their steps.  After a failure it is not put back;
    the backup file in ``<work_dir>/queue_backups/`` lists the commands to restore it by hand.
    """
    work = Path(setup.work_dir)
    done = load_existing_manifests(work)
    previous = [m for m in done.values() if m.get("theta")]
    prev_manifest = max(previous, key=lambda m: m["date_start"]) if previous else None   # where the boards are now

    (work / "packets").mkdir(parents=True, exist_ok=True)
    (work / f"{config['scan']['name']}_plan.json").write_text(
        json.dumps([{"packet_id": pid, "theta": theta} for pid, theta in plan], indent=2))

    suspend = setup.repeat_queue == "suspend"
    with repeat_queue_suspended(controller, repeat_queue_backup_file(setup, config["scan"]["name"]),
                                enabled=suspend) as saved:
        for packet_id, theta in plan:
            if done.get(packet_id, {}).get("status") == "ok":
                log.info("skipping %s (already ok)", packet_id)
                continue
            prev_manifest = take_packet(controller, make_request(packet_id, theta, config, setup), setup,
                                        limits, prev_manifest, allow_unreviewed_limits=allow_unreviewed,
                                        repeat_queue_suspended=saved if suspend else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scan", required=True, help="scan definition TOML")
    parser.add_argument("--setup", default=str(CONFIG_DIR / "daq_setup.toml"))
    parser.add_argument("--limits", default=str(CONFIG_DIR / "limits.toml"))
    parser.add_argument("--backend", choices=["cdaq", "fake"], default="fake")
    parser.add_argument("--work-dir", help="override [daq] work_dir")
    parser.add_argument("--fake-data", help="data/ directory whose real files the fake boards copy")
    parser.add_argument("--fake-time-scale", type=float, default=0.0,
                        help="fake boards sleep this fraction of the real durations (watch with fake_cdaq_monitor.py)")
    parser.add_argument("--dry-run", action="store_true", help="plan every packet and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.scan, "rb") as handle:
        config = tomllib.load(handle)
    setup = DaqSetup.load(args.setup)
    if args.work_dir:
        setup.work_dir = args.work_dir
    limits = Limits.load(args.limits)
    plan = build_plan(config, setup, limits)

    plan_ok = validate_plan(plan, limits, config, setup)
    if args.dry_run or not plan_ok:
        sys.exit(0 if plan_ok else 1)

    allow_unreviewed = args.backend == "fake"
    if not limits.reviewed and not allow_unreviewed:
        sys.exit(f"{limits.source} has reviewed = false; refusing to drive real hardware")

    controller = make_controller(args.backend, setup, args.fake_data, args.fake_time_scale, echo=True)
    try:
        run_scan(controller, plan, limits, config, setup, allow_unreviewed)
    except (LimitsError, StepError, PacketError) as err:
        sys.exit(f"scan stopped: {err}\nFix the problem and rerun; finished packets are skipped.")


if __name__ == "__main__":
    main()

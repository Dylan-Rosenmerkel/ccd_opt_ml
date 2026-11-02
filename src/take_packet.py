"""
take_packet.py -- acquire ONE training "data packet" at a given set of CCD parameters.

A packet is everything the ML pipeline needs for one operating point.  Each board in
``theta_boards`` has its OWN parameter set: ``theta[board]`` (new) and ``theta_ref[board]``
(reference).  A packet contains

    * 3 flushes at the reference parameters (the CCDs start from the same, emptied state);
    * 3 sets of REFERENCE traces at theta_ref, a flush between sets (one set = one trace per
      channel on every traced board).  They measure each amplifier's reference state now; the
      training plan works with every amplifier's change relative to its own reference, and
      the 3 repeats give the run-to-run scatter;
    * a change to theta, a flush, then 3 sets of traces at theta, a flush between sets;
    * one IMAGE at theta of a specified size ([image] table, like an image_params TOML file),
      read synchronously by all boards in master/slave mode, then back to standalone mode and
      a flush (like ``take_image.py -m <id> -f``);
    * a MANIFEST (``<work_dir>/packets/<packet_id>.json``) that ties the files together with
      theta, theta_ref, the change, the previous packet, timestamps and every step's receipt.

The counts (3, 3, 3, flushes) are packet options.  The calibrated image (WADERs) is produced
later by ``process_packet.py``.

How the hardware is driven
--------------------------
Like the group's ``scripts/take_image.py`` and ``scripts/take_trace.py``, this script does not
open the boards itself.  Every step is a short piece of Python text (``daq_commands.py``)
that is put in the cdaq queue for one board's client ("acm101") or for the pool of all boards
("acmpool").  The client runs it with its own ``daq`` object and writes a small JSON receipt
when it is done (``daq_control.py``).  The script waits for the receipts before sending the
next step.

Order of the steps (default counts; the reasons are in the walkthrough notebook)
-------------------------------------------------------------------------------
     0. checks on this computer: theta and theta_ref are COMPLETED (every voltage and timing
        constant the packet file does not give takes its nominal value, see
        :func:`complete_theta`), then the limits of every board's theta and theta_ref, of the
        image and trace parameters, and of every intermediate voltage of the ramps.  Nothing is
        sent if any check fails.
     1. status         (pool)           read run directory, loaded sequencer, sync/clock bits
     2. standalone     (pool)           daq.standalone(); external clock set if not yet set
     3. apply_ref      (each board)     ramp to that board's theta_ref, then read every value back
                                        (a mismatch stops the packet: :func:`readback_problems`)
        [erase_ref]    (theta boards)   only if the erase policy asks (default: never)
     4. flush x3       (pool)           daq.flush_ccd() three times
     5. ref_traces 1   (each traced board)  one trace per channel
     6. flush          (pool)
     7. ref_traces 2, flush, ref_traces 3
     8. apply          (each board)     ramp theta_ref -> theta, then read every value back
        [erase]        (theta boards)   only if the erase policy asks
     9. flush          (pool)
    10. traces 1, flush, traces 2, flush, traces 3
    11. image          (pool)           ONE queue entry: daq.master(leader), every board takes the
                                        image together, daq.standalone(), flush x n_final_flushes
    12. FITS wait, manifest

The cdaq repeat queue
---------------------
The group keeps ``daq.flush_ccd()`` in the cdaq REPEAT queue: the queue runner runs it whenever
the simple queue is empty, i.e. between any two steps of a packet (this script queues a step
only after the previous one has finished).  Two consequences, both handled:

* extra flushes between the steps.  ``main()`` (and ``scan_packets.py``, once per scan) saves
  and clears the repeat queue before the first step and puts it back afterwards
  (``[daq] repeat_queue = "suspend"``, the default; ``daq_control.repeat_queue_suspended``).
  After an error it is NOT put back automatically: the board state must be checked first.
* a repeat flush while the boards are in leader/follower mode would hang.  The image step is
  therefore one queue entry from ``daq.master`` back to ``daq.standalone``, so even with the
  repeat queue kept (``repeat_queue = "keep"``) nothing can run in between.

Usage
-----
    # print every step and the exact text each board would run (no hardware, no files)
    python take_packet.py --params ../configs/packet_example.toml --show-commands

    # rehearsal on fake boards (writes files under --work-dir)
    python take_packet.py --params ../configs/packet_example.toml --backend fake --work-dir /tmp/rehearsal

    # the same, slowed down to 5 % of real time, watched in a second terminal like the real
    # tmux session (queue, board log, client status, published queue commands)
    python fake_cdaq_monitor.py --state /tmp/rehearsal/fake_cdaq_state.json
    python take_packet.py --params ../configs/packet_example.toml --backend fake \
           --work-dir /tmp/rehearsal --fake-time-scale 0.05

    # first contact with the real DAQ: only the read-only status step
    python take_packet.py --params packet.toml --backend cdaq --status-only

    # real DAQ (on the DAQ machine, cdaq running, limits reviewed)
    python take_packet.py --params packet.toml --backend cdaq

Common options: ``--setup configs/daq_setup.toml`` (boards, pool, commands),
``--limits configs/limits.toml``, ``--prev-manifest`` (manifest of the packet taken before:
used for the voltage ramps, the erase decision and the time since the last acquisition).

Packet file format (``configs/packet_example.toml``)
----------------------------------------------------
    packet_id = "..."
    [options]               sequencer, counts of flushes/traces, erase_policy, ...
    [image]                 image size and readout (geometry keys, acq_level, subroutine)
    [trace]                 trace parameters (defaults: scripts/take_trace.py)
    [theta]                 operating parameters shared by every board in theta_boards
    [theta_board.102]       replacements for board 102 only (any board, any key of theta)
    [theta_ref]             reference parameters shared by every board
    [theta_ref_board.102]   reference replacements for board 102 only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import tomllib
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Mapping

from ccd_limits import BIAS_PINS, DAC_PINS, SEQUENCER_KEYS, Limits, LimitsError, expand_key, split_theta
import daq_commands as cmd
from daq_control import StepError, repeat_queue_suspended, wait_for_receipts

log = logging.getLogger("take_packet")

MANIFEST_VERSION = 3
DEFAULT_SEQUENCER_FILE = "reference_v5_pumping.seq"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

#: Values of sequencer constants that are not in theta (from reference_v5_pumping.seq and the
#: 2026-10-08 .meta files); only used for the time ESTIMATE when acmpy is not importable.
_SEQ_DEFAULTS = {
    "NDCMPRE": 0, "NROWPRE": 0, "NROWPOST": 0, "NCOLPRE": 0, "NCOLPOST": 0,
    "BOOLCLR": 0, "NCOLCLR": 616, "NSBINCLR": 10, "DUMPLESS": 0,
    "NROWFLS": 20, "NCOLFLS": 6160, "NPBINFLS": 1536, "NSBINFLS": 1,
    "delayRGL": 27, "delayRGH": 800, "delaySWH": 30, "delaySWL": 350,
    "delayOGL": 48, "delayOGH": 3, "delayDGL": 27, "delayDGH": 47, "delayInt": 1000,
    "delay_Va": 6000, "delay_Vb": 6000, "delay_Ha": 1800, "delay_Hb": 1800, "delay_Hc": 1800,
}


#: The timing constants every packet sets (reference_v5_pumping.seq, 10 ns cycles); with the
#: voltages of limits.toml [nominal] they are the values a parameter missing from a packet takes.
NOMINAL_TIMING = {k: v for k, v in _SEQ_DEFAULTS.items() if k.startswith("delay")}

#: Every parameter a complete theta sets: 7 biases, 40 DAC rails, 14 timing constants.  (The clock
#: rail offsets OFS_* are not parameters: they stay at the .bcf value.)
COMPLETE_KEYS = frozenset(BIAS_PINS) | frozenset(DAC_PINS) | frozenset(NOMINAL_TIMING)

#: Readback tolerances: a bias reads back within one potentiometer step or so (45 V reads 45.095 V);
#: a DAC rail within a few mV; timing constants exactly.
READBACK_TOL_BIAS_V = 0.3
READBACK_TOL_RAIL_V = 0.05


class PacketError(RuntimeError):
    """The packet could not be completed (wrong sequencer, failed step, ...)."""


# --------------------------------------------------------------------------- #
# Configuration objects                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class DaqSetup:
    """Board layout and command texts (``configs/daq_setup.toml``)."""
    queue: str
    pool: str
    boards: list[int]
    client: str
    leader: int
    theta_boards: list[int]
    trace_boards: list[int]
    trace_channels: list[int]
    work_dir: str
    timeout_margin_s: float
    poll_interval_s: float
    fits_wait_s: float
    commands: dict[str, str]
    repeat_queue: str = "suspend"          # "suspend" (save, clear, restore) or "keep"

    @classmethod
    def load(cls, path: str | Path) -> "DaqSetup":
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
        daq = data["daq"]
        setup = cls(queue=daq.get("queue", "default"), pool=daq["pool"],
                    boards=[int(b) for b in daq["boards"]], client=daq.get("client", "acm{board}"),
                    leader=int(daq["leader"]),
                    theta_boards=[int(b) for b in daq.get("theta_boards", daq["boards"])],
                    trace_boards=[int(b) for b in daq.get("trace_boards", daq["boards"])],
                    trace_channels=[int(c) for c in daq.get("trace_channels", [0, 1, 2, 3])],
                    work_dir=daq["work_dir"],
                    timeout_margin_s=float(daq.get("timeout_margin_s", 300)),
                    poll_interval_s=float(daq.get("poll_interval_s", 1.0)),
                    fits_wait_s=float(daq.get("fits_wait_s", 180)),
                    commands=dict(data["commands"]),
                    repeat_queue=daq.get("repeat_queue", "suspend"))
        if setup.repeat_queue not in ("suspend", "keep"):
            raise ValueError(f"{path}: repeat_queue must be 'suspend' or 'keep', not {setup.repeat_queue!r}")
        unknown = set(setup.theta_boards + setup.trace_boards + [setup.leader]) - set(setup.boards)
        if unknown:
            raise ValueError(f"{path}: boards {sorted(unknown)} are not in [daq] boards")
        return setup

    def client_name(self, board: int) -> str:
        return self.client.format(board=board)


@dataclass
class PacketRequest:
    """
    One packet.

    theta, theta_ref : per board, the COMPLETE operating parameters (voltages and timing
        constants, NOT the image/trace geometry): ``{101: {...}, 102: {...}}``.  theta_ref =
        None means "no reference traces" (not recommended for training data).
    image : the image parameters (geometry, acq_level, subroutine), like an image_params TOML.
    trace : the trace parameters; missing keys come from ``daq_commands.DEFAULT_TRACE``.  With
        ``match_image = true`` the trace reads its pixels like the image (same skips, binning
        and routine; ``daq_commands.MATCHED_KEYS``), with a small size of its own.
    n_initial_flushes : flushes at theta_ref before the first reference trace set.
    n_reference_traces, n_traces : trace sets at theta_ref and at theta.
    flushes_between_traces : flushes between two trace sets.
    flushes_after_change : flushes after going to theta, before the first trace set at theta.
    n_final_flushes : flushes after the image, in the same queue entry, once the boards are back
        in standalone mode (``take_image.py -f`` does one).
    erase_policy : "never" (default) | "on_voltage_change" | "always" | "first_only".
    reference_traces : "always" | "when_different" (skip when theta == theta_ref) | "never".
    retake_damaged_traces : how many times a trace step is repeated when a decoded trace shows
        lost data (only checked when the files are readable from this computer).
    """
    packet_id: str
    theta: dict[int, dict[str, float]]
    theta_ref: dict[int, dict[str, float]] | None
    image: dict[str, Any]
    trace: dict[str, Any] = field(default_factory=dict)
    sequencer: str = DEFAULT_SEQUENCER_FILE
    n_initial_flushes: int = 3
    n_reference_traces: int = 3
    n_traces: int = 3
    flushes_between_traces: int = 1
    flushes_after_change: int = 1
    n_final_flushes: int = 1
    erase_policy: str = "never"
    reference_traces: str = "always"
    retake_damaged_traces: int = 1
    ramp_settle_s: float = 0.5
    scan_name: str = ""


@dataclass
class PlannedStep:
    """One queued action: what text, for which target, expected from which boards."""
    kind: str                    # status, standalone, apply, erase, flush, ref_traces, traces, image
    name: str                    # readable name, e.g. "ref_traces_2"
    target: str                  # cdaq client ("acm101") or pool ("acmpool")
    boards: list[int]            # boards expected to write a receipt
    lines: list[str]             # the step's own text (without the receipt frame)
    expected_s: float            # expected duration (planning)
    timeout_s: float             # give up waiting for the receipts after this long
    purpose: str
    repeat: int = 0              # trace set number (1, 2, 3) for trace steps
    expected: dict | None = None # apply steps: the values the board must read back

    def code(self, step_id: str, receipt_dir: str) -> str:
        return cmd.wrap_step(step_id, self.lines, receipt_dir)


def board_thetas(common: Mapping[str, float], per_board: Mapping, boards: list[int]) -> dict[int, dict]:
    """
    Every board's parameter set: the shared table with that board's replacements on top.

    ``per_board`` is the ``[theta_board]`` table, e.g. ``{"102": {"VSUB": 50.0}}``.
    """
    per_board = {int(b): dict(values) for b, values in (per_board or {}).items()}
    stray = set(per_board) - set(boards)
    if stray:
        raise ValueError(f"settings given for board(s) {sorted(stray)}, which are not in theta_boards {boards}")
    return {b: {**dict(common), **per_board.get(b, {})} for b in boards}


# --------------------------------------------------------------------------- #
# Time estimates                                                               #
# --------------------------------------------------------------------------- #

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _cycles(p: Mapping[str, float]) -> dict[str, float]:
    """Durations (10 ns clock cycles) of the basic sequencer functions of reference_v5_pumping.seq."""
    front = p["delayOGH"] + p["delayOGL"] + p["delayRGL"] + p["delayRGH"]
    return {"h": p["delayRGL"] + 3 * p["delay_Ha"] + 3 * p["delay_Hb"] + p["delay_Hc"],   # Htransfer
            "v": p["delayRGL"] + 3 * p["delay_Va"] + 3 * p["delay_Vb"],                    # Vtransfer
            "dg": p["delayDGL"] + p["delayDGH"],                                           # drain/send
            "cds": front + 2 * p["delayInt"] + p["delaySWH"] + p["delaySWL"],              # ReadoutCDS
            "ped": front + p["delayInt"] + p["delaySWH"]}                                  # ReadoutPedestal


def formula_acquisition_seconds(params: Mapping[str, float]) -> float:
    """
    Readout time + EXPOSURE of AcquireImage, hand-coded from the sequencer (10 ns cycles).

    AcquireImage = SkipRow x NROWPRE + ReadRow x NROW + SkipRow x NROWPOST; ReadRow =
    [ClearRegister if BOOLCLR] + Vtransfer x NPBIN + SkipPixel x (NCOLPRE + NCOLPOST) + ReadPixel
    x NCOL.  Checked against measurements (2x5x500 trace 0.20 s, 80x320x10 image 138 s,
    100x130x500 image about 6 min).  Used for timeouts and planning only.
    """
    p = {**_SEQ_DEFAULTS, **{k: v for k, v in params.items() if k in SEQUENCER_KEYS}}
    c = _cycles(p)
    t_read_pixel = p["NSBIN"] * c["h"] + (p["NDCMPRE"] + p["NDCM"]) * c["cds"] + c["ped"] + c["dg"]
    t_skip_pixel = p["NSBIN"] * c["h"] + c["dg"]
    t_flush_pixel = p["NSBINFLS"] * c["h"] + c["dg"]
    t_clear = p["NCOLCLR"] * (p["NSBINCLR"] * c["h"] + c["dg"]) if p["BOOLCLR"] == 1 else 0
    t_read_row = (t_clear + p["NPBIN"] * c["v"] + (p["NCOLPRE"] + p["NCOLPOST"]) * t_skip_pixel
                  + p["NCOL"] * t_read_pixel + (c["dg"] if p["DUMPLESS"] == 1 else 0))
    t_skip_row = p["NPBIN"] * c["v"] + p["NCOL"] * t_flush_pixel
    cycles = p["NROW"] * t_read_row + (p["NROWPRE"] + p["NROWPOST"]) * t_skip_row
    return cycles * 1e-8 + p.get("EXPOSURE", 0)


def formula_flush_seconds(params: Mapping[str, float]) -> float:
    """FlushCCD = NROWFLS x (Vtransfer x NPBINFLS + FlushPixel x NCOLFLS); about 27 s at the reference."""
    p = {**_SEQ_DEFAULTS, **{k: v for k, v in params.items() if k in SEQUENCER_KEYS}}
    c = _cycles(p)
    row = p["NPBINFLS"] * c["v"] + p["NCOLFLS"] * (p["NSBINFLS"] * c["h"] + c["dg"])
    return p["NROWFLS"] * row * 1e-8


_exact_cache: dict[tuple, float] = {}


def acmpy_acquisition_seconds(params: Mapping[str, float], seqfile: str,
                              subroutine: str = "AcquireImage") -> float:
    """Exact readout time (+ EXPOSURE) from acmpy's offline sequencer compiler (like image_time.py)."""
    from acmpy.sequencer.seqrunner import SequencerRunner

    constants = {k: int(v) for k, v in params.items() if k in SEQUENCER_KEYS}
    key = (seqfile, subroutine, tuple(sorted(constants.items())))
    if key not in _exact_cache:
        seq = SequencerRunner.standalone()
        seq.read(seqfile)
        with redirect_stdout(StringIO()):
            seq.compile()
        for name, value in constants.items():
            try:
                seq[name] = value
            except KeyError:
                pass
        with redirect_stdout(StringIO()):
            seq.compile()
        exposure = float(constants.get("EXPOSURE", 0)) if subroutine == "AcquireImage" else 0.0
        _exact_cache[key] = seq.seq.timing_clocks(subroutine) * 1e-8 + exposure
    return _exact_cache[key]


def estimate_acquisition_seconds(params: Mapping[str, float], seqfile: str | None = None,
                                 subroutine: str = "AcquireImage") -> float:
    """acmpy's exact value when importable, else the formula.  Excludes ~1 s of overhead."""
    if seqfile:
        try:
            return acmpy_acquisition_seconds(params, seqfile, subroutine)
        except Exception as err:               # noqa: BLE001 - any failure -> formula
            log.debug("exact timing unavailable (%s); using the formula", err)
    if subroutine == "FlushCCD":
        return formula_flush_seconds(params)
    return formula_acquisition_seconds(params)


def flush_call_seconds(params: Mapping[str, float], seqfile: str | None) -> float:
    """How long ``daq.flush_ccd()`` blocks: acmpy sleeps int(t) + 1 + int(min(0.1 t, 10)) seconds."""
    t = estimate_acquisition_seconds(params, seqfile, "FlushCCD")
    return int(t) + 1 + int(min(0.1 * t, 10)) + 0.5


# --------------------------------------------------------------------------- #
# Decisions                                                                    #
# --------------------------------------------------------------------------- #

def should_erase(policy: str, voltages_changed: bool, first_packet: bool) -> bool:
    """Decide whether the erase procedure runs after a parameter change."""
    if policy == "always":
        return True
    if policy == "never":
        return False
    if policy == "first_only":
        return first_packet
    if policy == "on_voltage_change":
        return first_packet or voltages_changed
    raise ValueError(f"unknown erase policy {policy!r}")


def voltages_differ(theta: Mapping[str, float], theta_prev: Mapping[str, float] | None) -> bool:
    """True if any voltage (not sequencer constant) differs between the two sets."""
    if theta_prev is None:
        return True
    _, now = split_theta(theta)
    _, before = split_theta(theta_prev)
    return any(abs(now[k] - before.get(k, float("inf"))) > 1e-9 for k in now)


def _primitive_keys(theta: Mapping[str, float]) -> set[str]:
    """The pins and timing constants a theta sets (groups such as VDD expanded to VDD1, VDD2)."""
    sequencer, pins = split_theta(theta)
    return set(sequencer) | set(pins)


def complete_theta(theta: Mapping[str, float], nominal: Mapping[str, float]) -> dict[str, float]:
    """
    ``theta`` plus every parameter of :data:`COMPLETE_KEYS` it does not set, at its nominal value
    (``nominal`` = limits.toml [nominal] for the voltages, :data:`NOMINAL_TIMING` for the timing).

    Why: a packet only sets what its theta holds, and acmpy keeps every value until it is set
    again.  A parameter that one packet changes and the next packet's theta_ref does not hold
    (e.g. delayInt or RG_1_H, missing from the example tables) would stay changed for the next
    packet's reference traces, and its manifest would not say so.  With complete thetas every
    packet sets all of them.  Raises ValueError if a parameter has no nominal value.
    """
    covered = set()
    for key in theta:
        covered |= {key} if key in SEQUENCER_KEYS else set(expand_key(key))
    full = dict(theta)
    defaults = {**nominal, **NOMINAL_TIMING}
    missing = sorted(k for k in COMPLETE_KEYS - covered if k not in defaults)
    if missing:
        raise ValueError(f"theta does not set {missing} and there is no nominal value for them "
                         "(limits.toml [nominal])")
    full.update({k: defaults[k] for k in sorted(COMPLETE_KEYS - covered)})
    return full


def readback_problems(expected: Mapping[str, float], readback: Mapping[str, Any] | None) -> list[str]:
    """
    Compare an apply step's readback (``_r['readback']``: ``daq[key]`` for every key set) with the
    values it set.  A DAC pin may read back as ``[side A, side B]``: both must match.  Empty = fine.
    """
    if readback is None:
        return ["the receipt has no readback"]
    problems = []
    for name, want in expected.items():
        if name not in readback:
            problems.append(f"{name}: not read back")
            continue
        got = readback[name]
        try:
            if name in SEQUENCER_KEYS:
                ok = int(got) == int(want)
            else:
                tol = READBACK_TOL_BIAS_V if name in BIAS_PINS else READBACK_TOL_RAIL_V
                values = got if isinstance(got, (list, tuple)) else [got]
                ok = all(abs(float(v) - float(want)) <= tol for v in values)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            problems.append(f"{name} reads {got}, set {want}")
    return problems


def theta_difference(theta: Mapping[str, float], theta_ref: Mapping[str, float]) -> dict[str, float]:
    """Parameters that differ from the reference: ``{name: theta - theta_ref}``."""
    return {k: float(v) - float(theta_ref[k]) for k, v in theta.items()
            if k in theta_ref and abs(float(v) - float(theta_ref[k])) > 1e-9}


def config_id(theta: Mapping[str, float]) -> str:
    """Short, stable name of a parameter set (the same set always gives the same id)."""
    text = json.dumps({k: float(v) for k, v in sorted(theta.items())})
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def _check_theta(board: int, theta: Mapping[str, float], limits: Limits, label: str) -> None:
    """theta holds operating parameters only; the geometry belongs to [image]/[trace]."""
    misplaced = [k for k in theta if k in cmd.GEOMETRY_KEYS or k in cmd.ACQUISITION_KEYS]
    if misplaced:
        raise ValueError(f"{label} of board {board} contains {misplaced}: the image and trace "
                         "size belong in [image] and [trace], not in theta")
    try:
        limits.check(theta)
    except LimitsError as err:
        raise LimitsError(f"{label} of board {board}: {err}") from None


# --------------------------------------------------------------------------- #
# Planning: everything that will be sent, built without touching the hardware   #
# --------------------------------------------------------------------------- #

def plan_packet(request: PacketRequest, setup: DaqSetup, limits: Limits,
                prev_theta: Mapping[int, Mapping[str, float]] | None) -> list[PlannedStep]:
    """
    Validate the request and return the ordered list of steps.  Nothing is sent.

    ``prev_theta`` is the previous packet's theta per board (where the voltages are now);
    ``None`` = unknown (first packet: the voltages are applied directly, without a ramp).
    Raises ``LimitsError`` (parameters not acceptable), ``KeyError`` (incomplete image
    geometry) or ``ValueError`` (inconsistent request).
    """
    # ---- checks ---------------------------------------------------------------------------
    # every packet sets EVERY voltage and timing constant (complete_theta), the reference too
    request.theta = {b: complete_theta(t, limits.nominal) for b, t in request.theta.items()}
    if request.theta_ref is not None:
        request.theta_ref = {b: complete_theta(t, limits.nominal) for b, t in request.theta_ref.items()}
        for board in request.theta:
            if board in request.theta_ref and \
                    _primitive_keys(request.theta[board]) != _primitive_keys(request.theta_ref[board]):
                raise ValueError(f"theta and theta_ref of board {board} set different parameters")
    if set(request.theta) != set(setup.theta_boards):
        raise ValueError(f"theta is given for boards {sorted(request.theta)}, "
                         f"theta_boards is {setup.theta_boards}")
    for board, theta in request.theta.items():
        _check_theta(board, theta, limits, "theta")
    if request.theta_ref is not None:
        if set(request.theta_ref) != set(setup.theta_boards):
            raise ValueError(f"theta_ref is given for boards {sorted(request.theta_ref)}, "
                             f"theta_boards is {setup.theta_boards}")
        for board, theta in request.theta_ref.items():
            _check_theta(board, theta, limits, "theta_ref")
    image = cmd.image_parameters(request.image)
    trace = cmd.trace_parameters(request.trace, image)
    mismatch = cmd.trace_image_differences(trace, image)
    if mismatch:
        log.warning("the traces read their pixels differently from the image: %s (trace, image); "
                    "set [trace] match_image = true to read them the same way", mismatch)
    limits.check(cmd.sequencer_part(image))          # image size within the allowed windows
    limits.check(cmd.sequencer_part(trace))

    changed = request.theta_ref is not None and any(
        theta_difference(request.theta[b], request.theta_ref[b]) for b in setup.theta_boards)
    take_ref = request.theta_ref is not None and (
        request.reference_traces == "always"
        or (request.reference_traces == "when_different" and changed))

    # ---- durations (the longest board decides; boards without theta use the defaults) -------
    def per_board_seconds(params: Mapping[str, Any], subroutine: str, thetas: Mapping[int, Mapping]) -> float:
        sets = [dict(t) for t in thetas.values()] or [{}]
        return max(estimate_acquisition_seconds({**t, **cmd.sequencer_part(params)}, request.sequencer, subroutine)
                   for t in sets)

    def flush_seconds(thetas: Mapping[int, Mapping]) -> float:
        sets = [dict(t) for t in thetas.values()] or [{}]
        return max(flush_call_seconds(t, request.sequencer) for t in sets)

    margin, c = setup.timeout_margin_s, setup.commands
    steps: list[PlannedStep] = []

    def add(kind, name, target, boards, lines, expected_s, timeout_s, purpose, repeat=0):
        steps.append(PlannedStep(kind, name, target, list(boards), lines, expected_s,
                                 timeout_s + margin, purpose, repeat))

    def pool(kind, name, lines, expected_s, timeout_s, purpose):
        add(kind, name, setup.pool, setup.boards, lines, expected_s, timeout_s, purpose)

    def apply(name, thetas, starts, purpose):
        """One step per board: that board's sequencer constants, then its voltage ramp."""
        for board, theta in thetas.items():
            sequencer, _ = split_theta(theta)
            ramp = limits.ramp_plan(theta, (starts or {}).get(board))
            for point in ramp[:-1]:
                limits.check(point)                 # every intermediate state must be safe too
            seconds = len(ramp) * request.ramp_settle_s + 1
            add("apply", name, setup.client_name(board), [board],
                cmd.apply_lines(sequencer, ramp, request.ramp_settle_s), seconds, seconds,
                f"{purpose} (acm{board}, {len(ramp)} ramp step(s)), then read back")
            steps[-1].expected = {**sequencer, **ramp[-1]}

    def erase(name, thetas, starts):
        """Erase on the theta boards if the policy asks for it (default policy: never)."""
        first = starts is None
        if any(should_erase(request.erase_policy, voltages_differ(t, (starts or {}).get(b)), first)
               for b, t in thetas.items()):
            for board in setup.theta_boards:
                add("erase", name, setup.client_name(board), [board], cmd.erase_lines(c["erase"]),
                    30, 120, "erase procedure (no e-purge)")

    def flush(count, thetas, purpose):
        if count > 0:
            seconds = count * flush_seconds(thetas)
            pool("flush", f"flush_x{count}", cmd.flush_lines(c["flush"], count), seconds, 2 * seconds,
                 purpose)

    def trace_sets(kind, count, thetas):
        one = per_board_seconds(trace, trace["subroutine"], thetas)
        seconds = len(setup.trace_channels) * (one + 1.5)        # + prepare and the 0.5 s wait
        for repeat in range(1, count + 1):
            if repeat > 1:
                flush(request.flushes_between_traces, thetas, "flush between trace sets")
            for board in setup.trace_boards:
                add(kind, f"{kind}_{repeat}", setup.client_name(board), [board],
                    cmd.trace_lines(f"{request.packet_id}_{kind}{repeat}", setup.trace_channels, trace),
                    seconds, 3 * seconds,
                    f"{'reference ' if kind == 'ref_traces' else ''}trace set {repeat} on acm{board}, "
                    "one per channel", repeat)

    # ---- the sequence -------------------------------------------------------------------------
    pool("status", "status", cmd.status_lines(), 1, 1,
         "read the run directory, the loaded sequencer and the synchronisation/clock bits")
    pool("standalone", "standalone", cmd.standalone_lines(c["standalone_mode"], c.get("external_clock", "")),
         1, 1, "standalone mode for flushes and traces; external clock if not yet set")

    current = dict(prev_theta) if prev_theta else None
    if take_ref:
        apply("apply_ref", request.theta_ref, current, "go to the reference parameters")
        erase("erase_ref", request.theta_ref, current)
        flush(request.n_initial_flushes, request.theta_ref, "initial flushes at the reference")
        trace_sets("ref_traces", request.n_reference_traces, request.theta_ref)
        apply("apply", request.theta, request.theta_ref, "go to the packet's parameters theta")
        erase("erase", request.theta, request.theta_ref)
        flush(request.flushes_after_change, request.theta, "flush after the change to theta")
    else:
        apply("apply", request.theta, current, "go to the packet's parameters theta")
        erase("erase", request.theta, current)
        flush(request.n_initial_flushes, request.theta, "initial flushes at theta")
    trace_sets("traces", request.n_traces, request.theta)

    # leader mode, image, standalone, final flush(es): ONE queue entry, so a repeat command of
    # the cdaq queue can never run while the boards are in leader/follower mode
    image_s = per_board_seconds(image, image["subroutine"], request.theta)
    final_s = request.n_final_flushes * flush_seconds(request.theta)
    pool("image", "image",
         cmd.synchronized_image_lines(f"{request.packet_id}_img", image,
                                      c["leader_mode"].format(leader=setup.leader), c["standalone_mode"],
                                      c["flush"], request.n_final_flushes),
         image_s + 2 + final_s, 2 * (image_s + final_s) + 60,
         f"master/slave mode (leader acm{setup.leader}), synchronised image on all boards, back to "
         f"standalone, {request.n_final_flushes} flush(es)")
    return steps


def estimate_packet_seconds(steps: list[PlannedStep]) -> float:
    """Expected duration of a packet: the steps' expected durations plus ~2 s queue overhead each.
    The FITS wait is not included."""
    return sum(step.expected_s + 2 for step in steps)


# --------------------------------------------------------------------------- #
# Running a packet                                                             #
# --------------------------------------------------------------------------- #

def trace_problems(stem: str, channel: int) -> list[str] | None:
    """
    Decode a trace written by this packet and return its quality problems.

    ``[]`` = healthy, a non-empty list = lost or corrupted data (the step is retaken), ``None`` =
    could not be checked here (file not visible from this computer, or numpy missing).
    """
    path = Path(f"{stem}.RJ45ADCDATA{channel}")
    if not path.exists():
        return None
    try:
        from decode_trace import decode_trace
    except ImportError:
        return None
    try:
        return decode_trace(path).quality_problems()
    except Exception as err:                       # noqa: BLE001 - an undecodable file is damaged
        return [f"could not decode: {err}"]


def wait_for_fits(directory: Path, binname: str, timeout_s: float) -> list[str]:
    """Poll for the image builder's FITS files of one image (both naming conventions)."""
    deadline = time.monotonic() + timeout_s
    patterns = (f"avg_{binname}.fits*", f"var_{binname}.fits*", f"cds_avg_{binname}*", f"cds_var_{binname}*")
    while True:
        found = sorted(str(p) for pattern in patterns for p in directory.glob(pattern))
        if found or time.monotonic() >= deadline:
            return found
        time.sleep(2.0)


def _int_keys(table: Mapping | None) -> dict[int, Any] | None:
    """JSON turns board numbers into strings; turn them back."""
    return None if table is None else {int(k): v for k, v in table.items()}


def take_packet(controller, request: PacketRequest, setup: DaqSetup, limits: Limits,
                prev_manifest: dict | None = None, allow_unreviewed_limits: bool = False,
                repeat_queue_suspended: list | None = None) -> dict:
    """
    Run all steps of one packet and return (and write) its manifest.

    ``controller`` is a ``CdaqController`` (real DAQ) or ``FakeController`` (rehearsal).
    ``prev_manifest`` is the manifest of the previous packet (``None`` = first packet): its
    theta is where each board's voltages are ramped from.  ``repeat_queue_suspended`` is what
    the caller removed from the cdaq repeat queue for this packet (recorded in the manifest;
    ``None`` = the repeat queue was not touched, so its commands ran between the steps).
    This function does not touch the repeat queue itself: ``main()`` and ``scan_packets.py``
    do (``daq_control.repeat_queue_suspended``).
    """
    if not limits.reviewed and not allow_unreviewed_limits:
        raise LimitsError(f"{limits.source} has reviewed = false; review the limits first")
    prev_theta = _int_keys(prev_manifest.get("theta")) if prev_manifest else None
    steps = plan_packet(request, setup, limits, prev_theta)          # raises before anything is sent

    work = Path(setup.work_dir)
    receipt_dir = work / "receipts" / request.packet_id
    receipt_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "packets" / f"{request.packet_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    seconds_since = None
    if prev_manifest and prev_manifest.get("date_end"):
        ended = datetime.fromisoformat(prev_manifest["date_end"]).replace(tzinfo=timezone.utc)
        seconds_since = round((datetime.now(timezone.utc) - ended).total_seconds(), 1)

    ref = request.theta_ref
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "packet_id": request.packet_id,
        "scan_name": request.scan_name,
        "status": "started",
        "date_start": utc_now(),
        # parameters, per board
        "theta": request.theta,
        "theta_ref": ref,
        "theta_id": {b: config_id(t) for b, t in request.theta.items()},
        "reference_id": {b: config_id(t) for b, t in ref.items()} if ref else None,
        "delta_theta": {b: theta_difference(request.theta[b], ref[b]) for b in request.theta} if ref else None,
        "image": cmd.image_parameters(request.image),
        "trace": cmd.trace_parameters(request.trace, cmd.image_parameters(request.image)),
        "trace_matches_image": not cmd.trace_image_differences(
            cmd.trace_parameters(request.trace, cmd.image_parameters(request.image)),
            cmd.image_parameters(request.image)),
        # history
        "prev_packet_id": prev_manifest.get("packet_id") if prev_manifest else None,
        "prev_theta": prev_theta,
        "seconds_since_previous": seconds_since,
        # how it was taken
        "daq": {"pool": setup.pool, "boards": setup.boards, "leader": setup.leader,
                "theta_boards": setup.theta_boards, "trace_boards": setup.trace_boards,
                "trace_channels": setup.trace_channels, "queue": setup.queue},
        "sequence": {k: getattr(request, k) for k in (
            "n_initial_flushes", "n_reference_traces", "n_traces", "flushes_between_traces",
            "flushes_after_change", "n_final_flushes", "erase_policy", "reference_traces",
            "retake_damaged_traces")},
        "repeat_queue_suspended": repeat_queue_suspended,
        "sequencer": request.sequencer,
        "erase_done": sorted({s.name for s in steps if s.kind == "erase"}),
        "estimated_seconds": round(estimate_packet_seconds(steps)),
        "limits_file": limits.source,
        "limits_reviewed": limits.reviewed,
        # results
        "steps": [],
        "external_clock_set_by_packet": [],
        "reference_traces": [],
        "traces": [],
        "images": [],
        "trace_quality": {},
        "warnings": [],
    }

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    def run(step: PlannedStep, attempt: int = 0) -> dict[int, dict]:
        """Queue one step and wait for its receipts; the step id is <index>_<name>[_<board>]."""
        suffix = f"_{step.target}" if step.target != setup.pool else ""
        step_id = f"{len(manifest['steps']):02d}_{step.name}{suffix}" + (f"_retake{attempt}" if attempt else "")
        log.info("[%s] %s (%s)", request.packet_id, step_id, step.purpose)
        started = time.monotonic()
        controller.submit(step.target, step.code(step_id, str(receipt_dir)))
        entry = {"step": step_id, "kind": step.kind, "target": step.target, "boards": step.boards}
        manifest["steps"].append(entry)
        save()
        receipts = wait_for_receipts(receipt_dir, step_id, step.boards, step.timeout_s,
                                     getattr(controller, "poll_interval_s", setup.poll_interval_s))
        entry.update(seconds=round(time.monotonic() - started, 1), receipts=receipts)
        save()
        return receipts

    def record_traces(kind: str, step: PlannedStep, receipts: dict[int, dict]) -> bool:
        """Store the trace files; return True if any decoded trace shows lost data."""
        damaged = False
        for board, receipt in receipts.items():
            for channel, stem in zip(setup.trace_channels, receipt["files"]):
                manifest[kind].append({"board": board, "channel": channel, "repeat": step.repeat,
                                       "stem": stem})
                problems = trace_problems(stem, channel)
                manifest["trace_quality"][stem] = "not checked" if problems is None else problems
                damaged |= bool(problems)
        return damaged

    try:
        for step in steps:
            receipts = run(step)

            if step.kind == "status":
                for board, receipt in receipts.items():
                    loaded = Path(receipt.get("seqfile") or "").name
                    if loaded and loaded != Path(request.sequencer).name:
                        raise PacketError(f"acm{board} has sequencer {loaded} loaded, "
                                          f"this packet expects {request.sequencer}")
                manifest["run_directories"] = {b: r.get("directory") for b, r in receipts.items()}

            elif step.kind == "apply":
                for board, receipt in receipts.items():
                    problems = readback_problems(step.expected or {}, receipt.get("readback"))
                    if problems:
                        raise PacketError(f"acm{board} {step.name}: the readback differs from what was set: "
                                          + "; ".join(problems))

            elif step.kind == "standalone":
                manifest["external_clock_set_by_packet"] = [
                    b for b, r in receipts.items() if "clock_before" in r and r["clock_before"] != "external"]

            elif step.kind in ("ref_traces", "traces"):
                kind = "reference_traces" if step.kind == "ref_traces" else "traces"
                attempt = 0
                while record_traces(kind, step, receipts) and attempt < request.retake_damaged_traces:
                    attempt += 1
                    manifest["warnings"].append(f"{step.name} on {step.target}: damaged trace(s), retake {attempt}")
                    manifest[kind] = [t for t in manifest[kind]          # drop the damaged set
                                      if not (t["board"] in receipts and t["repeat"] == step.repeat)]
                    receipts = run(step, attempt)

            elif step.kind == "image":
                for board, receipt in receipts.items():
                    manifest["images"].append({"board": board, "stem": receipt["files"][0], "fits": []})
                    if not receipt.get("standalone_after_image"):
                        manifest["warnings"].append(f"acm{board} did not confirm standalone mode after the image")
                save()
                idle = getattr(controller, "idle", None)
                if idle:
                    idle()                  # fake queue: the repeat queue runs while we wait for the FITS
                for image in manifest["images"]:
                    stem = Path(image["stem"])
                    image["fits"] = wait_for_fits(stem.parent, stem.name, setup.fits_wait_s)
                    if not image["fits"]:
                        manifest["warnings"].append(f"no FITS file yet for {stem.name}")
        manifest["status"] = "ok"

    except BaseException as err:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(err).__name__}: {err}"
        raise
    finally:
        manifest["date_end"] = utc_now()
        save()

    log.info("[%s] done", request.packet_id)
    return manifest


def run_status_only(controller, status_step: PlannedStep, setup: DaqSetup, packet_id: str) -> dict:
    """
    Send only the status step (it reads, it changes nothing) and return the receipts.

    Use it as the first contact with the real DAQ: it shows that the queue, the clients, the
    pool and the receipt directory work, and what each board reports (run directory, loaded
    sequencer, synchronisation and clock bits).
    """
    assert status_step.kind == "status"
    receipt_dir = Path(setup.work_dir) / "receipts" / f"{packet_id}_status_only"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    step_id = f"status_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    controller.submit(status_step.target, status_step.code(step_id, str(receipt_dir)))
    return wait_for_receipts(receipt_dir, step_id, status_step.boards, status_step.timeout_s,
                             setup.poll_interval_s)


# --------------------------------------------------------------------------- #
# Command-line interface                                                       #
# --------------------------------------------------------------------------- #

def load_toml_or_json(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def request_from_params(params: dict, setup: DaqSetup) -> PacketRequest:
    """Build a :class:`PacketRequest` from a packet file (``configs/packet_example.toml``)."""
    options = params.get("options", {})
    theta = board_thetas(params["theta"], params.get("theta_board", {}), setup.theta_boards)
    theta_ref = (board_thetas(params["theta_ref"], params.get("theta_ref_board", {}), setup.theta_boards)
                 if "theta_ref" in params else None)
    counts = {k: int(options[k]) for k in ("n_initial_flushes", "n_reference_traces", "n_traces",
                                           "flushes_between_traces", "flushes_after_change",
                                           "n_final_flushes", "retake_damaged_traces") if k in options}
    return PacketRequest(
        packet_id=params["packet_id"], theta=theta, theta_ref=theta_ref,
        image=dict(params["image"]), trace=dict(params.get("trace", {})),
        sequencer=options.get("sequencer", DEFAULT_SEQUENCER_FILE),
        erase_policy=options.get("erase_policy", "never"),
        reference_traces=options.get("reference_traces", "always"),
        scan_name=params.get("scan_name", ""), **counts)


def make_controller(backend: str, setup: DaqSetup, fake_data: str | None = None,
                    time_scale: float = 0.0, fake_repeat_flush: bool = True, echo: bool = False):
    """
    ``cdaq`` -> real queue; ``fake`` -> FakeDAQ boards (copies of real files if ``fake_data``).

    The fake queue starts like the group's session: ``daq.flush_ccd()`` for the pool in the
    repeat queue (``fake_repeat_flush``).  It writes its state to
    ``<work_dir>/fake_cdaq_state.json`` for ``fake_cdaq_monitor.py``.  ``time_scale``: fraction
    of the real durations the fake boards sleep (0 = instant).  ``echo``: print the
    "Publishing: ..." line of every queue command, as ``ccd_cdaq`` does on the DAQ machine.
    """
    if backend == "cdaq":
        from daq_control import CdaqController
        return CdaqController(setup.queue)
    from daq_control import FakeController
    from fake_daq import FakeDAQ, example_sources
    run_dir = Path(setup.work_dir) / "fake_run"
    fakes = {b: FakeDAQ(b, run_dir, examples=example_sources(fake_data, b) if fake_data else None,
                        time_scale=time_scale)
             for b in setup.boards}
    targets = {setup.pool: list(setup.boards), **{setup.client_name(b): [b] for b in setup.boards}}
    repeat = [{"to": setup.pool, "cmd": setup.commands["flush"]}] if fake_repeat_flush else []
    return FakeController(fakes, targets, queue=setup.queue, repeat=repeat,
                          state_file=Path(setup.work_dir) / "fake_cdaq_state.json",
                          pools={setup.pool: [setup.client_name(b) for b in setup.boards]},
                          echo_publishing=echo)


def repeat_queue_backup_file(setup: DaqSetup, name: str) -> Path:
    """Where the saved repeat queue goes: ``<work_dir>/queue_backups/<name>_<UTC>.json``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(setup.work_dir) / "queue_backups" / f"{name}_{stamp}.json"


def print_plan(steps: list[PlannedStep], request: PacketRequest, setup: DaqSetup, full: bool) -> None:
    for index, step in enumerate(steps):
        print(f"\n[{index:02d}] {step.name:14s} -> {step.target:8s} expected {step.expected_s:6.1f} s, "
              f"timeout {step.timeout_s / 60:.1f} min   ({step.purpose})")
        text = step.code(f"{index:02d}_{step.name}", f"{setup.work_dir}/receipts/{request.packet_id}") \
            if full else "\n".join(step.lines)
        print("\n".join("      " + line for line in text.splitlines()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--params", required=True, help="packet TOML/JSON (see the module docstring)")
    parser.add_argument("--setup", default=str(CONFIG_DIR / "daq_setup.toml"))
    parser.add_argument("--limits", default=str(CONFIG_DIR / "limits.toml"))
    parser.add_argument("--backend", choices=["cdaq", "fake"], default="fake")
    parser.add_argument("--work-dir", help="override [daq] work_dir (e.g. for a rehearsal)")
    parser.add_argument("--fake-data", help="data/ directory whose real files the fake boards copy")
    parser.add_argument("--fake-time-scale", type=float, default=0.0,
                        help="fake boards sleep this fraction of the real durations (e.g. 0.05), "
                             "to watch the rehearsal with fake_cdaq_monitor.py")
    parser.add_argument("--prev-manifest", help="manifest of the previous packet")
    parser.add_argument("--show-commands", action="store_true",
                        help="print every step and the text each board would run, then stop")
    parser.add_argument("--full", action="store_true", help="with --show-commands: include the receipt frame")
    parser.add_argument("--status-only", action="store_true",
                        help="send ONLY the read-only status step and print what the boards report "
                             "(safe first contact with the real DAQ)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    setup = DaqSetup.load(args.setup)
    if args.work_dir:
        setup.work_dir = args.work_dir
    limits = Limits.load(args.limits)
    prev = load_toml_or_json(args.prev_manifest) if args.prev_manifest else None

    try:
        request = request_from_params(load_toml_or_json(args.params), setup)
        steps = plan_packet(request, setup, limits, _int_keys(prev.get("theta")) if prev else None)
    except (LimitsError, KeyError, ValueError) as err:
        raise SystemExit(f"packet rejected: {err}")

    if args.show_commands:
        print_plan(steps, request, setup, args.full)
        print(f"\n{len(steps)} steps, about {estimate_packet_seconds(steps) / 60:.1f} min")
        if not limits.reviewed:
            print(f"NOTE: {limits.source} has reviewed = false; the cdaq backend refuses to run")
        return

    controller = make_controller(args.backend, setup, args.fake_data, args.fake_time_scale, echo=True)
    if args.status_only:
        receipts = run_status_only(controller, steps[0], setup, request.packet_id)
        print(json.dumps(receipts, indent=2, default=str))
        return
    try:
        with repeat_queue_suspended(controller, repeat_queue_backup_file(setup, request.packet_id),
                                    enabled=setup.repeat_queue == "suspend") as saved:
            manifest = take_packet(controller, request, setup, limits, prev,
                                   allow_unreviewed_limits=(args.backend == "fake"),
                                   repeat_queue_suspended=saved if setup.repeat_queue == "suspend" else None)
    except (StepError, PacketError, LimitsError) as err:
        raise SystemExit(f"packet {request.packet_id} FAILED: {err}")
    print(json.dumps({k: manifest[k] for k in ("packet_id", "status", "external_clock_set_by_packet",
                                                "warnings")}, indent=2))


if __name__ == "__main__":
    main()

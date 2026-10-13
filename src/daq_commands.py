"""
daq_commands.py -- the exact Python text that each DAQ board is asked to execute.

The packet scripts never talk to an ACM board directly.  Like the group's
``scripts/take_image.py`` and ``scripts/take_trace.py`` they put short pieces of Python in
the cdaq queue (``ccd_cdaq.add2queue``); the cdaq client of a board runs them with
``exec`` in its own process, where ``daq`` is the ``acmpy.ACM`` object of that board.  This
module only BUILDS those pieces of text.  It has no side effects, so everything a packet
will send can be printed and read before anything is run (``take_packet.py
--show-commands`` and the walkthrough notebook do this).

The texts follow the group's scripts (2026-10-09 versions shown by the user):

    take_trace.py   daq.standalone(); time.sleep(.1); then per channel
                    daq.take_trace(fname="trace", subroutine="AcquireImage", EXPOSURE=0,
                                   NROW=2, NCOL=5, NSBIN=20, NPBIN=20, NDCM=500, BOOLCLR=0, ch=i)
    take_image.py   daq.master(<id>) (when a master is given), then
                    daq.take_image(fname="img", <all keys of the image_params TOML>)
                    and with -f: daq.standalone(); daq.flush_ccd()
    session start   cdaq qadd "daq.use_external_clock()" acmpool   (set once, never undone)
    idle            cdaq qadd "daq.flush_ccd()" acmpool -t repeat  (the REPEAT queue: run again
                    and again whenever the simple queue is empty; see take_packet.py)

Every step is wrapped the same way (:func:`wrap_step`)::

    import json as _json, time as _time, traceback as _tb, pathlib as _pl
    _r = {"step": "<step id>", "board": daq.id, "t_start": _time.time(), "files": []}
    try:
        <the step's own lines>
        _r["ok"] = True
    except Exception:
        _r["ok"] = False
        _r["error"] = _tb.format_exc()
    _r["t_end"] = _time.time()
    _pl.Path("<receipt dir>", "<step id>.acm" + str(daq.id) + ".json").write_text(_json.dumps(_r, default=str))

The small JSON file written at the end is the step's **receipt**.  The cdaq queue is
fire-and-forget (it returns neither results nor errors), so the receipt is how the packet
script learns that a board finished a step, whether it raised an error, and which files it
wrote (``daq.acquire._filepath`` after each acquisition).
"""

from __future__ import annotations

from typing import Mapping

# --------------------------------------------------------------------------- #
# Acquisition parameters                                                       #
# --------------------------------------------------------------------------- #

#: Sequencer constants that define the geometry of one acquisition.  Every acquisition (trace
#: or image) sends ALL of them, so nothing is inherited from the previous acquisition: e.g. an
#: image with NCOLPRE = 5300 followed by a trace that does not set NCOLPRE would read the 5300
#: prescan columns in the trace too.
GEOMETRY_KEYS = ("NROW", "NCOL", "NDCM", "NPBIN", "NSBIN", "EXPOSURE",
                 "NDCMPRE", "NROWPRE", "NROWPOST", "NCOLPRE", "NCOLPOST", "BOOLCLR")

#: Keys of an acquisition table that are not sequencer constants.
ACQUISITION_KEYS = ("subroutine", "acq_level")

#: Default trace: the values of the group's scripts/take_trace.py, plus the cuts it does not
#: set, forced to 0 (see GEOMETRY_KEYS).  A packet's [trace] table overrides any of them.
DEFAULT_TRACE = {"subroutine": "AcquireImage", "EXPOSURE": 0, "NROW": 2, "NCOL": 5,
                 "NSBIN": 20, "NPBIN": 20, "NDCM": 500, "BOOLCLR": 0,
                 "NDCMPRE": 0, "NROWPRE": 0, "NROWPOST": 0, "NCOLPRE": 0, "NCOLPOST": 0}


def trace_parameters(table: Mapping[str, object] | None) -> dict[str, object]:
    """The trace parameters: :data:`DEFAULT_TRACE` with the packet's ``[trace]`` values on top."""
    params = {**DEFAULT_TRACE, **dict(table or {})}
    params.pop("acq_level", None)                  # take_trace sets the level itself (channel = 0)
    return params


def image_parameters(table: Mapping[str, object]) -> dict[str, object]:
    """
    The image parameters from the packet's ``[image]`` table (the content of an image_params
    TOML file).  Every key of :data:`GEOMETRY_KEYS` must be present; ``acq_level`` defaults to
    2 and ``subroutine`` to ``"AcquireImage"``, as in take_image.py.
    """
    params = {"subroutine": "AcquireImage", "acq_level": 2, **dict(table)}
    missing = [k for k in GEOMETRY_KEYS if k not in params]
    if missing:
        raise KeyError(f"[image] must define the complete image geometry; missing {missing}")
    return params


def sequencer_part(params: Mapping[str, object]) -> dict[str, int]:
    """The sequencer constants of an acquisition table (used for the limits and the timing)."""
    return {k: int(v) for k, v in params.items() if k not in ACQUISITION_KEYS}


# --------------------------------------------------------------------------- #
# The receipt frame                                                            #
# --------------------------------------------------------------------------- #

_HEADER = (
    "import json as _json, time as _time, traceback as _tb, pathlib as _pl\n"
    "_r = {{'step': {step!r}, 'board': daq.id, 't_start': _time.time(), 'files': []}}\n"
    "try:\n"
)
_FOOTER = (
    "    _r['ok'] = True\n"
    "except Exception:\n"
    "    _r['ok'] = False\n"
    "    _r['error'] = _tb.format_exc()\n"
    "_r['t_end'] = _time.time()\n"
    "_pl.Path({receipt_dir!r}, {step!r} + '.acm' + str(daq.id) + '.json')"
    ".write_text(_json.dumps(_r, default=str))\n"
)


def wrap_step(step: str, lines: list[str], receipt_dir: str) -> str:
    """Wrap the step's own ``lines`` in the receipt-writing frame (see the module docstring)."""
    body = "".join(f"    {line}\n" for line in lines) or "    pass\n"
    return _HEADER.format(step=step) + body + _FOOTER.format(step=step, receipt_dir=receipt_dir)


def _kwargs(params: Mapping[str, object]) -> str:
    """``{'NROW': 2, 'subroutine': 'AcquireImage'}`` -> ``"NROW=2, subroutine='AcquireImage'"``."""
    return ", ".join(f"{k}={v!r}" for k, v in params.items())


def _record_state() -> list[str]:
    """Lines that store the board's own view of its state in the receipt."""
    return [
        "_r['directory'] = str(daq.acquire.directory)",
        "_r['seqfile'] = str(getattr(daq.sequencer, 'seqfile', ''))",
        "try:",
        "    _r['system_status'] = daq.command.get_system_status()",
        "except Exception as _e:",
        "    _r['system_status'] = 'unavailable: ' + repr(_e)",
    ]


# --------------------------------------------------------------------------- #
# One function per kind of step                                                #
# --------------------------------------------------------------------------- #

def status_lines() -> list[str]:
    """Only read: run directory, loaded sequencer file, synchronisation/clock status bits."""
    return _record_state()


def standalone_lines(standalone_command: str, clock_command: str = "") -> list[str]:
    """
    Standalone mode (needed for flushes and traces), as in take_trace.py: the command, then a
    0.1 s pause.  With ``clock_command`` the external clock is also made sure of: it is sent
    ONLY if the board does not already report the external clock (the group sets it once per
    session and never switches back).  If the status cannot be read, it is sent anyway; that
    is harmless when the clock already is external.
    """
    lines = [standalone_command, "_time.sleep(0.1)"]
    if clock_command:
        lines += [
            "try:",
            "    _clock = daq.command.get_system_status().get('activeclk')",
            "except Exception:",
            "    _clock = None",
            "_r['clock_before'] = _clock",
            "if _clock != 'external':",
            f"    {clock_command}",
            "    _time.sleep(0.5)",
        ]
    return lines + _record_state()


def apply_lines(sequencer: Mapping[str, int], ramp: list[dict[str, float]],
                settle_s: float) -> list[str]:
    """
    Set the sequencer constants of theta, then walk the voltages through the ramp steps.

    ``ramp`` comes from ``Limits.ramp_plan`` (every step already checked against the limits);
    the board waits ``settle_s`` seconds after every step except the last.
    """
    lines = [f"daq[{key!r}] = {int(value)!r}" for key, value in sequencer.items()]
    for index, step in enumerate(ramp):
        lines += [f"daq[{pin!r}] = {round(float(volts), 6)!r}" for pin, volts in step.items()]
        if index < len(ramp) - 1:
            lines.append(f"_time.sleep({settle_s!r})")
    lines.append(f"_r['ramp_steps'] = {len(ramp)}")
    return lines


def flush_lines(flush_command: str, count: int) -> list[str]:
    """
    ``count`` flushes, one after the other.  ``daq.flush_ccd()`` runs the sequencer routine
    FlushCCD (NROWFLS x [NPBINFLS vertical shifts + NCOLFLS horizontal pixels], about 27 s
    with the reference timing) and returns only when it has finished: acmpy's
    ``start_sequencer(wait=True)`` sleeps for the routine's duration.
    """
    return [flush_command] * count


def erase_lines(erase_command: str) -> list[str]:
    """Run the erase procedure (the e-purge is never used)."""
    return [erase_command]


def trace_lines(prefix: str, channels: list[int], params: Mapping[str, object]) -> list[str]:
    """
    One trace per channel on this board, one after the other (take_trace.py sends the same
    four calls as four queue entries; here they are one entry so one receipt lists all files).

    ``daq.take_trace(ch=c, ...)`` sets the filter of channel ``c`` to level 0 (every ADC
    sample) and the other three to level 3 (no output), then runs the normal sequencer
    readout of the trace geometry.  It does NOT wait for EXPOSURE.  It must run in standalone
    mode: in follower mode the sequencer would wait for an external start that never comes.
    """
    lines = []
    for ch in channels:
        lines.append(f"daq.take_trace(fname={prefix!r}, ch={int(ch)}, {_kwargs(params)})")
        lines.append("_r['files'].append(str(daq.acquire._filepath))")
    return lines


def leader_lines(leader_command: str) -> list[str]:
    """Leader/follower mode for the image (take_image.py -m <id> sends daq.master(<id>))."""
    return [leader_command, "_time.sleep(0.5)"] + _record_state()


def image_lines(prefix: str, params: Mapping[str, object]) -> list[str]:
    """
    Take the image on this board (sent to the pool, so every board runs it at the same time).

    ``take_image`` sets the parameters, waits 0.5 s, sleeps EXPOSURE seconds in Python
    (``acquire.start_exposure``), then starts the sequencer.  In leader/follower mode the
    followers' sequencers wait for the start trigger that the leader sends through the
    synchronisation board, so all CCDs are read out together.  ``acq_level`` is ALWAYS
    passed: after a trace the board's stored level is still the trace's ``[0, 3, 3, 3]``.
    """
    return [f"daq.take_image(fname={prefix!r}, {_kwargs(params)})",
            "_r['files'].append(str(daq.acquire._filepath))"]


def synchronized_image_lines(prefix: str, params: Mapping[str, object], leader_command: str,
                             standalone_command: str, flush_command: str,
                             n_final_flushes: int) -> list[str]:
    """
    The whole synchronized image as ONE queue entry: leader mode, the image, back to standalone
    mode, then ``n_final_flushes`` flushes (``take_image.py -m <id> -f`` does the same).

    Why one entry: the cdaq queue runs its REPEAT queue (the group keeps ``daq.flush_ccd()``
    there) whenever the simple queue is empty.  The packet script queues one step, waits for
    its receipt, then queues the next, so the simple queue is empty between two steps.  With
    the leader mode, the image and the return to standalone as separate entries, a repeat flush
    could start while the boards are in leader/follower mode, where a flush waits for an
    external start that never comes.  In one entry nothing can run in between.

    The return to standalone is in a ``finally``: it also runs when ``take_image`` raises (an
    image that hangs on the hardware never returns, so that case cannot be helped here).  The
    flushes run only after a successful image.
    """
    lines = leader_lines(leader_command)
    lines.append("try:")
    lines += ["    " + line for line in image_lines(prefix, params)]
    lines += ["finally:",
              f"    {standalone_command}",
              "    _time.sleep(0.1)",
              "    _r['standalone_after_image'] = True"]
    return lines + flush_lines(flush_command, n_final_flushes)

"""
fake_daq.py -- stand-in for ``acmpy.ACM`` boards, used to rehearse the packet scripts.

``FakeDAQ`` implements the part of the ACM interface that the packet commands use
(item get/set, ``take_trace``, ``take_image``, ``erase_ccd``, ``master``/``set_leader``,
``set_standalone``, ``use_external_clock``/``use_internal_clock``, ``command.get_system_status``,
``acquire.directory``/``acquire._filepath``, ``sequencer.seqfile``).  ``FakeController`` (in
``daq_control.py``) runs the very same command text that would be queued for the real cdaq
clients against these objects, so a rehearsal exercises the real commands.

The fake is deliberately strict where the real hardware would misbehave silently:

* ``take_trace``, ``flush_ccd`` and ``erase_ccd`` raise if the board is not in standalone
  mode (on the real board the sequencer would wait for an external start that never comes);
* ``take_image`` raises if the boards of a synchronised image are not in leader/follower mode
  with one leader, or if no acquisition level is given after a trace;
* ``epurge_ccd`` raises (the scripts must never call it).

Content of the files
--------------------
By default the data files are only placeholders (the 16-byte START/BUSY/header/END frame of
an empty channel, and an empty FITS marker).  ``FakeDAQ(..., examples=...)`` instead COPIES
real files taken earlier (the 2026-10-08 traces and the 2026-10-06 image) under the new
names, so the processing script can be run end to end.  Successive traces of a channel cycle
through the five 2026-10-08 repeats (one of which lost data, so the retake is exercised).
These copies are NOT measurements at the requested settings: every ``.meta`` written by the
fake says ``"FAKE": true`` and names the source file.

The fake boards start on the INTERNAL clock, so a rehearsal shows the packet switching to the
external clock once (and not again in the next packet).

Time and log
------------
By default every call returns at once.  ``time_scale`` > 0 makes the board sleep for that
fraction of the real duration (flush ~27 s, trace and image from the sequencer formulas of
``take_packet.py``), so a rehearsal can be watched with ``fake_cdaq_monitor.py``.  Each call
also sends a log line to ``log_callback`` (set by ``FakeController``), in the style of the
real board logs (``acm.101.sequencer - INFO - Setting subroutine to FlushCCD (26 sec)``).  The
wording of these lines is an imitation; only the flush line was seen in a real session.
"""

from __future__ import annotations

import json
import shutil
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

_START, _BUSY, _END = 0xD3544152, 0xC2555359, 0xA0454E44


class FakeHangError(RuntimeError):
    """Raised where the real board would hang or misbehave."""


def _utc() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


class _FakeAcquire:
    def __init__(self) -> None:
        self.directory = Path(".")
        self._filepath = Path(".")
        self.level = [2, 2, 2, 2]             # acquisition level per channel (acmpy's default)


class _FakeSequencer:
    def __init__(self, seqfile: str) -> None:
        self.seqfile = Path("/fake/seqfiles") / seqfile


class _FakeCommand:
    """Mimics ``daq.command`` (FPGACommand): only the status read-back is implemented."""

    def __init__(self, parent: "FakeDAQ") -> None:
        self.parent = parent

    def get_system_status(self) -> dict:
        mode = self.parent.mode
        return {"master": mode == "leader", "slave": mode == "follower",
                "external_start": mode in ("leader", "follower"),
                "activeclk": self.parent.clock, "set_ext_clk": self.parent.clock == "external"}


class FakeDAQ:
    """Minimal fake of one ``acmpy.ACM`` board; every call is recorded in ``self.calls``."""

    def __init__(self, board_number: int = 101, directory: str | Path = "fake_run",
                 seqfile: str = "reference_v5_pumping.seq", examples: dict | None = None,
                 time_scale: float = 0.0) -> None:
        self.id = int(board_number)
        self.state: dict[str, object] = {}
        self.acquire = _FakeAcquire()
        self.acquire.directory = Path(directory)
        self.acquire.directory.mkdir(parents=True, exist_ok=True)
        self.sequencer = _FakeSequencer(seqfile)
        self.command = _FakeCommand(self)
        self.mode = "standalone"
        self.clock = "internal"               # the real session sets external once (CEX)
        self.examples = examples or {}
        self._trace_index: dict[int, int] = {}   # which example file the next trace of a channel copies
        self.require_sync = True              # images must be taken in leader/follower mode
        self.calls: list[tuple] = []
        self.ignore_sets: set[str] = set()   # keys this board silently does not apply (tests of the readback)
        self._counter = 0
        self.time_scale = float(time_scale)   # fraction of the real durations to sleep (0 = none)
        self.log_callback = None              # set by FakeController: (who, level, message)

    # -- log and time ------------------------------------------------------------
    def _log(self, part: str, message: str, level: str = "INFO") -> None:
        if self.log_callback:
            self.log_callback(f"acm.{self.id}.{part}", level, message)

    def _routine(self, name: str, seconds: float) -> None:
        """Announce a sequencer routine and wait for (a fraction of) its duration."""
        self._log("sequencer", f"Setting subroutine to {name} ({seconds:.0f} sec)")
        if self.time_scale > 0:
            time.sleep(seconds * self.time_scale)

    def _acquisition_seconds(self) -> float:
        try:
            from take_packet import formula_acquisition_seconds   # same formula as the planning
            return formula_acquisition_seconds({k: v for k, v in self.state.items() if isinstance(v, (int, float))})
        except Exception:                      # noqa: BLE001 - incomplete parameters: a nominal time
            return 1.0

    # -- parameters ----------------------------------------------------------
    def __setitem__(self, key, value):
        self.calls.append(("set", key, value))
        if key in ("acq_level", "level"):
            self.acquire.level = [int(value)] * 4 if isinstance(value, int) else list(value)
        if key not in self.ignore_sets:
            self.state[key] = value

    def __getitem__(self, key):
        return self.state[key]

    def load_sequencer(self, seqfile=None):
        """acmpy: compile the edited sequencer text and upload it (here: only recorded)."""
        self.calls.append(("load_sequencer", seqfile))

    # -- synchronisation -------------------------------------------------------
    def master(self, masterid: int):
        self.mode = "leader" if int(masterid) == self.id else "follower"
        self.calls.append(("master", masterid))
        self._log("command", f"Synchronisation: {self.mode} (master = acm{masterid})")

    set_leader = master

    def set_standalone(self):
        self.mode = "standalone"
        self.calls.append(("set_standalone",))
        self._log("command", "Synchronisation: standalone")

    def standalone(self):                     # the name used by take_trace.py / take_image.py -f
        self.mode = "standalone"
        self.calls.append(("standalone",))
        self._log("command", "Synchronisation: standalone")

    def use_external_clock(self):
        self.clock = "external"
        self.calls.append(("use_external_clock",))
        self._log("command", "Clock: external (CEX)")

    def use_internal_clock(self):
        self.clock = "internal"
        self.calls.append(("use_internal_clock",))
        self._log("command", "Clock: internal")

    # -- CCD operations ----------------------------------------------------------
    def flush_ccd(self):
        if self.mode != "standalone":
            raise FakeHangError(f"acm{self.id}: flush in {self.mode} mode -- the FlushCCD routine "
                                "would wait for an external start")
        self.calls.append(("flush_ccd",))
        self._routine("FlushCCD", 26)

    def erase_ccd(self, **kwargs):
        if self.mode != "standalone":
            raise FakeHangError(f"acm{self.id}: erase in {self.mode} mode -- the EraseCCD flush "
                                "would wait for an external start")
        self.calls.append(("erase_ccd", kwargs))
        self._routine("EraseCCD", 30)

    def epurge_ccd(self, **kwargs):
        self.calls.append(("epurge_ccd", kwargs))
        raise FakeHangError("epurge_ccd must never be called by the packet scripts")

    # -- acquisitions ------------------------------------------------------------
    def _geometry(self) -> str:
        s = self.state
        return f"{s['NROW']}x{s['NCOL']}x{s['NDCM']}_{s['NPBIN']}x{s['NSBIN']}"

    def _stamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _write_frames(self, stem: Path, filters: list[int], copy_channel: dict[int, str]) -> None:
        for ch in range(4):
            out = stem.parent / f"{stem.name}.RJ45ADCDATA{ch}"
            if ch in copy_channel:
                shutil.copyfile(copy_channel[ch], out)
            else:
                frame = [_START, _BUSY, 0xE0B403E8 | (filters[ch] << 16), _END]
                out.write_bytes(b"".join(struct.pack(">I", w) for w in frame))

    def _write_meta(self, stem: Path, filters: list[int], source: str | None) -> None:
        meta = {"FAKE": True, "FAKE_SOURCE": source, "ACMID": self.id, "binname": stem.name,
                "bindir": str(stem.parent), "acqcount": self._counter,
                "SEQNAME": self.sequencer.seqfile.name, "ROUTINE": self.state.get("subroutine"),
                "FILTER": sum(f << (2 * i) for i, f in enumerate(filters)),
                **{f"FILTER{i}": f for i, f in enumerate(filters)},
                **{f"ch{i}": f"CCD_{'ABCD'[i]}" for i in range(4)},
                "DATEini": _utc(), "DATEend": _utc(),
                **{k: v for k, v in self.state.items() if k not in ("acq_level", "level", "subroutine")}}
        stem.with_suffix(".meta").write_text(json.dumps(meta, indent=4, default=str))

    def take_trace(self, fname="trace", ch=0, **other_params):
        if self.mode != "standalone":
            raise FakeHangError(f"acm{self.id}: trace in {self.mode} mode -- the sequencer would "
                                "wait for an external start that never comes")
        for key, value in other_params.items():
            self[key] = value
        stem = self.acquire.directory / (f"{fname}_{self._geometry()}_ch{ch}_{self.id}_"
                                         f"{self._stamp()}_{self._counter}")
        filters = [0 if i == ch else 3 for i in range(4)]
        sources = self.examples.get("traces", {}).get(int(ch), [])
        source = None
        if sources:                                # cycle through the available repeats
            index = self._trace_index.get(int(ch), 0)
            source = sources[index % len(sources)]
            self._trace_index[int(ch)] = index + 1
        self._write_frames(stem, filters, {int(ch): source} if source else {})
        self._write_meta(stem, filters, source)
        self.acquire.level = filters
        self.acquire._filepath = stem
        self._counter += 1
        self.calls.append(("take_trace", ch))
        self._routine("AcquireImage", self._acquisition_seconds() + 1)
        self._log("acquire", f"Trace ch{ch} written: {stem.name}")

    def take_image(self, fname="img", **other_params):
        for key, value in other_params.items():
            self[key] = value
        levels = self.acquire.level
        if len(set(levels)) != 1 or levels[0] not in (1, 2):
            raise FakeHangError(f"acm{self.id}: image with acquisition levels {levels}; pass "
                                "acq_level=1 or 2 (after a trace the level is still the trace's)")
        level = levels[0]
        if self.mode == "standalone" and self.require_sync:
            raise FakeHangError(f"acm{self.id}: image in standalone mode; images must be taken in "
                                "leader/follower mode")
        stem = self.acquire.directory / (f"{fname}_{self._geometry()}_L{level}_{self.id}_"
                                         f"{self._stamp()}_{self._counter}")
        filters = [level] * 4
        self._write_frames(stem, filters, {})
        source = self.examples.get("image")
        self._write_meta(stem, filters, source)
        fits = stem.parent / f"avg_{stem.name}.fits.bz2"
        if source:
            shutil.copyfile(source, fits)
        else:
            fits.write_bytes(b"")                   # empty marker
        self.acquire._filepath = stem
        self._counter += 1
        self.calls.append(("take_image", level))
        self._routine("AcquireImage", self._acquisition_seconds())
        self._log("acquire", f"Image written: {stem.name}")


def example_sources(data_dir: str | Path, board: int) -> dict:
    """
    Real files already in ``data/`` that the fake copies, per board.

    Traces: all repeats of each channel of this board from the 2026-10-08 session (module 101)
    or, if that has none, from the 2026-10-06 run (module 102).
    Image: the 2026-10-06 100x1000x3000 L2 image of the same board (both boards were read
    together).
    Returns ``{}`` (placeholders) when the files are not there.
    """
    data_dir = Path(data_dir)
    traces = {}
    # the 2026-10-08 session (5 repeats, module 101 only), else the 2026-10-06 run (1 repeat)
    sessions = [data_dir / "2026-10-08-PA03_PQ04_traces",
                data_dir / "2026-10-06-PA03_PQ04_Installation_Check130K_1"]
    for ch in range(4):
        for session in sessions:
            found = sorted(session.glob(f"trace_*_ch{ch}_{board}_*.RJ45ADCDATA{ch}"))
            if found:
                traces[ch] = [str(f) for f in found]
                break
    image = sorted((data_dir / "2026-10-06-PA03_PQ04_Installation_Check130K_1").glob(
        f"avg_img_100x1000x3000_1x1_L2_{board}_*.fits.bz2"))
    out: dict = {}
    if traces:
        out["traces"] = traces
    if image:
        out["image"] = str(image[0])
    return out

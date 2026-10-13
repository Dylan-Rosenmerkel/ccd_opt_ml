"""
daq_control.py -- send a step's command text to the boards and wait for their receipts.

Two controllers with the same interface, ``submit(target, code)``:

``CdaqController``  the real one.  ``ccd_cdaq.add2queue(code, target, queue)`` puts the text
                    in the cdaq queue (exactly what ``cdaq qadd "<code>" <target>`` does).
                    The queue runner sends it to the client(s) of ``target`` when they are
                    free; each client runs it with ``exec`` and its own ``daq`` object.
                    ``target`` is a client ("acm101") or a pool ("acmpool": every board in
                    the pool runs the code at the same time).
``FakeController``  rehearsal.  Runs the same text on ``FakeDAQ`` objects, through a model of
                    the cdaq queue (below).

After submitting, :func:`wait_for_receipts` polls the receipt directory until every board
that was asked has written ``<step id>.acm<board>.json`` (see ``daq_commands.wrap_step``).
A missing receipt after the timeout means the board did not finish: the queue may be
paused, the client offline, or the acquisition hung.  The board state is then unknown and
an operator must look (``cdaq print queues``, the client log) before continuing.

The cdaq queue (``Soft_JHU/cdaq/src/ccd_cdaq/_cdaqqueue.py`` and ``_main.py``)
-----------------------------------------------------------------------------
A queue has a SIMPLE part (run once, in order) and a REPEAT part (run again and again).  The
queue runner takes the next simple command if there is one; otherwise it takes the first
repeat command and rotates the repeat list.  It runs ONE command at a time and waits until the
target client is back in standby.  The group keeps ``daq.flush_ccd()`` (target ``acmpool``) in
the repeat queue, so the CCDs are flushed continuously (about every 27 s) while nothing else
is queued.

Consequence for the packet script, which queues one step and waits for its receipt before it
queues the next: between two steps the simple queue is EMPTY, so the runner starts a repeat
flush, and the next step waits behind it (up to ~27 s).  This (1) adds flushes the packet did
not ask for, and (2) would flush while the boards are in leader/follower mode if that mode
were left between two steps.  Hence:

* :func:`repeat_queue_suspended` saves the repeat queue, clears it for the duration of a packet
  or scan, and puts it back afterwards (``[daq] repeat_queue = "suspend"``, the default);
* the image step is one queue entry from leader mode back to standalone mode
  (``daq_commands.synchronized_image_lines``), so a repeat flush can never run in between.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("daq_control")


class StepError(RuntimeError):
    """A board reported an error, or did not report back in time."""


# --------------------------------------------------------------------------- #
# The real queue                                                               #
# --------------------------------------------------------------------------- #

class CdaqController:
    """Queue the code for the real cdaq clients (needs the ``ccd_cdaq`` package)."""

    def __init__(self, queue: str = "default") -> None:
        import ccd_cdaq                      # imported here: only needed on the DAQ machine
        self._cdaq = ccd_cdaq
        self.queue = queue
        self.sent: list[tuple[str, str]] = []

    def submit(self, target: str, code: str) -> None:
        self._cdaq.add2queue(code, target, queue_name=self.queue, queue_type="simple")
        self.sent.append((target, code))

    # -- repeat queue ------------------------------------------------------------
    def read_repeat_queue(self, dump_file: str | Path, timeout_s: float = 30.0) -> list[dict]:
        """
        The current repeat queue, as a list of ``{"to": target, "cmd": text}``.

        ``ccd_cdaq.queue2file`` asks the cdaq main process to write the queue (simple and repeat)
        to ``dump_file`` (``CDAQQueue.dump_to_file``); this waits for the file.  The path must be
        absolute and writable by the cdaq main process (it runs on the DAQ machine).
        """
        dump_file = Path(dump_file).resolve()
        dump_file.parent.mkdir(parents=True, exist_ok=True)
        dump_file.unlink(missing_ok=True)
        self._cdaq.queue2file(str(dump_file), self.queue)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if dump_file.exists():
                try:
                    return list(json.loads(dump_file.read_text())["repeat"])
                except (json.JSONDecodeError, KeyError):
                    pass                            # still being written
            time.sleep(0.5)
        raise StepError(f"the cdaq main process did not write the queue to {dump_file} within "
                        f"{timeout_s:.0f} s; the repeat queue is unknown. Check `cdaq print queues` "
                        "and, to continue without it, clear it by hand: cdaq qdel -t repeat")

    def clear_repeat_queue(self) -> None:
        """``cdaq qdel -t repeat``: a repeat command that is running now still finishes."""
        self._cdaq.clearqueue(self.queue, "repeat")

    def restore_repeat_queue(self, commands: list[dict]) -> None:
        """Add the saved commands back, in their order (``cdaq qadd "<cmd>" <to> -t repeat``)."""
        for command in commands:
            self._cdaq.add2queue(command["cmd"], command["to"], queue_name=self.queue, queue_type="repeat")


# --------------------------------------------------------------------------- #
# Suspending the repeat queue during a packet or scan                          #
# --------------------------------------------------------------------------- #

def restore_instructions(commands: list[dict], queue: str) -> str:
    """The cdaq commands an operator types to put a saved repeat queue back by hand."""
    lines = [f'cdaq qadd "{c["cmd"]}" {c["to"]} -q {queue} -t repeat' for c in commands]
    return "\n".join(lines) or "(the repeat queue was empty)"


@contextmanager
def repeat_queue_suspended(controller, backup_file: str | Path, enabled: bool = True):
    """
    Save and clear the repeat queue on entry; put it back on a normal exit.

    Yields the saved commands.  After an ERROR the queue is NOT restored: the board state is
    then unknown (a board may still be in leader/follower mode, where the group's repeat flush
    would hang).  The saved commands and the ``cdaq qadd`` lines to restore them are logged
    and are in ``backup_file``.  ``enabled = False`` (``[daq] repeat_queue = "keep"``) does
    nothing: the repeat commands then run between the packet's steps.
    """
    if not enabled or not hasattr(controller, "read_repeat_queue"):
        yield []
        return
    backup_file = Path(backup_file)
    saved = controller.read_repeat_queue(backup_file)
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    backup_file.write_text(json.dumps({"queue": getattr(controller, "queue", ""), "repeat": saved,
                                       "restore": restore_instructions(saved, getattr(controller, "queue", "default"))},
                                      indent=2))
    if saved:
        controller.clear_repeat_queue()
        log.info("repeat queue suspended (%d command(s), saved in %s)", len(saved), backup_file)
    finished = False
    try:
        yield saved
        finished = True
    finally:
        if saved and finished:
            controller.restore_repeat_queue(saved)
            log.info("repeat queue restored (%d command(s))", len(saved))
        elif saved:
            log.warning("the repeat queue was NOT restored because the packet/scan stopped with an "
                        "error; check the boards first (they may still be in leader/follower mode), "
                        "then restore it by hand:\n%s",
                        restore_instructions(saved, getattr(controller, "queue", "default")))


# --------------------------------------------------------------------------- #
# The fake queue                                                               #
# --------------------------------------------------------------------------- #

class FakeController:
    """
    Run the code on ``FakeDAQ`` objects through a model of the cdaq queue.

    ``targets`` maps a target name (client or pool) to board ids.  The model follows the real
    runner (``ccd_cdaq/_main.py``):

    * ``submit`` puts the text on the SIMPLE queue.  If the REPEAT queue is not empty, the
      runner was idle before this submission (the packet script only submits after the previous
      step finished), so it had started the next repeat command: that command runs FIRST, then
      the submitted one.  (On the real DAQ the submitted step waits up to ~27 s behind it.)
    * :meth:`idle` lets the runner run one repeat command, as it does while the script waits
      (e.g. for the image builder's FITS file at the end of a packet).
    * a pool target runs on all its boards at the same time (one thread per board).
    * a repeat command that raises (e.g. a flush in leader/follower mode) leaves that board
      HUNG, as the real client would be.  The next submission for it raises :class:`StepError`
      at once: on the real DAQ that step would time out after its full timeout instead.

    ``repeat`` is the initial repeat queue, e.g. ``[{"to": "acmpool", "cmd": "daq.flush_ccd()"}]``
    (what the group keeps there).  ``state_file``: if given, the queue, client and pool status,
    the log and the "Publishing" lines are written there as JSON after every change, for
    ``fake_cdaq_monitor.py`` (a terminal display like the real tmux panes).
    ``echo_publishing`` prints those lines too, as ``ccd_cdaq`` does in the operator's shell.
    """

    poll_interval_s = 0.05                    # the fake answers at once: poll the receipts quickly

    def __init__(self, fakes: dict, targets: dict[str, list[int]], queue: str = "default",
                 repeat: list[dict] | None = None, state_file: str | Path | None = None,
                 pools: dict[str, list[str]] | None = None, echo_publishing: bool = False) -> None:
        self.fakes = fakes
        self.targets = targets
        self.queue = queue
        self.sent: list[tuple[str, str]] = []
        self.simple: deque = deque()
        self.repeat: deque = deque(repeat or [])
        self.hung: dict[int, str] = {}                 # board -> why it hangs
        self.state_file = Path(state_file) if state_file else None
        self.echo_publishing = echo_publishing          # print the "Publishing" lines, like ccd_cdaq does
        self._lock = threading.RLock()
        self._current: dict | None = None
        self._ctime: float | None = None
        self.log: deque = deque(maxlen=400)
        self.publishing: deque = deque(maxlen=200)
        self.last_warning: dict | None = None
        if pools is None:                               # a target with several boards is a pool
            pools = {t: [f"acm{b}" for b in boards] for t, boards in targets.items() if len(boards) > 1}
        self.pools = pools
        self.clients = {f"acm{b}": {"status": 1, "current": None, "ctime": time.time(),
                                    "pool": next((p for p, members in pools.items() if f"acm{b}" in members), None)}
                        for b in fakes}
        for board, fake in fakes.items():
            fake.log_callback = self._board_log
        self._write_state()

    # -- the queue interface (what CdaqController offers too) ---------------------------
    def submit(self, target: str, code: str) -> None:
        self.sent.append((target, code))
        command = {"to": target, "cmd": code}
        self._publish(f"add_simple({command!r})")
        with self._lock:
            self.simple.append(command)
            self._write_state()
        if self.repeat and not self.hung:
            self._run_next_repeat()                     # the runner had started a repeat command
        stuck = sorted(b for b in self.targets[target] if b in self.hung)
        if stuck:
            reason = "; ".join(f"acm{b}: {self.hung[b]}" for b in stuck)
            self._log(f"CDAQQueue.{self.queue}.Runner", "INFO",
                      f"Client \"{target}\" is busy, waiting...")
            raise StepError(f"board(s) {stuck} are hung ({reason}). On the real DAQ this step would "
                            "wait for its full timeout and then stop with 'no receipt'.")
        with self._lock:
            self.simple.popleft()
        self._run(command, "simple")

    def idle(self, n_commands: int = 1) -> None:
        """The script waits and the simple queue is empty: the runner runs repeat commands."""
        for _ in range(n_commands):
            if not self.repeat:
                return
            self._run_next_repeat()

    def read_repeat_queue(self, dump_file: str | Path, timeout_s: float = 30.0) -> list[dict]:
        self._publish(f"dump_to_file({str(dump_file)!r})")
        with self._lock:
            return list(self.repeat)

    def clear_repeat_queue(self) -> None:
        self._publish("clear_repeat()")
        with self._lock:
            self.repeat.clear()
            self._log(f"CDAQQueue.{self.queue}", "INFO", "Cleared repeat queue.")

    def restore_repeat_queue(self, commands: list[dict]) -> None:
        for command in commands:
            self._publish(f"add_repeat({dict(command)!r})")
            with self._lock:
                self.repeat.append(dict(command))
                self._log(f"CDAQQueue.{self.queue}", "INFO", f"Added {dict(command)} to repeat queue.")

    # -- internals ------------------------------------------------------------------------
    def _run_next_repeat(self) -> None:
        with self._lock:
            command = self.repeat[0]
            self.repeat.rotate(-1)                      # as CDAQQueue.advance('repeat')
        self._run(command, "repeat")

    def _run(self, command: dict, queue_type: str) -> None:
        """Run one command on every board of its target at the same time."""
        boards = self.targets[command["to"]]
        with self._lock:
            self._current, self._ctime = command, time.time()
            for b in boards:
                self.clients[f"acm{b}"].update(status=2, current=_short(command["cmd"]), ctime=time.time())
            self._write_state()

        errors: dict[int, str] = {}

        def run_on(board: int) -> None:
            try:
                exec(command["cmd"], {"daq": self.fakes[board]})   # same text a cdaq client would exec
            except Exception as err:                     # noqa: BLE001 - recorded like the client log
                errors[board] = f"{type(err).__name__}: {err}"

        threads = [threading.Thread(target=run_on, args=(b,)) for b in boards]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        with self._lock:
            for board, message in errors.items():
                self._log(f"acm.{board}.client", "ERROR", f"[EXEC] Could not execute: {message}")
                if queue_type == "repeat":
                    self.hung[board] = f"repeat command {command['cmd']!r} failed: {message}"
            for b in boards:
                busy = b in self.hung
                self.clients[f"acm{b}"].update(status=2 if busy else 1,
                                               current=_short(command["cmd"]) if busy else None,
                                               ctime=time.time())
            if not any(b in self.hung for b in boards):  # a hung command stays "currently running"
                self._current, self._ctime = None, None
            self._write_state()

    def _board_log(self, who: str, level: str, message: str) -> None:
        with self._lock:
            self._log(who, level, message)

    def _log(self, who: str, level: str, message: str) -> None:
        entry = {"time": time.time(), "who": who, "level": level, "msg": message}
        self.log.append(entry)
        if level not in ("INFO", "DEBUG"):
            self.last_warning = entry
        self._write_state()

    def _publish(self, queue_call: str) -> None:
        """The line the real ``ccd_cdaq.main_sender`` prints for every queue command."""
        text = f"send2queue({self.queue!r}, {queue_call!r})"
        line = f'Publishing: "{text}" to topic "CDAQ/cdaq_main/input"'
        with self._lock:
            self.publishing.append({"time": time.time(), "text": line})
            self._write_state()
        if self.echo_publishing:
            print(line if len(line) < 300 else line[:296] + " ...", flush=True)

    def state(self) -> dict:
        """Everything the monitor shows (JSON-serialisable)."""
        pool_status = {}
        for pool, members in self.pools.items():
            statuses = [self.clients[m]["status"] for m in members if m in self.clients]
            pool_status[pool] = {"status": 2 if any(s == 2 for s in statuses) else 1, "clients": members}
        return {"time": time.time(),
                "queues": {self.queue: {"current": self._current and {"to": self._current["to"],
                                                                      "cmd": _short(self._current["cmd"])},
                                        "ctime": self._ctime,
                                        "simple": [{"to": c["to"], "cmd": _short(c["cmd"])} for c in self.simple],
                                        "repeat": [{"to": c["to"], "cmd": _short(c["cmd"])} for c in self.repeat]}},
                "clients": self.clients, "pools": pool_status,
                "log": list(self.log), "last_warning": self.last_warning,
                "publishing": list(self.publishing), "hung": {f"acm{b}": r for b, r in self.hung.items()}}

    def _write_state(self) -> None:
        if not self.state_file:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state(), default=str))
        temporary.replace(self.state_file)              # atomic: the monitor never reads half a file


def _short(code: str) -> str:
    """A queued text as the monitor shows it: the step id instead of the long receipt frame."""
    if code.startswith("import json as _json"):
        try:
            step = code.split("'step': ", 1)[1].split(",", 1)[0].strip("'\"")
            body = [line.strip() for line in code.splitlines()[3:] if line.startswith("    ")
                    and not line.strip().startswith(("_r[", "_time.sleep", "try:", "except", "finally:"))]
            first = body[0] if body else ""
            return f"[{step}] {first[:70]}" + (" ..." if len(body) > 1 else "")
        except IndexError:
            pass
    return code


# --------------------------------------------------------------------------- #
# Receipts                                                                     #
# --------------------------------------------------------------------------- #

def receipt_path(receipt_dir: str | Path, step_id: str, board: int) -> Path:
    return Path(receipt_dir) / f"{step_id}.acm{board}.json"


def wait_for_receipts(receipt_dir: str | Path, step_id: str, boards: list[int],
                      timeout_s: float, poll_s: float = 1.0) -> dict[int, dict]:
    """
    Wait until every board in ``boards`` has written its receipt for ``step_id``.

    Returns ``{board: receipt}``.  Raises :class:`StepError` if a receipt says the step failed
    (the board's traceback is in the message) or if not all receipts arrived in ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    receipts: dict[int, dict] = {}
    while True:
        for board in boards:
            path = receipt_path(receipt_dir, step_id, board)
            if board not in receipts and path.exists():
                try:
                    receipts[board] = json.loads(path.read_text())
                except json.JSONDecodeError:
                    pass                            # still being written; read it next round
        if len(receipts) == len(boards):
            break
        if time.monotonic() >= deadline:
            missing = sorted(set(boards) - set(receipts))
            raise StepError(
                f"step {step_id}: no receipt from board(s) {missing} after {timeout_s:.0f} s. "
                "The board state is UNKNOWN: check `cdaq print queues` and the client log "
                "before doing anything else.")
        time.sleep(poll_s)

    failed = {b: r.get("error", "") for b, r in receipts.items() if not r.get("ok")}
    if failed:
        details = "\n".join(f"--- acm{b} ---\n{err}" for b, err in failed.items())
        raise StepError(f"step {step_id} failed on board(s) {sorted(failed)}:\n{details}")
    return receipts

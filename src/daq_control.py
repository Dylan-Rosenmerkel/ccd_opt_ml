"""
daq_control.py -- send a step's command text to the boards and wait for their receipts.

Two controllers with the same interface, ``submit(target, code)``:

``CdaqController``  the real one.  ``ccd_cdaq.add2queue(code, target, queue)`` puts the text
                    in the cdaq queue (exactly what ``cdaq qadd "<code>" <target>`` does).
                    The queue runner sends it to the client(s) of ``target`` when they are
                    free; each client runs it with ``exec`` and its own ``daq`` object.
                    ``target`` is a client ("acm101") or a pool ("acmpool": every board in
                    the pool runs the code at the same time).
``FakeController``  rehearsal.  Runs the same text immediately on ``FakeDAQ`` objects.

After submitting, :func:`wait_for_receipts` polls the receipt directory until every board
that was asked has written ``<step id>.acm<board>.json`` (see ``daq_commands.wrap_step``).
A missing receipt after the timeout means the board did not finish: the queue may be
paused, the client offline, or the acquisition hung.  The board state is then unknown and
an operator must look (``cdaq print queues``, the client log) before continuing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class StepError(RuntimeError):
    """A board reported an error, or did not report back in time."""


class CdaqController:
    """Queue the code for the real cdaq clients (needs the ``ccd_cdaq`` package)."""

    def __init__(self, queue: str = "default") -> None:
        from ccd_cdaq import add2queue      # imported here: only needed on the DAQ machine
        self._add2queue = add2queue
        self.queue = queue
        self.sent: list[tuple[str, str]] = []

    def submit(self, target: str, code: str) -> None:
        self._add2queue(code, target, queue_name=self.queue, queue_type="simple")
        self.sent.append((target, code))


class FakeController:
    """Run the code on ``FakeDAQ`` objects; ``targets`` maps a target name to board ids."""

    def __init__(self, fakes: dict, targets: dict[str, list[int]]) -> None:
        self.fakes = fakes
        self.targets = targets
        self.sent: list[tuple[str, str]] = []

    def submit(self, target: str, code: str) -> None:
        self.sent.append((target, code))
        for board in self.targets[target]:
            exec(code, {"daq": self.fakes[board]})       # same text a cdaq client would exec


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

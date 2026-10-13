"""
fake_cdaq_monitor.py -- watch a FAKE-board rehearsal the way the real tmux session shows the DAQ.

The real DAQ machine runs a tmux session with panes like

    +----------------------+-------------------------------+-------------------------+
    | cdaq print queues    | cdaq print logs               | cdaq print clients      |
    | (queue: current,     | (board and cdaq log lines,    | (client status, current |
    |  simple, repeat)     |  last warning at the bottom)  |  command, pools)        |
    +----------------------+-------------------------------+-------------------------+
    | operator shell: cdaq qadd ..., take_image.py ...  ("Publishing: ..." lines)    |
    +--------------------------------------------------------------------------------+

``FakeController`` (``daq_control.py``) writes its state -- the queue, the clients and pool,
the log and the "Publishing" lines -- to a JSON file after every change
(``<work_dir>/fake_cdaq_state.json`` when created by ``take_packet.make_controller``).  This
script draws that file in the same layout and formats as the group's monitoring commands
(``ccd_cdaq/monitoring.py``: ``print_queues``, ``print_clients``, ``print_logs_from_mqtt``).

It is ONLY a display of the fake.  Nothing here talks to MQTT or the real cdaq: on the DAQ
machine use the real ``cdaq print ...`` commands.

Usage
-----
    # one terminal with all panes (no tmux needed), refreshed every 0.5 s
    python fake_cdaq_monitor.py --state /tmp/rehearsal/fake_cdaq_state.json

    # in another terminal, a slowed-down rehearsal (5 % of real time) to watch
    python take_packet.py --params ../configs/packet_example.toml --backend fake \\
           --work-dir /tmp/rehearsal --fake-time-scale 0.05

    # or a tmux session laid out like the real one (needs tmux); --run starts the rehearsal
    # in the bottom pane
    python fake_cdaq_monitor.py --state /tmp/rehearsal/fake_cdaq_state.json --tmux \\
           --run "python take_packet.py --params ../configs/packet_example.toml --backend fake --work-dir /tmp/rehearsal --fake-time-scale 0.05"

    # one pane only (what each tmux pane runs), or a single snapshot
    python fake_cdaq_monitor.py --state FILE --pane queues|logs|clients|commands
    python fake_cdaq_monitor.py --state FILE --once
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

COLORS = {"INFO": "\033[32m", "WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[1;31m"}
RESET = "\033[0m"
PANES = ("queues", "logs", "clients", "commands")


def pretty_print_duration(total_seconds: float) -> str:
    """As ``ccd_cdaq.monitoring.pretty_print_duration``: 1h 10min 35.2sec."""
    hours, minutes, seconds = total_seconds // 3600, (total_seconds % 3600) // 60, total_seconds % 60
    parts = []
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0:
        parts.append(f"{int(minutes)}min")
    if seconds > 0 or not parts:
        parts.append(f"{seconds:.1f}sec")
    return " ".join(parts)


def _stamp(t: float, ms: bool = False) -> str:
    text = datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")
    return text[:-3] if ms else text[:19]


# --------------------------------------------------------------------------- #
# The panes: each returns a list of (text, colour) lines                       #
# --------------------------------------------------------------------------- #

def pane_queues(state: dict, now: float) -> list[tuple[str, str]]:
    """``cdaq print queues``."""
    lines = [(f"printqueue execution time: {_stamp(now)}   [FAKE]", ""), ("", "")]
    for index, (name, queue) in enumerate(state["queues"].items()):
        lines.append((f"[{index}] {name}:", ""))
        current = f"Currently running: {queue['current']}"
        if queue.get("ctime"):
            current += f" (elapsed: {pretty_print_duration(now - queue['ctime'])})"
        lines.append((current, ""))
        for label, items in (("On simple queue:", queue["simple"]), ("On repeat queue:", queue["repeat"])):
            lines.append((label, ""))
            if not items:
                lines.append(("Empty!", ""))
            for i, item in enumerate(items[:20]):
                lines.append((f"[{i}] {item}", ""))
            if len(items) > 20:
                lines.append((f"... {len(items) - 20} more command(s) ...", ""))
        lines.append(("", ""))
    return lines


def pane_clients(state: dict, now: float) -> list[tuple[str, str]]:
    """``cdaq print clients``."""
    names = {0: "Uninitialized", 1: "Standby", 2: "Busy", -1: "Shutdown"}
    lines = [(f"Client status on {_stamp(now)}   [FAKE]", ""), ("Clients:", "")]
    clients = sorted(state["clients"].items(), key=lambda item: item[1].get("ctime", 0), reverse=True)
    for i, (name, c) in enumerate(clients):
        text = f"[{i}] {name}: {names.get(c['status'], 'Unknown')}"
        if c.get("current"):
            text += f"; Current: {c['current']}"
        if c.get("pool"):
            text += f"; Pool: {c['pool']}"
        text += f" ({pretty_print_duration(now - c['ctime'])})"
        lines.append((text, "ERROR" if name in state.get("hung", {}) else ""))
    lines += [("", ""), ("Pools:", "")]
    for i, (name, p) in enumerate(state["pools"].items()):
        lines.append((f"[{i}] {name}: {names.get(p['status'], 'Unknown')}", ""))
        lines.append((f"    {len(p['clients'])} clients: {', '.join(p['clients'])}", ""))
    for name, reason in state.get("hung", {}).items():
        lines += [("", ""), (f"{name} HUNG: {reason}", "ERROR")]
    return lines


def pane_logs(state: dict, now: float, height: int) -> list[tuple[str, str]]:
    """``cdaq print logs``: the newest lines, and the last warning/error at the bottom."""
    lines = [(f"{_stamp(e['time'], ms=True)} - {e['who']: <15} - {e['level'].upper(): <8} - {e['msg']}",
              e["level"].upper()) for e in state["log"]]
    lines = lines[-max(1, height - 2):]
    warning = state.get("last_warning")
    if warning and now - warning["time"] < 600:
        footer = (f"Last {warning['level'].upper()} - {datetime.fromtimestamp(warning['time']).strftime('%H:%M:%S')}"
                  f" - {warning['who']: <15} - {warning['msg']}", warning["level"].upper())
    else:
        footer = ("No warnings or errors in the last 10 minutes!", "")
    return lines + [("", ""), footer]


def pane_commands(state: dict, now: float, height: int, width: int = 200) -> list[tuple[str, str]]:
    """
    The operator shell: what every queue command published (as ccd_cdaq.main_sender prints it).
    The packet steps are long (the receipt frame), so each line is cut to two screen lines.
    """
    limit = max(40, 2 * width - 4)
    lines = [((p["text"] if len(p["text"]) <= limit else p["text"][:limit - 4] + " ..."), "")
             for p in state["publishing"]]
    return lines[-max(1, height):]


# --------------------------------------------------------------------------- #
# Drawing                                                                      #
# --------------------------------------------------------------------------- #

def _fit(lines: list[tuple[str, str]], width: int, height: int, wrap: bool, tail: bool = False) -> list[str]:
    """Wrap or cut each line to ``width``, keep ``height`` lines, pad, and colour."""
    out: list[tuple[str, str]] = []
    for text, colour in lines:
        pieces = textwrap.wrap(text, width, break_long_words=True, break_on_hyphens=False) if (wrap and text) else [text[:width]]
        out += [(piece, colour) for piece in (pieces or [""])]
    out = out[-height:] if tail else out[:height]
    out += [("", "")] * (height - len(out))
    return [(COLORS.get(colour, "") + text.ljust(width) + (RESET if colour in COLORS else "")) for text, colour in out]


def draw_all(state: dict, now: float, width: int, height: int) -> str:
    """All four panes in the layout of the real tmux session."""
    bottom_h = max(6, height // 4)
    top_h = height - bottom_h - 1
    w1 = int(width * 0.30)
    w3 = int(width * 0.30)
    w2 = width - w1 - w3 - 2
    left = _fit(pane_queues(state, now), w1, top_h, wrap=True)
    middle = _fit(pane_logs(state, now, top_h), w2, top_h, wrap=True, tail=True)
    right = _fit(pane_clients(state, now), w3, top_h, wrap=True)
    rows = [f"{a}│{b}│{c}" for a, b, c in zip(left, middle, right)]
    rows.append("\033[32m" + "─" * width + RESET)
    rows += _fit(pane_commands(state, now, bottom_h, width), width, bottom_h, wrap=True, tail=True)
    return "\n".join(rows)


def draw_pane(state: dict, now: float, pane: str, width: int, height: int) -> str:
    if pane == "queues":
        lines = _fit(pane_queues(state, now), width, height, wrap=True)
    elif pane == "clients":
        lines = _fit(pane_clients(state, now), width, height, wrap=True)
    elif pane == "logs":
        lines = _fit(pane_logs(state, now, height), width, height, wrap=True, tail=True)
    else:
        lines = _fit(pane_commands(state, now, height, width), width, height, wrap=True, tail=True)
    return "\n".join(lines)


def read_state(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def render(path: Path, pane: str, width: int, height: int) -> str:
    state = read_state(path)
    if state is None:
        return f"waiting for {path} (start a rehearsal with --backend fake) ..."
    now = time.time()
    return draw_all(state, now, width, height) if pane == "all" else draw_pane(state, now, pane, width, height)


def start_tmux(state: Path, run: str | None, session: str) -> None:
    """A tmux session laid out like the real one: three panes on top, the shell below."""
    if not shutil.which("tmux"):
        sys.exit("tmux is not installed here (macOS: brew install tmux). Use the single-terminal "
                 "view instead: python fake_cdaq_monitor.py --state FILE")
    me = f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} --state {shlex.quote(str(state))}"
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)

    def tmux(*arguments: str) -> str:
        """Run a tmux command that creates a pane and return the new pane's id (%N)."""
        out = subprocess.run(["tmux", *arguments, "-P", "-F", "#{pane_id}"], check=True,
                             capture_output=True, text=True)
        return out.stdout.strip()

    # pane ids, not indices: they do not depend on the user's pane-base-index setting
    shell = tmux("new-session", "-d", "-s", session, "-x", "240", "-y", "60")
    queues = tmux("split-window", "-v", "-b", "-p", "75", "-t", shell, f"{me} --pane queues")
    logs = tmux("split-window", "-h", "-p", "70", "-t", queues, f"{me} --pane logs")
    tmux("split-window", "-h", "-p", "40", "-t", logs, f"{me} --pane clients")
    if run:
        subprocess.run(["tmux", "send-keys", "-t", shell, run, "Enter"], check=True)
    subprocess.run(["tmux", "select-pane", "-t", shell], check=True)
    if sys.stdout.isatty():
        subprocess.run(["tmux", "attach", "-t", session])
    else:
        print(f"tmux session '{session}' started: tmux attach -t {session}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--state", required=True, help="the fake controller's state file (fake_cdaq_state.json)")
    parser.add_argument("--pane", choices=("all",) + PANES, default="all")
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--interval", type=float, default=0.5, help="refresh interval (s)")
    parser.add_argument("--width", type=int, help="override the terminal width")
    parser.add_argument("--height", type=int, help="override the terminal height")
    parser.add_argument("--tmux", action="store_true", help="open a tmux session laid out like the real one")
    parser.add_argument("--run", help="with --tmux: command to run in the bottom (shell) pane")
    parser.add_argument("--session", default="fake-cdaq", help="tmux session name")
    args = parser.parse_args()
    state = Path(args.state)

    if args.tmux:
        start_tmux(state, args.run, args.session)
        return

    def size() -> tuple[int, int]:
        columns, lines = shutil.get_terminal_size((200, 50))
        return args.width or columns, (args.height or lines) - 1

    if args.once:
        print(render(state, args.pane, *size()))
        return
    try:
        while True:
            frame = render(state, args.pane, *size())
            sys.stdout.write("\033[H\033[2J" + frame)       # clear, then draw (like refresh_screen)
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()

"""
check_trace_files.py -- are the traces of a packet phase complete and undamaged?  (Used by
daq_scripts/manual_packet.sh on the DAQ machine while the packet runs.)

For every expected trace (board, channel, set) it looks in the run directory for
``<prefix><set>_*_ch<channel>_<board>_*.meta`` and its ``.RJ45ADCDATA<channel>`` and reports the trace as
damaged if a file is missing, the .meta has no DATEend (the acquisition did not finish), or the decoded trace
fails ``DecodedTrace.quality_problems`` (lost UDP packets, uneven windows, inline sums that disagree with the raw
samples: the same check process_packet.py uses to drop a trace).  If several files match (a retake under the same
name), the newest counts.

Output: one line per damaged trace, ``<board> <channel> <set> <reason>``; nothing if all are fine.  The exit code is
0 in both cases (2 on a usage error), so the caller counts the lines.

    python src/check_trace_files.py --dir RUN_DIR --prefix L2_05_ref --sets 1 2 3 --channels 0 1 2 3 --boards 101
    python src/check_trace_files.py --dir RUN_DIR --prefix L2_05_new --expect 101:2:6 101:0:7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_trace import decode_trace  # noqa: E402


def trace_problem(run_dir: Path, prefix: str, board: int, channel: int, set_no: int) -> str | None:
    """None if the trace exists, finished and decodes cleanly; else a short reason."""
    metas = sorted(run_dir.glob(f"{prefix}{set_no}_*_ch{channel}_{board}_*.meta"), key=lambda p: p.stat().st_mtime)
    if not metas:
        return "missing"
    meta_path = metas[-1]
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return "unreadable .meta"
    if not meta.get("DATEend"):
        return "not finished (no DATEend)"
    data = meta_path.with_suffix(f".RJ45ADCDATA{channel}")
    if not data.exists():
        return "no data file"
    try:
        problems = decode_trace(data).quality_problems()
    except Exception as err:                    # noqa: BLE001 - an undecodable file is damaged
        return f"decode failed: {type(err).__name__}"
    return "; ".join(problems) if problems else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", required=True, help="run directory (where cdaq writes the traces)")
    parser.add_argument("--prefix", required=True, help="trace name prefix without the set number, e.g. L2_05_new")
    parser.add_argument("--sets", type=int, nargs="*", default=[])
    parser.add_argument("--channels", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--boards", type=int, nargs="*", default=[101])
    parser.add_argument("--expect", nargs="*", default=[], help="explicit board:channel:set triples (retakes)")
    args = parser.parse_args()
    run_dir = Path(args.dir)
    if not run_dir.is_dir():
        print(f"no such run directory: {run_dir}", file=sys.stderr)
        return 2
    expected = [(b, c, s) for s in args.sets for b in args.boards for c in args.channels]
    expected += [tuple(int(x) for x in e.split(":")) for e in args.expect]
    for board, channel, set_no in expected:
        problem = trace_problem(run_dir, args.prefix, board, channel, set_no)
        if problem:
            print(f"{board} {channel} {set_no} {problem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

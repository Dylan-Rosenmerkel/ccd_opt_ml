"""
drift_correct.py -- the command-line wrapper of src/image_drift.py (the along-row drift: detect, cut, correct).

    python lvl1/drift_correct.py <avg image(s)> --out <dir> [--cut 200] [--channels 0 1 3] [--report report.json]

process_packet.py applies the same correction to every image (configs/processing.toml [drift]).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from image_drift import main  # noqa: E402

if __name__ == "__main__":
    main()

"""
ranking_tables.py -- the "what matters most" tables of the level-1 story: for the noise, the dark charge and the gain,
the five settings that matter most in each view (measured effect, correlations, and what each model relies on or
predicts), and the five trace features the trace models use most.  Writes a LaTeX file for the story document and a
Markdown copy.

Sources: story.json (lvl1/story_figures.py) and model_diagnostics.json (lvl1/model_diagnostics_v2.py).

    .venv/bin/python lvl1/ranking_tables.py --story docs/overleaf_level1_story_2026-11-10/story.json \\
        --diag processed_v2_trainD/lvl1_results/model_diagnostics/model_diagnostics.json \\
        --out docs/overleaf_level1_story_2026-11-10/tables_rankings.tex
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

T = ["sigma_e", "lambda", "gain"]
NICE = {"sigma_e": "noise $\\sigma_e$", "lambda": "dark charge $\\lambda$", "gain": "gain"}
NICE_MD = {"sigma_e": "noise σe", "lambda": "dark charge λ", "gain": "gain"}
# a knob is named after the first pin of its group; these groups moved together in v2
PRETTY = {"V1_B_L": "V lows", "H1_B_L": "H lows", "TG_B_H": "TG", "SW_1_L": "SW\\_L", "RG_1_L": "RG\\_L",
          "RG_1_H": "RG\\_H", "VDD1": "VDD", "VR1": "VR", "delay_Ha": "delay\\_H", "delayInt": "delayInt",
          "delayRGH": "delayRGH", "VSUB": "VSUB", "V1_B_H": "V highs"}


def pretty(name: str, md: bool = False) -> str:
    base = re.sub(r"^theta\.", "", name).split(" (")[0]
    p = PRETTY.get(base, base.replace("_", "\\_"))
    return p.replace("\\_", "_") if md else p


def feat(name: str, md: bool = False) -> str:
    ref = name.startswith("ref: ")
    n = re.sub(r"^(own|ref): ", "", name)
    n = n.split(": ", 1)[-1] if ": " in n else n             # drop the feature family ("noise:", "edge:", ...)
    n = n if len(n) <= 32 else n[:30] + "…"
    n = ("ref.: " if ref else "") + n
    return n if md else n.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("…", "\\dots{}")


def not_learned(d: dict) -> bool:
    """A GP whose length scales never left their start value (1): each group's relevance is then exactly
    sqrt(number of pins in the group)."""
    return all(abs(v ** 2 - round(v ** 2)) < 0.02 and round(v ** 2) >= 1 for v in d.values())


def top5(d: dict, signed: bool, fmt: str, md: bool, norm: bool = False):
    if norm and not_learned(d):
        return ["not learned: the GP fitted this label as noise (all length scales at their start value)", "", "", "", ""]
    items = sorted(d.items(), key=lambda kv: -abs(kv[1]))[:5]
    mx = max(abs(v) for _, v in items) if norm and items else 1.0
    out = []
    for k, v in items:
        v = v / mx if norm else v
        s = (f"{v:+{fmt}}" if signed else f"{v:{fmt}}").replace("-", "−" if md else "$-$")
        out.append(f"{pretty(k, md)} {s}")
    return out + [""] * (5 - len(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--story", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = json.loads(Path(args.story).read_text())
    m = json.loads(Path(args.diag).read_text())
    tex, md = [], ["# What matters most: top five per view", ""]
    for t in T:
        views = [
            ("Measured effect on a change (\\%/SD, all settings fitted together)", "Measured effect on a change (%/SD, all settings together)",
             s["effects_per_setting"][t]["measured (images)"], True, ".1f", False),
            ("Correlation: setting change vs label change", "Correlation: setting change vs label change",
             s["corr_changes"] and {k: v[t] for k, v in s["corr_changes"].items()}, True, ".2f", False),
            ("Correlation: setting vs label (levels)", "Correlation: setting vs label (levels)",
             {k: v[t] for k, v in s["corr_levels"].items()}, True, ".2f", False),
            ("Settings GP (absolute): relevance (largest = 1)", "Settings GP (absolute): relevance (largest = 1)",
             m["settings_gp"][t]["relevance_top"], False, ".2f", True),
            ("Settings change GP: relevance (largest = 1)", "Settings change GP: relevance (largest = 1)",
             m["settings_change_gp"][t]["relevance_top"], False, ".2f", True),
            ("Settings change GP: predicted effect (\\%/SD)", "Settings change GP: predicted effect (%/SD)",
             s["effects_per_setting"][t]["settings change GP"], True, ".1f", False),
            ("Trace combined trees: predicted effect (\\%/SD)", "Trace combined trees: predicted effect (%/SD)",
             s["effects_per_setting"][t]["trace combined trees"], True, ".1f", False),
        ]
        tex += ["\\begin{table}[H]\\centering\\small",
                f"\\caption{{What matters most for the {NICE[t]}: the five largest settings in each view "
                "(v2; changes = two packets of one amplifier; new amplifier for the models). "
                "V lows / H lows: the vertical / horizontal clock lows, which moved together in v2; "
                "1 SD of a setting = its standard deviation over the v2 packets.}",
                "\\begin{tabular}{p{5.2cm}lllll}\\toprule",
                "view & 1st & 2nd & 3rd & 4th & 5th\\\\\\midrule"]
        md += [f"## {NICE_MD[t]}", "", "| view | 1st | 2nd | 3rd | 4th | 5th |", "|---|---|---|---|---|---|"]
        for lt, lm, d, signed, fmt, norm in views:
            cells = top5(d, signed, fmt, False, norm)
            tex.append(lt + " & " + (f"\\multicolumn{{5}}{{l}}{{\\textit{{{cells[0]}}}}}" if cells[1] == "" and cells[0].startswith("not")
                                     else " & ".join(cells)) + "\\\\")
            md.append(f"| {lm} | " + " | ".join(top5(d, signed, fmt, True, norm)) + " |")
        tex += ["\\bottomrule\\end{tabular}\\end{table}", ""]
        md.append("")
    # trace features
    tex += ["\\begin{table}[H]\\centering\\small",
            "\\caption{The trace features the trace models use most (permutation importance; ``ref.:'' = the "
            "reference trace taken at nominal).}",
            "\\begin{tabular}{p{3.6cm}p{2.3cm}p{2.3cm}p{2.3cm}p{2.3cm}p{2.3cm}}\\toprule",
            "model, label & 1st & 2nd & 3rd & 4th & 5th\\\\\\midrule"]
    md += ["## The trace features the trace models use most", "", "| model, label | 1st | 2nd | 3rd | 4th | 5th |",
           "|---|---|---|---|---|---|"]
    for key, name in (("trees", "trace trees (absolute)"), ("combined_trees", "trace combined trees (change)")):
        for t in T:
            f = m[key][t]["top"][:5]
            tex.append(f"{name}, {NICE[t]} & " + " & ".join(feat(x) for x in f) + "\\\\")
            md.append(f"| {name}, {NICE_MD[t]} | " + " | ".join(feat(x, True) for x in f) + " |")
    tex += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    Path(args.out).write_text("\n".join(tex) + "\n")
    Path(args.out).with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# run_fleet_tests.sh -- the tests of the ML plan (docs/overleaf_ml_plan_2026-11-10/ml_plan.tex) on the v2 data.
# Run from the repository root, with the v2 processed folder (dataset.csv) at $DATA:
#   DATA=processed_v2 bash lvl1/run_fleet_tests.sh
# Writes $OUT/<run>/{fewshot,bo,strips,residual}.{md,json,png}; about 5-20 min in all on a laptop.
# Send back the $OUT folder (the .md files are enough to read the results).
set -euo pipefail
PY="${PY:-.venv/bin/python}"
DATA="${DATA:-processed_v2}"
PLAN="${PLAN:-lvl1/packet_plan_v2/plan.json}"
OUT="${OUT:-$DATA/lvl1_results/fleet_tests_$(date +%Y-%m-%d)}"
COMMON=(--data "$DATA" --plan "$PLAN" --sessions D E)

# 1. the main run: board 101 ch0/1/3 (ch2's image fit is poor), drift removed with each amplifier's nominal packets
$PY lvl1/fleet_tests.py all "${COMMON[@]}" --channels 0 1 3 --drift --out "$OUT/main"
# 2. the same without the drift correction (the left-out amplifier's nominal packets not used at all)
$PY lvl1/fleet_tests.py all "${COMMON[@]}" --channels 0 1 3 --out "$OUT/no_drift"
# 3. with ch2: four amplifiers, one of them odd (a harder "new CCD")
$PY lvl1/fleet_tests.py fewshot "${COMMON[@]}" --channels 0 1 2 3 --drift --out "$OUT/with_ch2"
$PY lvl1/fleet_tests.py bo "${COMMON[@]}" --channels 0 1 2 3 --drift --out "$OUT/with_ch2"
# 4. BO from scratch (no calibration image at nominal first)
$PY lvl1/fleet_tests.py bo "${COMMON[@]}" --channels 0 1 3 --drift --start none --out "$OUT/bo_no_start"
echo "done: $OUT"

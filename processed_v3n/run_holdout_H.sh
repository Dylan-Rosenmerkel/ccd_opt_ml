#!/bin/bash
# session H: scored ONCE, with the models frozen in lvl1/frozen_choice_v3.json
cd /pool/homes/drosenm1/software/ccd_opt_ml
.venv/bin/python lvl1/scorecard_v2.py --data processed_v3n --plan lvl1/packet_plan_v3n/plan.json --train-session G \
  --holdout-session H --cells 'trained E' 'new E' --abs-models 'amplifier mean' mean 'settings + amplifier: gp' 'settings + amplifier: trees' 'settings: gp' 'settings: trees' 'trace: nn' 'trace: physics' 'trace: trees' --rel-models 'no change' 'settings + amplifier: gp' 'settings + amplifier: trees' 'settings + trace: combined trees' 'settings + trace: direction classifier' 'settings + trace: residual trees' 'settings change: gp' 'settings change: trees' 'trace change: difference curves PCA + trees' 'trace change: trees' 'trace: combined trees [A, B, B-A]' 'trace: physics difference (sigma_e)' 'trace: physics difference + correction (sigma_e)' \
  --out processed_v3n/lvl1_results/scorecard_H > processed_v3n/logs/scorecard_H.log 2>&1
echo "$(date +%T) holdout H exit $?" >> processed_v3n/logs/analysis.txt

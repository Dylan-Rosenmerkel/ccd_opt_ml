#!/bin/bash
# pooled v1+v2+v3: (1) session-G cross-validation with v1/v2 (A-F) always in training, every model;
# (2) exploratory: session H (already opened once) with the frozen v3 models, trained on A-G
cd /pool/homes/drosenm1/software/ccd_opt_ml
.venv/bin/python lvl1/scorecard_v2.py --data processed_v123 --plan lvl1/packet_plan_v123/plan.json --train-session G \
  --extra-train-sessions A B C D E F --out processed_v123/lvl1_results/scorecard_G_pooled > processed_v123/logs/scorecard_G_pooled.log 2>&1
echo "$(date +%T) G pooled exit $?" >> processed_v123/logs/runs.txt
.venv/bin/python lvl1/scorecard_v2.py --data processed_v123 --plan lvl1/packet_plan_v123/plan.json --train-session G \
  --extra-train-sessions A B C D E F --holdout-session H --cells 'trained E' 'new E' --abs-models 'amplifier mean' mean 'settings + amplifier: gp' 'settings + amplifier: trees' 'settings: gp' 'settings: trees' 'trace: nn' 'trace: physics' 'trace: trees' --rel-models 'no change' 'settings + amplifier: gp' 'settings + amplifier: trees' 'settings + trace: combined trees' 'settings + trace: direction classifier' 'settings + trace: residual trees' 'settings change: gp' 'settings change: trees' 'trace change: difference curves PCA + trees' 'trace change: trees' 'trace: combined trees [A, B, B-A]' 'trace: physics difference (sigma_e)' 'trace: physics difference + correction (sigma_e)' \
  --out processed_v123/lvl1_results/scorecard_H_pooled > processed_v123/logs/scorecard_H_pooled.log 2>&1
echo "$(date +%T) H pooled exit $?" >> processed_v123/logs/runs.txt

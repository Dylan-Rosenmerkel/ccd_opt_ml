#!/bin/bash
# waits for the processing, then runs the first analyses in parallel (session G only; H stays sealed)
cd /pool/homes/drosenm1/software/ccd_opt_ml
P=processed_v3n/logs/progress.txt; A=processed_v3n/logs/analysis.txt
: # processing done
R=processed_v3n/lvl1_results; mkdir -p $R
PLAN=lvl1/packet_plan_v3n/plan.json
log(){ echo "$(date +%T) $*" >> $A; }
( OMP_NUM_THREADS=8 .venv/bin/python -u lvl1/parameter_influence.py --data processed_v3n --plan $PLAN --sessions G --channels 0 1 3 --out $R/parameter_influence > processed_v3n/logs/influence.log 2>&1; log "influence exit $?" ) &
( OMP_NUM_THREADS=8 .venv/bin/python -u lvl1/parameter_influence.py --data processed_v3n --plan $PLAN --sessions G --channels 0 1 3 --noise-components --n-boot 100 --out $R/parameter_influence_noise > processed_v3n/logs/influence_noise.log 2>&1; log "influence noise exit $?" ) &
( OMP_NUM_THREADS=16 .venv/bin/python -u lvl1/scorecard_v2.py --data processed_v3n --plan $PLAN --train-session G --out $R/scorecard_G > processed_v3n/logs/scorecard_G.log 2>&1; log "scorecard exit $?" ) &
wait; log "ALL DONE"

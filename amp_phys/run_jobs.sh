#!/bin/bash
# long amp_phys / pooled jobs; each line logs the PROGRAM's exit code to amp_phys/logs/runs.txt
cd /pool/homes/drosenm1/software/ccd_opt_ml
L=amp_phys/logs
run() { name=$1; shift; "$@" > $L/$name.log 2>&1; echo "$(date +%T) $name exit $?" >> $L/runs.txt; }
# 1 oxide traps (strips, board 101, v3 nominal reference strips)
run oxide_traps .venv/bin/python amp_phys/oxide_traps.py --files $(ls /pool/storage/DAQ1_Data/run/PA_PQ_testing/2026-11-10-PA03_PQ04_v3_packets/*l1ref*_L1_101_*.RJ45ADCDATA0 | head -24) --out amp_phys/results/v3n/oxide_traps &
# 2 whole-trace anatomy: live, dead, v1, October
rm -rf amp_phys/results/anatomy
run anatomy_v3 .venv/bin/python amp_phys/trace_anatomy.py --traces $(ls processed_v3n/*/traces/ref1_acm101_ch1.npz | head -12) --label "Nov v3 101ch1" --out amp_phys/results/anatomy --plot &
run anatomy_dead .venv/bin/python amp_phys/trace_anatomy.py --traces $(ls processed_v3n/*/traces/ref1_acm102_ch0.npz | head -12) --label "Nov v3 dead 102ch0" --out amp_phys/results/anatomy --plot &
run anatomy_v1 .venv/bin/python amp_phys/trace_anatomy.py --traces $(ls processed_v1/*/traces/ref1_acm101_ch1.npz | head -12) --label "Nov v1 101ch1" --out amp_phys/results/anatomy &
run anatomy_oct .venv/bin/python amp_phys/trace_anatomy.py --traces $(ls /pool/storage/DAQ1_Data/run/PA_PQ_testing/2026-10-12-PA03_PQ04_Packet_Tests/trace_*_ch1_101_*.RJ45ADCDATA1 | head -12) --label "Oct 101ch1" --out amp_phys/results/anatomy &
wait
# 3 pooled v1+v2+v3 scorecards (after the merge fix)
run pooled_G .venv/bin/python lvl1/scorecard_v2.py --data processed_v123 --plan lvl1/packet_plan_v123/plan.json --train-session G --extra-train-sessions A B C D E F --out processed_v123/lvl1_results/scorecard_G_pooled
run pooled_H .venv/bin/python lvl1/scorecard_v2.py --data processed_v123 --plan lvl1/packet_plan_v123/plan.json --train-session G --extra-train-sessions A B C D E F --holdout-session H --cells 'trained E' 'new E' --abs-models 'amplifier mean' mean 'settings + amplifier: gp' 'settings + amplifier: trees' 'settings: gp' 'settings: trees' 'trace: nn' 'trace: physics' 'trace: trees' --rel-models 'no change' 'settings + amplifier: gp' 'settings + amplifier: trees' 'settings + trace: combined trees' 'settings + trace: direction classifier' 'settings + trace: residual trees' 'settings change: gp' 'settings change: trees' 'trace change: difference curves PCA + trees' 'trace change: trees' 'trace: combined trees [A, B, B-A]' 'trace: physics difference (sigma_e)' 'trace: physics difference + correction (sigma_e)' --out processed_v123/lvl1_results/scorecard_H_pooled

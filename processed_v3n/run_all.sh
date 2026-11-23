#!/bin/bash
cd /pool/homes/drosenm1/software/ccd_opt_ml/src
ls ../manifests_v3n/*.json | grep -v N3_002 | xargs -P 16 -I{} sh -c 'p=$(basename {} .json); OMP_NUM_THREADS=3 ../.venv/bin/python process_packet.py {} --out ../processed_v3n --no-rebuild > ../processed_v3n/logs/$p.log 2>&1; echo "$(date +%T) $p exit $?" >> ../processed_v3n/logs/progress.txt'
../.venv/bin/python process_packet.py --rebuild-only --out ../processed_v3n > ../processed_v3n/logs/rebuild.log 2>&1
echo "$(date +%T) REBUILD exit $?" >> ../processed_v3n/logs/progress.txt

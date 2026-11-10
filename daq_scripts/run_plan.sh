#!/usr/bin/env bash
# =============================================================================
# run_plan.sh -- take the packets of a plan (lvl1/packet_plan_v2/) one after the other with manual_packet.sh
#
# For every packet of plan.json, in the plan's order:
#   * skip it if it is finished: a record <id>.json, or of a retry (<id>_r1, <id>b ...), with status "complete";
#   * stop if it was started but did not finish (a record that is not "complete": queued, failed, or restarted until
#     manual_packet.sh gave up): something went wrong there (a hang, an error, many damaged traces).  Look at it first;
#     then rerun with RETRY=1 to take it again as <id>b (<id>c ...);
#   * otherwise take it: manual_packet.sh with the plan's N_REF_SETS / N_SETS / N_POST_SETS / IMAGE_TOML (plan.json
#     "run"), the packet's changes file, its reference file <id>_ref.txt when the plan has one (the reference at the
#     previous packet's settings), SCREEN=1 with the plan's screen trace sets for a trace-only funnel screen
#     (plan.json "screen": true), the plan's and the packet's "env" (any manual_packet.sh setting, e.g. IMAGE_TOML,
#     PAUSE_S, L1_STRIP; the packet's wins), <id>_undo.txt / <id>_ref_undo.txt when the plan has them (UNDO_FILE /
#     REF_UNDO_FILE: the rule-safe way back to nominal), and the answer "y" to its question.  manual_packet.sh waits for and checks the traces,
#     retakes damaged ones and restarts the packet itself when too many are damaged.
# Stops on the first packet that fails (manual_packet.sh exit code != 0), or before the next packet when the file
# $RECORD_DIR/STOP exists (touch it to end a night cleanly; the runner removes it when it stops), or after TO.
#
# Usage (on the DAQ machine, ccd-cdaq environment active, inside tmux):
#   DRY_RUN=1 bash daq_scripts/run_plan.sh                    # dry-run every remaining packet, send nothing
#   bash daq_scripts/run_plan.sh                              # take every remaining packet
#   FROM=L2_10 TO=L2_61 bash daq_scripts/run_plan.sh          # only this range (e.g. session D: up to L2_61)
#   RETRY=1 bash daq_scripts/run_plan.sh                      # also retake unfinished packets under a new name
#   touch ~/packet_records/STOP                               # (another terminal) stop after the current packet
# Every other manual_packet.sh setting (RECORD_DIR, CDAQ_DIR, PYTHON, CHECK_BOARDS, MAX_DAMAGED ...) passes through.
# A log of every packet goes to $RECORD_DIR/run_plan.log.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PLAN_DIR="${PLAN_DIR:-$ROOT/lvl1/packet_plan_v2}"
RECORD_DIR="${RECORD_DIR:-$HOME/packet_records}"
PYTHON="${PYTHON:-python}"
FROM="${FROM:-}"
TO="${TO:-}"
RETRY="${RETRY:-0}"
export RECORD_DIR PYTHON

[[ -r $PLAN_DIR/plan.json ]] || { echo "ERROR: no plan.json in $PLAN_DIR"; exit 1; }
# the plan's order and run settings (python's json: no jq needed)
read -r -a IDS <<< "$($PYTHON -c "import json,sys; print(' '.join(p['packet_id'] for p in json.load(open(sys.argv[1]))['packets']))" "$PLAN_DIR/plan.json")"
eval "$($PYTHON -c "import json,sys; r=json.load(open(sys.argv[1])).get('run', {}); print(' '.join(f'PLAN_{k}={v!r}' for k, v in r.items()))" "$PLAN_DIR/plan.json")"
export N_REF_SETS="${N_REF_SETS:-${PLAN_N_REF_SETS:-5}}" N_SETS="${N_SETS:-${PLAN_N_SETS:-10}}"
export N_POST_SETS="${N_POST_SETS:-${PLAN_N_POST_SETS:-0}}"
SCREEN_REF_SETS="${PLAN_SCREEN_REF_SETS:-1}" SCREEN_SETS="${PLAN_SCREEN_SETS:-2}"
# the plan's screen packets (trace-only, session F)
SCREENS=" $($PYTHON -c "import json,sys; print(' '.join(p['packet_id'] for p in json.load(open(sys.argv[1]))['packets'] if p.get('screen')))" "$PLAN_DIR/plan.json") "
export IMAGE_TOML="${IMAGE_TOML:-${PLAN_IMAGE_TOML:-image_params/tc1/img.toml}}"
mkdir -p "$RECORD_DIR"
LOG="$RECORD_DIR/run_plan.log"
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG"; }

status_of() {   # status_of <packet id>: the "status" of its record, "" if there is none
  local f="$RECORD_DIR/$1.json"
  [[ -e $f ]] || { echo ""; return; }
  $PYTHON -c "import json,sys; print(json.load(open(sys.argv[1])).get('status', 'complete (record v1)'))" "$f"
}
finished() {    # finished <id>: true if <id> or one of its retries (<id>_rN, <id>b.., <id>b_rN ..) is complete
  local f
  for f in "$RECORD_DIR/$1.json" "$RECORD_DIR/$1"_r*.json "$RECORD_DIR/$1"[b-z].json "$RECORD_DIR/$1"[b-z]_r*.json; do
    [[ -e $f ]] || continue
    [[ $(status_of "$(basename "$f" .json)") == complete* ]] && return 0
  done
  return 1
}
unfinished() {  # unfinished <id>: the latest attempt (<id>, <id>b ..., with their restarts <id>_r1 ...) that has a
                # record but did not complete ("" if none was started): queued, failed, or given up after its restarts
  local f last=""
  for f in "$RECORD_DIR/$1.json" "$RECORD_DIR/$1"_r*.json "$RECORD_DIR/$1"[b-z].json "$RECORD_DIR/$1"[b-z]_r*.json; do
    [[ -e $f ]] && last=$(basename "$f" .json)
  done
  echo "$last"
}
next_name() {   # next_name <id>: <id>b, <id>c ... (the first without a record)
  local c
  for c in b c d e f g h; do [[ -e "$RECORD_DIR/$1$c.json" ]] || { echo "$1$c"; return; }; done
  echo "ERROR"; return 1
}

started=0
[[ -z $FROM ]] && started=1
n_done=0; n_taken=0
log "run_plan: $PLAN_DIR (${#IDS[@]} packets), N_REF_SETS=$N_REF_SETS N_SETS=$N_SETS N_POST_SETS=$N_POST_SETS IMAGE_TOML=$IMAGE_TOML DRY_RUN=${DRY_RUN:-0}"
for id in "${IDS[@]}"; do
  [[ $id == "$FROM" ]] && started=1
  (( started )) || continue
  if [[ -e $RECORD_DIR/STOP ]]; then
    rm -f "$RECORD_DIR/STOP"
    log "STOP file found: stopping before $id ($n_taken taken in this run)"
    exit 0
  fi
  if finished "$id"; then
    n_done=$(( n_done + 1 ))
  else
    name=$id
    bad=$(unfinished "$id")
    if [[ -n $bad ]]; then
      if [[ $RETRY != 1 ]]; then
        log "STOP: $bad was started but did not finish (status: $(status_of "$bad")).  Check it (cdaq print queues /"
        log "      logs), then rerun with RETRY=1 to take it again under a new name."
        exit 1
      fi
      name=$(next_name "$id")
      log "retrying $id as $name (the earlier attempt $bad: $(status_of "$bad"))"
    fi
    if [[ ! -s $PLAN_DIR/$id.txt ]]; then
      log "STOP: $PLAN_DIR/$id.txt is missing or empty.  If $id is a session-F SHORTLIST packet: process the screens"
      log "      and sessions D+E, run lvl1/funnel_shortlist.py --plan $PLAN_DIR/plan.json (see PLAN.md), then rerun."
      exit 1
    fi
    extra=()
    [[ -e $PLAN_DIR/${id}_ref.txt ]] && extra+=("REF_CHANGES_FILE=$PLAN_DIR/${id}_ref.txt")
    # the rule-safe way back to nominal (plans that write <id>_undo.txt): sent by manual_packet.sh before its resets
    [[ -e $PLAN_DIR/${id}_undo.txt ]] && extra+=("UNDO_FILE=$PLAN_DIR/${id}_undo.txt")
    [[ -e $PLAN_DIR/${id}_ref_undo.txt ]] && extra+=("REF_UNDO_FILE=$PLAN_DIR/${id}_ref_undo.txt")
    # plan.json "env" (plan level) and the packet's own "env" (it wins): any manual_packet.sh setting, per packet
    while IFS= read -r kv; do [[ -n $kv ]] && extra+=("$kv"); done < <($PYTHON -c "
import json, sys
plan = json.load(open(sys.argv[1]))
env = {**plan.get('env', {}), **next(p for p in plan['packets'] if p['packet_id'] == sys.argv[2]).get('env', {})}
print('\\n'.join(f'{k}={v}' for k, v in env.items()))" "$PLAN_DIR/plan.json" "$id")
    [[ $SCREENS == *" $id "* ]] && extra+=(SCREEN=1 "N_REF_SETS=$SCREEN_REF_SETS" "N_SETS=$SCREEN_SETS" N_POST_SETS=0)
    log "taking $name ($PLAN_DIR/$id.txt${extra[*]+; ${extra[*]}})"
    if echo y | env ${extra[@]+"${extra[@]}"} PACKET_ID="$name" CHANGES_FILE="$PLAN_DIR/$id.txt" bash "$HERE/manual_packet.sh" >> "$LOG" 2>&1; then
      n_taken=$(( n_taken + 1 ))
      if [[ ${DRY_RUN:-0} == 1 ]]; then
        log "$name: dry run OK"
      else
        log "$name: done; $(grep -o "$name[_r0-9]* queued to the end ([0-9]* damaged traces retaken)" "$LOG" | tail -1)"
      fi
    else
      log "STOP: manual_packet.sh failed on $name (see the end of $LOG)"
      exit 1
    fi
  fi
  [[ $id == "$TO" ]] && { log "reached TO=$TO"; break; }
done
log "run_plan done: $n_taken taken now, $n_done already finished"

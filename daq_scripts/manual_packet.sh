#!/usr/bin/env bash
# =============================================================================
# manual_packet.sh -- one packet by hand, with the group's cdaq commands
#
#   standalone -> nominal (reset + readback) [-> the reference settings, REF_CHANGES] -> flush -> reference strip ->
#   reference traces -> flush -> change parameters -> settling traces -> new traces -> new strip -> flush ->
#   master/slave -> image (img.toml) -> standalone -> post-image traces -> flush -> nominal (reset + readback);
#   a PAUSE_S wait after every trace and after the image.  SCREEN=1: a trace-only packet (no strips, no image).
#
# Order (2026-11-04): the reference traces come right before the change and the new traces right after it, so the two
# sides of a within-packet comparison are minutes apart; the strips bracket that pair; the new traces bracket the
# strip and the image (N_SETS before, N_POST_SETS after), so a drift of the trace across the image can be seen.
# The reference is nominal unless REF_CHANGES / REF_CHANGES_FILE is given (the previous packet's settings, so that
# within-packet changes go both up and down, as between a level-2 incumbent and a candidate): then the change goes
# through a reset to nominal (seconds) before the new settings.
#
# Every line below is a plain `cdaq qadd "..." <target>` (or take_image.py), the same
# commands typed by hand.  The cdaq queue runs them ONE AFTER THE OTHER in the order
# they are added, so the script only queues them; it finishes in seconds while the
# queue works for ~15-20 min.  Watch with:  cdaq print queues / cdaq print logs
#
# Usage (on the DAQ machine, ccd-cdaq environment active):
#   DRY_RUN=1 bash manual_packet.sh     # print every command, send nothing  <- do this first
#   bash manual_packet.sh               # check, show the plan, ask, then queue
#   PACKET_ID=m0007 bash manual_packet.sh   # choose the packet's name (default: m<date>_<time>)
#   PACKET_ID=L1_07 CHANGES_FILE=lvl1/packet_plan/L1_07.txt bash manual_packet.sh
#                                           # take the changes from a file (one NAME=VALUE or acm101:NAME=VALUE per
#                                           # line, # comments allowed) instead of the CHANGES block below
#   REF_CHANGES_FILE=lvl1/packet_plan_v2/L2_09_ref.txt ...   # the reference at these settings instead of nominal
#   SCREEN=1 ...                                              # trace-only packet (the funnel screen of session F)
#
# Edit the SETTINGS block, nothing else.
#
# PROCESSING: every trace of the packet is named <PACKET_ID>_ref<set> / <PACKET_ID>_new<set> / <PACKET_ID>_post<set>
# (after the image; cdaq appends
# the geometry, channel, board and time), and a small RECORD of the packet (its name, the changes, the
# trace and image settings) is written to $RECORD_DIR/<PACKET_ID>.json when the commands are queued.  After
# the queue has run, on the analysis computer:
#   python src/make_manual_manifest.py <record.json> --run-dir <directory with the raw files>
#   python src/process_packet.py <the manifest it writes>
# The manifest takes the settings of every trace and image from their .meta files (the values the boards
# report), and the image is the first image of each board taken after the packet's last trace.
#
# Flushes (2026-11-03, user): one before the reference strip and traces, one after them, one after the new traces and
# strip (= before the image), one after the image and the post-image traces; none between trace sets.  After the change, N_SETTLE_TRACES traces (named
# <PACKET_ID>_settle<n>, not used for training) are taken before the new traces.
# Duration with the defaults: 4 flushes (~30 s each) = 2 min, 15 sets x 4 channels x 2 boards traces + a 2 s pause
# after every trace (~6 min), the img.toml image (16x1000x500, 4.8 min readout + its EXPOSURE): about 14 min.
# Every packet starts AND ends by restoring the .bcf voltages and the sequencer's timing on the boards, then
# reading them back (see nominal() below): a failed reset shows as "NOT NOMINAL" in cdaq print logs.
# =============================================================================
set -euo pipefail

# ------------------------------- SETTINGS ------------------------------------
CDAQ_DIR="${CDAQ_DIR:-$HOME/Soft_JHU/cdaq}"     # where scripts/take_image.py lives
PYTHON="${PYTHON:-python}"                      # or "pixi run python"
IMAGE_TOML="${IMAGE_TOML:-image_params/tc1/img.toml}"   # relative to CDAQ_DIR (16 x 1000 x 500, 1x1, prescan 5300;
                                # v2 plan: a 32-row copy, configs/image_params_32rows.toml)
LEADER=101                                      # master board for the image
POOL=acmpool
PACKET_ID="${PACKET_ID:-m$(date +%Y%m%d_%H%M%S)}" # unique name of this packet: the prefix of all its traces
RECORD_DIR="${RECORD_DIR:-$HOME/packet_records}" # where the packet record (JSON) is written
SCAN_NAME="${SCAN_NAME:-manual}"                # free text: which scan / series this packet belongs to

N_REF_SETS="${N_REF_SETS:-5}"   # reference trace sets (nominal parameters)
N_SETS="${N_SETS:-10}"          # trace sets at the new parameters, before the image
N_POST_SETS="${N_POST_SETS:-0}" # trace sets at the new parameters AFTER the image (v2 plan: 2), named <PACKET_ID>_post<set>
SCREEN="${SCREEN:-0}"           # 1: trace-only packet: reference and new traces, no strips, no image
N_SETTLE_TRACES="${N_SETTLE_TRACES:-2}"  # traces after the change, before the new sets (first channel; not used)
FLUSH_BEFORE_REF=1        # flushes before the first reference trace set
FLUSH_AFTER_REF=1         # flushes after the reference traces (before the change)
FLUSH_BEFORE_IMAGE=1      # flushes after the new traces, before the image (in standalone)
FLUSH_AFTER_IMAGE=1       # flushes after the image (back in standalone), before the final reset
FLUSH_BETWEEN="none"      # "set":   a flush between trace sets, none between the channels of a set
                          # "trace": a flush before every trace;  "none": no flushes between traces
TRACE_CHANNELS="${TRACE_CHANNELS:-0 1 2 3}"  # one trace per channel; override e.g. TRACE_CHANNELS="1 0 2 3"
TRACE_BOARDS="acm101 acm102"  # traces are sent to each board separately (not to the pool; DAQ edit of 2026-11-02)
PAUSE_S="${PAUSE_S:-2.0}"       # seconds of waiting (both boards) after every trace and after the image

# Safeguards (2026-11-03): after each trace phase (reference, new) the script WAITS until both boards have finished it,
# checks every trace file (src/check_trace_files.py: missing, unfinished, or failing the decoder's quality check, as in
# process_packet.py), and retakes each damaged trace under the next set number.  More than MAX_DAMAGED damaged traces
# in one packet (retakes included): the packet is stopped, the boards reset, and the whole packet restarted as
# <PACKET_ID>_r1 (then _r2 ...; its record is marked "restarted" and make_manual_manifest.py skips it).
# The script therefore runs until the image is queued (~10 min): keep the terminal open (tmux).
# Waiting needs the boards' cdaq clients on this machine (they write a small marker file into MARK_DIR).  While the
# script waits the simple queue can be empty, so the cdaq REPEAT queue (e.g. an idle flush) may run then: suspend it
# for the packets, or accept an extra flush between the phases.
CHECK_TRACES="${CHECK_TRACES:-1}"   # 0: no waiting, no checks (queue everything at once, as before)
CHECK_BOARDS="${CHECK_BOARDS:-101}" # boards whose traces are checked, retaken and counted (102 is not analysed)
MAX_DAMAGED="${MAX_DAMAGED:-5}"     # more damaged traces than this in one packet: restart the packet from the reset
MAX_RESTARTS="${MAX_RESTARTS:-2}"   # then give up (exit 1)
RETAKE_ROUNDS="${RETAKE_ROUNDS:-2}" # retakes of retakes at most this often per phase; then go on with what there is
PHASE_TIMEOUT_S="${PHASE_TIMEOUT_S:-900}"   # a phase not finished after this long: stop (a trace may hang, HDU END)
MARK_DIR="${MARK_DIR:-$RECORD_DIR/markers}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"

# Trace that reads its pixels like the image (docs/packet_design.md, section 6): the same skips
# (NDCM 500) and no binning (1x1), as img.toml; its own small size: 2 x 5 pixels after a
# 100-pixel prescan burst, no register clear, no exposure.  0.19 s, ~11 MB per channel.
# Every cut is given, so nothing is inherited from the last image.
# If the image's NDCM / NSBIN / NPBIN change, change them here too.
TRACE_ARGS="subroutine='AcquireImage', EXPOSURE=0, NROW=2, NCOL=5, NPBIN=1, NSBIN=1, NDCM=500, BOOLCLR=0, NDCMPRE=0, NROWPRE=0, NROWPOST=0, NCOLPRE=100, NCOLPOST=0"
# Level-1 STRIPS (2026-11-03): a small image with every skip kept (acq_level 1, all 4 channels at once), read like the
# image (its prescan NCOLPRE 5300, register clear BOOLCLR 1, NDCM 500, 1x1), one at the reference after the reference
# traces and one at the new settings after the new traces, so the same packets answer: what do the traces predict,
# what does the strip predict, and both together (docs/context/2026-11-03, section 13).  Named <PACKET_ID>_l1ref /
# <PACKET_ID>_l1new; checked like the traces (src/check_trace_files.py --l1) and retaken once if incomplete.
# If the image's NDCM / binning / prescan / clear change, change them here too.  L1_STRIP=0: no strips.
L1_STRIP="${L1_STRIP:-1}"
L1_ARGS="subroutine='AcquireImage', EXPOSURE=0, NROW=${L1_NROW:-2}, NCOL=${L1_NCOL:-1000}, NPBIN=1, NSBIN=1, NDCM=500, BOOLCLR=1, NDCMPRE=0, NROWPRE=0, NROWPOST=0, NCOLPRE=5300, NCOLPOST=0"
# The group's earlier trace command (2x5, 5x5 binning, 100 skips), with the cuts set to 0:
# TRACE_ARGS="subroutine='AcquireImage', EXPOSURE=0, NROW=2, NCOL=5, NPBIN=5, NSBIN=5, NDCM=100, BOOLCLR=0, NDCMPRE=0, NROWPRE=0, NROWPOST=0, NCOLPRE=0, NCOLPOST=0"

# The parameter changes, one per line:  NAME=VALUE  (both boards, sent to acmpool)
#                                   or  acm101:NAME=VALUE  (one board only)
CHANGES=(
  "H1_A_L=6.2"
  "H1_B_L=6.0"
  "H2_C_L=6.2"
  "H3_A_L=6.2"
  "H3_B_L=6.0"
)

# The reference settings (optional; empty = nominal), in the same format: the previous packet's changes, so the
# within-packet change is "incumbent -> candidate" and goes up as often as down.
REF_CHANGES=()

read_changes() {   # read_changes <file>: its NAME=VALUE lines (comments and blank lines dropped) into LINES
  [[ -r $1 ]] || { echo "ERROR: changes file $1 cannot be read"; exit 1; }
  LINES=()
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%%#*}; line=$(echo $line)          # strip comments and spaces
    [[ -n $line ]] && LINES+=("$line")
  done < "$1"
}
# CHANGES_FILE (optional): read the changes from a file instead (the planned packets of lvl1/packet_plan/).
if [[ -n ${CHANGES_FILE:-} ]]; then
  read_changes "$CHANGES_FILE"; CHANGES=("${LINES[@]}")
  [[ ${#CHANGES[@]} -gt 0 ]] || { echo "ERROR: no changes in $CHANGES_FILE"; exit 1; }
fi
if [[ -n ${REF_CHANGES_FILE:-} ]]; then
  read_changes "$REF_CHANGES_FILE"; REF_CHANGES=("${LINES[@]}")
fi

# Nominal configuration files, restored at the start and at the end:  board=file
BCF="acm101=module_PA03.bcf acm102=ccd_PQ04.bcf"
SEQ="reference_v5_pumping.seq"   # the sequencer the boards must run (checked by the readback, not loaded)
# -----------------------------------------------------------------------------

# Nominal values (the .bcf content shown on 2026-10-14), used only for the checks below.
NOMINAL="
VDD1 -19  VDD2 -19  VR1 -3  VR2 -3  VDRAIN1 -23  VDRAIN2 -23  VSUB 45
OFS_RS1 -10  OFS_RS2 -10  OFS_CLK -10
V1_A_H 9.9  V1_A_L 7.9  V1_B_H 9.9  V1_B_L 7.7  V2_C_H 9.9  V2_C_L 7.9
V3_A_H 9.9  V3_A_L 7.9  V3_B_H 9.9  V3_B_L 7.7
TG_A_H 7.4  TG_A_L 7.4  TG_B_H 7.4  TG_B_L 7.4
H1_A_H 9.5  H1_A_L 6.7  H1_B_H 9.5  H1_B_L 6.5  H2_C_H 9.0  H2_C_L 6.7
H3_A_H 9.5  H3_A_L 6.7  H3_B_H 9.5  H3_B_L 6.5
SW_1_H 1.0  SW_1_L -6.0  OG_1_H 0.0  OG_1_L -6.0  RG_1_H 9.0  RG_1_L 3.0  DG_1_H 0.0  DG_1_L -5.0
SW_2_H 1.0  SW_2_L -6.0  OG_2_H 0.0  OG_2_L -6.0  RG_2_H 9.0  RG_2_L 3.0  DG_2_H 0.0  DG_2_L -5.0
delay_Va 6000  delay_Vb 6000  delay_Ha 1800  delay_Hb 1800  delay_Hc 1800  delayInt 1000
delayRGL 27  delayRGH 800  delaySWH 30  delaySWL 350  delayOGL 48  delayOGH 3  delayDGL 27  delayDGH 47
"
# (the timing values are those of reference_v5_pumping.seq, in 10 ns clock cycles; whole numbers only)

# --------------------------------- checks ------------------------------------
# 1. every changed name is a known parameter (catches typos: an unknown key would be sent as is)
# 2. every value is a number
# 3. after the change every rail keeps HIGH >= LOW (on 2026-10-12 TG_B_H < TG_B_L slipped through)
# 4. the group's safe-operation rules (docs/voltage_rules.txt; the same as src/ccd_limits.py), on the
#    COMPLETE configuration (nominal + changes), for readout stage / bias set i = 1, 2:
#      VDDi > VDRAINi,  SW_i_H > OG_i_H,  DG_i_H > VRi,  DG_i_L < VRi,  OG_i_L < VRi,  OG_i_L >= SW_i_L,
#      VDRAINi < DG_i_L,  VDDi < VRi;  every V clock low above every TG level;
#      every H clock low below every TG level;  H3 lows above SW_1_H and SW_2_H.
#    (The rules file's line "DGL<=SWL" is a typo and is disregarded: the group, 2026-10-14.)
# 5. hardware ranges of the ACM manual: clock rails -10..+10 V; VR -13.8..-2, VDD and VDRAIN -27.9..-6,
#    VSUB 6.7..88.3 V.
# The allowed WINDOWS of configs/limits.toml are not checked here (scan_packets.py / take_packet.py do).
# Prints the table of changes; exits non-zero on any problem.  (awk: works with any bash.)
check_and_show() {   # check_and_show <title> <NAME=VALUE> ...: the checks above on nominal + these changes
  local title=$1; shift
  printf '%s\n' "$@" | awk -v title="$title" -v nominal="$(echo $NOMINAL)" -v pool="$POOL" '
    function rule(a, op, b) {          # a op b on the complete configuration; prints and flags a violation
      if (!(a in new) || !(b in new)) return
      x = new[a] + 0; y = new[b] + 0
      ok = (op == ">") ? (x > y) : (op == ">=") ? (x >= y) : (op == "<") ? (x < y) : (x <= y)
      if (!ok) { print "ERROR: " a " = " new[a] " must be " op " " b " = " new[b]; bad = 1 }
    }
    function trange(k, lo, hi) {
      if (new[k] + 0 < lo || new[k] + 0 > hi) { print "ERROR: " k " = " new[k] " outside the timing window [" lo ", " hi "] cycles"; bad = 1 }
    }
    function range(k, lo, hi) {
      if ((k in new) && (new[k] + 0 < lo || new[k] + 0 > hi)) { print "ERROR: " k " = " new[k] " outside the hardware range [" lo ", " hi "] V"; bad = 1 }
    }
    BEGIN { n = split(nominal, w, / +/); for (i = 1; i < n; i += 2) if (w[i] != "") nom[w[i]] = w[i+1]
            for (k in nom) new[k] = nom[k] }
    NF == 0 { next }
    { target = pool; spec = $0
      if (index(spec, ":")) { target = substr(spec, 1, index(spec, ":") - 1); spec = substr(spec, index(spec, ":") + 1) }
      name = substr(spec, 1, index(spec, "=") - 1); value = substr(spec, index(spec, "=") + 1)
      if (!(name in nom)) { print "ERROR: unknown parameter \"" name "\" in \"" $0 "\""; bad = 1; next }
      if (value !~ /^-?[0-9]+(\.[0-9]+)?$/) { print "ERROR: \"" value "\" is not a number in \"" $0 "\""; bad = 1; next }
      if (name ~ /^delay/ && value !~ /^[0-9]+$/) { print "ERROR: timing " name " must be a whole number of clock cycles: \"" $0 "\""; bad = 1; next }
      new[name] = value
      line[++m] = sprintf("  %-10s %8s -> %-8s (%s)", name, nom[name], value, target) }
    END { for (h in new) if (h ~ /_H$/) { l = substr(h, 1, length(h) - 1) "L"
            if ((l in new) && new[h] + 0 < new[l] + 0) { print "ERROR: " h " = " new[h] " is below " l " = " new[l]; bad = 1 } }
          for (i = 1; i <= 2; i++) {
            rule("VDD" i, ">", "VDRAIN" i);      rule("SW_" i "_H", ">", "OG_" i "_H")
            rule("DG_" i "_H", ">", "VR" i);     rule("DG_" i "_L", "<", "VR" i)
            rule("OG_" i "_L", "<", "VR" i);     rule("OG_" i "_L", ">=", "SW_" i "_L")
            rule("VDRAIN" i, "<", "DG_" i "_L")
            rule("VDD" i, "<", "VR" i)
            range("VR" i, -13.8, -2); range("VDD" i, -27.9, -6); range("VDRAIN" i, -27.9, -6)
          }
          range("VSUB", 6.7, 88.3)
          # timing windows of configs/limits.toml (10 ns cycles)
          for (k in new) {
            if (k ~ /^delay_V/) trange(k, 1000, 20000); if (k ~ /^delay_H/) trange(k, 200, 6000)
            if (k == "delayInt") trange(k, 50, 20000);  if (k ~ /^delay(RG|SW|OG|DG)/) trange(k, 1, 10000)
          }
          for (k in new) {
            if (k ~ /^(V[123]|H[123]|TG|SW|OG|RG|DG)_[A-C12]_[HL]$/) range(k, -10, 10)
            if (k ~ /^V[123]_[A-C]_L$/) for (t in new) if (t ~ /^TG_/) rule(k, ">", t)
            if (k ~ /^H[123]_[A-C]_L$/) for (t in new) if (t ~ /^TG_/) rule(k, "<", t)
          }
          rule("H3_A_L", ">", "SW_1_H"); rule("H3_A_L", ">", "SW_2_H"); rule("H3_B_L", ">", "SW_1_H"); rule("H3_B_L", ">", "SW_2_H")
          if (bad) exit 1
          print title; for (i = 1; i <= m; i++) print line[i] }'
}

# --------------------------------- sending -----------------------------------
q() {   # queue one command:  q "<python text>" <target>
  if [[ ${DRY_RUN:-0} == 1 ]]; then echo "cdaq qadd \"$1\" $2"; else cdaq qadd "$1" "$2"; fi
}
flushes() { for ((i = 0; i < $1; i++)); do q "daq.flush_ccd()" "$POOL"; done; }
pause() { q "time.sleep($PAUSE_S)" "$POOL"; }   # PAUSE_S seconds of waiting on both boards
traces() {   # traces <prefix> <number of sets> [channels]: one trace per channel per board per set (each sent to
             # its own board), then a PAUSE_S wait
  local first=1 s ch board chans=${3:-$TRACE_CHANNELS}
  for ((s = 1; s <= $2; s++)); do
    if (( ! first )) && [[ $FLUSH_BETWEEN == set ]]; then flushes 1; fi
    for ch in $chans; do
      if (( ! first )) && [[ $FLUSH_BETWEEN == trace ]]; then flushes 1; fi
      for board in $TRACE_BOARDS; do
        q "daq.take_trace(fname='$1$s', ch=$ch, $TRACE_ARGS)" "$board"
      done
      pause
      first=0
    done
  done
}
# nominal: put each board back to its .bcf and the sequencer back to its file, then read everything back.
# (acmpy, checked 2026-11-02: read_config ONLY parses the file into daq.config -- nothing is sent to the
#  board; voltages are applied by control.restore_defaults(); daq['delay...'] = v only edits the cached
#  sequencer text, which the next trace recompiles; sequencer.restore_defaults() returns that text to the
#  file read at startup and load_sequencer() compiles and uploads it now, so the readback sees it.)
# The readback compares every bias, offset and rail with the board's .bcf (bias readback within 0.3 V:
# potentiometer steps; rails within 0.05 V), every timing constant with $SEQ's values (NOMINAL table), and
# the loaded sequencer's name with $SEQ.  On a mismatch the command raises: the cdaq log shows
# "Could not execute ... AssertionError: NOT NOMINAL ..." with the offending values; otherwise it logs
# "readback: nominal OK".
TIMING_NOMINAL=$(echo $NOMINAL | awk '{ s = ""; for (i = 1; i < NF; i += 2) if ($i ~ /^delay/) s = s (s == "" ? "" : ", ") "'"'"'" $i "'"'"': " $(i+1); print "{" s "}" }')
READBACK="_f = lambda z: [float(x) for x in (z if isinstance(z, (tuple, list)) else (z, z))]; \
_bad = [k + ' ' + str(daq[k]) + ' (file ' + v + ')' for s, t in (('bias', 0.3), ('clock_offsets', 0.05), ('clock_rails', 0.05)) \
for k, v in daq.config[s].items() if any(abs(a - b) > t for a, b in zip(_f(daq[k]), _f(__import__('ast').literal_eval(v))))] \
+ [k + ' ' + str(daq[k]) + ' (seq ' + str(v) + ')' for k, v in ${TIMING_NOMINAL}.items() if daq[k] != v] \
+ ([] if daq.sequencer.seqfile.name == '$SEQ' else ['sequencer ' + daq.sequencer.seqfile.name + ' (expected $SEQ)']); \
assert not _bad, 'NOT NOMINAL after the reset: ' + ', '.join(_bad); daq.logger.info('readback: nominal OK')"
READBACK=${READBACK//\\$'\n'/}
nominal() {
  for pair in $BCF; do q "daq.read_config('${pair#*=}')" "${pair%%=*}"; done
  q "daq.control.restore_defaults()" "$POOL"
  q "daq.sequencer.restore_defaults()" "$POOL"
  q "daq.load_sequencer()" "$POOL"
  q "$READBACK" "$POOL"
}

# ------------------------------ packet record --------------------------------
# A JSON record of what this packet does, read by src/make_manual_manifest.py.  No numbers are parsed from
# it later except the requested changes (compared with what the .meta files report).
json_list() { local first=1 x; printf '['; for x in "$@"; do ((first)) || printf ', '; printf '"%s"' "$x"; first=0; done; printf ']'; }
write_record() {
  local file="$RECORD_DIR/$PACKET_ID.json"
  if [[ ${DRY_RUN:-0} == 1 ]]; then echo "(dry run: the record would be written to $file)"; return; fi
  mkdir -p "$RECORD_DIR"
  if [[ -e $file ]]; then echo "ERROR: $file exists: choose another PACKET_ID"; exit 1; fi
  {
    printf '{\n'
    printf '  "record_version": 2,\n'
    printf '  "status": "queued",\n'
    printf '  "packet_id": "%s",\n' "$PACKET_ID"
    printf '  "scan_name": "%s",\n' "$SCAN_NAME"
    printf '  "queued_at": "%s",\n' "$(date +%Y-%m-%dT%H:%M:%S%z)"
    printf '  "host": "%s",\n' "$(hostname)"
    printf '  "script": "manual_packet.sh",\n'
    printf '  "changes": %s,\n' "$(json_list "${CHANGES[@]}")"
    printf '  "changes_file": "%s",\n' "${CHANGES_FILE:-}"
    printf '  "ref_changes": %s,\n' "$(json_list ${REF_CHANGES[@]+"${REF_CHANGES[@]}"})"
    printf '  "ref_changes_file": "%s",\n' "${REF_CHANGES_FILE:-}"
    printf '  "screen": %s,\n' "$( ((SCREEN)) && echo true || echo false)"
    printf '  "bcf": %s,\n' "$(json_list $BCF)"
    printf '  "trace_prefix_ref": "%s_ref",\n' "$PACKET_ID"
    printf '  "trace_prefix_new": "%s_new",\n' "$PACKET_ID"
    printf '  "trace_prefix_post": "%s_post",\n' "$PACKET_ID"
    printf '  "n_post_sets": %d,\n' "$N_POST_SETS"
    printf '  "n_ref_sets": %d,\n' "$N_REF_SETS"
    printf '  "n_sets": %d,\n' "$N_SETS"
    printf '  "trace_channels": %s,\n' "$(json_list $TRACE_CHANNELS)"
    printf '  "trace_args": "%s",\n' "${TRACE_ARGS//\'/}"
    printf '  "flush_between": "%s",\n' "$FLUSH_BETWEEN"
    printf '  "flushes": {"before_ref": %d, "after_ref": %d, "before_image": %d, "after_image": %d},\n' \
      "$FLUSH_BEFORE_REF" "$FLUSH_AFTER_REF" "$FLUSH_BEFORE_IMAGE" "$FLUSH_AFTER_IMAGE"
    printf '  "n_settle_traces": %d,\n' "$N_SETTLE_TRACES"
    printf '  "pause_s": %s,\n' "$PAUSE_S"
    printf '  "trace_boards": %s,\n' "$(json_list $TRACE_BOARDS)"
    if [[ $L1_STRIP == 1 && $SCREEN != 1 ]]; then
      printf '  "l1_prefix_ref": "%s_l1ref",\n' "$PACKET_ID"
      printf '  "l1_prefix_new": "%s_l1new",\n' "$PACKET_ID"
      printf '  "l1_args": "%s",\n' "${L1_ARGS//\'/}"
    fi
    printf '  "image_toml": "%s",\n' "$( ((SCREEN)) || echo "$IMAGE_TOML")"
    printf '  "leader": %d\n' "$LEADER"
    printf '}\n'
  } > "$file"
  echo "Packet record written: $file"
}
set_status() {   # set_status <text>: the record's "status" (queued -> complete / restarted: ... / failed: ...)
  local file="$RECORD_DIR/$PACKET_ID.json"
  [[ ${DRY_RUN:-0} == 1 || ! -e $file ]] && return 0
  sed "s/\"status\": \"[^\"]*\"/\"status\": \"$1\"/" "$file" > "$file.tmp" && mv "$file.tmp" "$file"
}

# ------------------------------ safeguards -----------------------------------
marker() {   # marker <name>: each board writes MARK_DIR/<name>.<board id> (its run directory) when the queue gets there
  q "import pathlib as _p; _p.Path('$MARK_DIR/$1.' + str(daq.id)).write_text(str(daq.acquire.directory))" "$POOL"
}
wait_marker() {   # wait_marker <name>: until every trace board has written its marker (PHASE_TIMEOUT_S at most)
  local t0=$SECONDS b all
  while :; do
    all=1
    for b in $TRACE_BOARDS; do [[ -e "$MARK_DIR/$1.${b#acm}" ]] || all=0; done
    (( all )) && return 0
    if (( SECONDS - t0 > PHASE_TIMEOUT_S )); then
      echo "ERROR: the queue has not reached '$1' after $PHASE_TIMEOUT_S s.  A command may hang (e.g. a trace without"
      echo "       HDU END): check 'cdaq print queues' and 'cdaq print logs' before doing anything else."
      return 1
    fi
    sleep 3
  done
}
check_phase() {   # check_phase <prefix> <number of sets>: wait for the phase, check its traces, retake damaged ones.
                  # Returns 0 (go on), 2 (too many damaged traces: restart the packet) or 1 (error: stop).
  local prefix=$1 n=$2 name dir lines count round=0 next=$(( $2 + 1 )) b ch s reason
  local -a expect
  if [[ ${DRY_RUN:-0} == 1 || $CHECK_TRACES != 1 ]]; then
    if [[ $CHECK_TRACES == 1 ]]; then echo "(here the script waits for ${prefix}1..$n, checks them and retakes damaged traces)"; fi
    return 0
  fi
  name="${PACKET_ID}_$(basename "$prefix")_check0"
  marker "$name"; wait_marker "$name" || return 1
  dir=$(cat "$MARK_DIR/$name.${CHECK_BOARDS%% *}")
  lines=$($PYTHON "$SRC_DIR/check_trace_files.py" --dir "$dir" --prefix "$prefix" --sets $(seq 1 "$n") \
          --channels $TRACE_CHANNELS --boards $CHECK_BOARDS) || return 1
  while [[ -n $lines ]]; do
    count=$(printf '%s\n' "$lines" | wc -l | tr -d ' ')
    DAMAGED=$(( DAMAGED + count ))
    printf '%s\n' "$lines" | sed "s/^/  damaged trace ($prefix, board channel set reason): /"
    if (( DAMAGED > MAX_DAMAGED )); then return 2; fi
    if (( round >= RETAKE_ROUNDS )); then echo "  $prefix: still damaged after $round retake rounds; going on"; return 0; fi
    round=$(( round + 1 )); expect=()
    while read -r b ch s reason; do
      q "daq.take_trace(fname='$prefix$next', ch=$ch, $TRACE_ARGS)" "acm$b"
      pause
      expect+=("$b:$ch:$next"); next=$(( next + 1 ))
    done <<< "$lines"
    name="${PACKET_ID}_$(basename "$prefix")_check$round"
    marker "$name"; wait_marker "$name" || return 1
    lines=$($PYTHON "$SRC_DIR/check_trace_files.py" --dir "$dir" --prefix "$prefix" --expect "${expect[@]}") || return 1
  done
  echo "  $prefix: all traces checked ($DAMAGED damaged in this packet so far)"
  return 0
}
strip() {   # strip <name>: a level-1 strip on both boards (standalone), then a pause; waited for, checked, retaken once
  [[ $L1_STRIP == 1 && $SCREEN != 1 ]] || return 0
  local name=$1 lines dir attempt
  for attempt in 1 2; do
    q "daq.take_image(fname='$name', acq_level=1, $L1_ARGS)" "$POOL"
    pause
    if [[ ${DRY_RUN:-0} == 1 || $CHECK_TRACES != 1 ]]; then
      [[ $CHECK_TRACES == 1 ]] && echo "(here the script waits for the strip $name and checks it)"
      return 0
    fi
    marker "${name}_check$attempt"; wait_marker "${name}_check$attempt" || return 1
    dir=$(cat "$MARK_DIR/${name}_check$attempt.${CHECK_BOARDS%% *}")
    lines=$($PYTHON "$SRC_DIR/check_trace_files.py" --dir "$dir" --prefix "$name" --l1 --boards $CHECK_BOARDS) || return 1
    if [[ -z $lines ]]; then echo "  $name: level-1 strip checked"; return 0; fi
    printf '%s\n' "$lines" | sed "s/^/  incomplete level-1 strip ($name): /"
  done
  echo "  $name: still incomplete after a retake; going on without it"
  return 0
}

# --------------------------------- the plan ----------------------------------
[[ $PACKET_ID =~ ^[A-Za-z0-9_-]+$ ]] || { echo "ERROR: PACKET_ID may only contain letters, digits, _ and -"; exit 1; }
check_and_show "Parameter changes (nominal -> new):" "${CHANGES[@]}" || exit 1
if (( ${#REF_CHANGES[@]} )); then
  check_and_show "Reference settings (nominal -> reference; the reference traces and strip are taken here):" "${REF_CHANGES[@]}" || exit 1
else
  echo "Reference: nominal"
fi
echo "Packet: $PACKET_ID   (traces ${PACKET_ID}_ref<set>, ${PACKET_ID}_new<set>, ${PACKET_ID}_post<set>; record in $RECORD_DIR)"
(( SCREEN )) && echo "SCREEN packet: traces only (no strips, no image)"
echo "Traces: $N_REF_SETS reference + $N_SETS new (before the image) + $N_POST_SETS new (after it) sets x channels [$TRACE_CHANNELS] x boards [$TRACE_BOARDS], $N_SETTLE_TRACES settling, flush between: $FLUSH_BETWEEN"
echo "Flushes: before ref $FLUSH_BEFORE_REF, after ref $FLUSH_AFTER_REF, before image $FLUSH_BEFORE_IMAGE, after image $FLUSH_AFTER_IMAGE; pause ${PAUSE_S} s after every trace and the image"
echo "Trace: $TRACE_ARGS"
echo "Image: $IMAGE_TOML"
[[ $L1_STRIP == 1 && $SCREEN != 1 ]] && echo "Level-1 strips (reference and new): $L1_ARGS"
case $FLUSH_BETWEEN in trace|set|none) ;; *) echo "ERROR: FLUSH_BETWEEN must be trace, set or none"; exit 1;; esac
if [[ ${DRY_RUN:-0} != 1 ]]; then
  read -r -p "Start the packet now? (with CHECK_TRACES=1 the script runs until the image is queued) [y/N] " answer
  [[ $answer == [yY] ]] || { echo "nothing sent"; exit 0; }
fi

apply() {   # apply <NAME=VALUE | acmNNN:NAME=VALUE> ...: queue the parameter changes
  local change target spec
  for change in "$@"; do
    target=$POOL; spec=$change
    [[ $change == *:* ]] && { target=${change%%:*}; spec=${change#*:}; }
    q "daq['${spec%%=*}'] = ${spec#*=}" "$target"
  done
}

run_packet() {   # the packet's commands; returns 0, or check_phase's 2 (restart) / 1 (stop)
  # 0. standalone mode, then the nominal configuration (the sequencer is uploaded in standalone).  (The external
  #    clock is set once per session by hand; the daq.use_external_clock() line was removed on the DAQ, 2026-11-02.)
  q "daq.set_standalone()" "$POOL"
  nominal

  # 1. the reference: nominal, or (REF_CHANGES) the previous packet's settings with their own settling traces; a flush,
  #    the reference strip, then the reference traces, so that they end right before the change
  if (( ${#REF_CHANGES[@]} )); then
    apply "${REF_CHANGES[@]}"
    if (( N_SETTLE_TRACES > 0 )); then traces "${PACKET_ID}_settleref" "$N_SETTLE_TRACES" "${TRACE_CHANNELS%% *}"; fi
  fi
  flushes "$FLUSH_BEFORE_REF"
  strip "${PACKET_ID}_l1ref" || return $?
  traces "${PACKET_ID}_ref" "$N_REF_SETS"
  check_phase "${PACKET_ID}_ref" "$N_REF_SETS" || return $?
  flushes "$FLUSH_AFTER_REF"

  # 2. change the parameters (from a non-nominal reference: through a reset to nominal first, so no reference knob
  #    is left behind)
  if (( ${#REF_CHANGES[@]} )); then nominal; fi
  apply "${CHANGES[@]}"

  # 3. settling traces (first channel; the first acquisition recompiles the sequencer), then the new traces right
  #    after the change, the new strip, a flush
  if (( N_SETTLE_TRACES > 0 )); then traces "${PACKET_ID}_settle" "$N_SETTLE_TRACES" "${TRACE_CHANNELS%% *}"; fi
  traces "${PACKET_ID}_new" "$N_SETS"
  check_phase "${PACKET_ID}_new" "$N_SETS" || return $?
  if (( SCREEN )); then            # trace-only packet: back to nominal, done
    nominal
    return 0
  fi
  strip "${PACKET_ID}_l1new" || return $?
  flushes "$FLUSH_BEFORE_IMAGE"

  # 4. master/slave mode and the image (take_image.py queues the take_image call itself)
  q "daq.master($LEADER)" "$POOL"
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    echo "(cd $CDAQ_DIR && $PYTHON scripts/take_image.py $IMAGE_TOML)"
  else
    (cd "$CDAQ_DIR" && $PYTHON scripts/take_image.py "$IMAGE_TOML")
  fi
  pause

  # 5. back to standalone FIRST (a flush -- e.g. the repeat queue's -- would hang in master mode), the post-image
  #    traces (still at the new settings; checked and retaken, but never a reason to restart the packet: its image
  #    is taken), then a flush and the nominal configuration
  q "daq.set_standalone()" "$POOL"
  if (( N_POST_SETS > 0 )); then
    traces "${PACKET_ID}_post" "$N_POST_SETS"
    local rc=0
    check_phase "${PACKET_ID}_post" "$N_POST_SETS" || rc=$?
    (( rc == 1 )) && return 1
    (( rc == 2 )) && echo "  ${PACKET_ID}_post: many damaged post-image traces; the packet is kept"
  fi
  flushes "$FLUSH_AFTER_IMAGE"
  nominal
}

[[ ${DRY_RUN:-0} == 1 ]] || mkdir -p "$MARK_DIR"
BASE_ID=$PACKET_ID
attempt=0
while :; do
  DAMAGED=0
  write_record
  status=0
  run_packet || status=$?
  if (( status == 2 )); then
    echo "$PACKET_ID: $DAMAGED damaged traces (more than $MAX_DAMAGED): stopping it, resetting the boards, restarting the packet"
    set_status "restarted: $DAMAGED damaged traces"
    q "daq.set_standalone()" "$POOL"
    nominal
    attempt=$(( attempt + 1 ))
    if (( attempt > MAX_RESTARTS )); then
      echo "ERROR: $BASE_ID restarted $MAX_RESTARTS times and still too many damaged traces: giving up"
      exit 1
    fi
    PACKET_ID="${BASE_ID}_r$attempt"
    continue
  elif (( status != 0 )); then
    set_status "failed: stopped while waiting"
    exit 1
  fi
  set_status "complete"
  break
done

if [[ ${DRY_RUN:-0} == 1 ]]; then
  echo "(dry run: nothing was sent)"
else
  echo "$PACKET_ID queued to the end ($DAMAGED damaged traces retaken).  Watch it with: cdaq print queues (and cdaq print logs)"
  echo "Afterwards: python src/make_manual_manifest.py $RECORD_DIR/$PACKET_ID.json --run-dir <raw-file directory>"
fi

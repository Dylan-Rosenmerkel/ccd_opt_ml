#!/usr/bin/env bash
# =============================================================================
# manual_packet.sh -- one packet by hand, with the group's cdaq commands
#
#   start from nominal -> standalone -> reference traces -> change parameters ->
#   traces -> master/slave -> image (img.toml) -> back to nominal -> standalone
#
# Every line below is a plain `cdaq qadd "..." <target>` (or take_image.py), the same
# commands typed by hand.  The cdaq queue runs them ONE AFTER THE OTHER in the order
# they are added, so the script only queues them; it finishes in seconds while the
# queue works for ~15-20 min.  Watch with:  cdaq print queues / cdaq print logs
#
# Usage (on the DAQ machine, ccd-cdaq environment active):
#   DRY_RUN=1 bash manual_packet.sh     # print every command, send nothing  <- do this first
#   bash manual_packet.sh               # check, show the plan, ask, then queue
#
# Edit the SETTINGS block, nothing else.
#
# Duration with the defaults: 18 flushes (~30 s each: 3 initial, one between trace sets, one after
# the change, one at the end) = 9 min, 60 traces (~2 min), the img.toml image (16x1000x500,
# 4.8 min readout + its EXPOSURE): about 16 min.
# The nominal .bcf files are read at the start too, so every packet starts from the same state.
# =============================================================================
set -euo pipefail

# ------------------------------- SETTINGS ------------------------------------
CDAQ_DIR="${CDAQ_DIR:-$HOME/Soft_JHU/cdaq}"     # where scripts/take_image.py lives
PYTHON="${PYTHON:-python}"                      # or "pixi run python"
IMAGE_TOML="image_params/tc1/img.toml"          # relative to CDAQ_DIR (16 x 1000 x 500, 1x1, prescan 5300)
LEADER=101                                      # master board for the image
POOL=acmpool

N_INITIAL_FLUSHES=3       # flushes before the first reference trace set
N_REF_SETS=5              # reference trace sets (nominal parameters)
N_SETS=10                 # trace sets at the new parameters
FLUSH_BETWEEN="set"       # "set":   a flush between trace sets, none between the channels of a set
                          # "trace": a flush before every trace;  "none": no flushes between traces
FLUSH_AFTER_CHANGE=1      # flushes after changing the parameters
TRACE_CHANNELS="0 1 2 3"  # one trace per channel (take_trace records one channel at a time)

# Trace that reads its pixels like the image (docs/packet_design.md, section 6): the same skips
# (NDCM 500) and no binning (1x1), as img.toml; its own small size: 2 x 5 pixels after a
# 100-pixel prescan burst, no register clear, no exposure.  0.19 s, ~11 MB per channel.
# Every cut is given, so nothing is inherited from the last image.
# If the image's NDCM / NSBIN / NPBIN change, change them here too.
TRACE_ARGS="subroutine='AcquireImage', EXPOSURE=0, NROW=2, NCOL=5, NPBIN=1, NSBIN=1, NDCM=500, BOOLCLR=0, NDCMPRE=0, NROWPRE=0, NROWPOST=0, NCOLPRE=100, NCOLPOST=0"
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

# Nominal configuration files, read at the start and at the end:  board=file
BCF="acm101=module_PA03.bcf acm102=ccd_PQ04.bcf"
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
"

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
check_and_show() {
  printf '%s\n' "${CHANGES[@]}" | awk -v nominal="$(echo $NOMINAL)" -v pool="$POOL" '
    function rule(a, op, b) {          # a op b on the complete configuration; prints and flags a violation
      if (!(a in new) || !(b in new)) return
      x = new[a] + 0; y = new[b] + 0
      ok = (op == ">") ? (x > y) : (op == ">=") ? (x >= y) : (op == "<") ? (x < y) : (x <= y)
      if (!ok) { print "ERROR: " a " = " new[a] " must be " op " " b " = " new[b]; bad = 1 }
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
          for (k in new) {
            if (k ~ /^(V[123]|H[123]|TG|SW|OG|RG|DG)_[A-C12]_[HL]$/) range(k, -10, 10)
            if (k ~ /^V[123]_[A-C]_L$/) for (t in new) if (t ~ /^TG_/) rule(k, ">", t)
            if (k ~ /^H[123]_[A-C]_L$/) for (t in new) if (t ~ /^TG_/) rule(k, "<", t)
          }
          rule("H3_A_L", ">", "SW_1_H"); rule("H3_A_L", ">", "SW_2_H"); rule("H3_B_L", ">", "SW_1_H"); rule("H3_B_L", ">", "SW_2_H")
          if (bad) exit 1
          print "Parameter changes (nominal -> new):"; for (i = 1; i <= m; i++) print line[i] }'
}

# --------------------------------- sending -----------------------------------
q() {   # queue one command:  q "<python text>" <target>
  if [[ ${DRY_RUN:-0} == 1 ]]; then echo "cdaq qadd \"$1\" $2"; else cdaq qadd "$1" "$2"; fi
}
flushes() { for ((i = 0; i < $1; i++)); do q "daq.flush_ccd()" "$POOL"; done; }
traces() {   # traces <prefix> <number of sets>: one trace per channel per set, every board of the pool
  local first=1 s ch
  for ((s = 1; s <= $2; s++)); do
    for ch in $TRACE_CHANNELS; do
      if (( ! first )); then
        if [[ $FLUSH_BETWEEN == trace ]] || [[ $FLUSH_BETWEEN == set && $ch == ${TRACE_CHANNELS%% *} ]]; then
          flushes 1
        fi
      fi
      q "daq.take_trace(fname='$1$s', ch=$ch, $TRACE_ARGS)" "$POOL"
      first=0
    done
  done
}
nominal() { for pair in $BCF; do q "daq.read_config('${pair#*=}')" "${pair%%=*}"; done; }

# --------------------------------- the plan ----------------------------------
check_and_show || exit 1
echo "Traces: $N_REF_SETS reference + $N_SETS new sets x channels [$TRACE_CHANNELS], flush between: $FLUSH_BETWEEN"
echo "Trace: $TRACE_ARGS"
echo "Image: $IMAGE_TOML"
case $FLUSH_BETWEEN in trace|set|none) ;; *) echo "ERROR: FLUSH_BETWEEN must be trace, set or none"; exit 1;; esac
if [[ ${DRY_RUN:-0} != 1 ]]; then
  read -r -p "Queue all commands now? [y/N] " answer
  [[ $answer == [yY] ]] || { echo "nothing sent"; exit 0; }
fi

# 0. start from the nominal configuration, standalone mode, external clock
nominal
q "daq.set_standalone()" "$POOL"
q "daq.use_external_clock()" "$POOL"          # set once per session; harmless if already set

# 1. reference traces at nominal
flushes "$N_INITIAL_FLUSHES"
traces trace_ref "$N_REF_SETS"

# 2. change the parameters
for change in "${CHANGES[@]}"; do
  target=$POOL; spec=$change
  [[ $change == *:* ]] && { target=${change%%:*}; spec=${change#*:}; }
  q "daq['${spec%%=*}'] = ${spec#*=}" "$target"
done
flushes "$FLUSH_AFTER_CHANGE"

# 3. traces at the new parameters
traces trace_new "$N_SETS"

# 4. master/slave mode and the image (take_image.py queues the take_image call itself)
q "daq.master($LEADER)" "$POOL"
if [[ ${DRY_RUN:-0} == 1 ]]; then
  echo "(cd $CDAQ_DIR && $PYTHON scripts/take_image.py $IMAGE_TOML)"
else
  (cd "$CDAQ_DIR" && $PYTHON scripts/take_image.py "$IMAGE_TOML")
fi

# 5. back to standalone FIRST (a flush -- e.g. the repeat queue's -- would hang in master mode),
#    then the nominal configuration and a flush
q "daq.set_standalone()" "$POOL"
nominal
flushes 1

if [[ ${DRY_RUN:-0} == 1 ]]; then
  echo "(dry run: nothing was sent)"
else
  echo "All commands queued.  Watch them with: cdaq print queues   (and cdaq print logs)"
fi

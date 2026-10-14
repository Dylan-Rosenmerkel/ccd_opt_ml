# Session context: 2026-10-14

Continuation of `2026-10-13_session_context.md` (which ends with the 5 + 10 trace-set packets, the scan
time estimates, the image-time check and the noise question).

## 1. Inputs from the user

* **Nominal configuration** (the .bcf content):
  * `[bias]`: VDD1/2 −19, VR1/2 −3, VDRAIN1/2 −23, VSUB 45;
  * `[clock_offsets]`: OFS_RS1/2, OFS_CLK −10;
  * `[clock_rails]`: V1/V2/V3 A/B/C high 9.9, low 7.9/7.7; TG A/B 7.4/7.4; H1/H2/H3 high 9.5 (H2_C 9.0),
    low 6.7/6.5; SW 1.0/−6.0, OG 0.0/−6.0, RG 9.0/3.0, DG 0.0/−5.0, for amplifier sets 1 and 2;
  * `[sequencer]`: vertical `Vtransfer22`, horizontal `HtransferUU`, ch0..3 = CCD_D, C, A, B.
* **Configuration files:** acm101 reads `module_PA03.bcf`, acm102 reads `ccd_PQ04.bcf`
  (`cdaq qadd "daq.read_config('<file>')" <board>`).
* **The group's command style:**
  * `cdaq qadd "daq['H1_A_L'] = 6.2" acmpool`;
  * `daq.take_trace(...)`, `daq.master(101)`, `daq.use_external_clock()`, `daq.set_standalone()`;
  * `python scripts/take_image.py image_params/tc1/img.toml`.
* `img.toml` is now **16×1000×500**, 1×1, prescan 5300 (updated by the user).
* **The user is compiling the voltage and timing limits** of these CCDs and the relations that must hold
  between them. These go into `configs/limits.toml` / `ccd_limits.py` (still `reviewed = false`) and
  the checks of the manual script.

## 2. `daq_scripts/manual_packet.sh`: one packet by hand

Plain bash (works with bash 3.2+). It only **queues** plain `cdaq qadd` commands; the cdaq queue runs
them in order. Current default sequence:

1. `daq.read_config('module_PA03.bcf')` → acm101, `daq.read_config('ccd_PQ04.bcf')` → acm102 (start
   from nominal); `daq.set_standalone()`; `daq.use_external_clock()`;
2. 3 flushes;
3. **5 reference trace sets**: one trace per channel 0–3 to `acmpool`, no flush between the channels of
   a set, **a flush between sets**;
4. the parameter changes, `daq['NAME'] = value` (`NAME=VALUE` for both boards via acmpool,
   `acm101:NAME=VALUE` for one board); a flush;
5. **10 trace sets** at the new parameters, a flush between sets;
6. `daq.master(101)`; `take_image.py image_params/tc1/img.toml`;
7. `daq.set_standalone()` **first** (a flush, e.g. the repeat queue's, would hang in master mode), then
   both nominal .bcf files again and a flush.

**Trace, matched to the image** (`docs/packet_design.md` section 6):
`daq.take_trace(fname=..., ch=c, subroutine='AcquireImage', EXPOSURE=0, NROW=2, NCOL=5, NPBIN=1, NSBIN=1,
NDCM=500, BOOLCLR=0, NDCMPRE=0, NROWPRE=0, NROWPOST=0, NCOLPRE=100, NCOLPOST=0)`.
* It has the same skips and binning as img.toml, with its own small size; every cut is given, so
  nothing is inherited from the last image; 0.19 s, ~11 MB per channel.
* The group's earlier command (5×5 binning, 100 skips) is kept as a comment.
* If the image's NDCM/NSBIN/NPBIN change, `TRACE_ARGS` must be changed with them.

**Checks before anything is sent:**
* unknown parameter names (typos);
* values that are not numbers;
* any rail with HIGH < LOW after the change (the TG_B case of 2026-10-12).

It then shows a plan table and asks for confirmation. `DRY_RUN=1 bash manual_packet.sh` prints every
command and sends nothing. The checks were tested with deliberate mistakes.

**Default run:** 18 flushes, 60 traces, the 16-row image: **about 16 min**.

**Settings** (top of the file):
* `N_REF_SETS` (5), `N_SETS` (10), `N_INITIAL_FLUSHES` (3);
* `FLUSH_BETWEEN`: `"set"` (default), `"trace"` (a flush before every trace, ~44 min with the old
  35-row image) or `"none"` (~9 min);
* `FLUSH_AFTER_CHANGE`, `TRACE_CHANNELS`, `TRACE_ARGS`, `CHANGES`, `IMAGE_TOML`, `BCF`, `CDAQ_DIR`,
  `PYTHON`.

`daq_scripts/manual_packet_commands.txt` holds the 92 commands of the default run, for reference.

**History of the defaults today** (on request):
1. first the group's 5×5/100-skip trace with a flush between sets only;
2. then the matched trace with a flush before every trace;
3. then the matched trace with no flush between the channels of a set (current).

Commits: `035d4bd`, `85f6852`, `d4b9d6a`, `3edf635`.

## 3. Not yet known or checked

* What `read_config` resets: the mode is set explicitly afterwards, but whether it touches the
  sequencer constants or the clock is unknown.
* Whether `take_image.py` without `-m` keeps the master mode set just before (it should: the commands
  run in queue order).
* The script has only been dry-run on the Mac; nothing has been sent to the DAQ.

## 4. Open

* Run `DRY_RUN=1 bash daq_scripts/manual_packet.sh` on the DAQ machine, then a first real packet; then
  process it (σ at 500 skips ≈ 0.26 e⁻? λ per row; matched traces decode without layout warnings).
* Put the user's limits and relations into `configs/limits.toml` / `ccd_limits.py` and the manual
  script's checks.
* Decide whether to keep the flush between trace sets (test sets with and without it, compare the
  trace-to-trace scatter).
* Everything open from 2026-10-13 (WADERs JSON with per-row dark current; the first scan design).

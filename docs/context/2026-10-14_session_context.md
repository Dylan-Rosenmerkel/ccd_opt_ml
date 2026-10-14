# Session context: 2026-10-14

Continuation of `2026-10-13_session_context.md`.

## 1. Manual packet script with the group's commands

The user gave:
* the nominal configuration (`[bias]`, `[clock_offsets]`, `[clock_rails]`; .bcf files `module_PA03.bcf`
  for acm101 and `ccd_PQ04.bcf` for acm102);
* the group's command style: `cdaq qadd "daq['H1_A_L'] = 6.2" acmpool`, `daq.take_trace(...)`,
  `daq.master(101)`, `daq.use_external_clock()`, `daq.set_standalone()`, `daq.read_config('<bcf>')`,
  and `python scripts/take_image.py image_params/tc1/img.toml`.

New `daq_scripts/manual_packet.sh` (plain bash, works with bash 3.2+). It **queues** plain `cdaq qadd`
commands in this order:
1. read the nominal .bcf on each board; standalone; external clock;
2. 3 flushes; 5 reference trace sets (one trace per channel, to acmpool), a flush between sets;
3. the parameter changes (`NAME=VALUE` for both boards, `acm101:NAME=VALUE` for one board); a flush;
4. 10 trace sets, a flush between sets;
5. `daq.master(101)`, then `take_image.py img.toml`;
6. `daq.set_standalone()` **before** reading the nominal .bcf again (a flush in master mode would
   hang), then a flush.

* **Checks before anything is sent:** unknown parameter names (typos), non-numbers, and any rail with
  HIGH < LOW after the change (the TG_B case of 2026-10-12). A plan table is shown and confirmation
  asked. `DRY_RUN=1` prints every command.
* **Trace command:** the user's (2×5, 5×5 binning, 100 skips) plus all cuts set to 0. The matched
  alternative (1×1, 500 skips, prescan 100) is given as a comment.
* **Duration:** ~21 min with img.toml (35×1000×500 = 10.6 min); ~15 min with 16 rows.
* `daq_scripts/manual_packet_commands.txt`: the 91 commands of the default run, for reference.
* **Not checked:** what `read_config` resets exactly (the mode is set explicitly afterwards); the real
  voltage limits and relations, which the user is compiling.

**Update (same day, on request):**
* **The trace now reads its pixels like the image** (the matched trace of `docs/packet_design.md`):
  `NROW=2, NCOL=5, NPBIN=1, NSBIN=1, NDCM=500, NCOLPRE=100, BOOLCLR=0`, all other cuts 0 (0.19 s,
  ~11 MB per channel). The group's 5×5/100-skip command is kept as a comment.
* **A flush before every trace**, between the channels of a set too (`FLUSH_BETWEEN="trace"`;
  `"set"` or `"none"` are the alternatives). That is 63 flushes and 60 traces, about 44 min with
  img.toml (about 21 min with `"set"`).
* `manual_packet_commands.txt` was regenerated (167 commands).

# Collaboration-meeting deck (Winter 2026), figures and build scripts

The deck itself is `~/Desktop/CollabMeet_filled.pptx` (built from the outline `~/Desktop/CollabMeet.pptx`, which is left unchanged).

- `figs_trace.py <repo> <outdir>`: trace-breakdown figures from raw October traces and the Nov 3 packet L1_23.
- `figs_more.py <repo> <outdir> <repo-checkout-with-amp_phys-results> [names]`: amplifier fingerprints, transfer function, tree diagram, noise result, usable/unusable images, dark charge per amplifier, GP/EI/multi-amplifier illustrations, BO results.
- `eqs.py <outdir>`: equation images (matplotlib mathtext).
- `build_deck.py <outline.pptx> <out.pptx> <dir with fig/ and eq/>`: fills the outline (python-pptx).
Numbers come from model_math.tex, trace_math.tex, amp_phys/results (physics_sheet, trap_from_trace_v3) and the talk deck.

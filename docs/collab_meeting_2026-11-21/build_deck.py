"""Fill the CollabMeet.pptx outline: text, figures, tables, equation images and simple diagrams."""
import copy, os, sys
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

SRC, OUT, D = sys.argv[1], sys.argv[2], sys.argv[3]
FIG, EQ = os.path.join(D, "fig"), os.path.join(D, "eq")
prs = Presentation(SRC)
S = list(prs.slides)
GRAY = RGBColor(0x59, 0x59, 0x59)
BLUE = RGBColor(0x1F, 0x4E, 0x79)
LIGHT = RGBColor(0xDB, 0xE9, 0xF6)
GREEN = RGBColor(0xE2, 0xF0, 0xD9)
ORANGE = RGBColor(0xFB, 0xE5, 0xD6)
RED = RGBColor(0xC0, 0x00, 0x00)


# ------------------------------------------------------------------ helpers
def clear(slide, keep_title=True):
    for sh in list(slide.shapes):
        if keep_title and sh.is_placeholder and sh.placeholder_format.type in (1, 3):   # TITLE, CENTER_TITLE
            continue
        sh._element.getparent().remove(sh._element)


def title(slide, text, size=30):
    t = [sh for sh in slide.shapes if sh.is_placeholder and sh.placeholder_format.type in (1, 3)][0]
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.25), Inches(12.1), Inches(0.95)
    tf = t.text_frame
    tf.clear()
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text; r.font.size = Pt(size)
    p.alignment = PP_ALIGN.LEFT


def _bullet(p, level):
    pPr = p._p.get_or_add_pPr()
    marL = 285750 + 285750 * level
    pPr.set("marL", str(marL)); pPr.set("indent", str(-228600))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    bu = etree.SubElement(pPr, qn("a:buChar")); bu.set("char", "•" if level == 0 else "–")


def text(slide, x, y, w, h, items, size=16, color=None, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, space=4):
    """items: list of str or (str, opts) with opts: level (bullet level; None = no bullet), bold, size, color, italic.
    A str may contain **bold** spans."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    first = True
    for it in items:
        s, o = (it, {}) if isinstance(it, str) else it
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        p.alignment = align
        p.space_after = Pt(o.get("space", space))
        lvl = o.get("level", 0)
        if lvl is not None:
            _bullet(p, lvl)
        parts = s.split("**")
        for i, part in enumerate(parts):
            if not part:
                continue
            r = p.add_run(); r.text = part
            r.font.size = Pt(o.get("size", size if (lvl in (None, 0)) else size - 2))
            r.font.bold = (i % 2 == 1) or o.get("bold", False)
            r.font.italic = o.get("italic", False)
            c = o.get("color", color)
            if c is not None:
                r.font.color.rgb = c
    return tb


def pic(slide, name, x, y, w=None, h=None, folder=None):
    path = os.path.join(folder or FIG, name)
    if w is None and h is None:
        raise ValueError
    W, H = Image.open(path).size
    if w is not None and h is not None:          # fit inside the box, keep aspect, centre
        s = min(w / W, h / H); ww, hh = W * s, H * s
        return slide.shapes.add_picture(path, Inches(x + (w - ww) / 2), Inches(y + (h - hh) / 2), Inches(ww), Inches(hh))
    if w is not None:
        return slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))
    return slide.shapes.add_picture(path, Inches(x), Inches(y), height=Inches(h))


def eq(slide, name, x, y, scale=0.8, center_w=None):
    path = os.path.join(EQ, name + ".png")
    W, H = Image.open(path).size
    w, h = W / 300 * scale, H / 300 * scale
    if center_w is not None:
        x = x + (center_w - w) / 2
    slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


def table(slide, data, x, y, w, col_w=None, size=13, row_h=0.32, header_fill=LIGHT, bold_first_col=False, align_num=True):
    rows, cols = len(data), len(data[0])
    gt = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(row_h * rows))
    tbl = gt.table
    # plain style: no banding, thin borders
    tblPr = tbl._tbl.tblPr
    tblPr.set("bandRow", "0"); tblPr.set("firstRow", "1")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"      # "No Style, Table Grid"
    if col_w:
        for j, cw in enumerate(col_w):
            tbl.columns[j].width = Inches(cw)
    for i in range(rows):
        tbl.rows[i].height = Inches(row_h)
        for j in range(cols):
            c = tbl.cell(i, j); c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            val = data[i][j]
            tf = c.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]; p.text = ""
            parts = str(val).split("**")
            for k, part in enumerate(parts):
                if not part:
                    continue
                r = p.add_run(); r.text = part; r.font.size = Pt(size)
                r.font.bold = (k % 2 == 1) or i == 0 or (bold_first_col and j == 0)
            p.alignment = PP_ALIGN.CENTER if (j > 0 and align_num) else PP_ALIGN.LEFT
            if i == 0 and header_fill is not None:
                c.fill.solid(); c.fill.fore_color.rgb = header_fill
            else:
                c.fill.background()
    return gt


def box(slide, x, y, w, h, txt, fill=LIGHT, size=14, bold=False, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = RGBColor(0x40, 0x40, 0x40); sh.line.width = Pt(1)
    sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.06)
    lines = txt.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = line; r.font.size = Pt(size if i == 0 else size - 2)
        r.font.bold = bold and i == 0; r.font.color.rgb = RGBColor(0, 0, 0)
    return sh


def arrow(slide, x0, y0, x1, y1, color=RGBColor(0x40, 0x40, 0x40), width=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y0), Inches(x1), Inches(y1))
    c.line.color.rgb = color; c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd")); tail.set("type", "triangle"); tail.set("w", "med"); tail.set("len", "med")
    return c


def caption(slide, x, y, w, s, size=11, h=0.35):
    return text(slide, x, y, w, h, [(s, {"level": None, "italic": False})], size=size, color=GRAY)


def cite(slide, s, y=7.05):
    return text(slide, 0.6, y, 12.1, 0.35, [(s, {"level": None})], size=10, color=GRAY)


# Slide indices in base.pptx (27 slides):
# 0 title, 1 intro, 2 full trace, 3 V, 4 H, 5 skips, 6 one skip/noise, 7 PCA, 8 amps, 9 oxide1, 10 oxide2, 11 oxide3,
# 12 mutual1, 13 mutual2, 14 table, 15 ML intro, 16 data set, 17 decision tree, 18 EDT, 19 noise, 20 usability,
# 21 why not, 22 GP, 23 multi-amp, 24 BO loop, 25 BO tests, 26 BO results

# ------------------------------------------------------------------ 1: introduction
s = S[1]; clear(s); title(s, "Introduction: Driving Questions")
text(s, 0.6, 1.4, 7.3, 5.3, [
    "The **trace**: the raw output of the CCD amplifier, sampled at 15 MHz while the CCD is read out",
    ("Under 1 s for all 4 amplifiers (≈0.2 s each)", {"level": 1}),
    ("≈3 million samples for 2 rows × 5 pixels × 500 skips", {"level": 1}),
    ("Nothing is averaged away: every clock edge and every skip is in it", {"level": 1}),
    ("**Three driving questions**", {"space": 6}),
    ("1.  Can the trace tell different amplifiers, or different operating voltages, apart?", {"level": None, "size": 17}),
    ("2.  Can the trace tell us something about the amplifiers themselves?", {"level": None, "size": 17}),
    ("3.  Can the trace predict image noise, or help find the best voltages faster?", {"level": None, "size": 17}),
], size=18, space=10)
pic(s, "two_skips.png", 8.1, 2.1, w=4.8)
caption(s, 8.1, 4.6, 4.8, "Two of the 500 measurements (skips) of one pixel, as seen in the trace.", size=12)
cite(s, "Skipper CCD readout: Tiffenberg et al., PRL 119, 131802 (2017). Data: PA03 module, board 101 (4 amplifiers), Oct–Nov 2026.")

# ------------------------------------------------------------------ 2: full trace
s = S[2]; clear(s); title(s, "Trace Breakdown: One Row of a Trace")
pic(s, "regions_overview.png", 0.6, 1.25, w=12.1)
caption(s, 0.6, 5.15, 12.1, "The regions are found automatically from the sequencer flag bits recorded with every sample, so the decoding works for any geometry and timing.", size=12)
for x, head, body, fill in [
        (0.6, "Vertical transfers", "The parallel clocks move the image down one row. They couple into the output: a large, repeatable waveform.", RGBColor(0xCF, 0xE0, 0xEF)),
        (4.75, "Horizontal transfers", "The serial register moves the next pixel to the amplifier, one pixel per transfer.", RGBColor(0xFD, 0xE6, 0xD2)),
        (8.9, "Skip region", "Each pixel is measured 500 times without destroying the charge. This is the measurement itself.", RGBColor(0xD5, 0xEC, 0xD4))]:
    box(s, x, 5.65, 3.8, 1.3, "", fill=fill)
    text(s, x + 0.1, 5.7, 3.6, 1.2, [(head, {"level": None, "bold": True, "size": 16}), (body, {"level": None, "size": 13})])


def two_fig_slide(s, ttl, bullets, left, right, cap):
    clear(s); title(s, ttl)
    text(s, 0.6, 1.3, 12.1, 1.4, bullets, size=17, space=6)
    pic(s, left, 0.6, 2.75, w=5.9)
    pic(s, right, 6.8, 2.75, w=5.9)
    caption(s, 0.6, 6.25, 12.1, cap, size=12, h=0.6)


# ------------------------------------------------------------------ 3-5
two_fig_slide(S[17], "Trace Breakdown: Vertical Clocks", [
    "V1–V3 and the transfer gate (TG) move every row one step toward the serial register; one transfer takes 360 µs",
    "The clock edges couple into the output: a waveform of ≈40 000 ADU peak-to-peak, repeated at every transfer",
    "Features: peak-to-peak, lowest and highest point, change from the first to the last transfer"],
    "v_zoom.png", "v_setting.png",
    "Right: lowering the V low rails by 0.5 V (TG by 0.4 V) deepens only this waveform, peak-to-peak 41 → 46 kADU. Bands start at the spike that begins each transfer. Board 101, ch0, 12 Oct.")
two_fig_slide(S[18], "Trace Breakdown: Horizontal Clocks", [
    "H1–H3 move the charge along the serial register to the sense node: one transfer per pixel, 126 µs each",
    "The largest waveform in the trace (≈130 000 ADU peak-to-peak); it sets how the output looks before the first skip",
    "Features: peak-to-peak, lowest and highest point, scatter between transfers, change from first to last transfer"],
    "h_zoom.png", "h_setting.png",
    "Right: lowering the H low rails by 0.5 V changes only the horizontal waveform, peak-to-peak 126 → 144 kADU. Bands start at the spike that begins each transfer. 12 Oct.")
two_fig_slide(S[19], "Trace Breakdown: Skips Region", [
    "After the horizontal transfer the pixel's charge is measured 500 times (skips), 32.6 µs per skip",
    "The baseline drifts by ≈3 000 ADU across a pixel, but each skip subtracts its own pedestal, so the CDS stays flat",
    "Features: CDS noise, drift along the pixel, first-skip transient, baseline recovery, pixel charge level"],
    "skips_zoom.png", "skips_drift.png",
    "Board 101, ch0, nominal voltages, 3 Nov. Left: each spike is one skip cycle. Right: the same pixel, one point per skip.")

# ------------------------------------------------------------------ 6: one skip and its noise
s = S[20]; clear(s); title(s, "Trace Breakdown: One Skip and Its Noise")
pic(s, "two_skips.png", 0.6, 1.3, w=5.9)
text(s, 0.6, 4.35, 5.9, 0.95, [
    "Pedestal (10 µs) → charge onto the sense node → signal (10 µs) → reset",
    "CDS = mean(pedestal) − mean(signal): one measurement",
], size=14)
eq(s, "sigma1", 0.8, 5.45, scale=0.95)
text(s, 0.6, 6.05, 5.9, 0.9, [("**Single-skip noise σ₁:** compare consecutive skips of one pixel; the charge cancels, the noise does not. Board 101 ch0: σ₁ ≈ 5.1 ADU.", {"level": None})], size=14)
pic(s, "allan.png", 7.0, 1.25, w=5.4)
text(s, 7.0, 4.85, 5.7, 0.95, [("**Allan deviation:** split the pixel's skips into blocks of N, average each block (ȳ₁, ȳ₂, …), and compare neighbouring blocks:", {"level": None})], size=13)
eq(s, "allan", 7.3, 5.55, scale=0.8)
text(s, 7.0, 6.05, 5.7, 0.95, [("Dashed: pure white noise, σ₁/√N. The 500-skip images (star) sit on that line: averaging keeps working all the way to 500 skips.", {"level": None})], size=13)
cite(s, "Allan, Proc. IEEE 54, 221 (1966). Board 101 ch0, 210 traces (2–3 Nov). The readout level drifts slowly during each pixel; a straight line is removed from each pixel first, since a drift would make this statistic grow with N.")

# ------------------------------------------------------------------ 7: settings apart (one PCA panel)
s = S[4]; clear(s); title(s, "Telling Settings Apart")
text(s, 0.6, 1.4, 5.4, 4.8, [
    "Describe each trace by 85 numbers (noise, waveform shapes, timing, levels)",
    "**PCA** finds the two combinations of those numbers in which the traces differ most (PC1, PC2); each trace is one dot",
    "PCA is never told the settings",
    "Repeats of a setting land together; different settings land apart",
    "Hide one trace and find its most similar trace: same setting in **44–46 of 47** cases",
], size=16, space=10)
pic(s, "pca_ch0.png", 6.3, 1.4, w=6.4)
caption(s, 6.3, 6.25, 6.4, "Board 101 ch0, 12 Oct: four voltage changes, each traced 3 times.", size=12)

# ------------------------------------------------------------------ 8: amplifiers apart
s = S[5]; clear(s); title(s, "Telling Amplifiers Apart")
pic(s, "amps.png", 0.6, 1.3, w=12.1)
text(s, 0.6, 5.7, 12.1, 1.35, [
    "Same voltages, each amplifier traced 5 times in a row (8 Oct): the repeats agree, the amplifiers differ (ch1's reset pulse decays ≈150 ns faster)",
    "Is a number good for this? Compare how much it **differs between two amplifiers** with how much it **wobbles between repeats**. 25 of our 141 numbers tell every pair of amplifiers clearly apart (difference ≥ 3 × the wobble)",
], size=15)

# ------------------------------------------------------------------ 9: oxide 1, the noise model
s = S[21]; clear(s); title(s, "Amplifier Oxide Quality (1): A Model of the Noise")
text(s, 0.6, 1.3, 6.2, 0.4, [("**What the trace gives:** σ₁, the noise of one skip", {"level": None})], size=16)
eq(s, "s1int", 0.8, 1.8, scale=0.8)
text(s, 0.6, 2.55, 6.2, 0.6, [("S(f): noise spectrum at the amplifier output. |H(f)|²: how much of each frequency survives our measurement.", {"level": None})], size=13)
text(s, 0.6, 3.25, 6.2, 2.4, [
    ("**|H|² is a product of three steps:**", {"level": None}),
    ("average over a window of length τ: sinc²(fτ), removes fast noise", {"level": 0}),
    ("pedestal − signal, Δt = 13.8 µs apart: 4 sin²(πfΔt), removes slow noise", {"level": 0}),
    ("difference of consecutive skips, T = 32.6 µs apart (÷√2): 2 sin²(πfT), removes even slower noise and the pixel charge", {"level": 0}),
], size=14)
text(s, 0.6, 5.7, 6.2, 0.4, [("**Noise model:** white (thermal) noise + 1/f noise (oxide traps)", {"level": None})], size=14)
eq(s, "Sf", 0.8, 6.15, scale=0.85)
pic(s, "transfer.png", 7.0, 1.4, w=5.7)
caption(s, 7.0, 4.95, 5.7, "Only noise between ≈10 kHz and ≈500 kHz reaches σ₁. At low frequency |H|² ∝ f⁴, so even 1/f noise gives a finite σ₁.", size=12, h=0.7)
cite(s, "Window average and pedestal − signal: Hopkinson & Lumb, J. Phys. E 15, 1214 (1982), Table 1 'differential averager'. The skip-difference factor is ours.")

# ------------------------------------------------------------------ 10: oxide 2, measuring K
s = S[22]; clear(s); title(s, "Amplifier Oxide Quality (2): Measuring the 1/f Level K")
text(s, 0.6, 1.3, 6.3, 2.0, [
    ("**1.** From one trace, compute σ₁ using only the last τ samples of each 10 µs window: τ = 2, 3, 4, 6, 8, 9.3 µs", {"level": None}),
    ("**2.** For each τ, compute how much white and how much 1/f noise passes the filter:", {"level": None}),
], size=15)
eq(s, "iwif", 0.8, 2.55, scale=0.72)
text(s, 0.6, 3.25, 6.3, 0.9, [("**3.** Fit the six points with two numbers, S_w (white) and K (1/f):", {"level": None})], size=15)
eq(s, "fit", 0.8, 3.8, scale=0.68)
text(s, 0.6, 4.4, 6.3, 0.9, [("White noise shrinks as the window gets longer (∝ 1/τ); 1/f noise barely changes. The shape of σ₁²(τ) separates them.", {"level": None})], size=14)
table(s, [["board 101", "ch0", "ch1", "ch2", "ch3"],
          ["K (ADU²)", "2.5", "4.0", "4.2", "5.1"],
          ["white level (nV/√Hz)", "14.2", "12.7", "14.0", "13.0"]],
      0.6, 5.45, 6.3, col_w=[2.3, 1.0, 1.0, 1.0, 1.0], size=13, row_h=0.36)
pic(s, "sigtau.png", 7.1, 1.4, w=5.6)
caption(s, 7.1, 5.2, 5.6, "One trace of board 101 ch1 (Nov 10–12): six measured points and the two-number fit.", size=12, h=0.6)
cite(s, "Medians over 8 traces per amplifier. Board 101 also carries narrow interference lines (~120 kHz, 1.19 MHz) that bias this K low; under study.")

# ------------------------------------------------------------------ 11: oxide 3, K -> N_t
s = S[23]; clear(s); title(s, "Amplifier Oxide Quality (3): From K to the Oxide Trap Density")
text(s, 0.6, 1.3, 6.5, 0.4, [("**1. Convert K from ADC counts to volts at the transistor gate**", {"level": None})], size=15)
eq(s, "kvg", 0.9, 1.75, scale=0.95)
text(s, 0.6, 2.3, 6.5, 1.6, [
    ("**G = 0.875 ADU per µV**: the readout amplifies the CCD output 27.34 times, and one ADC count is 31.25 µV, so 1 µV at the CCD → 27.34 / 31.25 = 0.875 counts (design values)", {"level": 0}),
    ("**A = 0.85**: the output transistor passes 85 % of its gate voltage to its output (estimated from the layout)", {"level": 0}),
], size=13)
text(s, 0.6, 4.0, 6.5, 0.4, [("**2. McWhorter: 1/f noise from charges in oxide traps**", {"level": None})], size=15)
eq(s, "mcw", 0.9, 4.45, scale=0.95)
eq(s, "nt", 3.6, 4.45, scale=0.95)
text(s, 0.6, 5.15, 6.5, 1.8, [
    ("N_t: traps per energy per volume. WL = 345 µm²: gate area. T = 130 K. λ_t = 0.1 nm: how deep a carrier tunnels", {"level": 0}),
    ("C_ox = 0.454 fF/µm²: capacitance per area of the gate insulator (50 nm oxide + 50 nm nitride):", {"level": 0}),
], size=13)
eq(s, "cox", 1.0, 6.35, scale=0.7)
table(s, [["board 101", "N_t (10¹⁶ eV⁻¹ cm⁻³)"],
          ["ch0", "1.1"], ["ch1", "1.8"], ["ch2", "1.9"], ["ch3", "2.3"]],
      7.6, 1.4, 5.1, col_w=[2.3, 2.8], size=14, row_h=0.4)
text(s, 7.6, 3.6, 5.1, 3.3, [
    ("For comparison: bulk CMOS n-MOS with a 12 nm oxide, N_t ≈ 2 × 10¹⁶ (Ioannidis et al.)", {"level": 0}),
    ("Caveats: other 1/f-like noise in K pushes N_t up; the interference lines on board 101 push it down; C_ox and A are design or model values (15 % in C_ox → 30 % in N_t)", {"level": 0}),
], size=13)
cite(s, "McWhorter, in Semiconductor Surface Physics (1957); Christensson et al., Solid-State Electron. 11, 797 (1968); Ioannidis et al., Solid-State Electron. 95, 28 (2014); permittivities 3.9 and 7.5: Sze & Ng (2007).")

# ------------------------------------------------------------------ 12: couplings, how
s = S[24]; clear(s); title(s, "Clock Couplings to the Amplifier Input (1): How")
text(s, 0.6, 1.25, 12.1, 0.5, [("Idea: when a clock line switches by ΔV, the output settles to a new level. If that charge sits on the amplifier input, it reads exactly like electrons.", {"level": None})], size=15)
box(s, 0.6, 1.95, 2.2, 0.85, "Clock line k\n(H1, H2, H3)", fill=ORANGE)
box(s, 3.7, 1.95, 2.2, 0.85, "Sense node /\namplifier input", fill=LIGHT)
box(s, 6.8, 1.95, 2.2, 0.85, "Output transistor\nand readout", fill=LIGHT)
box(s, 9.9, 1.95, 2.8, 0.85, "Trace (ADU)", fill=GREEN)
arrow(s, 2.8, 2.37, 3.7, 2.37); arrow(s, 5.9, 2.37, 6.8, 2.37); arrow(s, 9.0, 2.37, 9.9, 2.37)
text(s, 2.8, 1.95, 0.9, 0.35, [("C_k", {"level": None, "bold": True})], size=13, align=PP_ALIGN.CENTER)
text(s, 0.6, 3.05, 7.1, 3.9, [
    ("**Step 1.** The sequencer file shows that each 18 µs state of a horizontal transfer switches exactly one phase: H3↓, H2↑, H1↓, H3↑, H2↓, H1↑, then a quiet state", {"level": None}),
    ("**Step 2.** In each state the output drifts to a new level (τ = 8.2 µs, the H line itself); fit that settled level, not the first jump", {"level": None}),
    ("**Step 3.** Across ≈100 voltage settings, fit the settled step against that phase's swing → ADU per volt (β_k)", {"level": None}),
    ("**Step 4.** Convert with the image gain g:", {"level": None}),
], size=14, space=6)
eq(s, "ck2", 1.0, 5.1, scale=0.9)
pic(s, "fig_H_state.png", 7.95, 3.0, w=4.75)
text(s, 7.95, 5.75, 4.75, 1.3, [
    ("**Worked example, H2, ch0:** β ≈ 26 500 ADU per volt of H2 swing ÷ g = 1.41 ADU/e⁻ → ≈ 18 800 e⁻ per volt; × q → **3.01 fF** (mean of the four amplifiers 3.09 fF)", {"level": None}),
    ("Reading the output 0.2–0.8 µs after the edge, long before it settles, had given 0.01 fF.", {"level": None, "size": 11}),
], size=13, space=4)

# ------------------------------------------------------------------ 13: couplings, results
s = S[25]; clear(s); title(s, "Clock Couplings to the Amplifier Input (2): Results")
table(s, [["line", "coupling (fF-eq.)", "4 amplifiers", "checked by"],
          ["H1", "3.41 ± 0.06", "3.33–3.47", "up = down within 1 %"],
          ["H2", "3.09 ± 0.07", "3.01–3.14", "up = down within 1 %"],
          ["H3", "3.49 ± 0.06", "3.42–3.54", "up step; down step 3.26"]],
      0.6, 1.35, 7.6, col_w=[1.1, 1.9, 1.7, 2.9], size=13, row_h=0.4)
caption(s, 0.6, 3.4, 7.6, "Board 101, mean of the four amplifiers. ± = fit systematics (start time taken conservatively, fit window); the amplifier range is shown separately.", size=11, h=0.45)
pic(s, "recon_ch0.png", 0.6, 3.85, w=7.6)
caption(s, 0.6, 6.4, 7.6, "Validation: couplings fitted on other settings predict the whole horizontal-transfer waveform (red) to 0.6 % rms; the residual (green, ×10) is only at the fast edge spikes.", size=11, h=0.6)
text(s, 8.55, 1.35, 4.2, 5.6, [
    ("**Effective couplings:** the step holds through the 18 µs state and follows VDD like the image gain, so it enters at the amplifier input; a single-CCD board (PQ04) is ≈18× weaker, so the size comes from the 4-CCD module's routing", {"level": 0}),
    ("**Not in the images:** the H clocks do not switch during the skips", {"level": 0}),
    ("**Not presented:** RG (a turn-off kick that drains back to VR, not stored charge) and V1–V3 (steps do not follow the swing)", {"level": 0}),
    ("**On hold:** reset-gate overlap limit, buried-channel depth", {"level": 0}),
], size=13, space=8)

# ------------------------------------------------------------------ 14: amplifier quantities (trace only, board 101)
s = S[26]; clear(s); title(s, "Amplifier Quantities from the Trace Alone")
table(s, [["board 101, Nov 10–12", "ch0", "ch1", "ch2", "ch3"],
          ["single-skip noise σ₁ at τ = 9.3 µs (ADU)", "5.2", "5.4", "5.7", "5.8"],
          ["white noise level (nV/√Hz)", "14.2", "12.7", "14.0", "13.0"],
          ["1/f noise level K (ADU²)", "2.5", "4.0", "4.2", "5.1"],
          ["oxide trap density N_t (10¹⁶ eV⁻¹ cm⁻³)", "1.1", "1.8", "1.9", "2.3"],
          ["reset settling time (µs)", "1.11", "0.93", "0.93", "1.16"]],
      1.4, 1.5, 10.5, col_w=[5.3, 1.3, 1.3, 1.3, 1.3], size=16, row_h=0.55)
text(s, 1.4, 5.0, 10.5, 1.9, [
    "Every number here comes from 0.2 s traces, with no image",
    "Converting to volts (white level, N_t) uses the design readout gain G; N_t also uses the model values A and C_ox",
    "The clock couplings, which also need one image gain, are on the previous slides",
], size=15)

# ------------------------------------------------------------------ 15: ML intro
s = S[28]; clear(s); title(s, "Can Traces Predict Image Noise and Other Image Quantities?")
text(s, 0.6, 1.3, 12.1, 2.4, [
    "Goal: use the traces, the quantities derived from them and the voltages to predict the image **before** it is taken",
    "We tried many machine-learning models: Gaussian processes, gradient-boosted trees, neural-network ensembles, CNNs on the trace curves, deep-kernel and sparse GPs",
    "This talk focuses on two:",
    ("**Extremely Randomized Decision Trees (EDT)**: predict image noise, and whether an image will be usable", {"level": 1, "size": 16}),
    ("**Gaussian Process (GP)**: learn how dark charge responds to the voltages, then search for the best voltages", {"level": 1, "size": 16}),
], size=17)
box(s, 0.9, 4.4, 2.3, 0.9, "Trace (0.2 s)", fill=LIGHT, size=15)
box(s, 0.9, 5.7, 2.3, 0.9, "Voltage settings", fill=ORANGE, size=15)
box(s, 4.0, 4.4, 2.6, 0.9, "Features\n141 numbers + physics", fill=LIGHT, size=15)
box(s, 7.4, 4.9, 2.2, 1.1, "Model\nEDT or GP", fill=GREEN, size=16, bold=True)
box(s, 10.4, 4.9, 2.3, 1.1, "Prediction\nnoise, usable?, dark charge", fill=RGBColor(0xF2, 0xF2, 0xF2), size=15)
arrow(s, 3.2, 4.85, 4.0, 4.85); arrow(s, 6.6, 4.85, 7.4, 5.25); arrow(s, 3.2, 6.15, 7.4, 5.7); arrow(s, 9.6, 5.45, 10.4, 5.45)

# ------------------------------------------------------------------ 16: data set
s = S[29]; clear(s); title(s, "Data Set")
text(s, 0.6, 1.3, 6.3, 0.5, [("**A packet** = one test of one voltage setting:", {"level": None})], size=16)
y0 = 1.9
for i, (t1, fill) in enumerate([("Reference traces\nat the nominal voltages", LIGHT), ("Set the new voltages", ORANGE),
                                ("Traces at the new setting\n(all amplifiers, 0.2 s each)", LIGHT), ("One 500-skip dark image", GREEN),
                                ("Image fit (WADERs) → labels\nσe noise, λ dark charge, g gain, usable?", RGBColor(0xF2, 0xF2, 0xF2))]):
    box(s, 0.9, y0 + i * 0.95, 5.0, 0.72, t1, fill=fill, size=14)
    if i < 4:
        arrow(s, 3.4, y0 + i * 0.95 + 0.72, 3.4, y0 + (i + 1) * 0.95)
text(s, 7.2, 1.3, 5.5, 2.5, [
    "**v3 night (Nov 10–12)**, PA03 board 101, amplifiers ch0, ch1, ch3",
    "18 voltages varied together: SW, RG, V1–V3 and H1–H3 high/low rails, VSUB, VDD, VR",
    "Settings drawn by Sobol sampling (spread evenly over the allowed range)",
], size=15)
table(s, [["", "settings", "packets", "use"],
          ["session G", "80 new + 14 nominal + 4 repeats", "98", "training"],
          ["session H", "24 new + 5 nominal", "29", "sealed test"]],
      7.2, 3.95, 5.5, col_w=[1.2, 2.3, 0.9, 1.1], size=12, row_h=0.42)
text(s, 7.2, 5.35, 5.5, 1.6, [
    "127 packets × 3 amplifiers = **381 images**; 82 were unusable",
    "**Sealed test:** session H was scored only after the model choice was frozen",
    "Each trace → features; the packet's traces are combined by the median",
], size=14)

# ------------------------------------------------------------------ 17: decision tree
s = S[30]; clear(s); title(s, "What Is a Decision Tree?")
pic(s, "tree.png", 0.6, 1.3, w=5.6)
pic(s, "tree_one.png", 0.9, 4.6, w=4.6)
text(s, 6.9, 1.3, 5.8, 5.7, [
    ("**Predicting:** start at the top, answer each yes/no question about the inputs, end in a leaf. The prediction is the average label of the training images in that leaf.", {"level": None, "space": 10}),
    ("**Growing it from training data:**", {"level": None}),
    ("1.  Put all training images in one node", {"level": None}),
    ("2.  Try questions of the form “input j < t ?”", {"level": None}),
    ("3.  Keep the question that splits the labels into the two most uniform groups (largest drop in spread):", {"level": None}),
], size=15, space=4)
eq(s, "split", 7.1, 3.75, scale=0.78)
text(s, 6.9, 4.3, 5.8, 1.5, [
    ("4.  Repeat in each new node; stop when a leaf has fewer than 3 images", {"level": None}),
    ("A single tree is easy to read but jumpy: its prediction is a step function, and a small change in the data can change the whole tree.", {"level": None, "size": 13}),
], size=15, space=6)
cite(s, "Breiman, Friedman, Olshen & Stone, Classification and Regression Trees (1984). Tree diagram: illustration with made-up numbers.")

# ------------------------------------------------------------------ 18: EDT
s = S[31]; clear(s); title(s, "Extremely Randomized Decision Trees (EDT)")
text(s, 0.6, 1.3, 6.4, 5.0, [
    ("**Grow one tree, step by step:**", {"level": None, "size": 16}),
    ("1.  Put all training images in one node", {"level": None}),
    ("2.  **Random:** pick a few inputs at random (half of the 41)", {"level": None}),
    ("3.  **Random:** for each picked input, pick one cut value t at random, anywhere between its smallest and largest value in the node", {"level": None}),
    ("4.  **Not random:** of these candidate questions, keep the one with the largest drop in spread Δ", {"level": None}),
    ("5.  Repeat in each new node until a leaf has fewer than 3 images", {"level": None}),
    ("**Then:** grow 400 trees, each with its own random picks, and average their predictions:", {"level": None, "space": 4}),
], size=15, space=7)
eq(s, "forest", 1.0, 4.55, scale=0.85)
text(s, 3.2, 4.7, 3.8, 0.6, [("T_b: tree b; B = 400", {"level": None})], size=14)
pic(s, "tree_forest.png", 7.2, 1.35, w=5.5)
text(s, 7.2, 4.4, 5.5, 2.5, [
    ("**Why it works:** each random tree is rough, but their errors differ, so the average is smooth and stable (right)", {"level": 0}),
    ("Works with ≈150 training images, needs no scaling of the inputs, and copes with many related inputs", {"level": 0}),
    ("Random forest: best cut and resampled images. EDT: random cut, all images; even more varied trees", {"level": 0}),
], size=14, space=8)
cite(s, "Geurts, Ernst & Wehenkel, Machine Learning 63, 3 (2006); Breiman, Machine Learning 45, 5 (2001); scikit-learn ExtraTreesRegressor. Right: illustration on made-up data.")

# ------------------------------------------------------------------ 19: noise prediction
s = S[9]; clear(s); title(s, "EDT on Our Data: Predicting Image Noise")
text(s, 0.6, 1.3, 7.0, 3.2, [
    "**Target:** ln σe, the read noise of a 500-skip image",
    "**Inputs (41):**",
    ("9 trace physics values: σ₁ at 6 window lengths, white level S_w, 1/f level K, predicted 500-skip noise σ₅₀₀", {"level": 1}),
    ("31 voltages, and σ₅₀₀ divided by a gain predicted from the voltages", {"level": 1}),
    "**Model:** EDT, 400 trees, ≥ 3 images per leaf, half the inputs tried at each split",
    "**Test, new amplifier:** train on session G of two amplifiers, test on the sealed session H of the third; rotate over all three",
], size=15)
text(s, 0.6, 4.55, 2.0, 0.5, [("**Score:**", {"level": None})], size=15)
eq(s, "err", 1.6, 4.5, scale=0.85)
text(s, 0.6, 5.35, 7.0, 1.5, [
    "**Standard:** the same EDT given the voltages only (what we know without a trace)",
    "The trace halves the error: **18.2 % → 8.6 %** (95 % interval 6.2–11.2 %)",
], size=15)
pic(s, "noise_bars.png", 7.9, 1.4, w=4.8)
caption(s, 7.9, 5.25, 4.8, "Pooled over 67 sealed test images on three held-out amplifiers. With the amplifier seen in training: 6.5 %.", size=12, h=0.6)
cite(s, "Interval: bootstrap over the 23 session-H packets (Efron 1979). One board, one night, three amplifiers.")

# ------------------------------------------------------------------ 20: usability
s = S[10]; clear(s); title(s, "EDT on Our Data: Usable or Not?")
pic(s, "usable.png", 0.6, 1.3, w=7.4)
caption(s, 0.6, 4.8, 7.4, "Left: usable, nominal setting (3 Nov, ch0): clear 0, 1, 2 e⁻ peaks. Right: unusable (2 Nov, ch2), horizontal clock delays halved (1800 → 900 cycles): the fit fails (χ²/ndf = 11). At that setting ch0, ch1 and ch3 were still usable.", size=11, h=0.55)
text(s, 8.3, 1.3, 4.4, 3.6, [
    "**82 of 381** v3 images (22 %) could not be used: the fit failed or raised a flag (dark charge not constrained, bad χ²)",
    "**Usable** = fit converged, no flags, λ > 0.005 e⁻/pixel",
    "**Classifier:** EDT on the 18 voltages + 8 trace features → P(usable)",
    "In the search: never image a setting with P < 0.5",
], size=15)
table(s, [["sealed test (H)", "AUC", "unusable caught", "usable blocked"],
          ["voltages only", "0.990", "11 of 20", "0 of 67"],
          ["voltages + trace", "0.985", "**18 of 20**", "0 of 67"]],
      0.6, 5.4, 7.4, col_w=[2.3, 1.2, 2.0, 1.9], size=14, row_h=0.42)
text(s, 8.3, 5.3, 4.4, 1.6, [("The trace catches most broken settings before an image is spent on them, and blocks none of the good ones.", {"level": None})], size=15)
cite(s, "AUC (area under the ROC curve): the chance that a randomly chosen usable image gets a higher P(usable) than a randomly chosen unusable one. 1 = perfect, 0.5 = guessing. Trained on session G, tested once on H, test amplifier left out.")

# ------------------------------------------------------------------ 21: why not directly
s = S[11]; clear(s); title(s, "Can ML Find Good Settings Faster? Not Directly")
pic(s, "dc_amp.png", 0.6, 1.4, w=6.2)
text(s, 7.1, 1.3, 5.6, 4.3, [
    "We tried to predict an image's **dark charge (DC)** from its traces. Three problems:",
    ("**Each amplifier has its own DC level.** Same settings, medians 0.27 / 0.33 / 0.61 / 0.49 e⁻/pixel. Train on some amplifiers, predict a new one: the level comes out wrong", {"level": 1, "size": 15}),
    ("**The trace is too short to collect DC:** 0.2 s, against minutes of image readout", {"level": 1, "size": 15}),
    ("**DC also depends on the environment** (temperature, history, light), not only on the voltages", {"level": 1, "size": 15}),
], size=16)
text(s, 0.6, 5.85, 12.1, 1.0, [
    ("**Idea:** do not predict the level. Learn the **shape** of the response to the voltages (where the minimum is) from the amplifiers already measured, and get the level from one image.", {"level": None}),
], size=17)

# ------------------------------------------------------------------ 22: GP
s = S[32]; clear(s); title(s, "Learning the Shape of the Voltage Response (Gaussian Process)")
text(s, 0.6, 1.3, 2.3, 0.45, [("**What we minimize:**", {"level": None})], size=15)
eq(s, "J", 2.75, 1.35, scale=0.85)
text(s, 0.6, 1.85, 6.4, 0.85, [("A GP is a prior over smooth functions: J at any set of settings is jointly Gaussian, with a covariance set by how close the settings are (Matérn-5/2):", {"level": None})], size=14)
eq(s, "kernel", 0.7, 2.75, scale=0.8)
text(s, 0.6, 3.4, 6.4, 0.5, [("ℓⱼ: how far voltage j must change to change J (one per voltage); s²: size of the response", {"level": None})], size=13)
text(s, 0.6, 3.95, 6.4, 0.4, [("**Prediction** at a new setting x* from the measured images y:", {"level": None})], size=14)
eq(s, "post", 0.7, 4.4, scale=0.75)
text(s, 0.6, 4.9, 6.4, 0.4, [("**Training:** choose s², ℓⱼ and the noise n² by maximizing", {"level": None})], size=14)
eq(s, "lml", 0.7, 5.35, scale=0.85)
text(s, 0.6, 5.85, 6.4, 0.9, [("First term: fit the data; second: penalize a too-flexible model. About 20 numbers are fitted, no weights. K_y = K + n²I.", {"level": None})], size=13)
pic(s, "gp.png", 7.2, 1.45, w=5.5)
caption(s, 7.2, 4.85, 5.5, "Illustration in one dimension (made-up function). The real model has 18 voltages.", size=12, h=0.5)
cite(s, "Rasmussen & Williams, Gaussian Processes for Machine Learning (MIT Press, 2006).")

# ------------------------------------------------------------------ 23: multi-amplifier GP
s = S[33]; clear(s); title(s, "Multi-Amplifier GP Model")
eq(s, "mt", 0.8, 1.35, scale=1.0)
text(s, 0.6, 1.85, 6.3, 1.0, [
    ("f(x): shape shared by all amplifiers;  gₐ(x): amplifier a's own small deviation", {"level": 0}),
    ("cₐ: amplifier a's own level (its DC offset);  ε: image-to-image scatter", {"level": 0}),
], size=14)
eq(s, "cov", 0.7, 2.95, scale=0.7)
text(s, 0.6, 3.35, 6.3, 0.45, [("ρ: how alike the shapes are;  c²: how much the levels differ", {"level": None})], size=14)
text(s, 0.6, 3.9, 6.3, 1.7, [
    ("**Training, two steps** (usable images of the amplifiers already measured)", {"level": None}),
    ("1.  One GP on all of them, amplifier labels ignored: fit the shape (s², ℓⱼ, n²) by maximum likelihood", {"level": None}),
    ("2.  Keep those fixed; choose ρ and c² on a grid, by maximum likelihood", {"level": None}),
], size=14)
text(s, 0.6, 5.65, 6.3, 1.2, [("**Main assumption:** the response has a similar shape on every amplifier (minima in similar places), and a new amplifier is like the ones already measured.", {"level": None})], size=14)
pic(s, "multi.png", 7.1, 1.35, w=5.6)
caption(s, 7.1, 3.4, 5.6, "Illustration in one dimension (made-up response).", size=12)
text(s, 7.1, 3.9, 5.6, 2.9, [
    ("**Why this fixes the DC-offset problem**", {"level": None}),
    ("The level cₐ is learned for each amplifier, not predicted", {"level": 0}),
    ("A constant offset does not move the minimum, so the other amplifiers already rank the settings of a new one", {"level": 0}),
    ("The first image fixes its level", {"level": 0}),
], size=14)
cite(s, "Multi-task GP: Bonilla, Chai & Williams, NeurIPS (2007); transfer between related optimisations: Swersky, Snoek & Adams, NeurIPS (2013).")

# ------------------------------------------------------------------ 24: BO loop
s = S[34]; clear(s); title(s, "Bayesian Optimization (BO) Loop")
steps = [("1. Fit the multi-amplifier GP", "other amplifiers + images so far", LIGHT),
         ("2. Score candidate settings", "m(x), s(x) → expected improvement", LIGHT),
         ("3. Take traces of the top candidates", "≈0.2 s each, no image yet", ORANGE),
         ("4. EDT gate: P(usable) ≥ 0.5?", "if not, never image it", ORANGE),
         ("5. Take one image at the best", "fit it: σe, λ → J", GREEN)]
for i, (a, b, fill) in enumerate(steps):
    box(s, 0.6, 1.4 + i * 1.05, 5.4, 0.8, a + "\n" + b, fill=fill, size=15, bold=True)
    if i < 4:
        arrow(s, 3.3, 1.4 + i * 1.05 + 0.8, 3.3, 1.4 + (i + 1) * 1.05)
G = RGBColor(0x80, 0x80, 0x80)
for (x, y, w, h) in [(6.4, 1.75, 0.08, 4.25), (6.0, 5.92, 0.48, 0.08)]:
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); r.fill.solid(); r.fill.fore_color.rgb = G; r.line.fill.background()
fb = s.shapes.add_shape(MSO_SHAPE.LEFT_ARROW, Inches(6.05), Inches(1.6), Inches(0.5), Inches(0.4)); fb.fill.solid(); fb.fill.fore_color.rgb = G; fb.line.fill.background()
caption(s, 0.6, 6.65, 5.6, "Repeat until the largest expected improvement left is below 1 %.", size=12)
eq(s, "ei", 7.0, 1.4, scale=0.85)
text(s, 6.9, 1.9, 5.8, 1.0, [("b: best J imaged so far; Φ, φ: normal CDF and density. First term: the guess is good (exploit); second: the model is unsure (explore).", {"level": None})], size=13)
pic(s, "ei.png", 7.2, 2.95, w=5.2)
cite(s, "Expected improvement: Jones, Schonlau & Welch, J. Global Optim. 13, 455 (1998). Usability gate: constrained BO, Gelbart, Snoek & Adams, UAI (2014).")

# ------------------------------------------------------------------ 25: BO tests, step by step
s = S[35]; clear(s); title(s, "BO Tests: Replaying a Night We Already Measured")
text(s, 0.6, 1.3, 7.0, 5.0, [
    ("**1.** Pick one amplifier, e.g. ch0, and pretend we have never imaged it", {"level": None}),
    ("**2.** Its settings from one half of the night become the **menu**: 63–64 settings, including the ones that gave unusable images. The search sees their voltages, not their results", {"level": None}),
    ("**3.** The model may only learn from the **other two amplifiers**, in the **other half** of the night", {"level": None}),
    ("**4.** The search picks a setting from the menu. “Taking a trace” = reading its stored traces; “taking an image” = revealing its stored result (dark charge, usable or not)", {"level": None}),
    ("**5.** Count the images until the lowest dark charge found is within **2 %** of the lowest in the menu", {"level": None}),
    ("**6.** Repeat for every amplifier and half: **6 searches**, averaged", {"level": None}),
], size=15, space=9)
text(s, 0.6, 5.75, 7.0, 1.2, [
    ("**Compared with:** random order (image the menu in a random order), and by hand (always image the untested setting closest to the best so far)", {"level": None}),
], size=14)
gx, gy, cw, ch = 9.2, 1.9, 1.75, 0.75
text(s, gx, gy - 0.5, cw, 0.4, [("first half", {"level": None, "bold": True})], size=13, align=PP_ALIGN.CENTER)
text(s, gx + cw, gy - 0.5, cw, 0.4, [("second half", {"level": None, "bold": True})], size=13, align=PP_ALIGN.CENTER)
for r, amp in enumerate(["ch0 (new)", "ch1", "ch3"]):
    text(s, gx - 1.45, gy + r * ch + 0.18, 1.4, 0.4, [(amp, {"level": None})], size=13, align=PP_ALIGN.RIGHT)
    for c in range(2):
        if r == 0 and c == 0:
            fill, lab = ORANGE, "menu"
        elif r > 0 and c == 1:
            fill, lab = GREEN, "model learns"
        else:
            fill, lab = RGBColor(0xF2, 0xF2, 0xF2), "not used"
        box(s, gx + c * cw, gy + r * ch, cw - 0.08, ch - 0.08, lab, fill=fill, size=13, shape=MSO_SHAPE.RECTANGLE)
text(s, gx - 1.4, gy + 3 * ch + 0.25, 4.9, 1.6, [("One of the 6 searches. The new amplifier never shares a moment of the night with what the model learned from.", {"level": None})], size=14)

# ------------------------------------------------------------------ 26: BO results (dark charge only, plain)
s = S[14]; clear(s); title(s, "BO Results: Finding the Lowest Dark Charge")
table(s, [["method", "what it does", "images", "wasted"],
          ["random order", "image the menu in a random order", "38.5", "5.7"],
          ["by hand", "always image the untested setting closest to the best so far", "26.8", "3.4"],
          ["BO, new amplifier only", "the GP learns only from this amplifier's own images", "4.0", "0.5"],
          ["**BO, multi-amplifier + EDT gate**", "the GP also learns from the other amplifiers; the EDT blocks likely-unusable settings", "**1.8**", "**0**"]],
      0.6, 1.4, 7.4, col_w=[2.2, 3.4, 0.9, 0.9], size=13, row_h=0.62)
text(s, 0.6, 4.7, 7.4, 2.3, [
    ("**Test:** replay of the v3 night. Each amplifier in turn is “new”: its 63–64 measured settings (broken ones included) are the menu, and the model learns only from the other two amplifiers in the other half of the night. 6 searches, averaged", {"level": 0}),
    ("**images:** until the lowest dark charge found is within 2 % of the best in the menu; **wasted:** images at unusable settings", {"level": 0}),
], size=13, space=6)
pic(s, "results.png", 8.3, 1.4, w=4.4)
text(s, 8.3, 4.15, 4.4, 2.8, [
    ("**Using the other amplifiers cuts the search from ≈27 images by hand to ≈2, and no image is wasted**", {"level": None, "space": 10}),
], size=15)

# ===================================================== main talk: new and merged slides (run after slides_v4_old.py)
GREY = RGBColor(0xF2, 0xF2, 0xF2)

# ------------------------------------------------------------------ 0: title
s = S[0]
for sh in s.shapes:
    if sh.is_placeholder and sh.placeholder_format.type in (1, 3):
        tf = sh.text_frame; tf.clear(); r = tf.paragraphs[0].add_run(); r.text = "Low-Level Analysis: Reading CCD Amplifiers from the Raw Trace"
        r.font.size = Pt(36)
        sh.left, sh.width = Inches(0.8), Inches(11.7)

# ------------------------------------------------------------------ 1: motivation + headline results
s = S[1]; clear(s); title(s, "Why Look at the Raw Trace?")
text(s, 0.6, 1.35, 6.6, 5.5, [
    "Finding good CCD voltages takes many images, minutes each: by hand, ≈27 images to get close to the lowest dark charge in our test",
    "The **trace** is the amplifier's raw output at 15 MHz while the CCD is read out: 0.2 s per amplifier, no image needed",
    ("**Three questions**", {"space": 4}),
    ("1.  Can the trace tell settings and amplifiers apart?", {"level": None, "size": 16}),
    ("2.  What does it tell us about the amplifier itself?", {"level": None, "size": 16}),
    ("3.  Can it predict image quality and find good voltages faster? (ML)", {"level": None, "size": 16}),
], size=17, space=10)
box(s, 7.6, 1.45, 5.1, 3.0, "", fill=GREY, shape=MSO_SHAPE.RECTANGLE)
text(s, 7.8, 1.55, 4.75, 5.0, [
    ("**Headline results** (PA03 board 101)", {"level": None, "size": 17, "space": 10}),
    ("Traces from the same setting group together: matched in **44–46 of 47** cases", {"level": 0}),
    ("Image noise on a new amplifier predicted to **8.6 %** (18.2 % from the voltages alone)", {"level": 0}),
    ("Broken settings flagged before imaging: **18 of 20**, no good setting blocked", {"level": 0}),
    ("Lowest dark charge reached in **≈2 images** instead of ≈27 by hand", {"level": 0}),
], size=15, space=10)

# ------------------------------------------------------------------ 3: one skip (simple)
s = S[3]; clear(s); title(s, "The Measurement: One Skip")
pic(s, "two_skips.png", 0.6, 1.4, w=6.3)
caption(s, 0.6, 4.6, 6.3, "Two of the 500 skips of one pixel in the raw trace (board 101, ch0).", size=12)
text(s, 7.2, 1.35, 5.5, 3.6, [
    "Each pixel is measured **500 times** (skips) without destroying the charge",
    "One skip: pedestal (10 µs) → charge moved onto the sense node → signal (10 µs)",
    "**CDS** = mean(pedestal) − mean(signal): one measurement of the charge",
    "Averaging 500 skips gives sub-electron noise",
], size=16, space=8)
text(s, 7.2, 4.9, 5.5, 0.5, [("**Single-skip noise σ₁:**", {"level": None})], size=16)
eq(s, "sigma1", 7.4, 5.35, scale=0.85)
text(s, 7.2, 5.9, 5.5, 1.1, [("j: every neighbouring pair of skips in a pixel, pooled over all pixels; the charge cancels in each difference. ÷√2: each difference holds two skips' noise. Board 101: σ₁ ≈ 3.7 e⁻ per skip.", {"level": None})], size=13)

# ------------------------------------------------------------------ 6: amplifier highlights
s = S[6]; clear(s); title(s, "What the Trace Tells Us About the Amplifier")
table(s, [["from 0.2 s of trace", "board 101, ch0–ch3"],
          ["white noise level", "12.7–14.2 nV/√Hz"],
          ["1/f noise → oxide trap density", "1.1–2.3 × 10¹⁶ eV⁻¹cm⁻³\n(bulk CMOS ≈ 2 × 10¹⁶)"],
          ["H clock coupling to the amplifier input", "3.1–3.5 fF-equivalent\n(single-CCD board ≈ 18× less)"],
          ["reset settling time", "0.9–1.2 µs"]],
      0.6, 1.5, 7.3, col_w=[3.9, 3.4], size=15, row_h=0.75)
pic(s, "sigtau.png", 8.3, 1.45, w=4.4)
caption(s, 8.3, 4.4, 4.4, "Example: white and 1/f noise separated by changing the length of the measurement window.", size=12, h=0.6)
caption(s, 0.6, 5.5, 7.3, "Methods and caveats are in the backup slides.", size=13)

# ------------------------------------------------------------------ 7: data set + what we predict
s = S[7]; clear(s); title(s, "Data Set and What We Predict")
text(s, 0.6, 1.3, 6.0, 0.45, [("**A packet** = one test of one voltage setting:", {"level": None})], size=16)
y0 = 1.85
for i, (t1, fill) in enumerate([("Reference traces at the nominal voltages", LIGHT), ("Set the new voltages", ORANGE),
                                ("Traces at the new setting (all amplifiers)", LIGHT), ("One 500-skip dark image", GREEN),
                                ("Image fit → noise σe, dark charge λ, usable?", GREY)]):
    box(s, 0.9, y0 + i * 0.9, 5.4, 0.62, t1, fill=fill, size=14)
    if i < 4:
        arrow(s, 3.6, y0 + i * 0.9 + 0.62, 3.6, y0 + (i + 1) * 0.9)
text(s, 7.0, 1.3, 5.7, 2.5, [
    "**v3 night (Nov 10–12):** board 101, amplifiers ch0, ch1, ch3",
    "18 voltages varied together, spread evenly over the allowed range (Sobol sampling)",
    "127 packets × 3 amplifiers = **381 images**; 82 unusable",
], size=15)
table(s, [["", "packets", "use"], ["session G", "98", "training"], ["session H", "29", "sealed test, scored once"]],
      7.0, 3.75, 5.7, col_w=[1.6, 1.3, 2.8], size=13, row_h=0.4)
text(s, 7.0, 5.15, 5.7, 1.8, [
    "**We predict:** image noise, whether the image is usable, and dark charge",
    "Tried GPs, boosted trees, neural networks and CNNs; this talk uses **extremely randomized trees** and a **Gaussian process**",
], size=15)

# ------------------------------------------------------------------ 8: trees
s = S[8]; clear(s); title(s, "Extremely Randomized Decision Trees (EDT)")
pic(s, "tree.png", 0.6, 1.35, w=5.9)
caption(s, 0.6, 4.75, 5.9, "A decision tree (illustration, made-up numbers).", size=12)
pic(s, "tree_forest.png", 7.0, 1.35, w=5.7)
text(s, 0.6, 5.2, 12.1, 1.8, [
    "A **decision tree** asks yes/no questions about the inputs and predicts the average of the training images in the leaf it ends in",
    "One tree is jumpy. **EDT** grows 400 trees; at each split it tries a few randomly chosen inputs, each with a random cut value, and keeps the best. The trees differ, and their **average** is smooth and stable",
    "Works with ≈150 training images; no tuning needed",
], size=15, space=6)
cite(s, "Geurts, Ernst & Wehenkel, Machine Learning 63, 3 (2006). Details in backup.")

# ------------------------------------------------------------------ 12: multi-amplifier GP
s = S[12]; clear(s); title(s, "Learning the Voltage Response from Other Amplifiers")
pic(s, "gp.png", 0.6, 1.3, w=4.6)
pic(s, "multi.png", 5.5, 1.35, w=7.2)
text(s, 0.6, 4.25, 12.1, 2.7, [
    "**Gaussian process (GP):** a smooth guess of J = ln(dark charge) over the 18 voltages, with an uncertainty that shrinks near measured images (left)",
    "**Multi-amplifier model:** J for amplifier a = shared shape f(voltages) + the amplifier's own level cₐ (+ a small own deviation)",
    "The amplifiers already measured give the **shape** (where the minimum is); one image of the new amplifier gives its **level** (right)",
    "Assumption: the response has a similar shape on every amplifier",
], size=15, space=6)
cite(s, "Rasmussen & Williams (2006); multi-task GP: Bonilla et al., NeurIPS (2007); Swersky et al., NeurIPS (2013). Illustrations in one dimension, made-up response.")

# ------------------------------------------------------------------ 13: BO loop + test
s = S[13]; clear(s); title(s, "Searching for the Best Voltages (Bayesian Optimization)")
steps = [("1. Fit the multi-amplifier GP", LIGHT), ("2. Score every candidate by expected improvement (EI)", LIGHT),
         ("3. Take traces; EDT gate: skip if P(usable) < 0.5", ORANGE), ("4. Take one image at the best candidate", GREEN),
         ("5. Add the result; repeat", GREY)]
for i, (a, fill) in enumerate(steps):
    box(s, 0.6, 1.4 + i * 0.95, 5.6, 0.68, a, fill=fill, size=14, bold=True)
    if i < 4:
        arrow(s, 3.4, 1.4 + i * 0.95 + 0.68, 3.4, 1.4 + (i + 1) * 0.95)
G = RGBColor(0x80, 0x80, 0x80)
for (x, y, w, h) in [(6.55, 1.7, 0.08, 4.1), (6.2, 5.75, 0.43, 0.08)]:
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); r.fill.solid(); r.fill.fore_color.rgb = G; r.line.fill.background()
fb = s.shapes.add_shape(MSO_SHAPE.LEFT_ARROW, Inches(6.22), Inches(1.55), Inches(0.45), Inches(0.36)); fb.fill.solid(); fb.fill.fore_color.rgb = G; fb.line.fill.background()
eq(s, "ei", 7.1, 1.4, scale=0.82)
text(s, 7.0, 1.9, 5.7, 0.95, [("Expected improvement over b, the best J imaged so far. First term: the predicted value m is good (exploit); second: the uncertainty s is large (explore).", {"level": None})], size=13)
pic(s, "ei.png", 7.25, 2.95, w=5.2)

# ------------------------------------------------------------------ 15: summary
s = S[15]; clear(s); title(s, "Summary and Next Steps")
text(s, 0.6, 1.4, 12.1, 5.5, [
    ("**1. Telling apart:** traces from the same setting group together (44–46 of 47); 25 of 141 trace numbers separate every pair of amplifiers", {"level": None, "space": 18}),
    ("**2. Amplifier physics from 0.2 s:** white and 1/f noise, oxide trap density ≈ 1–2 × 10¹⁶ eV⁻¹cm⁻³, clock couplings ≈ 3 fF-equivalent", {"level": None, "space": 18}),
    ("**3. ML:** image noise on a new amplifier 18.2 % → 8.6 %; broken settings flagged 18 of 20, no good one blocked; lowest dark charge in ≈2 images instead of ≈27 by hand", {"level": None, "space": 28}),
    ("**Limits:** one board (PA03 101), one night, sibling amplifiers", {"level": None, "space": 12}),
    ("**Next:** a live search on a CCD the model has never seen; short scans to pin down the RG and V couplings", {"level": None}),
], size=20)

text(s, 0.6, 6.0, 12.1, 0.7, [("Thank you. Questions?", {"level": None, "bold": True})], size=26, align=PP_ALIGN.CENTER)

# ------------------------------------------------------------------ 16: backup divider
s = S[16]; clear(s)
t_ = [sh for sh in s.shapes if sh.is_placeholder and sh.placeholder_format.type in (1, 3)][0]
t_.left, t_.top, t_.width, t_.height = Inches(0.6), Inches(2.8), Inches(12.1), Inches(1.4)
tf = t_.text_frame; tf.clear(); p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "Backup Slides"; r.font.size = Pt(44)

# ------------------------------------------------------------------ 27 (backup): predicted 500-skip noise
s = S[27]; clear(s); title(s, "Predicted 500-Skip Noise σ₅₀₀ (a Trace Physics Input)")
text(s, 0.6, 1.3, 6.6, 0.75, [("**1.** From the trace: fit the white level S_w and the 1/f level K from σ₁ at six window lengths", {"level": None})], size=14)
text(s, 0.6, 2.05, 6.6, 0.75, [("**2.** Filter of a 500-skip image pixel: one CDS (window average, pedestal − signal), then the average of N = 500 skips, T = 32.6 µs apart:", {"level": None})], size=14)
eq(s, "H500", 0.75, 2.85, scale=0.64)
text(s, 0.6, 3.35, 6.6, 0.45, [("**3.** Pass the fitted noise spectrum through that filter:", {"level": None})], size=14)
eq(s, "s500", 0.75, 3.8, scale=0.62)
text(s, 0.6, 4.4, 6.6, 2.5, [
    ("Both parts fall almost as 1/√N: the CDS removes the slowest 1/f noise, so what is left averages down nearly like white noise. Here the 1/f part raises σ₅₀₀ by ≈1.4× over white alone", {"level": 0}),
    ("In ADU; the model converts it to electrons with a gain predicted from the voltages", {"level": 0}),
    ("Assumes white + 1/f only: interference lines and drifts slower than a pixel are not included, so the EDT learns the remaining correction", {"level": 0}),
], size=13, space=6)
pic(s, "sigma500.png", 7.4, 1.35, w=5.3)
caption(s, 7.4, 4.95, 5.3, "Board 101 ch1 (Nov 10–12): from one trace, σ₅₀₀ = 0.22 ADU predicted; the 500-skip images give 0.20 ADU (σe = 0.147 e⁻ × g = 1.39 ADU/e⁻).", size=12, h=0.8)

# ------------------------------------------------------------------ 36 (backup): BO results, every objective
s = S[36]; clear(s); title(s, "BO Results for Every Objective")
text(s, 0.6, 1.2, 12.1, 0.5, [("Images until within 2 % of the best usable setting (unusable images among them in brackets). Nothing known about the new amplifier; mean of 6 replay searches", {"level": None})], size=13)
table(s, [["method", "what it does", "noise", "noise + dark charge", "dark charge"],
          ["random order", "image the menu in a random order", "6.5 (1.5)", "31.0 (4.8)", "38.5 (5.7)"],
          ["by hand", "always image the untested setting closest to the best so far", "11.5 (4.4)", "24.0 (3.1)", "26.8 (3.4)"],
          ["BO, new amplifier only", "GP learns only from this amplifier's own images", "8.2 (2.2)", "3.2 (0.3)", "4.0 (0.5)"],
          ["multi-amplifier ranking", "image in the order the multi-amplifier model predicts; no updates", "4.3 (0.5)", "8.3 (1.0)", "8.3 (1.0)"],
          ["trace ranking", "take traces of every candidate; rank by EDT-predicted noise; image to confirm", "**1.3** (0)", "7.7 (0)", "7.3 (0)"],
          ["multi-amplifier BO + EDT gate", "the search on the main slides", "3.5 (0.2)", "**1.7** (0)", "**1.8** (0)"],
          ["hybrid", "first image = trace ranking's top pick, then multi-amplifier BO + gate", "2.8 (0)", "2.2 (0)", "**1.8** (0)"]],
      0.6, 1.75, 12.1, col_w=[2.7, 5.2, 1.3, 1.6, 1.3], size=12, row_h=0.42)
text(s, 0.6, 5.25, 6.4, 1.9, [
    ("**Objective:** J = w ln σe + (1 − w) ln λ, with w = 1 (noise), 0.5, 0 (dark charge)", {"level": 0}),
    ("**Starting from one nominal image:** by hand 1.5 / 26.8 / 26.0; multi-amplifier BO + gate 3.7 / 2.5 / 2.7", {"level": 0}),
], size=12, space=5)
text(s, 7.2, 5.25, 5.5, 1.9, [
    ("**Noise:** each candidate's trace measures its noise, so ranking by the trace finds it in ≈1 image", {"level": 0}),
    ("**Dark charge:** the trace cannot see charge; the best (low VSUB) is far from nominal, so by hand ≈27, BO ≈2", {"level": 0}),
    ("**Hybrid:** ≤ 3 images for every objective, no image wasted", {"level": 0}),
], size=12, space=5)


# ===================================================== v6: ML-focused main talk (run after slides_main.py)

# ------------------------------------------------------------------ 1: motivation, two questions, flow
s = S[1]; clear(s); title(s, "Why Look at the Raw Trace?")
text(s, 0.6, 1.35, 6.5, 5.5, [
    "Finding good CCD voltages takes many images, minutes each: by hand, ≈27 images to get close to the lowest dark charge in our test",
    "The **trace** is the amplifier's raw output at 15 MHz while the CCD is read out: 0.2 s per amplifier, no image needed",
    ("**Two questions**", {"space": 4}),
    ("1.  What numbers can we calculate from the trace?", {"level": None, "size": 16}),
    ("2.  With the voltages, can they predict image quality and find good voltages faster? (ML)", {"level": None, "size": 16}),
], size=17, space=10)
box(s, 7.5, 1.6, 2.4, 0.8, "Trace (0.2 s)", fill=LIGHT, size=15)
box(s, 10.3, 1.6, 2.4, 0.8, "Voltage settings", fill=ORANGE, size=15)
box(s, 7.5, 3.0, 2.4, 1.0, "Numbers from the trace\nnoise levels, signal sizes", fill=LIGHT, size=13)
box(s, 8.6, 4.6, 3.0, 0.8, "Model (EDT, GP)", fill=GREEN, size=16, bold=True)
box(s, 8.3, 5.95, 3.6, 0.95, "Prediction: noise, gain, usable?\nwhich setting to image next", fill=GREY, size=13)
arrow(s, 8.7, 2.4, 8.7, 3.0); arrow(s, 8.7, 4.0, 9.4, 4.6); arrow(s, 11.5, 2.4, 10.8, 4.6); arrow(s, 10.1, 5.4, 10.1, 5.95)

# ------------------------------------------------------------------ V / H clocks: point to the setting effect
for k, ttl, b3 in [(17, "Trace Breakdown: Vertical Clocks",
                    "Its size follows the V clock voltages (right): the trace records the setting before any image is taken"),
                   (18, "Trace Breakdown: Horizontal Clocks",
                    "Its size follows the H clock voltages (right); its peak-to-peak is one of the gain inputs (“signal sizes”)")]:
    s = S[k]
    tb = [sh for sh in s.shapes if not sh.is_placeholder and sh.has_text_frame and sh.text_frame.text.startswith(("V1", "H1"))][0]
    p = tb.text_frame.paragraphs[2]
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    p.runs[0].text = b3

# ------------------------------------------------------------------ 6: what we calculate (model inputs)
s = S[6]; clear(s); title(s, "What We Calculate from the Trace (Model Inputs)")
table(s, [["number (ADU)", "how", "used to predict"],
          ["single-skip noise σ₁ at 6 window lengths (2–9.3 µs)", "spread of neighbouring-skip differences", "noise"],
          ["white level S_w and 1/f level K", "fit of σ₁² vs window length (right)", "noise, usable?"],
          ["predicted 500-skip noise σ₅₀₀", "S_w and K passed through the 500-skip filter", "noise, usable?"],
          ["signal sizes: reset jump, H waveform peak-to-peak, skip pulse, CDS offset", "sizes of fixed steps in the waveform (charge × gain)", "gain, usable?"]],
      0.6, 1.4, 7.6, col_w=[3.3, 2.75, 1.55], size=13, row_h=0.7, align_num=False)
caption(s, 0.6, 5.0, 7.6, "From 0.2 s of trace at each setting. The models also get the voltages. Details of each step in the backup slides.", size=12, h=0.5)
pic(s, "sigtau.png", 8.5, 1.4, w=4.2)
caption(s, 8.5, 4.3, 4.2, "White and 1/f noise separated by changing the length of the measurement window (board 101 ch1).", size=12, h=0.6)
box(s, 0.6, 5.65, 12.1, 0.85, "", fill=GREY, shape=MSO_SHAPE.RECTANGLE)
text(s, 0.75, 5.75, 11.8, 1.0, [("**Work in progress:** the same numbers also describe the amplifier itself: oxide trap density from K (≈1–2 × 10¹⁶ eV⁻¹cm⁻³), clock couplings to the sense node (≈3 fF-equivalent), and possibly the reset-gate overlap and implant depth (backup).", {"level": None})], size=14)

# ------------------------------------------------------------------ 30: what is a decision tree (main)
s = S[30]; clear(s); title(s, "What Is a Decision Tree?")
pic(s, "tree.png", 0.6, 1.3, w=5.6)
pic(s, "tree_one.png", 0.9, 4.6, w=4.6)
text(s, 6.9, 1.25, 5.8, 3.0, [
    ("**Training data:** ≈150 images. Each has inputs x (trace numbers, voltages) and a label y (here ln σe)", {"level": None, "space": 8}),
    ("**Predicting:** answer each yes/no question from the top, end in a leaf; the prediction is the average label of the training images in that leaf", {"level": None, "space": 8}),
    ("**Growing it:**", {"level": None}),
    ("1.  Put all training images in one node", {"level": None}),
    ("2.  Try questions of the form “input j < t ?”", {"level": None}),
    ("3.  Keep the one that leaves the smallest squared error, Σ (y − its group's average)², i.e. the largest drop:", {"level": None}),
], size=14, space=3)
eq(s, "split", 7.1, 4.4, scale=0.78)
text(s, 6.9, 4.95, 5.8, 2.0, [
    ("4.  Repeat in each new node; stop when a leaf would have fewer than 3 images", {"level": None, "space": 8}),
    ("Var: spread of the labels in a group; |S|: number of images. A single tree is easy to read but jumpy: a step function, and a small change in the data can change the whole tree.", {"level": None, "size": 13}),
], size=14)
cite(s, "Breiman, Friedman, Olshen & Stone, Classification and Regression Trees (1984). Tree diagram: illustration with made-up numbers.")

# ------------------------------------------------------------------ 31: EDT (main)
s = S[31]; clear(s); title(s, "Extremely Randomized Decision Trees (EDT)")
text(s, 0.6, 1.25, 6.4, 4.0, [
    ("**Grow one tree:**", {"level": None, "size": 15}),
    ("1.  Put all training images in one node", {"level": None}),
    ("2.  **Random inputs:** from the pool of all 41 inputs (9 trace numbers, 31 voltage values, σ₅₀₀ ÷ gain), draw 20", {"level": None}),
    ("3.  **Random cuts:** for each of the 20, draw one cut t anywhere between its smallest and largest value in the node", {"level": None}),
    ("4.  **Not random:** of these 20 questions, keep the one with the largest drop in squared error", {"level": None}),
    ("5.  Repeat in each new node until a leaf would have fewer than 3 images", {"level": None}),
    ("**Then:** grow 400 trees, each with its own draws, all on the same images, and average:", {"level": None, "space": 4}),
], size=14, space=6)
eq(s, "forest", 1.0, 4.5, scale=0.85)
text(s, 3.2, 4.65, 3.8, 0.6, [("T_b: tree b; B = 400", {"level": None})], size=14)
text(s, 0.6, 5.45, 6.4, 0.9, [("Random forest: best cut on resampled images. EDT: random cuts, all images, so the trees differ even more.", {"level": None})], size=13)
pic(s, "tree_extrap.png", 7.2, 1.3, w=5.5)
text(s, 7.2, 4.55, 5.5, 2.4, [
    ("**Why it works:** each tree is rough, but their errors differ, so the average is smooth and stable", {"level": 0}),
    ("**Caveat: trees cannot extrapolate.** A prediction is an average of training labels, so outside the trained range it stays flat (grey). Trust it only for voltages and amplifiers like those it was trained on", {"level": 0}),
], size=14, space=8)
cite(s, "Geurts, Ernst & Wehenkel, Machine Learning 63, 3 (2006); Breiman, Machine Learning 45, 5 (2001); scikit-learn ExtraTreesRegressor. Right: made-up data.")

# ------------------------------------------------------------------ 9: noise + gain
s = S[9]; clear(s); title(s, "EDT on Our Data: Predicting Image Noise and Gain")
text(s, 0.6, 1.25, 6.7, 5.4, [
    ("**Goal:** the read noise σe (e⁻) of a 500-skip image, before taking it", {"level": None, "space": 9}),
    ("**1. From the trace:** 9 noise numbers in ADU: σ₁ at 6 windows, S_w, K, σ₅₀₀", {"level": None, "space": 9}),
    ("**2. Gain EDT:** predicts the gain (ADU per e⁻) from the voltages, to **3.6 %** on a new amplifier (≈5 % on 104 LSM amplifiers). Needed because the trace is in ADU", {"level": None, "space": 9}),
    ("**3. Formula, no ML for the noise:** σe ≈ σ₅₀₀ ÷ gain: 11.9 %", {"level": None, "space": 9}),
    ("**4. Noise EDT:** 41 inputs = 9 trace numbers + 31 voltage values + σ₅₀₀ ÷ gain → ln σe: **8.6 %** (95 % interval 6.2–11.2 %)", {"level": None, "space": 9}),
    ("**What the trace adds:** this amplifier's noise at this setting is measured, not guessed from the voltages: 18.2 % → 8.6 %", {"level": None}),
], size=15)
pic(s, "noise_gain_bars.png", 7.5, 1.3, w=5.2)
caption(s, 8.9, 4.4, 3.8, "Typical error = RMS of ln(predicted / measured). Test: train on session G of two amplifiers, test once on the sealed session H of the third; rotate (67 test images). Amplifier seen in training: noise 6.5 %.", size=12, h=1.0)
for i, (t1, fill) in enumerate([("9 trace noise numbers (ADU)", LIGHT), ("σ₅₀₀ ÷ g  (g from the gain EDT)", GREY), ("31 voltage values", ORANGE)]):
    box(s, 0.6, 4.85 + i * 0.68, 3.4, 0.52, t1, fill=fill, size=13)
    arrow(s, 4.0, 5.11 + i * 0.68, 4.7, 5.53)
box(s, 4.7, 5.18, 1.8, 0.7, "Noise EDT", fill=GREEN, size=15, bold=True)
arrow(s, 6.5, 5.53, 7.0, 5.53)
box(s, 7.0, 5.18, 1.6, 0.7, "ln σe", fill=LIGHT, size=15, bold=True)
cite(s, "Interval: bootstrap over the 23 session-H packets (Efron 1979). LSM: offset of ≈5 % between test stands. One board, one night, three amplifiers.")

# ------------------------------------------------------------------ 10: usability
s = S[10]; clear(s); title(s, "EDT on Our Data: Usable or Not?")
pic(s, "usable.png", 0.6, 1.3, w=7.4)
caption(s, 0.6, 4.8, 7.4, "Left: usable, nominal setting (3 Nov, ch0): clear 0, 1, 2 e⁻ peaks. Right: unusable (2 Nov, ch2), horizontal clock delays halved (1800 → 900 cycles): the fit fails (χ²/ndf = 11). At that setting ch0, ch1 and ch3 were still usable.", size=11, h=0.55)
text(s, 8.3, 1.3, 4.4, 3.0, [
    "**82 of 381** v3 images (22 %) unusable: the fit failed or raised a flag",
    "**Classifier:** EDT on the voltages + 8 trace numbers (σ₅₀₀ total and white only, S_w, K, 4 signal sizes) → P(usable)",
    "In the search: never image a setting with P < 0.5",
], size=14, space=6)
text(s, 8.3, 3.4, 4.4, 3.1, [
    ("**Same setting, different amplifiers?** Of 106 settings: 74 usable on all three amplifiers, 22 on none, **10 on only some** (mostly borderline: χ²/ndf just above 3, or dark charge too small to fit)", {"level": None, "space": 8}),
    ("The trace catches settings that break every amplifier. It missed both amplifier-specific failures: it sees a broken readout, it has not learned which amplifier will fail", {"level": None}),
], size=13)
table(s, [["sealed test (H)", "AUC", "unusable caught", "broken on all 3", "broken on 1", "usable blocked"],
          ["voltages only", "0.990", "11 of 20", "11 of 18", "0 of 2", "0 of 67"],
          ["voltages + trace", "0.985", "**18 of 20**", "18 of 18", "0 of 2", "0 of 67"]],
      0.6, 5.45, 7.4, col_w=[1.75, 0.8, 1.3, 1.3, 1.1, 1.15], size=12, row_h=0.42)
cite(s, "AUC: the chance that a random usable image gets a higher P(usable) than a random unusable one (1 = perfect, 0.5 = guessing). Trained on session G, tested once on H, test amplifier left out. Split by setting type: re-run of the classifier.")

# ------------------------------------------------------------------ 33: multi-amplifier GP, why it suits us
s = S[33]
for sh in s.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith("Why this fixes"):
        p = sh.text_frame.add_paragraph()
        _bullet(p, 0)
        p.space_before = Pt(8)
        for k2, part in enumerate("Uses the **≈200 images** of the amplifiers already measured; the new amplifier needs only a few".split("**")):
            r = p.add_run(); r.text = part; r.font.size = Pt(14); r.font.bold = (k2 % 2 == 1)
title(s, "Multi-Amplifier GP: Why It Suits Our Case")

# ------------------------------------------------------------------ 37: caveats
s = S[37]; clear(s); title(s, "Caveats: What Has Not Been Tested Yet")
table(s, [["", "tested so far", "still needed"],
          ["data", "one board (PA03 101), 3 amplifiers, one night: 381 images", "more amplifiers and CCDs"],
          ["search (BO)", "replay: choose among 63–64 settings already measured", "a live search on a new CCD in a new setup, over a broader range of settings"],
          ["voltage response", "similar shape on 3 sibling amplifiers", "check the same-shape assumption on more amplifiers"],
          ["other modules", "LSM (104 amplifiers): typical noise right, no ranking of amplifiers", "train or calibrate per test stand"]],
      0.6, 1.4, 12.1, col_w=[2.0, 5.0, 5.1], size=14, row_h=0.72, bold_first_col=True, align_num=False)
text(s, 0.6, 5.2, 12.1, 1.8, [
    "**Trees cannot extrapolate:** predictions hold only inside the voltages and kinds of amplifier they were trained on",
    "**Dark charge** depends on temperature and history, which are not inputs: on LSM (180 K vs 130 K) the predicted 0.33 e⁻/pixel was 0.53 measured",
], size=15, space=8)

# ------------------------------------------------------------------ 15: summary
s = S[15]; clear(s); title(s, "Summary and Next Steps")
text(s, 0.6, 1.4, 12.1, 5.0, [
    ("**1. From the trace:** 0.2 s gives the amplifier's white and 1/f noise, its predicted 500-skip noise and signal sizes, the inputs to the models", {"level": None, "space": 16}),
    ("**2. Noise:** image noise on a new amplifier predicted to 8.6 % (18.2 % from the voltages alone); gain to 3.6 %", {"level": None, "space": 16}),
    ("**3. Usability:** 18 of 20 broken settings flagged before imaging, no good one blocked; amplifier-specific failures not yet caught", {"level": None, "space": 16}),
    ("**4. Search:** lowest dark charge in ≈2 images instead of ≈27 by hand", {"level": None, "space": 24}),
    ("**Next:** a live search on a CCD the model has never seen; more amplifiers to test the same-shape assumption", {"level": None}),
], size=19)
text(s, 0.6, 6.0, 12.1, 0.7, [("Thank you. Questions?", {"level": None, "bold": True})], size=26, align=PP_ALIGN.CENTER)


# ===================================================== v7 (run after slides_v6_body.py); base40: 38 = new GP slide, 39 = formula backup

def pics_off(slide):
    for sh in list(slide.shapes):
        if sh.shape_type == 13:                      # picture
            sh._element.getparent().remove(sh._element)

# ------------------------------------------------------------------ 30: decision tree, what the drop means
s = S[30]; clear(s); title(s, "What Is a Decision Tree?")
pic(s, "tree.png", 0.6, 1.3, w=5.6)
pic(s, "tree_one.png", 0.9, 4.6, w=4.6)
text(s, 6.9, 1.2, 5.8, 2.6, [
    ("**Training data:** ≈150 images, each with inputs x (trace numbers, voltages) and a label y (e.g. ln σe)", {"level": None, "space": 6}),
    ("**Predicting:** answer the yes/no questions from the top; the prediction is the average label of the training images in the leaf", {"level": None, "space": 6}),
    ("**Growing it:**  1. all training images in one node.  2. try questions “input j < t ?”.  3. keep the one with the largest drop Δ in squared error:", {"level": None}),
], size=13, space=3)
eq(s, "split", 7.1, 3.55, scale=0.72)
text(s, 6.9, 4.05, 5.8, 3.0, [
    ("**What Δ means:** |S| Var(S) = Σ (y − ȳ)², the error if every image in S is predicted by S's average. Δ = error with one average for the node − error with one average per side", {"level": None, "space": 6}),
    ("**Example,** labels 1, 2, 8, 9:  one average (5): error 16 + 9 + 9 + 16 = 50.  Split {1, 2} | {8, 9}: averages 1.5 and 8.5, error 1, so **Δ = 49**.  Split {1, 8} | {2, 9}: error 49, Δ = 1. The first question is kept", {"level": None, "space": 6}),
    ("4.  Repeat in each new node; stop when a leaf would have fewer than 3 images", {"level": None, "space": 6}),
    ("A single tree is jumpy: a step function that changes a lot with small changes in the data.", {"level": None, "size": 12}),
], size=13)
cite(s, "Breiman, Friedman, Olshen & Stone, Classification and Regression Trees (1984). Tree diagram: illustration with made-up numbers.")

# ------------------------------------------------------------------ 31: EDT, general
s = S[31]; clear(s); title(s, "Extremely Randomized Decision Trees (EDT)")
text(s, 0.6, 1.25, 6.4, 4.0, [
    ("**Grow one tree:**", {"level": None, "size": 15}),
    ("1.  Put all training images in one node", {"level": None}),
    ("2.  **Random inputs:** from the pool of all inputs, draw a random subset (we draw half)", {"level": None}),
    ("3.  **Random cuts:** for each drawn input, one cut t, drawn anywhere between its smallest and largest value in the node", {"level": None}),
    ("4.  **Not random:** of these questions, keep the one with the largest drop Δ (previous slide)", {"level": None}),
    ("5.  Repeat in each new node until a leaf would have too few images (we use 3)", {"level": None}),
    ("**Then:** grow many trees (we use 400), each with its own draws, all on the same images, and average:", {"level": None, "space": 4}),
], size=14, space=6)
eq(s, "forest", 1.0, 4.5, scale=0.85)
text(s, 3.2, 4.65, 3.8, 0.6, [("T_b: tree b; B trees", {"level": None})], size=14)
text(s, 0.6, 5.45, 6.4, 0.9, [("Random forest: best cut on resampled images. EDT: random cuts, all images, so the trees differ even more.", {"level": None})], size=13)
pic(s, "tree_extrap.png", 7.2, 1.3, w=5.5)
text(s, 7.2, 4.55, 5.5, 2.4, [
    ("**Why it works:** each tree is rough, but their errors differ, so the average is smooth and stable", {"level": 0}),
    ("**Caveat: trees cannot extrapolate.** A prediction is an average of training labels, so outside the trained range it stays flat (grey). Trust it only for voltages and amplifiers like those it was trained on", {"level": 0}),
], size=14, space=8)
cite(s, "Geurts, Ernst & Wehenkel, Machine Learning 63, 3 (2006); Breiman, Machine Learning 45, 5 (2001); scikit-learn ExtraTreesRegressor. Right: made-up data.")

# ------------------------------------------------------------------ 9: noise EDT (no gain plot, no formula)
s = S[9]; clear(s); title(s, "EDT on Our Data: Predicting Image Noise")
text(s, 0.6, 1.2, 7.4, 2.6, [
    ("**Goal:** the read noise σe (e⁻) of a 500-skip image, before taking it", {"level": None, "space": 7}),
    ("**Test:** train on session G of two amplifiers, test once on the sealed session H of the third; rotate (67 test images)", {"level": None, "space": 7}),
    ("**New amplifier:** voltages only **18.2 %** → trace + voltages **8.6 %** (95 % interval 6.2–11.2 %)", {"level": None, "space": 7}),
    ("**Seen amplifier:** adding everything (also the 141 general trace features and the signal sizes) does best, **6.1 %**, but falls to 12.7 % on a new amplifier", {"level": None}),
], size=14)
for i, (t1, fill) in enumerate([("9 trace noise numbers (ADU)", LIGHT), ("σ₅₀₀ in e⁻ (gain from a gain EDT, 3.6 %)", GREY), ("31 voltage values", ORANGE)]):
    box(s, 0.6, 3.5 + i * 0.52, 3.6, 0.42, t1, fill=fill, size=12)
    arrow(s, 4.2, 3.71 + i * 0.52, 4.8, 4.23)
box(s, 4.8, 3.9, 1.7, 0.66, "Noise EDT", fill=GREEN, size=15, bold=True)
arrow(s, 6.5, 4.23, 7.0, 4.23)
box(s, 7.0, 3.9, 1.0, 0.66, "ln σe", fill=LIGHT, size=15, bold=True)
box(s, 0.6, 5.15, 7.4, 1.8, "", fill=GREY, shape=MSO_SHAPE.RECTANGLE)
text(s, 0.7, 5.2, 7.2, 1.75, [
    ("**What the trace adds**", {"level": None, "space": 3}),
    ("Voltages only: learns “these voltages → this noise” from the training amplifiers. It does well here because the new amplifier is a sibling: same CCD, electronics and night", {"level": 0, "size": 13}),
    ("The trace measures this amplifier's own noise at this setting, so it does not need the new amplifier to look like the training ones. A voltages-only model should break down on CCDs less similar to the training ones; to be tested", {"level": 0, "size": 13}),
], size=14, space=3)
pic(s, "noise_new.png", 8.4, 1.15, w=4.3)
pic(s, "noise_seen.png", 8.4, 4.1, w=4.3)
cite(s, "Typical error = RMS of ln(predicted / measured) on the 67 sealed test images. Interval: bootstrap over the 23 session-H packets (Efron 1979). “Everything”: same trees, amp_phys/ert_inputs.py. Physics formula without ML: backup.")

# ------------------------------------------------------------------ 39 (backup): the physics formula
s = S[39]; clear(s); title(s, "Image Noise Without ML: σ₅₀₀ ÷ Gain")
text(s, 0.6, 1.3, 6.9, 5.5, [
    "**Formula:** σe ≈ σ₅₀₀ ÷ g. σ₅₀₀ is the 500-skip noise predicted from the trace (ADU); g is the gain (ADU per e⁻). No ML for the noise itself",
    ("gain from the gain EDT (voltages): **11.9 %**", {"level": 1}),
    ("gain measured on the new amplifier at nominal: 9.2 %; each image's own gain: 9.0 %", {"level": 1}),
    "**Noise EDT:** 8.6 %, so with a good gain the formula does almost as well; most of the trace's value is σ₅₀₀ itself",
    "**Why not below ≈8–9 %:** each amplifier has its own offset (−6 % to +2 %), and the image sees charge-related effects the trace cannot. The label itself repeats to 2.3 %",
], size=15, space=8)
pic(s, "noise_formula.png", 7.8, 1.4, w=4.9)
caption(s, 7.8, 4.25, 4.9, "Same sealed test as the main slide (new amplifier, 67 images).", size=12)

# ------------------------------------------------------------------ 10: broken settings
s = S[10]; clear(s); title(s, "EDT on Our Data: Spotting Broken Settings")
pic(s, "img_good.png", 0.6, 1.2, w=3.05)
pic(s, "img_bad.png", 3.75, 1.2, w=3.05)
caption(s, 0.6, 4.05, 6.2, "Good: clear 0, 1, 2 e⁻ peaks. Bad: the fit fails (here the horizontal clock delays were halved).", size=11, h=0.45)
text(s, 0.6, 4.55, 6.3, 2.5, [
    "**Classifier:** EDT on the voltages + 8 trace numbers → P(good image); the search never images a setting with P < 0.5",
    "**It predicts settings at which every amplifier gives a bad image:** settings that break the readout",
    "**New settings, not remembered ones:** the test session shares 1 of its 25 settings with training, and none of its 6 broken ones",
    "**The trace makes it sure:** P drops from ≈0.45 (a coin flip) to ≈0.13 and catches the bad images the voltages let through",
], size=13, space=5)
pic(s, "pgood.png", 7.4, 1.15, w=4.6)
table(s, [["sealed test", "bad images caught", "good images blocked"],
          ["voltages only", "11–13 of 18", "0 of 67"],
          ["voltages + trace", "**17–18 of 18**", "0 of 67"]],
      7.1, 4.85, 5.6, col_w=[1.9, 1.9, 1.8], size=13, row_h=0.42)
caption(s, 7.1, 6.15, 5.6, "6 broken settings × 3 amplifiers = 18 bad images. Trained on session G, tested once on H, test amplifier left out; range over 10 random seeds.", size=11, h=0.8)

# ------------------------------------------------------------------ 32 -> backup: GP math (title only changes)
title(S[32], "Gaussian Process: The Math")

# ------------------------------------------------------------------ 38: GP, what it does
s = S[38]; clear(s); title(s, "Gaussian Process (GP): A Smooth Guess with Error Bars")
text(s, 0.6, 1.25, 6.6, 5.7, [
    ("**What it fits:** J(x) = ln(dark charge) as a function of the 18 voltages x, from the images taken so far. No fixed formula: the data decide the shape", {"level": None, "space": 8}),
    ("**What it gives:** at any setting a guess m(x) and an uncertainty s(x); s is small near measured settings and grows away from them (top right)", {"level": None, "space": 8}),
    ("**Assumptions:**", {"level": None}),
    ("J is smooth: settings close in voltage have similar J. How close is “close” is a length scale ℓⱼ for each voltage (bottom right)", {"level": 0}),
    ("Each image's J has random scatter of size n", {"level": 0, "space": 8}),
    ("**What is fitted:** about 20 numbers: the 18 ℓⱼ, the size of the response s² and the scatter n², chosen to make the measured J most probable. No weights, nothing set by hand", {"level": None, "space": 8}),
    ("**The guess** is a weighted average of the measured J, with more weight on nearby settings:", {"level": None}),
], size=14, space=4)
eq(s, "post", 0.8, 5.0, scale=0.72)
text(s, 0.6, 5.55, 6.6, 0.45, [("**Smoothness** (correlation of J at two settings; bottom right):", {"level": None})], size=14)
eq(s, "kernel", 0.8, 6.0, scale=0.72)
pic(s, "gp.png", 7.5, 1.2, w=5.2)
pic(s, "gp_kernel.png", 7.75, 4.4, w=4.7)
cite(s, "Rasmussen & Williams, Gaussian Processes for Machine Learning (MIT Press, 2006). Plots: one voltage, made-up response. Kernel: Matérn-5/2; full math in backup.")

# ------------------------------------------------------------------ 33: multi-amplifier GP, less math, figure per step
s = S[33]; clear(s); title(s, "Multi-Amplifier GP: Why It Suits Our Case")
text(s, 0.6, 1.2, 6.3, 0.9, [("**Model:** each amplifier's J = a shape f(x) shared by all amplifiers + its own level cₐ (+ a small own deviation)", {"level": None})], size=14)
eq(s, "mt", 0.8, 2.0, scale=0.85)
text(s, 0.6, 2.6, 6.3, 1.1, [("**Training** on the amplifiers already measured, in two steps (below): first the shape, then how far the levels differ (c²) and how alike the shapes are (ρ)", {"level": None})], size=14)
text(s, 7.1, 1.2, 5.6, 2.6, [
    ("**Why it suits our case**", {"level": None}),
    ("The dark-charge level differs by amplifier and cannot be predicted (previous slide); here cₐ is learned, not predicted", {"level": 0}),
    ("A constant level does not move the minimum, so the ≈200 images of the amplifiers already measured rank the settings for a new one", {"level": 0}),
    ("The new amplifier needs only a few images: the first fixes its level", {"level": 0}),
], size=13, space=4)
for k, f in enumerate(["mt_step1.png", "mt_step2.png", "mt_new.png"]):
    pic(s, f, 0.6 + k * 4.08, 3.8, w=3.95)
cite(s, "Multi-task GP: Bonilla, Chai & Williams, NeurIPS (2007); Swersky, Snoek & Adams, NeurIPS (2013). Plots: one voltage, made-up response.")

# ------------------------------------------------------------------ 13: BO loop, b / m / s, what makes it BO
s = S[13]
pics_off(s)
for sh in list(s.shapes):
    if sh.has_text_frame and sh.text_frame.text.startswith("Expected improvement over b"):
        sh._element.getparent().remove(sh._element)
eq(s, "ei", 7.1, 1.3, scale=0.78)
text(s, 7.0, 1.8, 5.7, 1.3, [
    ("**b:** the best (lowest) J of the images so far", {"level": 0}),
    ("**m(x), s(x):** the GP's guess of J at setting x, and its uncertainty", {"level": 0}),
    ("**EI(x):** how much better than b we expect x to be. Large if m is low (exploit) or s is large (explore)", {"level": 0}),
], size=12, space=2)
pic(s, "bo_gp.png", 7.3, 3.0, w=4.6)
pic(s, "bo_ei.png", 7.3, 5.2, w=4.6)
text(s, 0.6, 6.05, 6.2, 1.0, [("**What makes it Bayesian optimization:** a probability model of the unknown J (the GP), updated after every image (prior + data → posterior), and a rule that picks the next image from that belief: guess and uncertainty both.", {"level": None})], size=13)

# ------------------------------------------------------------------ 34 (backup BO loop): separate GP and EI images
s = S[34]
for sh in list(s.shapes):
    if sh.shape_type == 13 and sh.top > Inches(2.5):
        sh._element.getparent().remove(sh._element)
pic(s, "bo_gp.png", 7.2, 2.95, w=4.4)
pic(s, "bo_ei.png", 7.2, 5.1, w=4.4)

# ------------------------------------------------------------------ 14: BO results, the test first
s = S[14]; clear(s); title(s, "BO Results: Finding the Lowest Dark Charge")
text(s, 0.6, 1.2, 12.1, 1.75, [
    ("**The test, a replay of the v3 night:** each amplifier in turn is treated as new. Its 63–64 measured settings, broken ones included, are the menu: the search sees their voltages and traces, and “taking an image” reveals the stored result. The model learns only from the other two amplifiers, in the other half of the night. 6 searches, averaged", {"level": 0}),
    ("**Score:** images until the lowest dark charge found is within 2 % of the best in the menu; **wasted:** images at broken settings", {"level": 0}),
], size=14, space=6)
table(s, [["method", "what it does", "images", "wasted"],
          ["random order", "image the menu in a random order", "38.5", "5.7"],
          ["by hand", "always image the untested setting closest to the best so far", "26.8", "3.4"],
          ["BO, new amplifier only", "the GP learns only from this amplifier's own images", "4.0", "0.5"],
          ["**BO, multi-amplifier + EDT gate**", "the GP also learns from the other amplifiers; the EDT blocks likely-broken settings", "**1.8**", "**0**"]],
      0.6, 2.8, 7.4, col_w=[2.2, 3.4, 0.9, 0.9], size=13, row_h=0.6)
pic(s, "results.png", 8.3, 2.8, w=4.4)
text(s, 0.6, 6.25, 12.1, 0.75, [("**Using the other amplifiers cuts the search from ≈27 images by hand to ≈2, and no image is wasted.** Noise and noise + dark charge: backup.", {"level": None})], size=15)

# ------------------------------------------------------------------ 37: caveats, no dark-charge bullet
s = S[37]
for sh in list(s.shapes):
    if sh.has_text_frame and sh.text_frame.text.startswith("Trees cannot extrapolate"):
        p = sh.text_frame.paragraphs[1]; p._p.getparent().remove(p._p)

# ------------------------------------------------------------------ 15: summary
s = S[15]; clear(s); title(s, "Summary and Next Steps")
text(s, 0.6, 1.4, 12.1, 5.0, [
    ("**1. From the trace:** 0.2 s gives the amplifier's white and 1/f noise, its predicted 500-skip noise and signal sizes, the inputs to the models", {"level": None, "space": 16}),
    ("**2. Noise:** image noise on a new amplifier predicted to 8.6 % (18.2 % from the voltages alone); gain to 3.6 %", {"level": None, "space": 16}),
    ("**3. Broken settings:** spotted before imaging, also at settings never seen (17–18 of 18 bad images), no good image blocked", {"level": None, "space": 16}),
    ("**4. Search:** lowest dark charge in ≈2 images instead of ≈27 by hand", {"level": None, "space": 24}),
    ("**Next:** a live search on a CCD the model has never seen; more amplifiers to test the same-shape assumption", {"level": None}),
], size=19)
text(s, 0.6, 6.0, 12.1, 0.7, [("Thank you. Questions?", {"level": None, "bold": True})], size=26, align=PP_ALIGN.CENTER)

# ------------------------------------------------------------------ 5 (backup): amplifiers apart, two separate images
s = S[5]
for sh in list(s.shapes):
    if sh.shape_type == 13:
        sh._element.getparent().remove(sh._element)
pic(s, "amps_pulse.png", 0.6, 1.3, h=4.25)
pic(s, "amps_decay.png", 8.1, 1.3, h=4.25)


# ------------------------------------------------------------------ 40 (backup): voltage correlations with dark charge and noise
import csv
s = S[40]; clear(s); title(s, "How Each Voltage Correlates with Dark Charge and Noise")
rows = list(csv.DictReader(open(os.path.join(D, "volt_corr.csv"))))
def fr(v): 
    v = float(v); t = f"{v:+.2f}".replace("+", "") if v >= 0 else f"−{abs(v):.2f}"
    return f"**{t}**" if abs(v) >= 0.3 else t
data = [["voltage", "ln DC", "per amp", "ln noise", "per amp"]]
for r in rows:
    lo = lambda a, b: f"{fr(a).strip('*')} to {fr(b).strip('*')}"
    data.append([r["voltage"], fr(r["r_dc"]), lo(r["dc_lo"], r["dc_hi"]), fr(r["r_noise"]), lo(r["n_lo"], r["n_hi"])])
table(s, data, 0.6, 1.2, 6.2, col_w=[1.2, 0.95, 1.6, 0.95, 1.5], size=10, row_h=0.27)
pic(s, "corr_matrix.png", 7.0, 1.15, w=5.7)
text(s, 7.0, 6.15, 5.7, 0.9, [
    ("Pearson r on the 302 good v3 images (PA03 ch0, ch1, ch3, sessions G + H); bold: |r| ≥ 0.3. “per amp”: range of r over the three amplifiers alone; the pooled r can be smaller, since each amplifier has its own level", {"level": None}),
], size=10, color=GRAY)
cite(s, "Only VSUB stands out for dark charge (r = 0.43). Voltages are not independent: H high and low rails move together (r ≈ 0.8), as do V1 and V3 low (0.68), so one voltage can borrow another's correlation. ln DC and ln noise: r = 0.36.", y=6.9)


# ------------------------------------------------------------------ 41, 42 (backup): the protected features (tier P), always given to every model
s = S[41]; clear(s); title(s, "Always-Included Trace Features (Tier P) 1: Noise")
cells = [("p_jump.png", 0.5, 1.15, ["**CDS rms (detrended)**", "plain spread of one pixel's CDS around its slow trend; every skip counts",
                                       "**jump-robust CDS σ**", "same spread, but each skip is judged by the median of it and its two neighbours, so a lone outlying skip is ignored"]),
         ("p_jumpratio.png", 6.75, 1.15, ["**On PA03 the two agree:**", "the plain rms is never more than 10 % above the robust one, so there are no jumps", "A jumpy amplifier (e.g. telegraph noise) would show a tail to the right"]),
         ("p_avg.png", 0.5, 4.1, ["**white noise per skip**", "from how the noise of N-skip averages falls with N", "**Allan deviation, n = 50**", "scatter between neighbouring 50-skip averages"]),
         ("p_null.png", 6.75, 4.1, ["**null-CDS noise**", "one pedestal window, first half minus second half", "No charge is moved, so it is the readout noise alone"])]
for f, x, y, lines in cells:
    pic(s, f, x, y, w=4.25, h=2.8)
    text(s, x + 4.35, y + 0.1, 1.8, 2.75, [(t, {"level": None, "space": 4}) for t in lines], size=11)
cite(s, "Tier P: direct estimators of the noise and gain, chosen by physics and given to every model in the training plan (lvl1/feature_study.py, train_lvl1.py); the other 133 trace features enter by evidence tier (A1, A2, B, C). Right top: 17 770 PA03 pixels.", y=6.95)

s = S[42]; clear(s); title(s, "Always-Included Trace Features (Tier P) 2: Gain-Related")
pic(s, "p_reset.png", 0.6, 1.3, w=5.9)
pic(s, "p_edges.png", 6.8, 1.3, w=5.9)
text(s, 0.6, 5.0, 5.9, 2.0, [
    ("**reset-level scatter:** each reset leaves the output at a new random level (kTC noise, ≈150 ADU here). Its size depends on the sense-node capacitance, so it tracks the gain", {"level": 0}),
    ("The CDS subtracts the pedestal from the signal of the same skip, so this level cancels (blue, ≈6 ADU)", {"level": 0}),
], size=13, space=6)
text(s, 6.8, 5.0, 5.9, 2.0, [
    ("**SW dip per volt, RG pulse per volt:** how much of the summing-well and reset-gate clocks leaks into the output, divided by the clock swing", {"level": 0}),
    ("The leak passes through the same amplifier as the charge, so it scales with the gain", {"level": 0}),
], size=13, space=6)
cite(s, "PA03 ch0, one nominal trace (3 Nov).", y=7.0)

# ------------------------------------------------------------------ wording: PA03 instead of board 101; good / bad image
REPL = [("PA03 module, board 101", "PA03 module"), ("PA03 board 101", "PA03"), ("(PA03 101)", "(PA03)"), ("PA03 101", "PA03"),
        ("one board (PA03)", "one CCD (PA03)"), ("Board 101", "PA03"), ("board 101", "PA03"),
        ("P(usable)", "P(good)"), ("usable?", "good image?"), ("best usable setting", "best setting with a good image"),
        ("unusable images", "bad images"), ("unusable", "bad"), ("Usable", "Good"), ("usable images", "good images"), ("usable", "good")]

def fix_runs(tf):
    for p in tf.paragraphs:
        for r in p.runs:
            t = r.text
            for a, b in REPL:
                t = t.replace(a, b)
            if t != r.text:
                r.text = t

for sl in prs.slides:
    for sh in sl.shapes:
        if sh.has_text_frame:
            fix_runs(sh.text_frame)
        if sh.has_table:
            for row in sh.table.rows:
                for c in row.cells:
                    fix_runs(c.text_frame)
    if sl.has_notes_slide:
        fix_runs(sl.notes_slide.notes_text_frame)

# ------------------------------------------------------------------ order: main talk, divider, backup
MAIN = [0, 1, 2, 17, 18, 3, 6, 7, 30, 31, 9, 10, 11, 38, 33, 13, 14, 37, 15]
BACK = [4, 5, 19, 20, 21, 22, 23, 24, 25, 26, 27, 41, 42, 39, 40, 28, 29, 32, 34, 35, 36]
DROP = [8, 12]
assert sorted(MAIN + [16] + BACK + DROP) == list(range(len(S)))
sldIdLst = prs.slides._sldIdLst
ids = list(sldIdLst)
for k in DROP:
    prs.part.drop_rel(ids[k].rId)
for e in ids:
    sldIdLst.remove(e)
for k in MAIN + [16] + BACK:
    sldIdLst.append(ids[k])

def set_prefix(sl, on):
    for sh in sl.shapes:
        if sh.is_placeholder and sh.placeholder_format.type in (1, 3):
            runs = [r for p in sh.text_frame.paragraphs for r in p.runs]
            if not runs:
                return
            t0 = runs[0].text
            if on and not t0.startswith("Backup"):
                runs[0].text = "Backup: " + t0
                for r in runs:
                    r.font.size = Pt(26)
            if not on and t0.startswith("Backup: "):
                runs[0].text = t0[len("Backup: "):]

for k in MAIN:
    set_prefix(S[k], False)
for k in BACK:
    set_prefix(S[k], True)

prs.save(OUT)
print("saved", OUT, len(prs.slides))

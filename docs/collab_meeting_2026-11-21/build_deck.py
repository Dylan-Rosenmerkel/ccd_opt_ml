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
two_fig_slide(S[3], "Trace Breakdown: Vertical Clocks", [
    "V1–V3 and the transfer gate (TG) move every row one step toward the serial register; one transfer takes 360 µs",
    "The clock edges couple into the output: a waveform of ≈40 000 ADU peak-to-peak, repeated at every transfer",
    "Features: peak-to-peak, lowest and highest point, change from the first to the last transfer"],
    "v_zoom.png", "v_setting.png",
    "Right: lowering the V low rails by 0.5 V (TG by 0.4 V) deepens only this waveform, peak-to-peak 41 → 46 kADU. Bands start at the spike that begins each transfer. Board 101, ch0, 12 Oct.")
two_fig_slide(S[4], "Trace Breakdown: Horizontal Clocks", [
    "H1–H3 move the charge along the serial register to the sense node: one transfer per pixel, 126 µs each",
    "The largest waveform in the trace (≈130 000 ADU peak-to-peak); it sets how the output looks before the first skip",
    "Features: peak-to-peak, lowest and highest point, scatter between transfers, change from first to last transfer"],
    "h_zoom.png", "h_setting.png",
    "Right: lowering the H low rails by 0.5 V changes only the horizontal waveform, peak-to-peak 126 → 144 kADU. Bands start at the spike that begins each transfer. 12 Oct.")
two_fig_slide(S[5], "Trace Breakdown: Skips Region", [
    "After the horizontal transfer the pixel's charge is measured 500 times (skips), 32.6 µs per skip",
    "The baseline drifts by ≈3 000 ADU across a pixel, but each skip subtracts its own pedestal, so the CDS stays flat",
    "Features: CDS noise, drift along the pixel, first-skip transient, baseline recovery, pixel charge level"],
    "skips_zoom.png", "skips_drift.png",
    "Board 101, ch0, nominal voltages, 3 Nov. Left: each spike is one skip cycle. Right: the same pixel, one point per skip.")

# ------------------------------------------------------------------ 6: one skip and its noise
s = S[6]; clear(s); title(s, "Trace Breakdown: One Skip and Its Noise")
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
s = S[7]; clear(s); title(s, "Telling Settings Apart")
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
s = S[8]; clear(s); title(s, "Telling Amplifiers Apart")
pic(s, "amps.png", 0.6, 1.3, w=12.1)
text(s, 0.6, 5.7, 12.1, 1.35, [
    "Same voltages, each amplifier traced 5 times in a row (8 Oct): the repeats agree, the amplifiers differ (ch1's reset pulse decays ≈150 ns faster)",
    "Is a number good for this? Compare how much it **differs between two amplifiers** with how much it **wobbles between repeats**. 25 of our 141 numbers tell every pair of amplifiers clearly apart (difference ≥ 3 × the wobble)",
], size=15)

# ------------------------------------------------------------------ 9: oxide 1, the noise model
s = S[9]; clear(s); title(s, "Amplifier Oxide Quality (1): A Model of the Noise")
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
s = S[10]; clear(s); title(s, "Amplifier Oxide Quality (2): Measuring the 1/f Level K")
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
s = S[11]; clear(s); title(s, "Amplifier Oxide Quality (3): From K to the Oxide Trap Density")
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
s = S[12]; clear(s); title(s, "Clock Couplings to the Amplifier Input (1): How")
text(s, 0.6, 1.25, 12.1, 0.5, [("Idea: when a clock line switches by ΔV, the output settles to a new level. If that charge sits on the amplifier input, it reads exactly like electrons.", {"level": None})], size=15)
box(s, 0.6, 1.95, 2.2, 0.85, "Clock line k\n(H1–H3, SW)", fill=ORANGE)
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
text(s, 0.6, 5.75, 7.1, 0.6, [("Summing well (SW): a 0.300 µs pulse, shorter than the readout's response, so use response area ÷ pulse width instead.", {"level": None})], size=13)
pic(s, "fig_H_state.png", 7.95, 3.0, w=4.75)
text(s, 7.95, 5.75, 4.75, 1.3, [
    ("**Worked example, H2, ch0:** β ≈ 26 500 ADU per volt of H2 swing ÷ g = 1.41 ADU/e⁻ → ≈ 18 800 e⁻ per volt; × q → **3.01 fF** (mean of the four amplifiers 3.09 fF)", {"level": None}),
    ("Reading the output 0.2–0.8 µs after the edge, long before it settles, had given 0.01 fF.", {"level": None, "size": 11}),
], size=13, space=4)

# ------------------------------------------------------------------ 13: couplings, results
s = S[13]; clear(s); title(s, "Clock Couplings to the Amplifier Input (2): Results")
table(s, [["line", "coupling (fF-eq.)", "4 amplifiers", "checked by"],
          ["H1", "3.41 ± 0.06", "3.33–3.47", "up = down within 1 %"],
          ["H2", "3.09 ± 0.07", "3.01–3.14", "up = down within 1 %"],
          ["H3", "3.49 ± 0.06", "3.42–3.54", "up step; down step 3.26"],
          ["SW", "0.54 ± 0.02", "0.53–0.55", "area ∝ swing (r = −0.97)"]],
      0.6, 1.35, 7.6, col_w=[1.1, 1.9, 1.7, 2.9], size=13, row_h=0.4)
caption(s, 0.6, 3.4, 7.6, "Board 101, mean of the four amplifiers. ± = fit systematics (start time taken conservatively, window, pulse width); the amplifier range is shown separately.", size=11, h=0.45)
pic(s, "recon_ch0.png", 0.6, 3.85, w=7.6)
caption(s, 0.6, 6.4, 7.6, "Validation: couplings fitted on other settings predict the whole horizontal-transfer waveform (red) to 0.6 % rms; the residual (green, ×10) is only at the fast edge spikes.", size=11, h=0.6)
text(s, 8.55, 1.35, 4.2, 5.6, [
    ("**Effective couplings:** the step holds through the 18 µs state and follows VDD like the image gain, so it enters at the amplifier input; a single-CCD board (PQ04) is ≈18× weaker, so the size comes from the 4-CCD module's routing", {"level": 0}),
    ("**Not in the images:** the H clocks do not switch during the skips", {"level": 0}),
    ("**Not presented:** RG (a turn-off kick that drains back to VR, not stored charge) and V1–V3 (steps do not follow the swing)", {"level": 0}),
    ("**On hold:** reset-gate overlap limit, buried-channel depth", {"level": 0}),
], size=13, space=8)

# ------------------------------------------------------------------ 14: amplifier quantities (trace only, board 101)
s = S[14]; clear(s); title(s, "Amplifier Quantities from the Trace Alone")
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
    "Numbers that need an image gain (the clock couplings) are on the previous slides",
], size=15)

# ------------------------------------------------------------------ 15: ML intro
s = S[15]; clear(s); title(s, "Can Traces Predict Image Noise and Other Image Quantities?")
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
s = S[16]; clear(s); title(s, "Data Set")
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
s = S[17]; clear(s); title(s, "What Is a Decision Tree?")
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
s = S[18]; clear(s); title(s, "Extremely Randomized Decision Trees (EDT)")
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
s = S[19]; clear(s); title(s, "EDT on Our Data: Predicting Image Noise")
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
s = S[20]; clear(s); title(s, "EDT on Our Data: Usable or Not?")
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
s = S[21]; clear(s); title(s, "Can ML Find Good Settings Faster? Not Directly")
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
s = S[22]; clear(s); title(s, "Learning the Shape of the Voltage Response (Gaussian Process)")
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
s = S[23]; clear(s); title(s, "Multi-Amplifier GP Model")
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
s = S[24]; clear(s); title(s, "Bayesian Optimization (BO) Loop")
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
s = S[25]; clear(s); title(s, "BO Tests: Replaying a Night We Already Measured")
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
s = S[26]; clear(s); title(s, "BO Results: Finding the Lowest Dark Charge")
table(s, [["method", "what it does", "images", "wasted"],
          ["random order", "image the menu in a random order", "38.5", "5.7"],
          ["by hand", "always image the untested setting closest to the best so far", "26.8", "3.4"],
          ["BO, new amplifier only", "the GP learns only from this amplifier's own images", "4.0", "0.5"],
          ["**BO, multi-amplifier + EDT gate**", "the GP also learns from the other amplifiers; the EDT blocks likely-unusable settings", "**1.8**", "**0**"]],
      0.6, 1.4, 7.4, col_w=[2.2, 3.4, 0.9, 0.9], size=13, row_h=0.62)
text(s, 0.6, 4.75, 7.4, 2.2, [
    ("**images:** number of images until the lowest dark charge found is within 2 % of the lowest in the menu", {"level": 0}),
    ("**wasted:** images at settings that turned out unusable", {"level": 0}),
    ("Average of the 6 searches, starting with nothing known about the new amplifier", {"level": 0}),
], size=14)
pic(s, "results.png", 8.3, 1.4, w=4.4)
text(s, 8.3, 4.15, 4.4, 2.8, [
    ("**Using the other amplifiers cuts the search from ≈27 images by hand to ≈2, and no image is wasted**", {"level": None, "space": 10}),
    ("Limits: one board, one night, sibling amplifiers. Next: a live run on a CCD the model has never seen.", {"level": None, "size": 14}),
], size=15)

prs.save(OUT)
print("saved", OUT)

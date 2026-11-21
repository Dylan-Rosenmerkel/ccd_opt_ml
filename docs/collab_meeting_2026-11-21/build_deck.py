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


# ------------------------------------------------------------------ slide 2: introduction
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

# ------------------------------------------------------------------ slide 3: full trace
s = S[2]; clear(s); title(s, "Trace Breakdown: One Row of a Trace")
pic(s, "regions_overview.png", 0.6, 1.25, w=12.1)
caption(s, 0.6, 5.15, 12.1, "The regions are found automatically from the sequencer flag bits recorded with every sample, so the decoding works for any geometry and timing.", size=12)
for x, head, body, fill in [
        (0.6, "Vertical transfers", "The parallel clocks move the image down one row. They couple into the output: a large, repeatable waveform.", RGBColor(0xCF, 0xE0, 0xEF)),
        (4.75, "Horizontal transfers", "The serial register moves the next pixel to the amplifier, one pixel per transfer.", RGBColor(0xFD, 0xE6, 0xD2)),
        (8.9, "Skip region", "Each pixel is measured 500 times without destroying the charge. This is the measurement itself.", RGBColor(0xD5, 0xEC, 0xD4))]:
    box(s, x, 5.65, 3.8, 1.3, "", fill=fill)
    text(s, x + 0.1, 5.7, 3.6, 1.2, [(head, {"level": None, "bold": True, "size": 16}), (body, {"level": None, "size": 13})])


def clocks_slide(s, ttl, bullets, left, right, cap):
    clear(s); title(s, ttl)
    text(s, 0.6, 1.3, 12.1, 1.4, bullets, size=17, space=6)
    pic(s, left, 0.6, 2.75, w=5.9)
    pic(s, right, 6.8, 2.75, w=5.9)
    caption(s, 0.6, 6.25, 12.1, cap, size=12, h=0.6)


# ------------------------------------------------------------------ slides 4-6
clocks_slide(S[3], "Trace Breakdown: Vertical Clocks", [
    "V1–V3 and the transfer gate (TG) move every row one step toward the serial register; one transfer takes 180 µs",
    "The clock edges couple into the output: a waveform of ≈25 000 ADU, repeated exactly at every transfer",
    "Features: peak-to-peak, lowest and highest point, change from the first to the last transfer"],
    "v_zoom.png", "v_setting.png",
    "Right: lowering the V low rails by 0.5 V (TG by 0.4 V) deepens only this waveform, peak-to-peak 25.3 → 31.3 kADU. Board 101, ch0, 12 Oct.")
clocks_slide(S[4], "Trace Breakdown: Horizontal Clocks", [
    "H1–H3 move the charge along the serial register to the sense node: one transfer per pixel, 54 µs each",
    "The largest waveform in the trace (≈90 000 ADU); it sets how the output looks before the first skip",
    "Features: peak-to-peak, lowest and highest point, scatter between transfers, change from first to last transfer"],
    "h_zoom.png", "h_setting.png",
    "Right: lowering the H low rails by 0.5 V changes only the horizontal waveform, peak-to-peak 86.6 → 101.6 kADU. Board 101, ch0, 12 Oct.")
clocks_slide(S[5], "Trace Breakdown: Skips Region", [
    "After the horizontal transfer the pixel's charge is measured 500 times (skips), 32.6 µs per skip",
    "The baseline drifts by ≈3 000 ADU across a pixel, but each skip is referenced to its own pedestal, so the CDS stays flat",
    "Features: CDS noise, drift along the pixel, first-skip transient, baseline recovery, pixel charge level"],
    "skips_zoom.png", "skips_drift.png",
    "Board 101, ch0, nominal voltages, 3 Nov (unbinned 2 × 5 pixel trace). Each spike in the left panel is one skip cycle.")

# ------------------------------------------------------------------ slide 7: one skip and its noise
s = S[6]; clear(s); title(s, "Trace Breakdown: One Skip and Its Noise")
pic(s, "two_skips.png", 0.6, 1.3, w=5.9)
text(s, 0.6, 4.35, 5.9, 0.95, [
    "Pedestal (10 µs) → charge onto the sense node → signal (10 µs) → reset",
    "CDS = mean(pedestal) − mean(signal): one measurement",
], size=14)
eq(s, "sigma1", 0.8, 5.45, scale=0.95)
text(s, 0.6, 6.05, 5.9, 0.9, [("Consecutive skips of the same pixel: the charge cancels, the noise does not. Here σ₁ ≈ 4.9 ADU ≈ 3.6 e⁻.", {"level": None})], size=14)
pic(s, "allan.png", 7.0, 1.3, w=5.6)
eq(s, "allan", 7.2, 5.0, scale=0.95)
text(s, 7.0, 5.6, 5.7, 1.3, [
    ("ȳₖ: mean of block k of N skips", {"level": None}),
    ("White noise falls as σ₁/√N; the curve flattens where slow noise (1/f, drift) takes over. That floor limits the 500-skip image.", {"level": None}),
], size=14)
cite(s, "Allan deviation: D. W. Allan, Proc. IEEE 54, 221 (1966).")

# ------------------------------------------------------------------ slide 8: settings apart
s = S[7]; clear(s); title(s, "Telling Settings Apart")
text(s, 0.6, 1.3, 12.1, 1.1, [
    "Reduce each trace to a vector of 85 hand-picked features (noise, waveform shapes, timing, levels)",
    "Principal component analysis (PCA): find the two directions in which the traces differ most (PC1, PC2) and plot every trace along them. PCA is never told the settings",
], size=16)
pic(s, "oct12_feature_space.png", 1.15, 2.4, w=11.0)
text(s, 0.6, 6.2, 12.1, 0.7, [
    ("Repeats of a setting sit together, different settings far apart. Test: hide one trace, find its most similar trace. It came from the same setting in **44–46 of 47** cases.", {"level": None}),
], size=15)
caption(s, 0.6, 6.95, 12.1, "First voltage scan, 12 Oct: four voltage changes, each traced 3 times. Caveat: only four settings, and some drift between them.", size=11)

# ------------------------------------------------------------------ slide 9: amplifiers apart
s = S[8]; clear(s); title(s, "Telling Amplifiers Apart")
pic(s, "amps.png", 0.6, 1.3, w=12.1)
text(s, 0.6, 5.85, 12.1, 1.2, [
    "Each amplifier traced 5 times in a row at the same voltages (8 Oct): repeats agree, amplifiers differ (ch1's reset pulse decays ≈150 ns faster)",
    "Separation = difference between amplifiers ÷ spread of repeats: **25 of 141 features** separate all 6 pairs by ≥ 3, led by the ringing after the reset (293) and the pulse decay (120)",
], size=16)

# ------------------------------------------------------------------ slide 10: oxide 1
s = S[9]; clear(s); title(s, "What the Trace Tells Us About the Amplifier: Oxide Quality (1)")
pic(s, "two_skips.png", 0.6, 1.3, w=5.6)
text(s, 0.6, 4.15, 6.0, 0.65, [("**Time domain:** σ₁ from consecutive skips. τ = window length, Δt = 13.8 µs between the windows, T = 32.6 µs per skip", {"level": None})], size=13)
pic(s, "transfer.png", 6.9, 1.3, w=5.8)
text(s, 0.6, 4.85, 6.6, 0.4, [("**Frequency domain:** each step is a filter on the noise spectrum S(f)", {"level": None})], size=13)
eq(s, "H", 0.6, 5.32, scale=0.72)
eq(s, "fit", 0.6, 5.95, scale=0.8)
text(s, 0.6, 6.4, 6.6, 0.5, [("Measure σ₁ at six window lengths τ = 2–9.3 µs and fit S_w (white) and K (1/f).", {"level": None})], size=13)
table(s, [["amplifier", "K (ADU²)", "white (nV/√Hz)"],
          ["101 ch0", "2.5", "14.2"], ["101 ch1", "4.0", "12.7"], ["101 ch2", "4.2", "14.0"],
          ["101 ch3", "5.1", "13.0"], ["102 ch3", "5.7", "7.7"]],
      7.7, 4.95, 4.6, col_w=[1.5, 1.4, 1.7], size=12, row_h=0.28)
cite(s, "Transfer function: Hopkinson & Lumb, J. Phys. E 15, 1214 (1982), Table 1 'differential averager'; the skip-difference factor is ours. Board 101 carries narrow interference lines that bias this K low (under study).", y=6.95)

# ------------------------------------------------------------------ slide 11: oxide 2
s = S[10]; clear(s); title(s, "What the Trace Tells Us About the Amplifier: Oxide Quality (2)")
text(s, 0.6, 1.3, 6.4, 0.4, [("**1. Refer K from the ADC to the transistor gate**", {"level": None})], size=16)
eq(s, "kvg", 0.9, 1.78, scale=1.0)
text(s, 0.6, 2.35, 6.4, 0.95, [
    ("G = 0.875 ADU/µV: readout-chain gain (design: 27.34 V/V ÷ 31.25 µV/LSB)", {"level": 0}),
    ("A = 0.85: source-follower gain (from the layout; a model value)", {"level": 0}),
], size=14)
text(s, 0.6, 3.4, 6.4, 0.4, [("**2. McWhorter: 1/f noise from charges in oxide traps**", {"level": None})], size=16)
eq(s, "mcw", 0.9, 3.85, scale=1.0)
text(s, 0.6, 4.55, 6.4, 0.4, [("**3. Solve for the trap density**", {"level": None})], size=16)
eq(s, "nt", 0.9, 5.0, scale=1.0)
text(s, 0.6, 5.7, 6.4, 0.7, [("WL = 345 µm², C_ox = 0.454 fF/µm², T = 130 K, λ_t = 0.1 nm (tunnelling depth)", {"level": None})], size=13)
table(s, [["amplifier", "N_t (10¹⁶ eV⁻¹ cm⁻³)"],
          ["101 ch0", "1.1"], ["101 ch1", "1.8"], ["101 ch2", "1.9"], ["101 ch3", "2.3"],
          ["102 ch3", "2.5"], ["LSM, 11 channels (180 K)", "1.2–2.4"]],
      7.4, 1.4, 5.3, col_w=[2.9, 2.4], size=13, row_h=0.36)
text(s, 7.4, 4.2, 5.3, 2.7, [
    ("For comparison: bulk CMOS n-MOS with a 12 nm oxide, N_t ≈ 2 × 10¹⁶ (Ioannidis et al.)", {"level": 0}),
    ("Caveats: K also contains any other 1/f-like noise (pushes N_t up); interference lines on board 101 push K the other way; C_ox and A are model values (±15 % in C_ox → ±30 % in N_t)", {"level": 0}),
], size=13)
cite(s, "McWhorter, in Semiconductor Surface Physics (1957); Christensson et al., Solid-State Electron. 11, 797 (1968); Ioannidis et al., Solid-State Electron. 95, 28 (2014).")

# ------------------------------------------------------------------ slide 12: mutual capacitance
s = S[11]; clear(s); title(s, "What the Trace Tells Us About the Amplifier: Mutual Capacitances")
text(s, 0.6, 1.25, 12.1, 0.5, [("A voltage step ΔV on a clock line couples a charge C_k ΔV onto the sense node, which reads it exactly like electrons.", {"level": None})], size=15)
box(s, 0.6, 2.05, 2.1, 0.9, "Clock line k\n(RG, SW, H, V)", fill=ORANGE)
box(s, 3.6, 2.05, 2.1, 0.9, "Sense node\nC_sn ≈ 85–95 fF", fill=LIGHT)
box(s, 6.6, 2.05, 2.1, 0.9, "Source follower\nA = 0.85", fill=LIGHT)
box(s, 9.6, 2.05, 2.9, 0.9, "Readout chain + ADC\ngain G → trace (ADU)", fill=GREEN)
arrow(s, 2.7, 2.5, 3.6, 2.5); arrow(s, 5.7, 2.5, 6.6, 2.5); arrow(s, 8.7, 2.5, 9.6, 2.5)
text(s, 2.72, 2.08, 0.9, 0.35, [("C_k", {"level": None, "bold": True})], size=13, align=PP_ALIGN.CENTER)
eq(s, "ck", 0.9, 3.15, scale=1.0)
text(s, 3.6, 3.2, 9.1, 0.6, [("N: electrons seen per step; y: trace response (ADU); g: image gain (ADU/e⁻). Only g is needed, not C_sn, A or G.", {"level": None})], size=14)
text(s, 0.6, 4.0, 5.9, 0.8, [("**Data needed:** traces at ≈100 voltage settings (all rails varied together) + one image gain. Regress the response on every rail at once.", {"level": None})], size=13)
table(s, [["line → sense node", "C_k (fF)"],
          ["SW (summing well)", "0.08–0.13"], ["H1–H3", "0.011–0.016"],
          ["RG low rail", "0.12–0.19 *"], ["V1–V3", "< 0.003 (not seen)"]],
      0.6, 4.9, 5.9, col_w=[3.3, 2.6], size=13, row_h=0.32)
text(s, 6.9, 4.0, 5.8, 3.0, [
    ("**Uses**", {"level": None}),
    ("Clock feedthrough into the CDS: ΔCDS = C_k ΔV_k / q electrons; predicts the offset when a swing changes [Janesick]", {"level": 0}),
    ("Sense-node budget: C_sn = gate + wiring + Σ C_k; the clock couplings are < 1 % of the 85–95 fF [Janesick]", {"level": 0}),
    ("Reset charge injection: Q = α C W L (V_G − V_ch − V_T) gives the buried-channel depth (≈0.6 µm) [Sheu & Hu; Wegmann et al.]", {"level": 0}),
], size=13)
caption(s, 0.6, 6.55, 5.9, "* measured within ≈1 µs of the edge; may still contain cable pickup.", size=11)
cite(s, "Janesick, Scientific Charge-Coupled Devices (SPIE, 2001); Sheu & Hu, IEEE JSSC 19, 519 (1984); Wegmann, Vittoz & Rahali, IEEE JSSC 22, 1091 (1987).")

# ------------------------------------------------------------------ slide 13: table of amplifier quantities
s = S[12]; clear(s); title(s, "Amplifier Quantities from the Traces")
table(s, [["quantity", "101 ch0", "101 ch1", "101 ch2", "101 ch3", "102 ch3", "LSM (12 ch.)"],
          ["image gain g (ADU/e⁻)", "1.41", "1.39", "1.33", "1.34", "1.29", "1.26–1.38"],
          ["conversion gain at the input (µV/e⁻)", "1.62", "1.59", "1.52", "1.53", "1.51", "1.44–1.58"],
          ["sense-node capacitance C_sn (fF)", "84", "85", "89", "89", "90", "86–95"],
          ["single-skip noise σ₁ (e⁻)", "3.7", "3.9", "4.3", "4.3", "4.0", "3.3–4.3"],
          ["white noise (nV/√Hz)", "14.2", "12.7", "14.0", "13.0", "7.7", "5.4–9.1 †"],
          ["oxide trap density N_t (10¹⁶ eV⁻¹cm⁻³)", "1.1", "1.8", "1.9", "2.3", "2.5", "1.2–2.4 †"],
          ["reset settling time (µs)", "1.11", "0.93", "0.93", "1.16", "1.85", "0.28–1.04"],
          ["image read noise, 500 skips (e⁻)", "0.144", "0.147", "0.164", "0.173", "0.200", "0.19–0.33"]],
      0.6, 1.45, 12.1, col_w=[4.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.8], size=14, row_h=0.46)
text(s, 0.6, 5.85, 12.1, 1.1, [
    "From the trace alone: σ₁, white noise, N_t, settling. With one image gain added: conversion gain and C_sn (C_sn = A q / conversion gain, A = 0.85)",
    "The 17 amplifiers agree to ±6 % in C_sn; most of it is wiring, not the transistor gate",
], size=15)
caption(s, 0.6, 7.0, 12.1, "PA03 at 130 K (Nov 10–12), LSM test stand at 180 K. † 11 of 12 LSM channels (ACM110 ch0 fit fails).", size=11)

# ------------------------------------------------------------------ slide 14: ML intro
s = S[13]; clear(s); title(s, "Can Traces Predict Image Noise and Other Image Quantities?")
text(s, 0.6, 1.3, 12.1, 2.4, [
    "Goal: use the traces, the quantities derived from them and the voltages to predict the image **before** it is taken",
    "We tried many machine-learning models: Gaussian processes, gradient-boosted trees, neural-network ensembles, CNNs on the trace curves, deep-kernel and sparse GPs",
    "This talk focuses on two:",
    ("**Extremely Randomized Decision Trees (EDT)**: predict image noise, and whether an image will be usable", {"level": 1, "size": 16}),
    ("**Gaussian Process (GP)**: learn how dark charge responds to the voltages, then search for the best voltages", {"level": 1, "size": 16}),
], size=17)
box(s, 0.9, 4.4, 2.3, 0.9, "Trace (0.2 s)", fill=LIGHT, size=15)
box(s, 0.9, 5.7, 2.3, 0.9, "Voltages\n(18 knobs)", fill=ORANGE, size=15)
box(s, 4.0, 4.4, 2.6, 0.9, "Features\n141 numbers + physics", fill=LIGHT, size=15)
box(s, 7.4, 4.9, 2.2, 1.1, "Model\nEDT or GP", fill=GREEN, size=16, bold=True)
box(s, 10.4, 4.9, 2.3, 1.1, "Prediction\nnoise, usable?, dark charge", fill=RGBColor(0xF2, 0xF2, 0xF2), size=15)
arrow(s, 3.2, 4.85, 4.0, 4.85); arrow(s, 6.6, 4.85, 7.4, 5.25); arrow(s, 3.2, 6.15, 7.4, 5.7); arrow(s, 9.6, 5.45, 10.4, 5.45)

# ------------------------------------------------------------------ slide 15: data set
s = S[14]; clear(s); title(s, "Data Set")
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
    "18 voltage knobs varied together: SW, RG, V1–V3 and H1–H3 high/low rails, VSUB, VDD, VR",
    "Settings drawn by Sobol sampling (spread evenly over the allowed box)",
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

# ------------------------------------------------------------------ slide 16: EDT intro
s = S[15]; clear(s); title(s, "Extremely Randomized Decision Trees (EDT)")
pic(s, "tree.png", 0.6, 1.3, w=6.0)
text(s, 0.6, 4.85, 6.0, 0.9, [("**One tree:** yes/no questions on one input at a time; prediction = mean label of the training rows in the leaf. A split is scored by how much it reduces the spread:", {"level": None})], size=14)
eq(s, "split", 0.8, 5.85, scale=0.9)
text(s, 6.9, 1.3, 5.8, 2.5, [
    ("**How the extremely randomized version grows a tree**", {"level": None, "size": 16}),
    ("At each node, draw a random subset of the inputs", {"level": 0}),
    ("For each, draw **one random cut point** (uniform between its min and max)", {"level": 0}),
    ("Keep the best of these random splits; stop when a leaf has < 3 rows", {"level": 0}),
    ("Every tree sees all training rows (no bootstrap)", {"level": 0}),
], size=14)
eq(s, "forest", 7.1, 3.8, scale=0.9)
text(s, 9.2, 4.0, 3.5, 0.5, [("average of B = 400 trees", {"level": None})], size=14)
text(s, 6.9, 4.85, 5.8, 2.0, [
    ("**Why this one:** random cuts make the trees differ, so averaging cancels their errors; works with ≈150 training rows; needs no scaling; handles many correlated inputs", {"level": 0}),
    ("A random forest (Breiman) picks the best cut and resamples rows; EDT is more random, so its average varies less", {"level": 0}),
], size=14)
cite(s, "Geurts, Ernst & Wehenkel, Machine Learning 63, 3 (2006); Breiman, Machine Learning 45, 5 (2001); scikit-learn ExtraTreesRegressor.")

# ------------------------------------------------------------------ slide 17: noise prediction
s = S[16]; clear(s); title(s, "EDT on Our Data: Predicting Image Noise")
text(s, 0.6, 1.3, 7.0, 3.2, [
    "**Target:** ln σe, the read noise of a 500-skip image",
    "**Inputs (41):**",
    ("9 trace physics values: σ₁ at 6 window lengths, white level S_w, 1/f level K, predicted 500-skip noise σ₅₀₀", {"level": 1}),
    ("31 voltages, and σ₅₀₀ divided by a gain predicted from the voltages", {"level": 1}),
    "**Model:** EDT, 400 trees, ≥ 3 rows per leaf, half the inputs tried at each split",
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

# ------------------------------------------------------------------ slide 18: usability
s = S[17]; clear(s); title(s, "EDT on Our Data: Usable or Not?")
pic(s, "usable.png", 0.6, 1.3, w=7.4)
caption(s, 0.6, 4.8, 7.4, "Left: a usable image, clear 0, 1, 2 e⁻ peaks. Right: an unusable one, the peaks are not resolved (examples from 2–3 Nov).", size=12, h=0.5)
text(s, 8.3, 1.3, 4.4, 3.6, [
    "**82 of 381** v3 images (22 %) could not be used: the fit failed or raised a flag (dark charge not constrained, bad χ²)",
    "**Usable** = fit converged, no flags, λ > 0.005 e⁻/pixel",
    "**Classifier:** EDT on the 18 voltage knobs + 8 trace features → P(usable)",
    "In the search: never image a setting with P < 0.5",
], size=15)
table(s, [["sealed test (H)", "AUC", "unusable caught", "usable blocked"],
          ["voltages only", "0.990", "11 of 20", "0 of 67"],
          ["voltages + trace", "0.985", "**18 of 20**", "0 of 67"]],
      0.6, 5.4, 7.4, col_w=[2.3, 1.2, 2.0, 1.9], size=14, row_h=0.42)
text(s, 8.3, 5.3, 4.4, 1.6, [("The trace catches most broken settings before an image is spent on them, and blocks none of the good ones.", {"level": None})], size=15)
cite(s, "Trained on session G, tested once on H, test amplifier left out. AUC: probability that a usable image is ranked above an unusable one.")

# ------------------------------------------------------------------ slide 19: why not directly
s = S[18]; clear(s); title(s, "Can ML Find Good Settings Faster? Not Directly")
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

# ------------------------------------------------------------------ slide 20: GP
s = S[19]; clear(s); title(s, "Learning the Shape of the Voltage Response (Gaussian Process)")
text(s, 0.6, 1.3, 2.3, 0.45, [("**What we minimize:**", {"level": None})], size=15)
eq(s, "J", 2.75, 1.35, scale=0.85)
text(s, 0.6, 1.85, 6.4, 0.85, [("A GP is a prior over smooth functions: J at any set of settings is jointly Gaussian, with a covariance set by how close the settings are (Matérn-5/2):", {"level": None})], size=14)
eq(s, "kernel", 0.7, 2.75, scale=0.8)
text(s, 0.6, 3.4, 6.4, 0.5, [("ℓⱼ: how far knob j must move to change J (one per knob); s²: size of the response", {"level": None})], size=13)
text(s, 0.6, 3.95, 6.4, 0.4, [("**Prediction** at a new setting x* from the measured images y:", {"level": None})], size=14)
eq(s, "post", 0.7, 4.4, scale=0.75)
text(s, 0.6, 4.9, 6.4, 0.4, [("**Training:** choose s², ℓⱼ and the noise n² by maximizing", {"level": None})], size=14)
eq(s, "lml", 0.7, 5.35, scale=0.85)
text(s, 0.6, 5.85, 6.4, 0.9, [("First term: fit the data; second: penalize a too-flexible model. About 20 numbers are fitted, no weights. K_y = K + n²I.", {"level": None})], size=13)
pic(s, "gp.png", 7.2, 1.45, w=5.5)
caption(s, 7.2, 4.85, 5.5, "Illustration in one dimension (made-up function). The real model has 18 voltage knobs.", size=12, h=0.5)
cite(s, "Rasmussen & Williams, Gaussian Processes for Machine Learning (MIT Press, 2006).")

# ------------------------------------------------------------------ slide 21: multi-amplifier GP
s = S[20]; clear(s); title(s, "Multi-Amplifier GP Model")
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

# ------------------------------------------------------------------ slide 22: BO loop
s = S[21]; clear(s); title(s, "Bayesian Optimization (BO) Loop")
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

# ------------------------------------------------------------------ slide 23: BO tests
s = S[22]; clear(s); title(s, "BO Tests")
text(s, 0.6, 1.3, 6.6, 2.6, [
    "**Replay of the v3 night:** each amplifier in turn plays the new CCD",
    ("Candidates: its 63–64 measured settings in one half of the night (unusable ones included)", {"level": 1}),
    ("Model data: the other two amplifiers in the other half", {"level": 1}),
    ("“Take a trace” = use its stored traces; “take an image” = reveal its stored σe, λ and usable flag", {"level": 1}),
    "3 amplifiers × 2 halves = **6 searches**",
], size=15)
text(s, 0.6, 3.95, 6.6, 1.3, [
    "**Standards:**",
    ("Random order (mean of 200 orders)", {"level": 1}),
    ("By hand: hill-climb, always image the untested setting nearest to the best so far", {"level": 1}),
], size=15)
text(s, 0.6, 5.35, 6.6, 0.75, [("**Metric:** number of images until the best found is within 2 % of the best usable value in the pool", {"level": None})], size=15)
eq(s, "gap", 0.9, 6.1, scale=0.9)
gx, gy, cw, ch = 8.6, 1.9, 1.9, 0.75
text(s, gx, gy - 0.5, cw, 0.4, [("first half", {"level": None, "bold": True})], size=13, align=PP_ALIGN.CENTER)
text(s, gx + cw, gy - 0.5, cw, 0.4, [("second half", {"level": None, "bold": True})], size=13, align=PP_ALIGN.CENTER)
for r, amp in enumerate(["ch0 (new)", "ch1", "ch3"]):
    text(s, gx - 1.45, gy + r * ch + 0.18, 1.4, 0.4, [(amp, {"level": None})], size=13, align=PP_ALIGN.RIGHT)
    for c in range(2):
        if r == 0 and c == 0:
            fill, lab = ORANGE, "candidates"
        elif r > 0 and c == 1:
            fill, lab = GREEN, "model data"
        else:
            fill, lab = RGBColor(0xF2, 0xF2, 0xF2), "not used"
        box(s, gx + c * cw, gy + r * ch, cw - 0.08, ch - 0.08, lab, fill=fill, size=13, shape=MSO_SHAPE.RECTANGLE)
text(s, gx - 1.4, gy + 3 * ch + 0.2, 4.7, 1.2, [("The new amplifier never shares a time block with its model data. Repeat for every amplifier and half.", {"level": None})], size=14)

# ------------------------------------------------------------------ slide 24: BO results
s = S[23]; clear(s); title(s, "BO Results")
table(s, [["images to 2 % (wasted)", "noise", "noise + DC", "DC only"],
          ["random order", "6.5 (1.5)", "31.0 (4.8)", "38.5 (5.7)"],
          ["by hand (hill-climb)", "11.5 (4.4)", "24.0 (3.1)", "26.8 (3.4)"],
          ["multi-amp model, ranking only", "4.3 (0.5)", "8.3 (1.0)", "8.3 (1.0)"],
          ["trace ranking (EDT)", "**1.3** (0)", "7.7 (0)", "7.3 (0)"],
          ["BO, new amplifier's data only", "8.2 (2.2)", "3.2 (0.3)", "4.0 (0.5)"],
          ["multi-amp BO + EDT gate", "3.5 (0.2)", "**1.7** (0)", "**1.8** (0)"],
          ["hybrid: trace pick, then BO", "2.8 (0)", "2.2 (0)", "**1.8** (0)"]],
      0.6, 1.35, 7.6, col_w=[3.4, 1.4, 1.4, 1.4], size=14, row_h=0.44)
caption(s, 0.6, 4.95, 7.6, "Nothing known about the new amplifier; mean of 6 searches; in brackets: images wasted on unusable settings. Differences below ≈1 image are not significant.", size=12, h=0.6)
text(s, 0.6, 5.7, 7.6, 1.2, [("With the nominal setting known, stepping out from it by hand is best for noise (1.5 images); for dark charge, BO still needs only 2.7 against 26.0.", {"level": None})], size=14)
text(s, 8.6, 1.35, 4.1, 5.5, [
    ("**Dark charge:** multi-amp BO + gate reaches 2 % in **1.8 images**, against 26.8 by hand and 38.5 at random, with no wasted image", {"level": 0, "space": 12}),
    ("**Noise:** the trace measures it directly, so ranking by the trace wins (1.3)", {"level": 0, "space": 12}),
    ("**Limits:** one board, one night, sibling amplifiers", {"level": 0, "space": 12}),
    ("**Next:** a live run on a CCD the model has never seen", {"level": 0}),
], size=16)

prs.save(OUT)
print("saved", OUT)

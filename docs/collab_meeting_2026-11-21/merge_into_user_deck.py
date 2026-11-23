"""Merge the copy's new content into the user's saved, animated deck without touching animated shapes.
usage: python merge.py USER.pptx COPY.pptx OUT.pptx FIGDIR"""
import copy, hashlib, io, sys
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.oxml.ns import qn
from PIL import Image

USER, COPY, OUT, FIG = sys.argv[1:5]
U = Presentation(USER); C = Presentation(COPY)
us, cs = list(U.slides), list(C.slides)
h = lambda b: hashlib.md5(b).hexdigest()[:6]
R_EMBED = qn("r:embed")


def swap_image(slide, old_hash, new_path, keep="width"):
    """Replace the picture whose image hashes to old_hash by new_path, same shape (id, position, animations)."""
    for sh in slide.shapes:
        if sh.shape_type == 13 and h(sh.image.blob) == old_hash:
            _, rId = slide.part.get_or_add_image_part(new_path)
            sh._element.find(".//" + qn("a:blip")).set(R_EMBED, rId)
            W, H = Image.open(new_path).size
            if keep == "width":
                sh.height = Emu(int(sh.width * H / W))
            return True
    raise SystemExit(f"image {old_hash} not found")


def copy_shapes(src, dst, skip_title=True):
    """Deep-copy the shapes of slide src into slide dst, re-linking pictures."""
    for sh in src.shapes:
        if skip_title and sh.is_placeholder and sh.placeholder_format.type in (1, 3):
            continue
        el = copy.deepcopy(sh._element)
        for blip in el.iter(qn("a:blip")):
            rid = blip.get(R_EMBED)
            if rid:
                blob = src.part.related_part(rid).blob
                _, new = dst.part.get_or_add_image_part(io.BytesIO(blob))
                blip.set(R_EMBED, new)
        dst.shapes._spTree.append(el)


def set_title(dst, src):
    ts = [s for s in src.shapes if s.is_placeholder and s.placeholder_format.type in (1, 3)][0]
    td = [s for s in dst.shapes if s.is_placeholder and s.placeholder_format.type in (1, 3)][0]
    td._element.getparent().replace(td._element, copy.deepcopy(ts._element))


def replace_slide(dst, src):
    assert dst._element.find(qn("p:timing")) is None, "slide has animations"
    for sh in list(dst.shapes):
        if not (sh.is_placeholder and sh.placeholder_format.type in (1, 3)):
            sh._element.getparent().remove(sh._element)
    set_title(dst, src)
    copy_shapes(src, dst)


def new_slide_from(src, after_slide):
    lay = [l for l in U.slide_layouts if l.name == src.slide_layout.name]
    s = U.slides.add_slide(lay[0] if lay else U.slide_layouts[0])
    for ph in list(s.placeholders):
        if ph.placeholder_format.type not in (1, 3):
            ph._element.getparent().remove(ph._element)
    if not [x for x in s.shapes if x.is_placeholder and x.placeholder_format.type in (1, 3)]:
        raise SystemExit("layout has no title")
    set_title(s, src); copy_shapes(src, s)
    lst = U.slides._sldIdLst; ids = list(lst)
    new = ids[-1]; lst.remove(new)
    pos = [i for i, e in enumerate(lst) if U.part.related_part(e.get(qn("r:id"))) is after_slide.part][0]
    lst.insert(pos + 1, new)
    return s


# ---- slides 4 and 5 (animated on 4): clock-setting plots with the 'kADU relative' axis, swapped in place
swap_image(us[3], "db811d", f"{FIG}/v_setting.png")
swap_image(us[4], "606c96", f"{FIG}/h_setting.png")
# ---- slide 7 (animated): the two-skip plot -> how sigma_1(tau) is measured, same shape
swap_image(us[6], "e08f68", f"{FIG}/sigma1_tau.png")

# ---- slide 9: new step-by-step tree slide, keeping the user's shorter 'Training data' line
replace_slide(us[8], cs[8])
for sh in us[8].shapes:
    if sh.has_text_frame:
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                if " (trace numbers, voltages)" in r.text:
                    r.text = r.text.replace(" (trace numbers, voltages)", "")

# ---- slide 11: keep the user's slide; two plots instead of one; add the seen-amplifier line
s = us[10]
for sh in list(s.shapes):
    if sh.shape_type == 13 and h(sh.image.blob) == "12ec0a":
        x, y, w = sh.left, sh.top, sh.width
        sh._element.getparent().remove(sh._element)
    elif sh.has_text_frame and sh.text_frame.text.startswith("Typical error = RMS"):
        sh._element.getparent().remove(sh._element)
s.shapes.add_picture(f"{FIG}/noise_new.png", Inches(8.4), Inches(1.15), width=Inches(4.3))
s.shapes.add_picture(f"{FIG}/noise_seen.png", Inches(8.4), Inches(4.1), width=Inches(4.3))
for sh in s.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith("Goal:"):
        tf = sh.text_frame
        res = [p for p in tf.paragraphs if p.text.startswith("Result:")][0]
        for r in res.runs:
            if r.text.startswith("Result"):
                r.text = r.text.replace("Result", "New amplifier")
        newp = copy.deepcopy(res._p); res._p.addnext(newp)
        from pptx.text.text import _Paragraph
        np_ = _Paragraph(newp, res._parent)
        for r in list(np_.runs)[1:]:
            r._r.getparent().remove(r._r)
        r0 = np_.runs[0]; r0.text = "Seen amplifier:"; r0.font.bold = True
        for part, bold in [(" adding everything (also the 141 general trace features and the signal sizes) does best, ", False), ("6.1 %", True), (", but falls to 12.7 % on a new amplifier", False)]:
            r = np_.add_run(); r.text = part; r.font.bold = bold; r.font.size = r0.font.size
        break

# ---- slide 14: new GP slide
replace_slide(us[13], cs[13])

# ---- backup: tier P 1 and 2 after the sigma_500 backup; correlations after the no-ML formula
t = lambda sl: [x for x in sl.shapes if x.is_placeholder and x.placeholder_format.type in (1, 3)][0].text_frame.text
src = {t(x)[:40]: x for x in cs}
s500 = [x for x in us if t(x).startswith("Backup: Predicted 500-Skip Noise")][0]
form = [x for x in us if t(x).startswith("Backup: Image Noise Without ML")][0]
p1 = new_slide_from([x for x in cs if t(x).startswith("Backup: Always-Included Trace Features (Tier P) 1")][0], s500)
p2 = new_slide_from([x for x in cs if t(x).startswith("Backup: Always-Included Trace Features (Tier P) 2")][0], p1)
new_slide_from([x for x in cs if t(x).startswith("Backup: How Each Voltage Correlates")][0], form)
U.save(OUT)
print("saved", OUT, len(U.slides))

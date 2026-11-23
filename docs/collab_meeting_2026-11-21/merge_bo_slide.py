import copy, sys
from pptx import Presentation
from pptx.util import Inches
from pptx.oxml.ns import qn
U = Presentation(sys.argv[1]); C = Presentation(sys.argv[2])
t = lambda sl: ([s for s in sl.shapes if s.is_placeholder and s.placeholder_format.type in (1, 3)] or [None])[0]
dst = [s for s in U.slides if t(s) and t(s).text_frame.text.startswith("BO Results: Finding")][0]
src = [s for s in C.slides if t(s) and t(s).text_frame.text.startswith("BO Results: Finding")][0]
assert dst._element.find(qn("p:timing")) is None
for sh in list(dst.shapes):
    txt = sh.text_frame.text if sh.has_text_frame else ""
    if txt.startswith("The test"):
        sh._element.getparent().remove(sh._element)                 # the user's short test description -> step-by-step version
    elif sh.has_table:
        sh.top = Inches(3.75)
        for r in sh.table.rows: r.height = Inches(0.5)
    elif sh.shape_type == 13:
        sh.top = Inches(3.75)
    elif txt.startswith("Using the other amplifiers"):
        sh.top = Inches(6.45)
n = 0
for sh in src.shapes:                                               # the steps text, the halves diagram and its caption
    if sh.is_placeholder or sh.top >= Inches(3.6):
        continue
    el = copy.deepcopy(sh._element); dst.shapes._spTree.append(el); n += 1
    new = dst.shapes[-1]
    if new.has_text_frame and new.text_frame.text.startswith("The test"):
        new.top = new.top + Inches(0.15)                            # clear the user's two-line title
U.save(sys.argv[3]); print("added", n, "shapes; saved", sys.argv[3])

"""Build five compact Run7 model architectures with the FACETpy toolkit.

Run from any directory: python3 /path/to/build_run7_compact_a.py
Requires rsvg-convert on PATH. Canvas units stay fixed; SVG print width is 16 cm.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "skills/facetpy-diagram/assets"))
from facetpy_svg import Diagram, C, pill, capsule, edge, text  # noqa: E402

OUT = ROOT / "docs/source/_static/diagrams/run7_kompakt"


def node(d, x, y, w, h, label, sub=None, terminal=False):
    box = (capsule if terminal else pill)(x, y, w, h, "")
    lines = label.split("\n")
    line_height = 29
    total = len(lines) * line_height + (29 if sub else 0)
    baseline = y + (h - total) / 2 + 24
    for i, line in enumerate(lines):
        box["svg"] += text(x+w/2, baseline+i*line_height, line, size=24,
                           weight=600, anchor="middle",
                           fill=C["header_fg"] if terminal else C["ink"])
    if sub:
        box["svg"] += text(x+w/2, baseline+len(lines)*line_height, sub,
                           size=22, anchor="middle",
                           fill=C["header_fg"] if terminal else C["slate"])
    d.add(box)
    return box


def wire(d, points, dashed=False, arrow=True):
    fragment = edge(points, marker_end="arrow" if arrow else None).replace('stroke-width="1.5"', 'stroke-width="2"')
    if dashed:
        fragment = fragment.replace('<polyline ', '<polyline stroke-dasharray="7 6" ')
    d.add_edge(fragment)


def label(d, x, y, value):
    d.add(text(x, y, value, size=22, fill=C["blue"], anchor="middle"))


def save(d, name):
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"{name}.svg"
    png = OUT / f"{name}.png"
    d.render_png(str(png), svg_path=str(svg), width=2400)
    content = svg.read_text()
    content = re.sub(r'width="1000" height="[^\"]+"',
                     f'width="160mm" height="{d.height * .16:g}mm"',
                     content, count=1)
    svg.write_text(content)
    print(f"{name}: 1000 × {d.height}; PNG 2400 × {round(d.height*2.4)}")


def cascade(context=False):
    d = Diagram(410, grid=False, footer_text=None)
    node(d, 40, 60, 210, 95, "Epoch context" if context else "EEG epoch", terminal=True)
    node(d, 340, 60, 240, 95, "Context DAE 1" if context else "DAE 1",
         "Encoder · Decoder")
    node(d, 130, 270, 60, 60, "−", terminal=True)
    node(d, 260, 260, 195, 80, "Residual\ncontext" if context else "Residual\nsignal")
    node(d, 510, 250, 230, 100, "Context DAE 2" if context else "DAE 2",
         "Encoder · Decoder")
    node(d, 790, 78, 60, 60, "+", terminal=True)
    node(d, 780, 250, 200, 110,
         "Center\nartifact" if context else "Artifact\nestimate", terminal=True)
    wire(d, [(253, 108), (336, 108)])
    wire(d, [(145, 158), (145, 190), (80, 190), (80, 300), (126, 300)])
    wire(d, [(460, 158), (460, 215), (160, 215), (160, 266)])
    wire(d, [(193, 300), (256, 300)])
    wire(d, [(458, 300), (506, 300)])
    wire(d, [(583, 108), (786, 108)])
    wire(d, [(625, 353), (625, 385), (760, 385), (760, 175), (820, 175), (820, 142)])
    wire(d, [(853, 108), (900, 108), (900, 246)])
    label(d, 690, 91, "Artifact 1")
    label(d, 688, 374, "Artifact 2")
    if context:
        label(d, 288, 204, "Center epoch only")
    save(d, "cascaded_context_dae" if context else "cascaded_dae")


def dpae():
    d = Diagram(360, grid=False, footer_text=None)
    node(d, 30, 140, 180, 80, "EEG epoch", terminal=True)
    node(d, 285, 45, 225, 90, "Local pathway", "Fine structure")
    node(d, 285, 235, 225, 90, "Global pathway", "Slow structure")
    node(d, 570, 140, 115, 80, "Fusion")
    node(d, 730, 140, 115, 80, "Decoder")
    node(d, 880, 130, 105, 100, "Artifact", terminal=True)
    wire(d, [(213,180),(245,180),(245,90),(281,90)])
    wire(d, [(245,180),(245,280),(281,280)])
    wire(d, [(513,90),(540,90),(540,180),(566,180)])
    wire(d, [(513,280),(540,280),(540,180)], arrow=False)
    wire(d, [(688,180),(726,180)])
    wire(d, [(848,180),(876,180)])
    save(d, "dpae")


def tasnet():
    d = Diagram(380, grid=False, footer_text=None)
    node(d, 30, 50, 180, 80, "EEG epoch", terminal=True)
    node(d, 260, 50, 155, 80, "Encoder")
    node(d, 470, 40, 230, 100, "TCN", "Temporal patterns")
    node(d, 760, 50, 210, 80, "Source masks")
    node(d, 835, 245, 60, 60, "×", terminal=True)
    node(d, 570, 235, 185, 80, "Decoder")
    node(d, 210, 225, 260, 100, "Source estimates", "EEG | Artifact", terminal=True)
    wire(d, [(213,90),(256,90)])
    wire(d, [(418,90),(466,90)])
    wire(d, [(703,90),(756,90)])
    wire(d, [(865,133),(865,160),(935,160),(935,275),(899,275)])
    wire(d, [(338,133),(338,195),(865,195),(865,241)])
    label(d, 565, 186, "Latent representation")
    wire(d, [(832,275),(759,275)])
    wire(d, [(567,275),(474,275)])
    save(d, "conv_tasnet")


def demucs():
    d = Diagram(330, grid=False, footer_text=None)
    node(d, 30, 170, 210, 80, "Epoch context", terminal=True)
    node(d, 275, 160, 160, 100, "CNN\nencoder")
    node(d, 490, 170, 135, 80, "BiLSTM")
    node(d, 680, 160, 160, 100, "CNN\ndecoder")
    node(d, 880, 160, 105, 100, "Artifact", terminal=True)
    wire(d, [(243,210),(271,210)])
    wire(d, [(438,210),(486,210)])
    wire(d, [(628,210),(676,210)])
    wire(d, [(843,210),(876,210)])
    wire(d, [(355,157),(355,80),(760,80),(760,156)], dashed=True)
    label(d, 560, 66, "Skip connections (+)")
    save(d, "demucs")


if __name__ == "__main__":
    cascade()
    cascade(context=True)
    dpae()
    tasnet()
    demucs()

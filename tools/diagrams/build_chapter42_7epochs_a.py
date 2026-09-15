"""Build five chapter 4.2 model architectures with seven-epoch inputs with the FACETpy toolkit.

Run from any directory: python3 /path/to/build_chapter42_7epochs_a.py
Requires rsvg-convert on PATH. Canvas units stay fixed; SVG print width is 16 cm.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "skills/facetpy-diagram/assets"))
from facetpy_svg import Diagram, C, pill, capsule, edge, text  # noqa: E402

OUT = ROOT / "docs/source/_static/diagrams/chapter4_2_7epochs"


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
    d = Diagram(430, grid=False, footer_text=None)
    node(d, 30, 50, 235, 110, "7 EEG epochs", "Context stack" if context else "Concatenated", terminal=True)
    node(d, 340, 55, 240, 100, "Context DAE 1" if context else "DAE 1", "Encoder → decoder")
    node(d, 130, 275, 60, 60, "−", terminal=True)
    node(d, 260, 265, 195, 80, "Updated\ncontext" if context else "Residual\nsignal")
    node(d, 510, 255, 230, 100, "Context DAE 2" if context else "DAE 2", "Encoder → decoder")
    node(d, 790, 78, 60, 60, "+", terminal=True)
    node(d, 780, 250, 200, 120, "Center\nartifact", None if context else "Select center", terminal=True)
    wire(d, [(268,105),(336,105)])
    wire(d, [(145,163),(145,190),(80,190),(80,305),(126,305)])
    wire(d, [(460,158),(460,215),(160,215),(160,271)])
    wire(d, [(193,305),(256,305)])
    wire(d, [(458,305),(506,305)])
    wire(d, [(583,108),(786,108)])
    wire(d, [(625,358),(625,395),(760,395),(760,175),(820,175),(820,142)])
    wire(d, [(853,108),(900,108),(900,246)])
    label(d, 690, 91, "Artifact 1")
    label(d, 688, 384, "Artifact 2")
    label(d, 305, 204, "Update center only" if context else "Subtract across 7 epochs")
    save(d, "cascaded_context_dae" if context else "cascaded_dae")


def dpae():
    d = Diagram(440, grid=False, footer_text=None)
    node(d, 30, 160, 200, 100, "7 EEG epochs", "Concatenated", terminal=True)
    node(d, 295, 40, 220, 100, "Local pathway", "Small kernels")
    node(d, 295, 290, 220, 100, "Global pathway", "Large kernels")
    node(d, 575, 160, 150, 100, "Feature\nfusion", "Concatenate")
    node(d, 780, 160, 190, 100, "Decoder")
    node(d, 755, 310, 225, 110, "Center artifact", "Select center", terminal=True)
    wire(d, [(233,210),(260,210),(260,90),(291,90)])
    wire(d, [(260,210),(260,340),(291,340)])
    wire(d, [(518,90),(545,90),(545,210),(571,210)])
    wire(d, [(518,340),(545,340),(545,210)], arrow=False)
    wire(d, [(728,210),(776,210)])
    wire(d, [(875,263),(875,306)])
    save(d, "dpae")


def tasnet():
    d = Diagram(460, grid=False, footer_text=None)
    node(d, 20, 45, 200, 105, "7 EEG epochs", "Concatenated", terminal=True)
    node(d, 265, 45, 170, 105, "Conv1D\nencoder")
    node(d, 485, 45, 205, 105, "TCN\nseparator")
    node(d, 755, 30, 220, 65, "EEG mask")
    node(d, 755, 125, 220, 65, "Artifact mask")
    node(d, 825, 290, 60, 60, "×", terminal=True)
    node(d, 520, 270, 230, 120, "Conv1D decoder", "Shared weights")
    node(d, 160, 270, 275, 120, "Center estimates", "EEG | Artifact", terminal=True)
    wire(d, [(223,97),(261,97)])
    wire(d, [(438,97),(481,97)])
    wire(d, [(693,97),(725,97),(725,62),(751,62)])
    wire(d, [(725,97),(725,157),(751,157)])
    wire(d, [(978,62),(990,62),(990,240),(920,240),(920,320),(889,320)])
    wire(d, [(865,193),(865,225),(955,225),(955,240)], arrow=False)
    wire(d, [(350,153),(350,225),(800,225),(800,260),(855,260),(855,286)])
    label(d, 557, 215, "Encoded features")
    label(d, 855, 418, "Apply each mask")
    wire(d, [(822,320),(754,320)])
    wire(d, [(517,330),(439,330)])
    save(d, "conv_tasnet")


def demucs():
    d = Diagram(430, grid=False, footer_text=None)
    node(d, 25, 160, 200, 110, "7 EEG epochs", "Concatenated", terminal=True)
    node(d, 265, 160, 210, 110, "CNN encoder", "GLU gates")
    node(d, 530, 160, 185, 110, "BiLSTM\nbottleneck")
    node(d, 765, 160, 210, 110, "CNN decoder", "GLU gates")
    node(d, 765, 320, 210, 70, "Select center")
    node(d, 440, 310, 230, 90, "Artifact estimate", terminal=True)
    wire(d, [(228,215),(261,215)])
    wire(d, [(478,215),(526,215)])
    wire(d, [(718,215),(761,215)])
    wire(d, [(870,273),(870,316)])
    wire(d, [(762,355),(674,355)])
    wire(d, [(370,157),(370,75),(870,75),(870,156)], dashed=True)
    label(d, 620, 61, "Skip connections (+)")
    save(d, "demucs")


if __name__ == "__main__":
    cascade()
    cascade(context=True)
    dpae()
    tasnet()
    demucs()

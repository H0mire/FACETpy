"""Rebuild four compact, methodological Run7 model diagrams."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "skills/facetpy-diagram/assets"))
from facetpy_svg import C, Diagram, capsule, container, edge, pill, text

OUT = ROOT / "docs/source/_static/diagrams/run7_kompakt"


def node(d, x, y, w, labels, terminal=False, h=72):
    """Use toolkit geometry and shapes with document-sized multiline text."""
    labels = labels.split("\n")
    box = (capsule if terminal else pill)(x, y, w, h, "")
    box["svg"] = re.sub(r"<text\b.*?</text>", "", box["svg"])
    for i, label in enumerate(labels):
        box["svg"] += text(
            x + w / 2, y + h / 2 + 8 + (i - (len(labels) - 1) / 2) * 28,
            label, size=24, weight=600 if terminal else 500,
            anchor="middle", fill=C["header_fg"] if terminal else C["ink"],
        )
    d.add(box)
    return box


def group(d, x, y, w, h):
    box = container(x, y, w, h, "")
    # A bare grouping frame has no header or title tab.
    box["svg"] = re.sub(r"<path\b[^>]*/>|<text\b.*?</text>", "", box["svg"])
    d.add(box)


def link(d, points, dashed=False):
    fragment = edge(points)
    if dashed:
        fragment = fragment.replace("<polyline ", '<polyline stroke-dasharray="7 6" ')
    d.add_edge(fragment)


def finish(d, name):
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"{name}.svg"
    png = OUT / f"{name}.png"
    markup = d.render_svg().replace(
        f'width="1000" height="{d.height}"',
        f'width="160mm" height="{d.height * 0.16:g}mm"', 1,
    )
    svg.write_text(markup)
    subprocess.run(["rsvg-convert", "-w", "2400", str(svg), "-o", str(png)], check=True)
    print(svg.relative_to(ROOT))


def sepformer():
    d = Diagram(420, grid=False, footer_text=None)
    group(d, 355, 35, 255, 350)
    node(d, 20, 50, 180, "Epoch\ncontext", True)
    node(d, 225, 50, 110, "Encoder")
    node(d, 370, 50, 225, "Chunks")
    node(d, 370, 175, 225, "Intra-attention")
    node(d, 370, 300, 225, "Inter-attention")
    node(d, 640, 300, 145, "Mask +\nDecoder")
    node(d, 820, 300, 160, "Center\nartifact", True)
    link(d, [(200, 86), (222, 86)])
    link(d, [(335, 86), (367, 86)])
    link(d, [(482.5, 122), (482.5, 172)])
    link(d, [(482.5, 247), (482.5, 297)])
    link(d, [(595, 336), (637, 336)])
    link(d, [(785, 336), (817, 336)])
    # Encoded features also supply the multiplicative latent mask path.
    link(d, [(280, 50), (280, 20), (712.5, 20), (712.5, 297)], True)
    finish(d, "sepformer")


def denoise_mamba():
    d = Diagram(445, grid=False, footer_text=None)
    group(d, 265, 170, 505, 120)
    node(d, 370, 35, 260, "EEG epoch", True)
    node(d, 30, 195, 175, "Projection")
    node(d, 285, 195, 160, "Convolution")
    node(d, 495, 195, 190, "Selective\nSSM")
    node(d, 715, 211, 40, "+", h=40)
    node(d, 805, 195, 175, "Projection")
    node(d, 360, 345, 280, "Artifact estimate", True)
    link(d, [(500, 107), (500, 140), (117.5, 140), (117.5, 192)])
    link(d, [(205, 231), (282, 231)])
    link(d, [(445, 231), (492, 231)])
    link(d, [(685, 231), (712, 231)])
    link(d, [(755, 231), (802, 231)])
    link(d, [(235, 231), (235, 310), (735, 310), (735, 254)], True)
    link(d, [(892.5, 267), (892.5, 325), (500, 325), (500, 342)])
    finish(d, "denoise_mamba")


def ic_unet():
    d = Diagram(485, grid=False, footer_text=None)
    node(d, 20, 45, 220, "Multichannel\ncontext", True)
    node(d, 290, 45, 135, "ICA")
    node(d, 485, 45, 190, "Encoder")
    node(d, 485, 305, 190, "Bottleneck")
    node(d, 740, 305, 210, "Decoder")
    node(d, 750, 45, 190, "Inverse ICA")
    node(d, 730, 405, 230, "Center EEG", True)
    link(d, [(240, 81), (287, 81)])
    link(d, [(425, 81), (482, 81)])
    link(d, [(580, 117), (580, 302)])
    link(d, [(675, 341), (737, 341)])
    link(d, [(845, 305), (845, 120)])
    # The IC residual is abstracted into the decoder; retain only U-Net skips.
    link(d, [(675, 81), (710, 81), (710, 270), (790, 270), (790, 302)], True)
    link(d, [(940, 81), (980, 81), (980, 390), (845, 390), (845, 402)])
    finish(d, "ic_unet")


def st_gnn():
    d = Diagram(300, grid=False, footer_text=None)
    group(d, 230, 150, 550, 110)
    node(d, 20, 170, 190, "Multichannel\ncontext", True)
    node(d, 250, 170, 150, "Temporal\nconvolution")
    node(d, 430, 170, 150, "Graph\nconvolution")
    node(d, 610, 170, 150, "Temporal\nconvolution")
    node(d, 795, 170, 185, "Center\nartifact", True)
    node(d, 382.5, 40, 245, "Electrode graph")
    link(d, [(210, 206), (247, 206)])
    link(d, [(400, 206), (427, 206)])
    link(d, [(580, 206), (607, 206)])
    link(d, [(760, 206), (792, 206)])
    link(d, [(505, 112), (505, 167)], True)
    finish(d, "st_gnn")


if __name__ == "__main__":
    sepformer()
    denoise_mamba()
    ic_unet()
    st_gnn()

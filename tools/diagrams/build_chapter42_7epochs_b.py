"""Build compact chapter 4.2 diagrams grounded in seven-epoch configurations.

Demucs MC is the documented Run6 fallback; the other four depict Run7.
Shapes, palette, typography and connectors come from the FACETpy toolkit.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "skills/facetpy-diagram/assets"))
from facetpy_svg import C, Diagram, capsule, container, edge, pill, text

OUT = ROOT / "docs/source/_static/diagrams/chapter4_2_7epochs"


def node(d, x, y, w, labels, terminal=False, h=80, secondary=False):
    labels = labels.split("\n")
    box = (capsule if terminal else pill)(x, y, w, h, "")
    box["svg"] = re.sub(r"<text\b.*?</text>", "", box["svg"])
    for i, label in enumerate(labels):
        box["svg"] += text(
            x + w / 2, y + h / 2 + 8 + (i - (len(labels) - 1) / 2) * 28,
            label, size=22 if secondary and i else 24,
            weight=600 if terminal else 500, anchor="middle",
            fill=C["header_fg"] if terminal else C["ink"],
        )
    d.add(box)
    return box


def group(d, x, y, w, h, label=None):
    box = container(x, y, w, h, "")
    box["svg"] = re.sub(r"<path\b[^>]*/>|<text\b.*?</text>", "", box["svg"])
    if label:
        box["svg"] += text(x + 18, y + 29, label, size=22, weight=600, fill=C["blue"])
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
    print(f"{name}: 160 × {d.height * 0.16:g} mm")


def sepformer():
    d = Diagram(545, grid=False, footer_text=None)
    group(d, 370, 35, 260, 365)
    node(d, 20, 50, 180, "7 EEG epochs\nConcatenated", True, secondary=True)
    node(d, 230, 50, 120, "Conv1D\nencoder")
    node(d, 385, 50, 230, "Overlapping\nchunks")
    node(d, 385, 175, 230, "Intra-chunk\nattention")
    node(d, 385, 300, 230, "Inter-chunk\nattention")
    node(d, 650, 300, 330, "Merge chunks\nMask × features")
    node(d, 650, 430, 155, "Conv1D\ndecoder")
    node(d, 835, 430, 145, "Center\nartifact", True)
    link(d, [(200, 90), (227, 90)])
    link(d, [(350, 90), (382, 90)])
    link(d, [(500, 130), (500, 172)])
    link(d, [(500, 255), (500, 297)])
    link(d, [(615, 340), (647, 340)])
    link(d, [(727.5, 380), (727.5, 427)])
    link(d, [(805, 470), (832, 470)])
    link(d, [(290, 50), (290, 20), (815, 20), (815, 297)], True)
    finish(d, "sepformer")


def denoise_mamba():
    d = Diagram(505, grid=False, footer_text=None)
    group(d, 265, 170, 505, 145, "ConvSSD blocks")
    node(d, 370, 35, 260, "7 EEG epochs\nConcatenated", True, secondary=True)
    node(d, 30, 220, 175, "Input\nprojection")
    node(d, 285, 220, 160, "Local\nconvolution")
    node(d, 495, 220, 190, "Selective\nSSM")
    node(d, 715, 240, 40, "+", h=40)
    node(d, 805, 220, 175, "Output\nprojection")
    node(d, 360, 390, 280, "Center artifact\nSelect center", True, secondary=True)
    link(d, [(500, 115), (500, 140), (117.5, 140), (117.5, 217)])
    link(d, [(205, 260), (282, 260)])
    link(d, [(445, 260), (492, 260)])
    link(d, [(685, 260), (712, 260)])
    link(d, [(755, 260), (802, 260)])
    link(d, [(235, 260), (235, 345), (735, 345), (735, 283)], True)
    link(d, [(892.5, 300), (892.5, 365), (500, 365), (500, 387)])
    finish(d, "denoise_mamba")


def ic_unet():
    d = Diagram(385, grid=False, footer_text=None)
    group(d, 380, 35, 470, 315, "U-Net")
    node(d, 20, 80, 170, "7 EEG epochs\nAll channels", True, secondary=True)
    node(d, 220, 80, 140, "ICA\ntransform")
    node(d, 400, 80, 180, "Encoder")
    node(d, 400, 240, 180, "Bottleneck")
    node(d, 650, 240, 180, "Decoder")
    node(d, 880, 240, 100, "Inverse\nICA")
    node(d, 880, 80, 100, "Center\nEEG", True)
    link(d, [(190, 120), (217, 120)])
    link(d, [(360, 120), (397, 120)])
    link(d, [(490, 160), (490, 237)])
    link(d, [(580, 280), (647, 280)])
    link(d, [(830, 280), (877, 280)])
    link(d, [(930, 240), (930, 163)])
    # U-Net skips; clean component residual is abstracted into reconstruction.
    link(d, [(580, 120), (740, 120), (740, 237)], True)
    finish(d, "ic_unet")


def st_gnn():
    d = Diagram(510, grid=False, footer_text=None)
    group(d, 255, 35, 520, 145, "ST-Conv blocks")
    node(d, 20, 80, 190, "7 EEG epochs\nAll channels", True, secondary=True)
    node(d, 270, 80, 140, "Temporal\nGLU")
    node(d, 445, 80, 140, "ChebConv")
    node(d, 620, 80, 140, "Temporal\nGLU")
    node(d, 820, 80, 160, "Prediction\nhead")
    node(d, 820, 245, 160, "Select\ncenter")
    node(d, 795, 390, 185, "Multichannel\nartifact", True)
    node(d, 380, 245, 270, "Electrode graph\nGraph Laplacian", secondary=True)
    link(d, [(210, 120), (267, 120)])
    link(d, [(410, 120), (442, 120)])
    link(d, [(585, 120), (617, 120)])
    link(d, [(760, 120), (817, 120)])
    link(d, [(900, 160), (900, 242)])
    link(d, [(900, 325), (900, 387)])
    link(d, [(515, 245), (515, 163)], True)
    finish(d, "st_gnn")


def demucs_mc():
    d = Diagram(395, grid=False, footer_text=None)
    node(d, 20, 50, 240, "FARM residual\n7 epochs × 3 channels", True, secondary=True)
    node(d, 290, 50, 185, "Shared CNN\nencoder")
    node(d, 510, 50, 235, "Cross-channel\nattention")
    node(d, 795, 50, 160, "BiLSTM")
    node(d, 795, 260, 160, "Shared CNN\ndecoder")
    node(d, 570, 260, 175, "Center\nepochs")
    node(d, 285, 260, 225, "1×1 channel\nmixing")
    node(d, 20, 260, 200, "Target\nresidual", True)
    link(d, [(260, 90), (287, 90)])
    link(d, [(475, 90), (507, 90)])
    link(d, [(745, 90), (792, 90)])
    link(d, [(955, 90), (980, 90), (980, 300), (958, 300)])
    link(d, [(795, 300), (748, 300)])
    link(d, [(570, 300), (513, 300)])
    link(d, [(285, 300), (223, 300)])
    link(d, [(382.5, 130), (382.5, 190), (875, 190), (875, 257)], True)
    finish(d, "demucs_mc")


if __name__ == "__main__":
    sepformer()
    denoise_mamba()
    ic_unet()
    st_gnn()
    demucs_mc()

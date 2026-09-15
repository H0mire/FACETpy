"""Compact methodological Run7 diagrams; rerun from any directory."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'skills/facetpy-diagram/assets'))
from facetpy_svg import Diagram, pill, capsule, text, edge, C

OUT = ROOT / 'docs/source/_static/diagrams/run7_kompakt'


def node(d, x, y, w, label, secondary=None, terminal=False, h=76):
    b = (capsule if terminal else pill)(x, y, w, h, '')
    d.add(b)
    d.add(text(b['cx'], y + (32 if secondary else h/2+8), label,
               size=24, weight=600, anchor='middle',
               fill=C['header_fg'] if terminal else C['ink']))
    if secondary:
        d.add(text(b['cx'], y+59, secondary, size=22, anchor='middle',
                   fill=C['header_fg'] if terminal else C['slate']))
    return b


def arrow(d, points, dashed=False):
    s = edge(points)
    if dashed:
        s = s.replace('stroke-width="1.5"', 'stroke-width="1.5" stroke-dasharray="7 6"')
    d.add_edge(s)


def label(d, x, y, s):
    d.add(text(x, y, s, size=22, anchor='middle', fill=C['slate']))


def diagram(h):
    return Diagram(h, grid=False, footer_text=None)


def save(d, name):
    OUT.mkdir(parents=True, exist_ok=True)
    svg, png = OUT / (name+'.svg'), OUT / (name+'.png')
    d.render_png(str(png), svg_path=str(svg), width=2400)
    body = svg.read_text()
    body = body.replace(f'width="1000" height="{d.height}"',
                        f'width="160mm" height="{d.height*.16:g}mm"', 1)
    svg.write_text(body)
    ET.parse(svg)
    print(name, f'1000×{d.height}; 160×{d.height*.16:g} mm')


def dhct():
    d = diagram(350)
    node(d, 40, 50, 240, 'EEG epoch', terminal=True)
    node(d, 350, 50, 290, 'CNN + Transformer', 'Local + global')
    node(d, 730, 50, 230, 'Bottleneck')
    node(d, 350, 240, 290, 'Artifact decoder')
    node(d, 40, 240, 240, 'Artifact estimate', terminal=True)
    arrow(d, [(283,88),(347,88)])
    arrow(d, [(643,88),(727,88)])
    arrow(d, [(845,129),(845,278),(643,278)])
    arrow(d, [(347,278),(283,278)])
    arrow(d, [(495,129),(495,237)], dashed=True)
    label(d, 568, 190, 'Skips')
    save(d, 'dhct_gan')


def dhct_v2():
    d = diagram(350)
    node(d, 40, 50, 240, 'Epoch context', terminal=True)
    node(d, 350, 50, 240, 'Context fusion', 'Channel stack')
    node(d, 670, 50, 290, 'CNN + Transformer')
    node(d, 670, 240, 290, 'Artifact decoder')
    node(d, 350, 240, 250, 'Center artifact', terminal=True)
    arrow(d, [(283,88),(347,88)])
    arrow(d, [(593,88),(667,88)])
    arrow(d, [(815,129),(815,237)])
    arrow(d, [(920,129),(920,237)], dashed=True)
    label(d, 869, 190, 'Skips')
    arrow(d, [(667,278),(603,278)])
    save(d, 'dhct_gan_v2')


def nested():
    d = diagram(400)
    node(d, 35, 80, 250, 'Epoch context', terminal=True)
    node(d, 345, 80, 300, 'Spectral Restormer', 'STFT → iSTFT')
    node(d, 710, 80, 250, 'Residual context', 'Center − artifact')
    node(d, 710, 285, 250, 'Temporal U-Net', 'Residual artifact')
    node(d, 430, 285, 130, '+')
    node(d, 35, 285, 250, 'Center artifact', terminal=True)
    arrow(d, [(288,118),(342,118)])
    label(d, 313, 68, 'Center')
    arrow(d, [(648,118),(707,118)])
    arrow(d, [(160,77),(160,35),(835,35),(835,77)])
    arrow(d, [(835,159),(835,282)])
    arrow(d, [(707,323),(563,323)])
    arrow(d, [(495,159),(495,282)])
    label(d, 565, 225, 'Artifact')
    arrow(d, [(427,323),(288,323)])
    save(d, 'nested_gan')


def vit():
    d = diagram(370)
    node(d, 35, 50, 250, 'Epoch context', terminal=True)
    node(d, 370, 50, 230, 'STFT')
    node(d, 685, 50, 280, 'Patches + ViT', 'Masked patches')
    node(d, 685, 255, 280, 'Mask × spectrum', 'Complex mask')
    node(d, 370, 255, 230, 'iSTFT', 'Select center')
    node(d, 35, 255, 250, 'Center EEG', terminal=True)
    arrow(d, [(288,88),(367,88)])
    arrow(d, [(603,88),(682,88)])
    arrow(d, [(825,129),(825,252)])
    arrow(d, [(485,129),(485,195),(725,195),(725,252)])
    label(d, 565, 179, 'Spectrum')
    arrow(d, [(682,293),(603,293)])
    arrow(d, [(367,293),(288,293)])
    save(d, 'vit_spectrogram')


def d4pm():
    d = diagram(510)
    node(d, 35, 45, 250, 'EEG epoch', terminal=True)
    node(d, 350, 45, 290, 'Noise state', terminal=True)
    node(d, 705, 45, 260, 'Diffusion step')
    node(d, 350, 220, 290, 'Noise predictor', 'Transformer')
    node(d, 350, 385, 290, 'Reverse step')
    node(d, 705, 385, 260, 'Artifact estimate', terminal=True)
    arrow(d, [(495,124),(495,217)])
    arrow(d, [(160,124),(160,170),(400,170),(400,217)])
    arrow(d, [(835,124),(835,170),(590,170),(590,217)])
    arrow(d, [(495,299),(495,382)])
    arrow(d, [(643,423),(702,423)])
    arrow(d, [(495,464),(495,485),(15,485),(15,20),(495,20),(495,42)])
    label(d, 110, 460, 'Iterative')
    save(d, 'd4pm')


if __name__ == '__main__':
    dhct()
    dhct_v2()
    nested()
    vit()
    d4pm()

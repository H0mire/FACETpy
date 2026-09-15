"""Chapter 4.2 methodological seven-epoch Run7 diagrams; rerun from any directory."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'skills/facetpy-diagram/assets'))
from facetpy_svg import Diagram, pill, capsule, text, edge, C

OUT = ROOT / 'docs/source/_static/diagrams/chapter4_2_7epochs'


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


def dhct(name, stacked=False):
    d = diagram(560)
    node(d, 40, 50, 240, '7 EEG epochs', 'Stacked as channels' if stacked else 'Concatenated in time', terminal=True)
    node(d, 350, 50, 320, 'CNN + LGTB encoder', 'Local + global attention')
    node(d, 750, 50, 220, 'Bottleneck')
    node(d, 350, 250, 320, 'Artifact decoder')
    node(d, 40, 250, 240, 'Center artifact', terminal=True)
    node(d, 750, 250, 220, 'EEG decoder', 'Auxiliary')
    node(d, 500, 450, 300, 'Gated fusion', 'Auxiliary EEG')
    arrow(d, [(283,88),(347,88)])
    arrow(d, [(673,88),(747,88)])
    arrow(d, [(790,129),(790,195),(510,195),(510,247)])
    arrow(d, [(347,288),(283,288)])
    arrow(d, [(860,129),(860,247)], dashed=True)
    arrow(d, [(860,329),(860,488),(803,488)], dashed=True)
    arrow(d, [(510,329),(510,400),(570,400),(570,447)], dashed=True)
    arrow(d, [(37,88),(20,88),(20,488),(497,488)], dashed=True)
    label(d, 170, 470, 'Center input')
    save(d, name)


def nested():
    d = diagram(440)
    node(d, 35, 80, 250, '7 EEG epochs', terminal=True)
    node(d, 345, 80, 300, 'Inner stage', 'Spectral Restormer', h=108)
    d.add(text(495, 167, 'STFT → iSTFT', size=22, anchor='middle', fill=C['slate']))
    node(d, 710, 80, 250, 'Update center', 'Subtract inner artifact', h=108)
    d.add(text(835, 167, 'Keep other epochs', size=22, anchor='middle', fill=C['slate']))
    node(d, 710, 325, 250, 'Outer stage', 'Temporal U-Net')
    node(d, 430, 325, 130, '+')
    node(d, 35, 325, 250, 'Center artifact', terminal=True)
    arrow(d, [(288,118),(342,118)])
    label(d, 313, 68, 'Center')
    arrow(d, [(648,134),(707,134)])
    arrow(d, [(160,77),(160,35),(835,35),(835,77)])
    arrow(d, [(835,191),(835,322)])
    label(d, 750, 262, '7 epochs')
    arrow(d, [(707,363),(563,363)])
    label(d, 633, 347, 'Residual')
    arrow(d, [(495,191),(495,322)])
    label(d, 577, 267, 'Inner artifact')
    arrow(d, [(427,363),(288,363)])
    save(d, 'nested_gan')


def vit():
    d = diagram(430)
    node(d, 30, 60, 190, '7 EEG epochs', terminal=True)
    node(d, 270, 60, 130, 'STFT')
    node(d, 450, 60, 240, 'Patch + mask', 'Center masked')
    node(d, 740, 60, 230, 'ViT encoder')
    node(d, 740, 310, 230, 'Complex mask', 'Linear head')
    node(d, 465, 310, 225, 'Mask × spectrum')
    node(d, 255, 310, 160, 'iSTFT', 'Select center')
    node(d, 30, 310, 175, 'Center EEG', terminal=True)
    arrow(d, [(223,98),(267,98)])
    arrow(d, [(403,98),(447,98)])
    arrow(d, [(693,98),(737,98)])
    arrow(d, [(855,139),(855,307)])
    arrow(d, [(335,139),(335,220),(577,220),(577,307)])
    label(d, 460, 200, 'Original spectrum')
    arrow(d, [(737,348),(693,348)])
    arrow(d, [(462,348),(418,348)])
    arrow(d, [(252,348),(208,348)])
    save(d, 'vit_spectrogram')


def d4pm():
    d = diagram(610)
    node(d, 35, 50, 250, '7 EEG epochs', 'Conditioning signal', terminal=True)
    node(d, 350, 50, 290, 'Diffusion state', terminal=True)
    node(d, 725, 50, 240, 'Noise level', 'FiLM conditioning')
    node(d, 180, 230, 300, 'Feature extraction', 'Two input streams')
    node(d, 610, 230, 355, 'Transformer + FiLM', 'Feature fusion')
    node(d, 665, 420, 300, 'Noise estimate')
    node(d, 300, 420, 300, 'Reverse diffusion step')
    node(d, 35, 420, 220, 'Center artifact', 'Select center', terminal=True)
    arrow(d, [(160,129),(160,180),(255,180),(255,227)])
    arrow(d, [(495,129),(495,180),(405,180),(405,227)])
    arrow(d, [(845,129),(845,227)])
    arrow(d, [(483,268),(607,268)])
    arrow(d, [(815,309),(815,417)])
    arrow(d, [(662,458),(603,458)])
    arrow(d, [(297,458),(258,458)])
    arrow(d, [(450,499),(450,570),(15,570),(15,20),(495,20),(495,47)])
    label(d, 255, 550, 'Iterative')
    save(d, 'd4pm')


if __name__ == '__main__':
    dhct('dhct_gan')
    dhct('dhct_gan_v2', stacked=True)
    nested()
    vit()
    d4pm()

"""Compose the IC-U-Net architecture diagram in the FACETpy visual language."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "skills/facetpy-diagram/assets"))

from facetpy_svg import (  # noqa: E402
    C, FONT, Diagram, capsule, card, container, edge, eeg_wave, pill, text,
    text_width,
)

OUT_SVG = REPO / "docs/source/_static/diagrams/ic_unet_architecture.svg"
OUT_PNG = REPO / "docs/source/_static/diagrams/ic_unet_architecture.png"

H = 1880
d = Diagram(H)   # custom two-line title block below (the subtitle is long)

TITLE = "IC-U-Net — U-Net in IC Space"
SUB = ("ICA prior + multichannel 1D U-Net · Chuang et al. 2022 · "
       "src/facet/models/ic_unet/")
_tw = text_width(TITLE, 24)
d.add(text(70, 52, TITLE, size=24, weight=700, fill=C["ink"]),
      f'<rect x="70" y="62" width="{22 + len(TITLE) * 9}" height="3" rx="1.5" '
      f'fill="url(#fp-header)"/>',
      eeg_wave(70 + _tw + 34, 44, 90),
      text(70, 84, SUB, size=13.5, fill=C["slate"]))

# --------------------------------------------------------------------------- #
#  Custom brand elements
# --------------------------------------------------------------------------- #
d.add_defs(
    f'<marker id="fp-skip" markerWidth="12" markerHeight="12" refX="9" refY="5" '
    f'orient="auto"><path d="M1 1 L10 5 L1 9 Z" fill="{C["blue400"]}"/></marker>'
)


def snowflake(cx, cy, r, color):
    """Tiny 3-axis snowflake — the 'frozen' mark."""
    import math
    p = []
    for a in (90, 30, 150):
        dx = r * math.cos(math.radians(a))
        dy = r * math.sin(math.radians(a))
        p.append(f'<line x1="{cx-dx:.1f}" y1="{cy-dy:.1f}" x2="{cx+dx:.1f}" '
                 f'y2="{cy+dy:.1f}" stroke="{color}" stroke-width="1.6" '
                 f'stroke-linecap="round"/>')
    return "".join(p)


def frozen_card(x, y, w, title, lines):
    """A card marked as frozen: dashed brand halo + snowflake 'frozen' chip.

    The shared treatment makes the two ICA blocks read as one pair.
    """
    g = card(x, y, w, 0, title, lines)
    h = g["h"]
    halo = (f'<rect x="{x-7}" y="{y-7}" width="{w+14}" height="{h+14}" rx="18" '
            f'fill="none" stroke="{C["blue400"]}" stroke-width="1.6" '
            f'stroke-dasharray="6 5" stroke-opacity="0.95"/>')
    chip_txt = "frozen"
    cw = text_width(chip_txt, 10.5) + 34
    cx0 = x + w - cw - 12
    cy0 = y + 18
    chip = (f'<rect x="{cx0}" y="{cy0-9.5}" width="{cw:.1f}" height="19" rx="9.5" '
            f'fill="{C["blue200"]}"/>'
            + snowflake(cx0 + 13, cy0, 5.2, C["navy_d"])
            + text(cx0 + 23, cy0 + 4, chip_txt, size=10.5, fill=C["navy_d"],
                   weight=600))
    g["svg"] = halo + g["svg"] + chip
    return g


def multi_note(x, y, w, lines, size=11.5):
    """Folded-corner annotation that carries several lines."""
    fold = 14
    h = 16 + len(lines) * 17 + 8
    p = [f'<path d="M{x} {y} L{x+w-fold} {y} L{x+w} {y+fold} L{x+w} {y+h} '
         f'L{x} {y+h} Z" fill="{C["tint"]}" stroke="{C["slate"]}" '
         f'stroke-opacity="0.45" stroke-width="1"/>',
         f'<path d="M{x+w-fold} {y} L{x+w-fold} {y+fold} L{x+w} {y+fold}" '
         f'fill="none" stroke="{C["slate"]}" stroke-opacity="0.45"/>']
    ly = y + 20
    for i, ln in enumerate(lines):
        p.append(text(x + 11, ly, ln, size=size, fill=C["slate"],
                      weight=600 if i == 0 else None))
        ly += 17
    from facetpy_svg import _geom
    return _geom("".join(p), x, y, w, h)


def skip_edge(x1, x2, y, label):
    """Horizontal skip connector — dashed, brand-blue, visually distinct."""
    line = (f'<path d="M{x1} {y} H{x2}" fill="none" stroke="{C["blue400"]}" '
            f'stroke-width="2" stroke-dasharray="7 5" '
            f'marker-end="url(#fp-skip)"/>')
    lx, ly = (x1 + x2) / 2, y + 4
    bw = text_width(label, 11) + 18
    chip = (f'<rect x="{lx-bw/2:.1f}" y="{ly-12}" width="{bw:.1f}" height="17" '
            f'rx="5" fill="{C["surface"]}" stroke="{C["blue400"]}" '
            f'stroke-opacity="0.8"/>'
            + text(lx, ly, label, size=11, fill=C["blue"], weight=600,
                   anchor="middle"))
    return line + chip


def minus_op(x, y, w, h, label):
    """Process node with a drawn circled-minus operator glyph."""
    tw = text_width(label, 13)
    tx = x + w / 2 + 16                      # text shifted right of the glyph
    gx = tx - tw / 2 - 22
    cy = y + h / 2
    svg = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
           f'fill="{C["tint"]}" stroke="{C["blue"]}" stroke-opacity="0.55" '
           f'stroke-width="1.5" filter="url(#fp-shadow)"/>'
           f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" '
           f'fill="{C["blue"]}"/>'
           f'<circle cx="{gx:.1f}" cy="{cy}" r="9" fill="none" '
           f'stroke="{C["blue"]}" stroke-width="1.8"/>'
           f'<line x1="{gx-4.5:.1f}" y1="{cy}" x2="{gx+4.5:.1f}" y2="{cy}" '
           f'stroke="{C["blue"]}" stroke-width="1.8" stroke-linecap="round"/>'
           + text(tx, cy + 4.5, label, size=13, weight=500, anchor="middle"))
    from facetpy_svg import _geom
    return _geom(svg, x, y, w, h)


# --------------------------------------------------------------------------- #
#  Geometry
# --------------------------------------------------------------------------- #
MID = 480          # centre line of the U-Net container
COL = 340          # width of the stacked stage cards
BW = 320           # width of an encoder / decoder block

# --- Stage 1 + 2: input, demean, forward ICA (slightly left of canvas centre)
SX = 340           # centre of the top stack

inp = card(SX - COL / 2, 110, COL, 0, "Noisy Context Window",
           ["(B, 30, 3584)", "7 context epochs × 512 samples"])
demean = pill(SX - 130, 250, 260, 38, "Per-Channel Demean")
fwd = frozen_card(SX - COL / 2, 322, COL, "Forward ICA",
                  ["ic = W · x", "W: 30×30, frozen (FastICA)"])

steck = card(560, 110, 400, 0, "Model Profile",
             ["30-channel Niazy montage · 4096 Hz",
              "retraining required for other montages",
              "Loss: MSE / MAE / IC-U-Net ensemble loss",
              "Unified Holdout: +11.11 dB SNR (rank 8/15)"])

# --- Stage 3: the U
CT, CB = 460, 1248
frame = container(44, CT, 872, CB - CT, "IcUnet1D — denoising in IC space")

L_OUT, L0, L1, L2, L3 = 502, 652, 802, 952, 1102   # row tops

e1 = card(64, L0, BW, 0, "inc: DoubleConv", ["30→64   k=7", "(B, 64, 3584)"])
e2 = card(79, L1, BW, 0, "down1: MaxPool2 + DoubleConv",
          ["64→128   k=7", "(B, 128, 1792)"])
e3 = card(94, L2, BW, 0, "down2: MaxPool2 + DoubleConv",
          ["128→256   k=5", "(B, 256, 896)"])
bn = card(310, L3, COL, 0, "down3: MaxPool2 + DoubleConv",
          ["256→512   k=3", "(B, 512, 448)"])
d3 = card(546, L2, BW, 0, "up1: Upsample×2 ⊕ skip",
          ["→ DoubleConv  512+256→256  k=3", "(B, 256, 896)"])
d2 = card(561, L1, BW, 0, "up2: Upsample×2 ⊕ skip",
          ["→ DoubleConv  256+128→128  k=3", "(B, 128, 1792)"])
d1 = card(576, L0, BW, 0, "up3: Upsample×2 ⊕ skip",
          ["→ DoubleConv  128+64→64  k=3", "(B, 64, 3584)"])
outc = card(576, L_OUT, BW, 0, "outc: Conv1d",
            ["64→30   k=1", "(B, 30, 3584)"])

dc_note = multi_note(54, 1100, 234,
                     ["DoubleConv =",
                      "[Conv1d → BatchNorm1d",
                      "→ LeakyReLU(0.1)] × 2"])

# --- Stages 4–6: back out of IC space
inv = frozen_card(MID - COL / 2, 1300, COL, "Inverse ICA",
                  ["clean = W⁺ · ic_clean", "pseudoinverse, frozen"])
centre = card(MID - COL / 2, 1438, COL, 0, "Center Extraction",
              ["epoch 4 of 7 → samples 1536:2048", "(B, 30, 512)"])
head = minus_op(MID - 200, 1576, 400, 46,
                "artifact = noisy_center − clean_center")
outbox = card(MID - COL / 2, 1656, COL, 0, "Predicted Artifact",
              ["(B, 30, 512)"])
final = capsule(MID - 260, 1771, 520, 48,
                "DeepLearningCorrection: corrected = noisy − artifact")

# --------------------------------------------------------------------------- #
#  Connectors (drawn beneath the nodes)
# --------------------------------------------------------------------------- #
# top stack
d.add_edge(edge([(SX, inp["y"] + inp["h"] + 3), (SX, demean["y"] - 3)]))
d.add_edge(edge([(SX, demean["y"] + demean["h"] + 3), (SX, fwd["y"] - 10)]))
# into the U (routed clear of the container title tab)
d.add_edge(edge([(SX, fwd["y"] + fwd["h"] + 10), (SX, 508),
                 (e1["cx"], 508), (e1["cx"], e1["y"] - 3)]))

# encoder descent
d.add_edge(edge([(e1["cx"], e1["y"] + e1["h"] + 3), (e1["cx"], L1 - 23),
                 (e2["cx"], L1 - 23), (e2["cx"], e2["y"] - 3)]))
d.add_edge(edge([(e2["cx"], e2["y"] + e2["h"] + 3), (e2["cx"], L2 - 23),
                 (e3["cx"], L2 - 23), (e3["cx"], e3["y"] - 3)]))
d.add_edge(edge([(e3["cx"], e3["y"] + e3["h"] + 3), (e3["cx"], L3 - 23),
                 (bn["cx"], L3 - 23), (bn["cx"], bn["y"] - 3)]))

# decoder ascent
d.add_edge(edge([(bn["x"] + bn["w"] + 3, bn["cy"]), (706, bn["cy"]),
                 (706, d3["y"] + d3["h"] + 3)]))
d.add_edge(edge([(d3["cx"], d3["y"] - 3), (d3["cx"], L2 - 23),
                 (d2["cx"], L2 - 23), (d2["cx"], d2["y"] + d2["h"] + 3)]))
d.add_edge(edge([(d2["cx"], d2["y"] - 3), (d2["cx"], L1 - 23),
                 (d1["cx"], L1 - 23), (d1["cx"], d1["y"] + d1["h"] + 3)]))
d.add_edge(edge([(d1["cx"], d1["y"] - 3),
                 (d1["cx"], outc["y"] + outc["h"] + 3)]))

# skip connections — horizontal, level n encoder → level n decoder
d.add_edge(skip_edge(e1["x"] + e1["w"] + 3, d1["x"] - 4, e1["cy"], "skip (concat)"))
d.add_edge(skip_edge(e2["x"] + e2["w"] + 3, d2["x"] - 4, e2["cy"], "skip (concat)"))
d.add_edge(skip_edge(e3["x"] + e3["w"] + 3, d3["x"] - 4, e3["cy"], "skip (concat)"))

# out of the U-Net, down the right-hand lane, back to the centre line
d.add_edge(edge([(outc["x"] + outc["w"] + 3, outc["cy"]), (946, outc["cy"]),
                 (946, 1274), (MID, 1274), (MID, inv["y"] - 10)]))

# tail stack
d.add_edge(edge([(MID, inv["y"] + inv["h"] + 10), (MID, centre["y"] - 3)]))
d.add_edge(edge([(MID, centre["y"] + centre["h"] + 3), (MID, head["y"] - 3)]))
d.add_edge(edge([(MID, head["y"] + head["h"] + 3), (MID, outbox["y"] - 3)]))
d.add_edge(edge([(MID, outbox["y"] + outbox["h"] + 3), (MID, final["y"] - 3)]))

# --------------------------------------------------------------------------- #
#  Nodes
# --------------------------------------------------------------------------- #
d.add(frame)                                     # container first
d.add(inp, demean, fwd, steck)
d.add(e1, e2, e3, bn, d3, d2, d1, outc, dc_note)
d.add(inv, centre, head, outbox, final)

d.render_png(str(OUT_PNG), svg_path=str(OUT_SVG), width=1280)
print("svg:", OUT_SVG)
print("png:", OUT_PNG)
print("container bottom", CB, "canvas", H)
for nm, g in [("e1", e1), ("d1", d1), ("bn", bn), ("outc", outc), ("inv", inv),
              ("steck", steck), ("dc_note", dc_note)]:
    print(f"  {nm}: x={g['x']} y={g['y']} w={g['w']} h={g['h']}")

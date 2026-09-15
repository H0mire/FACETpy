"""Build the DPAE architecture diagram (FACETpy brand system)."""
import sys, os

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "skills/facetpy-diagram/assets"))

from facetpy_svg import (Diagram, card, pill, container, note, edge,
                         text, text_width, node_dot, C, FONT)

OUT_DIR = str(REPO / "docs/source/_static/diagrams")
SVG = os.path.join(OUT_DIR, "dpae_architecture.svg")
PNG = os.path.join(OUT_DIR, "dpae_architecture.png")

H = 1920
d = Diagram(H,
            title="DPAE — Dual-Pathway Autoencoder",
            subtitle="1D-CNN, channel-wise artifact regression")

# --- custom defs: accent skip marker + branch dot ---------------------------
d.add_defs(
    f'<marker id="fp-skip" markerWidth="12" markerHeight="12" refX="9" '
    f'refY="5" orient="auto">'
    f'<path d="M1 1 L10 5 L1 9 Z" fill="{C["blue"]}"/></marker>')


def skip_edge(points):
    """Dashed accent bypass connector (visually distinct from the main flow)."""
    pstr = " ".join(f"{px},{py}" for px, py in points)
    return (f'<polyline points="{pstr}" fill="none" stroke="{C["blue"]}" '
            f'stroke-opacity="0.9" stroke-width="2.2" stroke-dasharray="9 5" '
            f'stroke-linejoin="round" marker-end="url(#fp-skip)"/>')


def skip_label_vertical(x, y, txt):
    """Accent chip rendered along a vertical connector (reads bottom-to-top)."""
    bw = text_width(txt, 11) + 18
    return (f'<g transform="rotate(-90 {x} {y})">'
            f'<rect x="{x-bw/2:.1f}" y="{y-10}" width="{bw:.1f}" height="20" '
            f'rx="6" fill="{C["surface"]}" stroke="{C["blue"]}" '
            f'stroke-opacity="0.55"/>'
            + text(x, y + 4, txt, size=11, fill=C["blue"], weight=600,
                   anchor="middle") + '</g>')


def backdrop(x, y, w, h):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
            f'fill="{C["tint"]}" fill-opacity="0.7"/>')


def desc(x, y, s, size=11.5, italic=False):
    return text(x, y, s, size=size, fill=C["slate"], italic=italic)


CONV = C["blue"]
POOL = C["blue400"]

# --- title second line ------------------------------------------------------
d.add(desc(70, 82, "Xiong et al. 2023  ·  src/facet/models/dpae/", size=12,
           italic=True))

# --------------------------------------------------------------------------- #
# INPUT
# --------------------------------------------------------------------------- #
inp = card(350, 100, 300, 104, "Noisy EEG Epoch",
           ["(B, 1, 512)", "single channel, demeaned"])
d.add(inp)

# --------------------------------------------------------------------------- #
# PATHWAY COLUMNS
# --------------------------------------------------------------------------- #
LX, RX, CW = 80, 605, 315          # container x / width
CY, CH = 250, 550                  # container y / height
LCX, RCX = LX + CW / 2, RX + CW / 2
PX_L, PX_R, PW, PH = 96, 621, 283, 44

d.add_edge(backdrop(LX, CY, CW, CH), backdrop(RX, CY, CW, CH))
d.add(container(LX, CY, CW, CH, "Local Pathway"),
      container(RX, CY, CW, CH, "Global Pathway"))
d.add(desc(LX + 127, 268, "— small kernels, dilated"),
      desc(LX + 14, 292, "fine temporal detail", size=11, italic=True),
      desc(RX + 135, 268, "— large kernels, pooled"),
      desc(RX + 14, 292, "slow trends & gradient envelope",
           size=11, italic=True))

local_layers = [
    ("Conv1d  1→F   k=3  d=1  ·  SELU", CONV),
    ("Conv1d  F→F   k=3  d=2  ·  SELU", CONV),
    ("Conv1d  F→2F  k=3  d=4  ·  SELU", CONV),
    ("MaxPool1d  k=2", POOL),
    ("Conv1d  2F→2F  k=3  d=8  ·  SELU", CONV),
    ("MaxPool1d  k=2", POOL),
    ("Conv1d  2F→128  k=3  ·  SELU", CONV),
]
global_layers = [
    ("Conv1d  1→F   k=15  ·  SELU", CONV),
    ("MaxPool1d  k=2", POOL),
    ("Conv1d  F→2F  k=11  ·  SELU", CONV),
    ("MaxPool1d  k=2", POOL),
    ("Conv1d  2F→128  k=7  ·  SELU", CONV),
]

L_Y0, L_PITCH = 310, 68
R_Y0, R_PITCH = 310, 102

for i, (t, acc) in enumerate(local_layers):
    y = L_Y0 + i * L_PITCH
    d.add(pill(PX_L, y, PW, PH, t, accent=acc))
    if i:
        prev = y - L_PITCH + PH
        d.add_edge(edge([(LCX, prev + 3), (LCX, y - 4)]))

for i, (t, acc) in enumerate(global_layers):
    y = R_Y0 + i * R_PITCH
    d.add(pill(PX_R, y, PW, PH, t, accent=acc))
    if i:
        prev = y - R_PITCH + PH
        d.add_edge(edge([(RCX, prev + 3), (RCX, y - 4)]))

L_END = L_Y0 + (len(local_layers) - 1) * L_PITCH + PH      # 762
R_END = R_Y0 + (len(global_layers) - 1) * R_PITCH + PH     # 762

d.add(text(LCX, 786, "Output:  (B, 128, 128)", size=11.5, fill=C["blue"],
           weight=600, anchor="middle"),
      text(RCX, 786, "Output:  (B, 128, 128)", size=11.5, fill=C["blue"],
           weight=600, anchor="middle"))

# fan-out from input into both pathways
d.add_edge(edge([(inp["cx"], inp["y"] + inp["h"] + 3), (inp["cx"], 228),
                 (LCX, 228), (LCX, CY - 4)]),
           edge([(inp["cx"], inp["y"] + inp["h"] + 3), (inp["cx"], 228),
                 (RCX, 228), (RCX, CY - 4)]))

# mid note between the columns
NX, NW = 405, 190
d.add(note(NX, 690, NW, 68, ""))
for k, ln in enumerate(["both pathways", "downsample ×4 →",
                        "identical bottleneck shape"]):
    d.add(text(NX + NW / 2, 712 + k * 16, ln, size=10.5, fill=C["slate"],
               anchor="middle"))

# --------------------------------------------------------------------------- #
# FUSION
# --------------------------------------------------------------------------- #
fusion = card(310, 870, 380, 104, "Fusion",
              ["concat (channel axis) → 256",
               "BatchNorm1d → Conv1d 1×1 → SELU"])
d.add(fusion)

d.add_edge(edge([(LCX, CY + CH + 3), (LCX, 835), (440, 835), (440, 867)],
                label=("(B, 128, 128)", 339, 839)),
           edge([(RCX, CY + CH + 3), (RCX, 835), (560, 835), (560, 867)],
                label=("(B, 128, 128)", 661, 839)))

# --------------------------------------------------------------------------- #
# DECODER
# --------------------------------------------------------------------------- #
DX, DW, DY, DH = 280, 440, 1030, 308
d.add_edge(backdrop(DX, DY, DW, DH))
d.add(container(DX, DY, DW, DH, "Decoder"))
d.add(desc(DX + 82, 1048, "— mirrored upsampling"))

dec_layers = [
    "ConvTranspose1d  256→2F   k=4  s=2  ·  SELU",
    "ConvTranspose1d  2F→F   k=4  s=2  ·  SELU",
    "Conv1d  F→F   k=3  ·  SELU",
    "Conv1d  F→1   k=1",
]
D_Y0, D_PITCH, DPW = 1080, 64, 400
for i, t in enumerate(dec_layers):
    y = D_Y0 + i * D_PITCH
    d.add(pill(DX + 20, y, DPW, PH, t, accent=CONV))
    if i:
        d.add_edge(edge([(500, y - D_PITCH + PH + 3), (500, y - 4)]))

d.add_edge(edge([(500, fusion["y"] + fusion["h"] + 3), (500, DY - 4)],
                label=("(B, 256, 128)", 500, 1006)))

# --------------------------------------------------------------------------- #
# RESIDUAL MERGE  +  OUTPUT
# --------------------------------------------------------------------------- #
merge = pill(250, 1390, 500, 54, "⊕    out = decoder(x) + residual_scale · x",
             accent=C["blue"])
d.add(merge)
d.add_edge(edge([(500, DY + DH + 3), (500, merge["y"] - 4)]))

out = card(350, 1486, 300, 81, "Predicted Artifact", ["(B, 1, 512)"])
d.add(out)
d.add_edge(edge([(500, merge["y"] + merge["h"] + 3), (500, out["y"] - 4)]))

corr = card(310, 1609, 380, 81, "DeepLearningCorrection",
            ["corrected = noisy − artifact"])
d.add(corr)
d.add_edge(edge([(500, out["y"] + out["h"] + 3), (500, corr["y"] - 4)]))

# --------------------------------------------------------------------------- #
# RESIDUAL BYPASS (far-left corridor, skipping the whole encoder/decoder)
# --------------------------------------------------------------------------- #
BX = 44
d.add_edge(skip_edge([(inp["x"], inp["cy"]), (BX, inp["cy"]),
                      (BX, merge["cy"]), (merge["x"] - 5, merge["cy"])]))
d.add(skip_label_vertical(BX, 1000, "residual_scale (learned, init 0)"))
d.add(f'<circle cx="{inp["x"]}" cy="{inp["cy"]}" r="4.5" fill="{C["blue"]}"/>')

# --------------------------------------------------------------------------- #
# SIDE-NOTE CARD
# --------------------------------------------------------------------------- #
d.add(card(60, 1730, 420, 127, "Setup & Result",
           ["F = base_filters = 32 · latent = 128 · ≈ 2 M parameters",
            "Loss: MSE on the artifact waveform",
            "Unified Holdout: +7.28 dB SNR (rank 12/15)"]))

d.render_png(PNG, svg_path=SVG, width=1280)
print("svg:", SVG)
print("png:", PNG)

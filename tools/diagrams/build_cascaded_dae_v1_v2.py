"""Cascaded DAE — v1 vs. v2 side-by-side comparison (FACETpy visual language).

Two symmetric columns share one cascade skeleton
(INPUT → STAGE 1 → RESIDUAL → STAGE 2 → OUTPUT) with the rows aligned at the
same y so the reader can compare them line-by-line. Only the residual row
differs, because that is where the two models actually diverge.

Verified against
    src/facet/models/masterthesis/cascaded_dae/training.py
    src/facet/models/masterthesis/cascaded_context_dae/training.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "skills/facetpy-diagram/assets"))

from facetpy_svg import (  # noqa: E402
    C, Diagram, card, edge, eeg_wave, text, text_width, _geom,
)

OUT_SVG = REPO / "docs/source/_static/diagrams/cascaded_dae_v1_v2.svg"
OUT_PNG = REPO / "docs/source/_static/diagrams/cascaded_dae_v1_v2.png"

# --------------------------------------------------------------------------- #
#  Canvas + two-line title block
# --------------------------------------------------------------------------- #
H = 1792
d = Diagram(H)   # custom two-line title (the subtitle is long)

TITLE = "Cascaded DAE — v1 vs. v2"
SUB = ("Two-stage residual denoising autoencoder · FACETpy internal baselines · "
       "src/facet/models/")
d.add(text(70, 52, TITLE, size=24, weight=700, fill=C["ink"]),
      f'<rect x="70" y="62" width="{22 + len(TITLE) * 9}" height="3" rx="1.5" '
      f'fill="url(#fp-header)"/>',
      eeg_wave(70 + text_width(TITLE, 24) + 34, 44, 90),
      text(70, 84, SUB, size=13.5, fill=C["slate"]))

# --------------------------------------------------------------------------- #
#  Custom brand elements (shared with the other diagrams in this series)
# --------------------------------------------------------------------------- #
d.add_defs(
    f'<marker id="fp-skip" markerWidth="12" markerHeight="12" refX="9" refY="5" '
    f'orient="auto"><path d="M1 1 L10 5 L1 9 Z" fill="{C["blue"]}"/></marker>'
)


def skip_edge(points):
    """Dashed accent bypass connector — visually distinct from the main chain."""
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


def multi_note(x, y, w, lines, size=12):
    """Folded-corner annotation that carries several lines."""
    fold = 14
    h = 16 + len(lines) * 18 + 8
    p = [f'<path d="M{x} {y} L{x+w-fold} {y} L{x+w} {y+fold} L{x+w} {y+h} '
         f'L{x} {y+h} Z" fill="{C["tint"]}" stroke="{C["slate"]}" '
         f'stroke-opacity="0.45" stroke-width="1"/>',
         f'<path d="M{x+w-fold} {y} L{x+w-fold} {y+fold} L{x+w} {y+fold}" '
         f'fill="none" stroke="{C["slate"]}" stroke-opacity="0.45"/>']
    ly = y + 21
    for i, ln in enumerate(lines):
        p.append(text(x + 14, ly, ln, size=size + (0.5 if i == 0 else 0),
                      fill=C["blue"] if i == 0 else C["slate"],
                      weight=700 if i == 0 else None))
        ly += 18
    return _geom("".join(p), x, y, w, h)


def layer_row(x, y, w, h, txt, accent=None, size=12.5):
    """Compact process row — pill(), but with a controllable text size so the
    long `Linear a→b · LeakyReLU · Dropout` strings fit inside the column."""
    accent = accent or C["blue"]
    svg = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
           f'fill="{C["tint"]}" stroke="{accent}" stroke-opacity="0.55" '
           f'stroke-width="1.4" filter="url(#fp-shadow)"/>'
           f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" fill="{accent}"/>'
           + text(x + w / 2 + 2, y + h / 2 + 4.5, txt, size=size, weight=500,
                  anchor="middle"))
    return _geom(svg, x, y, w, h)


def mini_chip(cx, cy, txt, size=11.5):
    """Small tinted inline chip used for the flatten/reshape notes."""
    bw = text_width(txt, size) + 22
    return (f'<rect x="{cx-bw/2:.1f}" y="{cy-11}" width="{bw:.1f}" height="22" '
            f'rx="11" fill="{C["surface"]}" stroke="{C["blue"]}" '
            f'stroke-opacity="0.35"/>'
            + text(cx, cy + 4, txt, size=size, fill=C["blue"], weight=600,
                   anchor="middle"))


def section_label(x, y, txt):
    return text(x, y, txt, size=11, fill=C["slate"], weight=700)


def halo(g, color, dashed=True, pad=7, width=2.0):
    """Emphasis ring around a card — marks the two residual nodes."""
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    ring = (f'<rect x="{g["x"]-pad}" y="{g["y"]-pad}" width="{g["w"]+2*pad}" '
            f'height="{g["h"]+2*pad}" rx="18" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-opacity="0.95"{dash}/>')
    g["svg"] = ring + g["svg"]
    return g


def badge(x_right, cy, txt):
    """Small light chip sitting inside a dark gradient header."""
    bw = text_width(txt, 10.5) + 22
    x0 = x_right - bw
    return (f'<rect x="{x0:.1f}" y="{cy-9.5}" width="{bw:.1f}" height="19" '
            f'rx="9.5" fill="{C["blue200"]}"/>'
            + text(x0 + bw / 2, cy + 4, txt, size=10.5, fill=C["navy_d"],
                   weight=700, anchor="middle"))


def backdrop(x, y, w, h, fill, op):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
            f'fill="{fill}" fill-opacity="{op}"/>')


# --------------------------------------------------------------------------- #
#  Column geometry — identical rows, mirrored lanes
# --------------------------------------------------------------------------- #
LX, RX, CW = 60, 510, 430                  # content x / width per column
LCX, RCX = LX + CW / 2, RX + CW / 2        # 275 / 725
PX_L, PX_R, PW = LX + 14, RX + 14, CW - 28  # inner row x / width
LANE_L, LANE_R = 34, 966                   # outer skip-connector lanes

HEAD_Y, HEAD_H = 120, 104
IN_Y, IN_H = 254, 104
S1_Y, S1_H = 400, 364
RES_Y, RES_H = 810, 184
S2_Y, S2_H = 1040, 148
OUT_Y, OUT_H = 1234, 104
CHIP_Y, CHIP_H = 1378, 42
NOTE_Y = 1452
SHARED_Y = 1588

# column tint + divider (background layer)
d.add_edge(backdrop(46, 108, 448, 1322, C["ink"], 0.032),
           backdrop(506, 108, 448, 1322, C["blue400"], 0.07),
           f'<line x1="500" y1="112" x2="500" y2="1426" stroke="{C["ink"]}" '
           f'stroke-opacity="0.13" stroke-width="1.2" stroke-dasharray="3 6"/>')

# --------------------------------------------------------------------------- #
#  Row 0 — column headers
# --------------------------------------------------------------------------- #
h_l = card(LX, HEAD_Y, CW, HEAD_H, "v1 · cascaded_dae",
           ["single epoch, channel-wise",
            "class CascadedDenoisingAutoencoder"])
h_r = card(RX, HEAD_Y, CW, HEAD_H, "v2 · cascaded_context_dae",
           ["7-epoch context window",
            "class CascadedContextDenoisingAutoencoder"])

# --------------------------------------------------------------------------- #
#  Row 1 — input
# --------------------------------------------------------------------------- #
in_l = card(LX, IN_Y, CW, IN_H, "Noisy Epoch",
            ["(B, 1, 512)", "one channel, demeaned"])
in_r = card(RX, IN_Y, CW, IN_H, "Noisy Context Window",
            ["(B, 7, 1, 512)", "7 epochs × one channel, demeaned"])

# --------------------------------------------------------------------------- #
#  Row 2 — Stage 1 (full layer stack, both columns)
# --------------------------------------------------------------------------- #
ROW_H = 42
Y_FLAT, Y_ENC, Y_A, Y_B = 458, 494, 500, 550
Y_DEC, Y_C, Y_D, Y_RESHAPE = 618, 624, 674, 740


def stage1(x, cx, title, flat_txt, enc_rows, dec_rows, reshape_txt):
    g = card(x, S1_Y, CW, S1_H, title)
    p = [g["svg"],
         f'<line x1="{x}" y1="{S1_Y+36}" x2="{x+CW}" y2="{S1_Y+36}" '
         f'stroke="{C["ink"]}" stroke-opacity="0.12"/>',
         mini_chip(cx, Y_FLAT, flat_txt),
         section_label(x + 18, Y_ENC, "ENCODER"),
         section_label(x + 18, Y_DEC, "DECODER")]
    for y, txt in zip((Y_A, Y_B), enc_rows):
        p.append(layer_row(x + 14, y, PW, ROW_H, txt, accent=C["blue"])["svg"])
    for y, txt in zip((Y_C, Y_D), dec_rows):
        p.append(layer_row(x + 14, y, PW, ROW_H, txt, accent=C["blue400"])["svg"])
    p.append(mini_chip(cx, Y_RESHAPE, reshape_txt))
    g["svg"] = "".join(p)
    return g


DO = "· LeakyReLU(0.2) · Dropout 0.2"
s1_l = stage1(LX, LCX, "Stage 1 — DenoisingAutoencoder", "flatten → 512",
              [f"Linear 512→512 {DO}", f"Linear 512→128 {DO}"],
              [f"Linear 128→512 {DO}", "Linear 512→512"],
              "reshape → (B, 1, 512)")
s1_r = stage1(RX, RCX, "Stage 1 — ContextDenoisingAutoencoder", "flatten → 3584",
              [f"Linear 3584→512 {DO}", f"Linear 512→128 {DO}"],
              [f"Linear 128→512 {DO}", "Linear 512→512"],
              "reshape → (B, 1, 512)")

# --------------------------------------------------------------------------- #
#  Row 3 — the residual node: where the two models diverge
# --------------------------------------------------------------------------- #
STRIP_Y, STRIP_H, STRIP_W = 886, 44, 390
Y_SHAPE, Y_SUB, Y_CAP = 870, 948, 972


def residual_shell(x, title, shape, hint):
    g = card(x, RES_Y, CW, RES_H, title)
    return [g,
            f'<line x1="{x}" y1="{RES_Y+36}" x2="{x+CW}" y2="{RES_Y+36}" '
            f'stroke="{C["ink"]}" stroke-opacity="0.12"/>',
            text(x + 18, Y_SHAPE, shape, size=12.5, fill=C["ink"], weight=600),
            text(x + CW - 18, Y_SHAPE, hint, size=11.5, fill=C["slate"],
                 italic=True, anchor="end")]


# --- v1: one epoch, the whole window is corrected --------------------------
g, *extra_l = residual_shell(LX, "residual = x − artifact₁", "(B, 1, 512)",
                             "full window is corrected")
sx = LX + 20
parts = [g["svg"], *extra_l,
         f'<rect x="{sx}" y="{STRIP_Y}" width="{STRIP_W}" height="{STRIP_H}" '
         f'rx="9" fill="url(#fp-header)" stroke="{C["blue"]}" '
         f'stroke-width="1.4"/>',
         text(LCX, STRIP_Y + 27, "the single epoch  ·  512 samples", size=12.5,
              fill=C["header_fg"], weight=600, anchor="middle"),
         mini_chip(LCX, Y_SUB, "− artifact₁"),
         text(LCX, Y_CAP, "no neighbouring context available", size=10.5,
              fill=C["slate"], anchor="middle")]
g["svg"] = "".join(parts)
# solid ring, not dashed: the dashed style is reserved for the skip connectors
res_l = halo(g, C["blue400"], dashed=False, width=2.0)

# --- v2: 7 epochs, only the center one is corrected -------------------------
g, *extra_r = residual_shell(RX, "residual context", "(B, 7, 1, 512)",
                             "only index 3 changes")
BW, GAP = 48, 9
sx = RX + 20
parts = [g["svg"], *extra_r]
for i in range(7):
    bx = sx + i * (BW + GAP)
    mid = i == 3
    if mid:
        parts.append(
            f'<rect x="{bx}" y="{STRIP_Y}" width="{BW}" height="{STRIP_H}" '
            f'rx="9" fill="url(#fp-header)" stroke="{C["blue"]}" '
            f'stroke-width="1.6"/>')
        parts.append(text(bx + BW / 2, STRIP_Y + 27, "3", size=13,
                          fill=C["header_fg"], weight=700, anchor="middle"))
    else:
        parts.append(
            f'<rect x="{bx}" y="{STRIP_Y}" width="{BW}" height="{STRIP_H}" '
            f'rx="9" fill="{C["tint"]}" stroke="{C["slate"]}" '
            f'stroke-opacity="0.45" stroke-width="1.2"/>')
        parts.append(text(bx + BW / 2, STRIP_Y + 27, str(i), size=13,
                          fill=C["slate"], weight=600, anchor="middle"))
c1 = sx + 1 * (BW + GAP) + BW / 2
c3 = sx + 3 * (BW + GAP) + BW / 2
c5 = sx + 5 * (BW + GAP) + BW / 2
parts += [text(c1, Y_SUB, "unchanged", size=10, fill=C["slate"], anchor="middle"),
          text(c5, Y_SUB, "unchanged", size=10, fill=C["slate"], anchor="middle"),
          mini_chip(c3, Y_SUB, "− artifact₁"),
          text(RCX, Y_CAP,
               "only the center epoch is corrected; the 6 neighbours pass through",
               size=10.5, fill=C["slate"], anchor="middle")]
g["svg"] = "".join(parts)
res_r = halo(g, C["blue"], dashed=False, width=2.4)
res_r["svg"] += badge(RX + CW - 14, RES_Y + 18, "key difference")

# --------------------------------------------------------------------------- #
#  Row 4 — Stage 2 (compact, separate weights)
# --------------------------------------------------------------------------- #
def stage2(x, title, stack):
    g = card(x, S2_Y, CW, S2_H, title)
    g["svg"] = "".join([
        g["svg"],
        f'<line x1="{x}" y1="{S2_Y+36}" x2="{x+CW}" y2="{S2_Y+36}" '
        f'stroke="{C["ink"]}" stroke-opacity="0.12"/>',
        text(x + 18, S2_Y + 58, "same shape, separate weights", size=11.5,
             fill=C["slate"], italic=True),
        layer_row(x + 14, S2_Y + 68, CW - 28, ROW_H, stack, accent=C["blue"])["svg"],
        text(x + CW / 2, S2_Y + 132, "LeakyReLU(0.2) · Dropout 0.2 between layers",
             size=10.5, fill=C["slate"], anchor="middle"),
    ])
    return g


s2_l = stage2(LX, "Stage 2 — DenoisingAutoencoder",
              "same 512→512→128→512→512 stack")
s2_r = stage2(RX, "Stage 2 — ContextDenoisingAutoencoder",
              "same 3584→512→128→512→512 stack")

# --------------------------------------------------------------------------- #
#  Row 5 — output
# --------------------------------------------------------------------------- #
out_l = card(LX, OUT_Y, CW, OUT_H, "artifact = artifact₁ + artifact₂",
             ["(B, 1, 512)", "single-epoch artifact"])
out_r = card(RX, OUT_Y, CW, OUT_H, "artifact = artifact₁ + artifact₂",
             ["(B, 1, 512)", "center-epoch artifact"])

# --------------------------------------------------------------------------- #
#  Row 6 — results strip
# --------------------------------------------------------------------------- #
chip_l = layer_row(LX + 20, CHIP_Y, STRIP_W, CHIP_H,
                   "Unified Holdout:  +18.06 dB SNR  (rank 5/15)",
                   accent=C["blue400"], size=13)
chip_r = layer_row(RX + 20, CHIP_Y, STRIP_W, CHIP_H,
                   "Unified Holdout:  +18.92 dB SNR  (rank 3/15)",
                   accent=C["blue"], size=13)

# --------------------------------------------------------------------------- #
#  Key-difference note + shared training recipe (full width)
# --------------------------------------------------------------------------- #
# full-width folded note: heading, then one statement under each column so the
# sentences line up with the model they describe.
keynote = multi_note(44, NOTE_Y, 912, ["Key difference"] + [""] * 2)
keynote["svg"] += "".join(
    text(x, NOTE_Y + 39 + i * 18, ln, size=12, fill=C["slate"])
    for x, lines in ((LX + 2, ["v1 sees one epoch and subtracts artifact₁",
                               "across the whole window."]),
                     (RX + 2, ["v2 sees ±3 neighbouring epochs and subtracts",
                               "artifact₁ only from the center."]))
    for i, ln in enumerate(lines))

shared = card(44, SHARED_Y, 912, 150, "Shared training recipe (Niazy proof-fit)", [
    "hidden_units = (512, 128, 512) · dropout 0.2 · LeakyReLU(0.2)",
    "Loss: L1 on the artifact waveform · AdamW · lr 1e-3 · weight decay 1e-4 · grad clip 1.0",
    "50 max epochs · seed 42 · val_ratio 0.2 · channel-wise inference (montage-independent)",
    "Correction: DeepLearningCorrection subtracts the predicted artifact",
])

# --------------------------------------------------------------------------- #
#  Connectors — identical chain in both columns
# --------------------------------------------------------------------------- #
for cx, a1_label in ((LCX, "artifact₁"), (RCX, "artifact₁  (center epoch only)")):
    d.add_edge(
        edge([(cx, IN_Y + IN_H + 3), (cx, S1_Y - 4)]),
        edge([(cx, S1_Y + S1_H + 3), (cx, RES_Y - 11)],
             label=(a1_label, cx, RES_Y - 20)),
        edge([(cx, RES_Y + RES_H + 11), (cx, S2_Y - 4)]),
        edge([(cx, S2_Y + S2_H + 3), (cx, OUT_Y - 4)],
             label=("artifact₂", cx, OUT_Y - 20)),
    )

# artifact₁ bypass — down the OUTER margin of each column, never crossing
# the main chain or the other column.
OUT_CY = OUT_Y + OUT_H / 2
d.add_edge(
    skip_edge([(LX + 40, S1_Y + S1_H + 3), (LX + 40, 782), (LANE_L, 782),
               (LANE_L, OUT_CY), (LX - 4, OUT_CY)]),
    skip_edge([(RX + CW - 40, S1_Y + S1_H + 3), (RX + CW - 40, 782),
               (LANE_R, 782), (LANE_R, OUT_CY), (RX + CW + 4, OUT_CY)]),
)
d.add(skip_label_vertical(LANE_L, 1020, "artifact₁"),
      skip_label_vertical(LANE_R, 1020, "artifact₁"),
      f'<circle cx="{LX+40}" cy="{S1_Y+S1_H}" r="4.5" fill="{C["blue"]}"/>',
      f'<circle cx="{RX+CW-40}" cy="{S1_Y+S1_H}" r="4.5" fill="{C["blue"]}"/>')

# --------------------------------------------------------------------------- #
#  Nodes
# --------------------------------------------------------------------------- #
d.add(h_l, h_r, in_l, in_r, s1_l, s1_r, res_l, res_r,
      s2_l, s2_r, out_l, out_r, chip_l, chip_r, keynote, shared)

d.render_png(str(OUT_PNG), svg_path=str(OUT_SVG), width=1280)
print("svg:", OUT_SVG)
print("png:", OUT_PNG)
print("canvas height:", H, "shared bottom ends:", SHARED_Y + shared["h"])
for nm, g in [("hdr", h_l), ("in", in_l), ("s1", s1_l), ("res_l", res_l),
              ("res_r", res_r), ("s2", s2_l), ("out_l", out_l),
              ("out_r", out_r), ("keynote", keynote), ("shared", shared)]:
    print(f"  {nm}: x={g['x']} y={g['y']} w={g['w']} h={g['h']}")

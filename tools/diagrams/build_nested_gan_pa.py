"""Nested-GAN — Paper-Accurate Edition: architecture diagram (FACETpy language).

Fourth diagram of the deep-learning model series. Reuses the custom helpers
introduced by the earlier build scripts so the whole set reads as one system:

    skip_edge()             — dashed brand-blue bypass polyline      (DPAE / cascaded)
    skip_link()             — horizontal U-Net skip with a chip      (IC-U-Net)
    skip_label_vertical()   — rotated accent chip on a vertical lane (DPAE / cascaded)
    multi_note()            — folded-corner note with several lines  (IC-U-Net / cascaded)
    layer_row() / multi_row — compact tinted process rows            (cascaded)
    mini_chip(), halo(), badge(), backdrop(), section_label()        (cascaded)
    two-line title block                                            (IC-U-Net / cascaded)

Verified against
    src/facet/models/experimental/paper_accurate/nested_gan/training.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "skills/facetpy-diagram/assets"))

from facetpy_svg import (  # noqa: E402
    C, HEAD, Diagram, card, container, edge, eeg_wave, node_dot, rounded_top,
    text, text_width, _geom,
)

OUT_SVG = REPO / "docs/source/_static/diagrams/nested_gan_pa_architecture.svg"
OUT_PNG = REPO / "docs/source/_static/diagrams/nested_gan_pa_architecture.png"

# --------------------------------------------------------------------------- #
#  Canvas + two-line title block (shared with the rest of the series)
# --------------------------------------------------------------------------- #
H = 3396
d = Diagram(H)

TITLE = "Nested-GAN — Paper-Accurate Edition"
SUB = ("Hierarchical spectral Restormer → outer time refiner · "
       "src/facet/models/experimental/paper_accurate/nested_gan/")
d.add(text(70, 52, TITLE, size=24, weight=700, fill=C["ink"]),
      f'<rect x="70" y="62" width="{22 + len(TITLE) * 9}" height="3" rx="1.5" '
      f'fill="url(#fp-header)"/>',
      eeg_wave(70 + text_width(TITLE, 24) + 34, 44, 90),
      text(70, 84, SUB, size=13.5, fill=C["slate"]))

# --------------------------------------------------------------------------- #
#  Custom brand elements
# --------------------------------------------------------------------------- #
d.add_defs(
    f'<marker id="fp-skip" markerWidth="12" markerHeight="12" refX="9" refY="5" '
    f'orient="auto"><path d="M1 1 L10 5 L1 9 Z" fill="{C["blue"]}"/></marker>',
    f'<marker id="fp-skip2" markerWidth="12" markerHeight="12" refX="9" refY="5" '
    f'orient="auto"><path d="M1 1 L10 5 L1 9 Z" fill="{C["blue400"]}"/></marker>',
)


def skip_edge(points, soft=False):
    """Dashed accent bypass connector — visually distinct from the main chain.

    `soft=True` uses the lighter blue reserved for connections that stay INSIDE
    one branch (the U-Net skips, the Restormer global residual); the strong blue
    marks the two bypasses that cross between branches."""
    col = C["blue400"] if soft else C["blue"]
    mk = "fp-skip2" if soft else "fp-skip"
    pstr = " ".join(f"{px},{py}" for px, py in points)
    return (f'<polyline points="{pstr}" fill="none" stroke="{col}" '
            f'stroke-opacity="0.9" stroke-width="2.2" stroke-dasharray="9 5" '
            f'stroke-linejoin="round" marker-end="url(#{mk})"/>')


def skip_link(x1, x2, y, label):
    """Horizontal U-Net skip connector — dashed, brand-blue, with a chip."""
    line = (f'<path d="M{x1} {y} H{x2}" fill="none" stroke="{C["blue400"]}" '
            f'stroke-width="2" stroke-dasharray="7 5" '
            f'marker-end="url(#fp-skip2)"/>')
    lx, ly = (x1 + x2) / 2, y + 4
    bw = text_width(label, 11) + 18
    chip = (f'<rect x="{lx-bw/2:.1f}" y="{ly-12}" width="{bw:.1f}" height="17" '
            f'rx="5" fill="{C["surface"]}" stroke="{C["blue400"]}" '
            f'stroke-opacity="0.8"/>'
            + text(lx, ly, label, size=11, fill=C["blue"], weight=600,
                   anchor="middle"))
    return line + chip


def skip_label_vertical(x, y, txt, soft=False):
    """Accent chip rendered along a vertical connector (reads bottom-to-top)."""
    col = C["blue400"] if soft else C["blue"]
    bw = text_width(txt, 11) + 18
    return (f'<g transform="rotate(-90 {x} {y})">'
            f'<rect x="{x-bw/2:.1f}" y="{y-10}" width="{bw:.1f}" height="20" '
            f'rx="6" fill="{C["surface"]}" stroke="{col}" '
            f'stroke-opacity="0.75"/>'
            + text(x, y + 4, txt, size=11, fill=C["blue"], weight=600,
                   anchor="middle") + '</g>')


def multi_note(x, y, w, lines, size=11.5):
    """Folded-corner annotation that carries several lines (first = heading)."""
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
    """Compact process row — pill(), with a controllable text size."""
    accent = accent or C["blue"]
    svg = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
           f'fill="{C["tint"]}" stroke="{accent}" stroke-opacity="0.55" '
           f'stroke-width="1.4" filter="url(#fp-shadow)"/>'
           f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" fill="{accent}"/>'
           + text(x + w / 2 + 2, y + h / 2 + 4.5, txt, size=size, weight=500,
                  anchor="middle"))
    return _geom(svg, x, y, w, h)


def multi_row(x, y, w, h, rows, accent=None):
    """layer_row() carrying several centred lines: rows = [(txt, size, fill, weight)]."""
    accent = accent or C["blue"]
    p = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
         f'fill="{C["tint"]}" stroke="{accent}" stroke-opacity="0.55" '
         f'stroke-width="1.4" filter="url(#fp-shadow)"/>'
         f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" fill="{accent}"/>']
    n = len(rows)
    ly = y + h / 2 - (n - 1) * 10 + 4.5
    for txt, size, fill, weight in rows:
        p.append(text(x + w / 2 + 2, ly, txt, size=size, fill=fill,
                      weight=weight, anchor="middle"))
        ly += 20
    return _geom("".join(p), x, y, w, h)


def mini_chip(cx, cy, txt, size=11.5):
    """Small tinted inline chip."""
    bw = text_width(txt, size) + 22
    return (f'<rect x="{cx-bw/2:.1f}" y="{cy-11}" width="{bw:.1f}" height="22" '
            f'rx="11" fill="{C["surface"]}" stroke="{C["blue"]}" '
            f'stroke-opacity="0.35"/>'
            + text(cx, cy + 4, txt, size=size, fill=C["blue"], weight=600,
                   anchor="middle"))


def halo(g, color, dashed=True, pad=7, width=2.0):
    """Emphasis ring around a node."""
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    ring = (f'<rect x="{g["x"]-pad}" y="{g["y"]-pad}" width="{g["w"]+2*pad}" '
            f'height="{g["h"]+2*pad}" rx="18" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-opacity="0.95"{dash}/>')
    g["svg"] = ring + g["svg"]
    return g


def badge(x_right, cy, txt, glyph=None):
    """Small light chip sitting inside a dark gradient header."""
    gw = 16 if glyph else 0
    bw = text_width(txt, 10.5) + 22 + gw
    x0 = x_right - bw
    p = [f'<rect x="{x0:.1f}" y="{cy-9.5}" width="{bw:.1f}" height="19" '
         f'rx="9.5" fill="{C["blue200"]}"/>']
    if glyph:
        p.append(f'<circle cx="{x0+15:.1f}" cy="{cy}" r="6.4" fill="none" '
                 f'stroke="{C["navy_d"]}" stroke-width="1.4"/>')
        p.append(text(x0 + 15, cy + 3.6, glyph, size=9.5, fill=C["navy_d"],
                      weight=700, anchor="middle"))
    p.append(text(x0 + gw + bw / 2 - gw / 2 + 0, cy + 4, txt, size=10.5,
                  fill=C["navy_d"], weight=700, anchor="middle"))
    return "".join(p)


def backdrop(x, y, w, h, fill, op):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
            f'fill="{fill}" fill-opacity="{op}"/>')


def caveat_card(x, y, w, title, lines, size=12.3, pitch=20):
    """Prominent honesty callout: branded card, solid ring, warning badge."""
    h = HEAD + 14 + len(lines) * pitch + 12
    p = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
         f'fill="{C["surface"]}" stroke="{C["ink"]}" stroke-opacity="0.16" '
         f'stroke-width="1.25" filter="url(#fp-shadow)"/>',
         f'<path d="{rounded_top(x, y, w, HEAD, 12)}" fill="url(#fp-header)"/>',
         node_dot(x + 18, y + HEAD / 2),
         text(x + 34, y + HEAD / 2 + 5, title, size=15.5,
              fill=C["header_fg"], weight=600),
         badge(x + w - 14, y + HEAD / 2, "honesty note", glyph="!"),
         f'<line x1="{x}" y1="{y+HEAD}" x2="{x+w}" y2="{y+HEAD}" '
         f'stroke="{C["ink"]}" stroke-opacity="0.12"/>']
    ly = y + HEAD + 14 + 12
    for i, ln in enumerate(lines):
        p.append(text(x + 18, ly, ln, size=size,
                      fill=C["ink"] if i == 0 else C["slate"],
                      weight=700 if i == 0 else None))
        ly += pitch
    g = _geom("".join(p), x, y, w, h)
    return halo(g, C["blue"], dashed=False, pad=7, width=2.4)


def emph_node(x, y, w, h, main, sub, glyph="+"):
    """Gradient emphasis node with a circled operator glyph (⊕ / merge)."""
    cy = y + h / 2
    gx = x + 34
    p = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
         f'fill="url(#fp-header)" filter="url(#fp-shadow)"/>',
         f'<circle cx="{gx}" cy="{cy}" r="11" fill="none" '
         f'stroke="{C["blue200"]}" stroke-width="1.8"/>',
         f'<line x1="{gx-5.5}" y1="{cy}" x2="{gx+5.5}" y2="{cy}" '
         f'stroke="{C["blue200"]}" stroke-width="1.8" stroke-linecap="round"/>']
    if glyph == "+":
        p.append(f'<line x1="{gx}" y1="{cy-5.5}" x2="{gx}" y2="{cy+5.5}" '
                 f'stroke="{C["blue200"]}" stroke-width="1.8" '
                 f'stroke-linecap="round"/>')
    p.append(text(x + w / 2 + 20, cy - 4, main, size=14.5,
                  fill=C["header_fg"], weight=700, anchor="middle"))
    p.append(text(x + w / 2 + 20, cy + 15, sub, size=11,
                  fill=C["blue200"], weight=500, anchor="middle"))
    return _geom("".join(p), x, y, w, h)


def leader(x1, y1, x2, y2):
    return (f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{C["slate"]}" '
            f'stroke-opacity="0.55" stroke-width="1.2" stroke-dasharray="4 4"/>')


MID = 500
CONT_X, CONT_W = 40, 920
CONT_R = CONT_X + CONT_W                      # 960
LANE_L_OUT = 20                               # outer-left lane  (context bypass)
LANE_R_OUT = 980                              # outer-right lane (inner_artifact)

# --------------------------------------------------------------------------- #
#  Honesty callout — the headline caveat, directly under the title
# --------------------------------------------------------------------------- #
cav = caveat_card(
    50, 104, 900, "Generator-only — no discriminators",
    ["No discriminators at all — this edition is generator-only; the two-GAN / "
     "four-discriminator structure is absent.",
     "The primary Nested-GAN paper (Biomed. Phys. Eng. Express 2025 · "
     "DOI 10.1088/2057-1976/ae1a8c) is paywalled and",
     "discloses no architecture details, so the nested GAN structure could not "
     "be verified and was deliberately NOT re-added.",
     "Faithful here: the Restormer generator backbone (Zamir et al., CVPR 2022).",
     "The multi-resolution STFT loss is a documented deterministic surrogate "
     "for the paper's multi-resolution discriminators."])

# --------------------------------------------------------------------------- #
#  Entry: input → squeeze → centre pick
# --------------------------------------------------------------------------- #
inp = card(330, 296, 340, 104, "Noisy Context Window",
           ["(B, 7, 1, 512)", "channel-wise, 7 epochs"])
sq = layer_row(350, 418, 300, 40, "squeeze → (B, 7, 512)", size=12.5)
pick = card(270, 486, 460, 104, "gather inner input",
            ["center epoch (index 3)",
             "optional ±N neighbours as extra STFT channels (default N=0)"])

# =========================================================================== #
#  INNER BRANCH — hierarchical spectrogram Restormer
# =========================================================================== #
IY, IH = 640, 1320
ENC_X, ENC_W = 96, 340                        # 96 .. 436
DEC_X, DEC_W = 554, 350                       # 554 .. 904
ENC_CX, DEC_CX = ENC_X + ENC_W / 2, DEC_X + DEC_W / 2
LANE_IN_L, LANE_IN_R = 66, 930                # inner corridors

inner_frame = container(CONT_X, IY, CONT_W, IH,
                        "Inner branch — HierarchicalSpectrogramRestormer")

stft = card(330, 690, 340, 104, "STFT",
            ["n_fft=64 · hop=16 · win=64", "→ (B, 2, F, frames)"])
proj = layer_row(300, 818, 400, 44, "input projection conv → 48 features")
proj_note = multi_note(712, 812, 230,
                       ["global-residual baseline",
                        "the projected copy is kept",
                        "and re-added at ⊕ below"])
padrow = layer_row(300, 888, 400, 42, "reflect-pad H,W to a multiple of 2³",
                   accent=C["blue400"], size=12.5)

E_Y = (968, 1110, 1252)
e1 = layer_row(ENC_X, E_Y[0], ENC_W, 48, "Level 1 — Restormer blocks · 48 ch")
e2 = layer_row(ENC_X, E_Y[1], ENC_W, 48, "Level 2 — Restormer blocks · 96 ch")
e3 = layer_row(ENC_X, E_Y[2], ENC_W, 48, "Level 3 — Restormer blocks · 192 ch")
ds1 = multi_row(ENC_X, 1038, ENC_W, 52,
                [("Downsample: 1×1 conv + pixel-unshuffle", 11.8, C["ink"], 500),
                 ("H,W ÷2   ·   C ×2", 10.5, C["slate"], None)],
                accent=C["blue400"])
ds2 = multi_row(ENC_X, 1180, ENC_W, 52,
                [("Downsample: 1×1 conv + pixel-unshuffle", 11.8, C["ink"], 500),
                 ("H,W ÷2   ·   C ×2", 10.5, C["slate"], None)],
                accent=C["blue400"])

D_Y = (961, 1103, 1245)
DEC_ROWS = (
    ("Upsample: 1×1 conv + pixel-shuffle  ·  H,W ×2 · C ÷2", 11.0),
    ("skip-concat + 1×1 reduce → Restormer blocks", 11.8),
)


def dec_block(y, chans):
    return multi_row(DEC_X, y, DEC_W, 62,
                     [(DEC_ROWS[0][0], DEC_ROWS[0][1], C["slate"], None),
                      (f"{DEC_ROWS[1][0]} · {chans}", DEC_ROWS[1][1],
                       C["ink"], 500)])


d1 = dec_block(D_Y[0], "48 ch")
d2 = dec_block(D_Y[1], "96 ch")
d3 = dec_block(D_Y[2], "192 ch")

bn = layer_row(290, 1344, 420, 52, "Bottleneck — Restormer blocks · 384 ch",
               size=13)

blocknote = multi_note(190, 1420, 620, [
    "Restormer block  =  MDTA → GDFN, each with a LayerNorm residual",
    "MDTA: multi-DConv head transposed attention, learnable per-head temperature",
    "GDFN: gated DConv feed-forward, expansion γ = 2.66"])

refine = layer_row(240, 1526, 520, 46,
                   "Refinement stage — 2 Restormer blocks at full resolution")
outproj = layer_row(240, 1600, 520, 44,
                    "output residual projection → 2-ch residual spectrogram R")
gres = halo(emph_node(240, 1676, 520, 66, "out_spec = projected_input + R",
                      "Restormer's global residual"),
            C["blue400"], dashed=False, pad=7, width=2.4)
istft = layer_row(270, 1776, 460, 44, "crop to un-padded size → iSTFT",
                  accent=C["blue400"])
inner_out = card(330, 1852, 340, 81, "inner_artifact", ["(B, 512)"])

# =========================================================================== #
#  CENTRE-SLOT INJECTION — the structural hinge
# =========================================================================== #
INJ_X, INJ_W, INJ_Y, INJ_H = 190, 620, 2000, 186
inj = card(INJ_X, INJ_Y, INJ_W, INJ_H, "refined_context = context.clone()")
BOXW, BOXGAP = 48, 9
SX = INJ_X + (INJ_W - (7 * BOXW + 6 * BOXGAP)) / 2
parts = [inj["svg"],
         f'<line x1="{INJ_X}" y1="{INJ_Y+36}" x2="{INJ_X+INJ_W}" y2="{INJ_Y+36}" '
         f'stroke="{C["ink"]}" stroke-opacity="0.12"/>',
         text(INJ_X + 18, 2062,
              "refined_context[:, 3, :] = context[:, 3, :] − inner_artifact",
              size=12.5, fill=C["ink"], weight=600),
         text(INJ_X + INJ_W - 18, 2062, "(B, 7, 512)", size=11.5,
              fill=C["slate"], italic=True, anchor="end")]
for i in range(7):
    bx = SX + i * (BOXW + BOXGAP)
    if i == 3:
        parts.append(f'<rect x="{bx}" y="2082" width="{BOXW}" height="44" '
                     f'rx="9" fill="url(#fp-header)" stroke="{C["blue"]}" '
                     f'stroke-width="1.6"/>')
        parts.append(text(bx + BOXW / 2, 2109, "3", size=13,
                          fill=C["header_fg"], weight=700, anchor="middle"))
    else:
        parts.append(f'<rect x="{bx}" y="2082" width="{BOXW}" height="44" '
                     f'rx="9" fill="{C["tint"]}" stroke="{C["slate"]}" '
                     f'stroke-opacity="0.45" stroke-width="1.2"/>')
        parts.append(text(bx + BOXW / 2, 2109, str(i), size=13, fill=C["slate"],
                          weight=600, anchor="middle"))
c1 = SX + 1 * (BOXW + BOXGAP) + BOXW / 2
c5 = SX + 5 * (BOXW + BOXGAP) + BOXW / 2
parts += [text(c1, 2146, "unchanged", size=10, fill=C["slate"], anchor="middle"),
          text(c5, 2146, "unchanged", size=10, fill=C["slate"], anchor="middle"),
          mini_chip(MID, 2142, "− inner_artifact", size=11),
          text(MID, 2172,
               "only the center epoch is corrected before the outer branch",
               size=10.5, fill=C["slate"], anchor="middle")]
inj["svg"] = "".join(parts)
inj = halo(inj, C["blue"], dashed=False, width=2.4)
inj["svg"] += badge(INJ_X + INJ_W - 14, INJ_Y + 18, "structural hinge")

# =========================================================================== #
#  OUTER BRANCH — 1-D residual U-Net over the multi-epoch context
# =========================================================================== #
OY, OH = 2230, 700
OENC_X, OENC_W = 96, 300                      # 96 .. 396
ODEC_X, ODEC_W = 580, 320                     # 580 .. 900
OENC_CX = OENC_X + OENC_W / 2                 # 246
ODEC_CX = ODEC_X + ODEC_W / 2                 # 740
LANE_OUT_R = 930

outer_frame = container(CONT_X, OY, CONT_W, OH,
                        "Outer branch — OuterTimeRefiner")

OE_Y = (2350, 2482, 2614)
OE_CY = tuple(y + 23 for y in OE_Y)
oe1 = layer_row(OENC_X, OE_Y[0], OENC_W, 46, "conv_block 7→32")
oe2 = layer_row(OENC_X, OE_Y[1], OENC_W, 46, "conv_block 32→64")
oe3 = layer_row(OENC_X, OE_Y[2], OENC_W, 46, "conv_block 64→128")
op1 = layer_row(OENC_X + 30, 2420, OENC_W - 60, 38, "avg_pool1d k=2",
                accent=C["blue400"], size=12)
op2 = layer_row(OENC_X + 30, 2552, OENC_W - 60, 38, "avg_pool1d k=2",
                accent=C["blue400"], size=12)
op3 = layer_row(OENC_X + 30, 2684, OENC_W - 60, 38, "avg_pool1d k=2",
                accent=C["blue400"], size=12)
obn = layer_row(310, 2754, 380, 48, "conv_block 128→256", size=13)


def odec(y, a, b, e):
    return multi_row(ODEC_X, y, ODEC_W, 62,
                     [(f"ConvTranspose1d {a}→{b}  k=2 s=2", 11.0, C["slate"], None),
                      (f"concat {e} → conv_block {a}→{b}", 11.8, C["ink"], 500)])


od1 = odec(2342, 64, 32, "e1")
od2 = odec(2474, 128, 64, "e2")
od3 = odec(2606, 256, 128, "e3")
ohead = layer_row(320, 2856, 360, 44, "head:  Conv1d 32→1  k=1")

# =========================================================================== #
#  OUTPUT
# =========================================================================== #
merge = emph_node(240, 2970, 520, 66, "artifact = inner_artifact + residual",
                  "(B, 1, 512)")
corr = card(300, 3072, 400, 81, "DeepLearningCorrection",
            ["corrected = noisy − artifact"])

recipe = card(50, 3196, 900, 150, "Training recipe (Niazy proof-fit)", [
    "Loss = 1.0 · L1(time) + 0.5 · multi-resolution log-magnitude STFT loss",
    "MR-STFT fft sizes 32 / 64 / 128 / 256 · hop fraction 0.25",
    "inner_channels 48 · inner_levels 3 · refinement blocks 2 · outer base 32",
    "The MR-STFT term is the documented surrogate for the paper's "
    "multi-resolution discriminators",
])

# --------------------------------------------------------------------------- #
#  Connectors (drawn beneath the nodes)
# --------------------------------------------------------------------------- #
# --- entry chain
d.add_edge(
    edge([(MID, inp["y"] + inp["h"] + 3), (MID, sq["y"] - 4)]),
    edge([(MID, sq["y"] + sq["h"] + 3), (MID, pick["y"] - 4)]),
    edge([(MID, pick["y"] + pick["h"] + 3), (MID, stft["y"] - 4)],
         label=("(B, 1, 512)", MID, 618)),
)

# --- inner: pre-U chain
d.add_edge(
    edge([(MID, stft["y"] + stft["h"] + 3), (MID, proj["y"] - 4)]),
    edge([(MID, proj["y"] + proj["h"] + 3), (MID, padrow["y"] - 4)]),
    leader(701, 840, 711, 851),
    # into level 1 (jog happens below the container title tab)
    edge([(MID, padrow["y"] + padrow["h"] + 3), (MID, 950),
          (ENC_CX, 950), (ENC_CX, e1["y"] - 4)]),
)

# --- inner: encoder descent
d.add_edge(
    edge([(ENC_CX, e1["y"] + e1["h"] + 3), (ENC_CX, ds1["y"] - 4)]),
    edge([(ENC_CX, ds1["y"] + ds1["h"] + 3), (ENC_CX, e2["y"] - 4)]),
    edge([(ENC_CX, e2["y"] + e2["h"] + 3), (ENC_CX, ds2["y"] - 4)]),
    edge([(ENC_CX, ds2["y"] + ds2["h"] + 3), (ENC_CX, e3["y"] - 4)]),
    edge([(ENC_CX, e3["y"] + e3["h"] + 3), (ENC_CX, 1324),
          (360, 1324), (360, bn["y"] - 4)]),
)

# --- inner: decoder ascent
d.add_edge(
    edge([(bn["x"] + bn["w"] + 3, bn["cy"]), (790, bn["cy"]),
          (790, d3["y"] + d3["h"] + 3)]),
    edge([(DEC_CX, d3["y"] - 3), (DEC_CX, d2["y"] + d2["h"] + 3)]),
    edge([(DEC_CX, d2["y"] - 3), (DEC_CX, d1["y"] + d1["h"] + 3)]),
    # out of the U, down the right inner corridor into the refinement stage
    edge([(d1["x"] + d1["w"] + 3, d1["cy"]), (LANE_IN_R, d1["cy"]),
          (LANE_IN_R, refine["cy"]), (refine["x"] + refine["w"] + 4,
                                      refine["cy"])]),
)

# --- inner: U-Net skips
d.add_edge(
    skip_link(ENC_X + ENC_W + 3, DEC_X - 5, e1["cy"], "skip"),
    skip_link(ENC_X + ENC_W + 3, DEC_X - 5, e2["cy"], "skip"),
    skip_link(ENC_X + ENC_W + 3, DEC_X - 5, e3["cy"], "skip"),
)

# --- inner: tail chain
d.add_edge(
    edge([(MID, refine["y"] + refine["h"] + 3), (MID, outproj["y"] - 4)]),
    edge([(MID, outproj["y"] + outproj["h"] + 3), (MID, gres["y"] - 11)]),
    edge([(MID, gres["y"] + gres["h"] + 11), (MID, istft["y"] - 4)]),
    edge([(MID, istft["y"] + istft["h"] + 3), (MID, inner_out["y"] - 4)]),
)

# --- inner: global-residual bypass (left inner corridor)
d.add_edge(skip_edge([(proj["x"], proj["cy"]), (LANE_IN_L, proj["cy"]),
                      (LANE_IN_L, gres["cy"]), (gres["x"] - 12, gres["cy"])],
                     soft=True))

# --- inner branch → injection
d.add_edge(edge([(MID, inner_out["y"] + inner_out["h"] + 3),
                 (MID, INJ_Y - 11)]))

# --- context bypass (left outer lane): the 7-epoch window reaches the hinge
d.add_edge(skip_edge([(sq["x"], sq["cy"]), (LANE_L_OUT, sq["cy"]),
                      (LANE_L_OUT, INJ_Y + INJ_H / 2),
                      (INJ_X - 12, INJ_Y + INJ_H / 2)]))

# --- injection → outer U (jog above the container, descend clear of the tab)
d.add_edge(edge([(MID, INJ_Y + INJ_H + 11), (MID, 2325),
                 (OENC_CX, 2325), (OENC_CX, oe1["y"] - 4)]))

# --- outer: encoder descent
d.add_edge(
    edge([(OENC_CX, oe1["y"] + oe1["h"] + 3), (OENC_CX, op1["y"] - 4)]),
    edge([(OENC_CX, op1["y"] + op1["h"] + 3), (OENC_CX, oe2["y"] - 4)]),
    edge([(OENC_CX, oe2["y"] + oe2["h"] + 3), (OENC_CX, op2["y"] - 4)]),
    edge([(OENC_CX, op2["y"] + op2["h"] + 3), (OENC_CX, oe3["y"] - 4)]),
    edge([(OENC_CX, oe3["y"] + oe3["h"] + 3), (OENC_CX, op3["y"] - 4)]),
    edge([(OENC_CX, op3["y"] + op3["h"] + 3), (OENC_CX, 2738),
          (360, 2738), (360, obn["y"] - 4)]),
)

# --- outer: decoder ascent
d.add_edge(
    edge([(obn["x"] + obn["w"] + 3, obn["cy"]), (ODEC_CX, obn["cy"]),
          (ODEC_CX, od3["y"] + od3["h"] + 3)]),
    edge([(ODEC_CX, od3["y"] - 3), (ODEC_CX, od2["y"] + od2["h"] + 3)]),
    edge([(ODEC_CX, od2["y"] - 3), (ODEC_CX, od1["y"] + od1["h"] + 3)]),
    edge([(od1["x"] + od1["w"] + 3, od1["cy"]), (LANE_OUT_R, od1["cy"]),
          (LANE_OUT_R, ohead["cy"]), (ohead["x"] + ohead["w"] + 4,
                                      ohead["cy"])]),
)

# --- outer: U-Net skips
d.add_edge(
    skip_link(OENC_X + OENC_W + 3, ODEC_X - 5, OE_CY[0], "skip"),
    skip_link(OENC_X + OENC_W + 3, ODEC_X - 5, OE_CY[1], "skip"),
    skip_link(OENC_X + OENC_W + 3, ODEC_X - 5, OE_CY[2], "skip"),
)

# --- outer → merge → correction
d.add_edge(
    edge([(MID, ohead["y"] + ohead["h"] + 3), (MID, merge["y"] - 4)],
         label=("residual", MID, 2938)),
    edge([(MID, merge["y"] + merge["h"] + 3), (MID, corr["y"] - 4)]),
)

# --- inner_artifact bypass down the outer-right lane into the merge node
d.add_edge(skip_edge([(inner_out["x"] + inner_out["w"], inner_out["cy"]),
                      (LANE_R_OUT, inner_out["cy"]),
                      (LANE_R_OUT, merge["cy"]),
                      (merge["x"] + merge["w"] + 5, merge["cy"])]))

# --------------------------------------------------------------------------- #
#  Nodes
# --------------------------------------------------------------------------- #
# branch tints (background layer) — inner = spectral, outer = time
d.add_edge(backdrop(CONT_X, IY, CONT_W, IH, C["blue400"], 0.16),
           backdrop(CONT_X, OY, CONT_W, OH, C["ink"], 0.028))

d.add(inner_frame, outer_frame)
d.add(cav, inp, sq, pick)

# inner branch labels
d.add(text(428, IY + 20, "operates on the STFT as a 2-channel real/imag image",
           size=12, fill=C["slate"], italic=True),
      mini_chip(845, IY + 16, "frequency domain", size=11))
d.add(stft, proj, proj_note, padrow,
      e1, ds1, e2, ds2, e3, d1, d2, d3, bn, blocknote,
      refine, outproj, gres, istft, inner_out)
d.add(skip_label_vertical(LANE_IN_L, 1250, "projected input (global residual)",
                          soft=True),
      f'<circle cx="{proj["x"]}" cy="{proj["cy"]}" r="4.5" '
      f'fill="{C["blue400"]}" stroke="{C["blue"]}" stroke-width="1.2"/>')

# outer branch labels
d.add(text(58, OY + 50, "1-D residual U-Net over the multi-epoch context",
           size=12, fill=C["slate"], italic=True),
      text(58, OY + 68, "fixes trigger-boundary phase discontinuities",
           size=12, fill=C["slate"], italic=True),
      mini_chip(860, OY + 16, "time domain", size=11))
d.add(oe1, op1, oe2, op2, oe3, op3, obn, od1, od2, od3, ohead)

d.add(inj, merge, corr, recipe)
d.add(skip_label_vertical(LANE_L_OUT, 1790, "context  (B, 7, 512)"),
      f'<circle cx="{sq["x"]}" cy="{sq["cy"]}" r="4.5" fill="{C["blue"]}"/>',
      skip_label_vertical(LANE_R_OUT, 2600, "inner_artifact"),
      f'<circle cx="{inner_out["x"]+inner_out["w"]}" cy="{inner_out["cy"]}" '
      f'r="4.5" fill="{C["blue"]}"/>')

d.render_png(str(OUT_PNG), svg_path=str(OUT_SVG), width=1280)
print("svg:", OUT_SVG)
print("png:", OUT_PNG)
print("canvas H:", H)
for nm, g in [("cav", cav), ("inp", inp), ("pick", pick), ("stft", stft),
              ("proj_note", proj_note), ("blocknote", blocknote),
              ("gres", gres), ("inner_out", inner_out), ("inj", inj),
              ("obn", obn), ("ohead", ohead), ("merge", merge),
              ("corr", corr), ("recipe", recipe)]:
    print(f"  {nm}: x={g['x']} y={g['y']} w={g['w']} h={g['h']} "
          f"bottom={g['y']+g['h']}")
print("inner container:", IY, "->", IY + IH)
print("outer container:", OY, "->", OY + OH)

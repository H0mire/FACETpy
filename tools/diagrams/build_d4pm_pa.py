"""D4PM — Paper-Accurate Edition: architecture diagram (FACETpy language).

Sixth diagram of the deep-learning model series. Three regions in one canvas:

    Region 1 — the shared epsilon-predictor (D4PMNoisePredictor), ×2 instances
    Region 2 — the training step (D4PMTrainingModule.forward)
    Region 3 — the inference reverse loop (joint posterior sampling, Algorithm 1)

Reuses the custom helpers introduced by the earlier build scripts so the whole
set reads as one system:

    skip_edge() / skip_label_vertical() / share_link()   (DPAE / IC-U-Net / cascaded)
    multi_note() / multi_row() / layer_row()             (IC-U-Net / cascaded)
    mini_chip(), halo(), badge(), backdrop(), jdot()     (cascaded / nested-GAN)
    stack_card(), hop_h()                                (DHCT-GAN)
    caveat_card(), emph_node()                           (nested-GAN)
    two-line title block                                 (IC-U-Net / cascaded)

Verified against
    src/facet/models/experimental/paper_accurate/d4pm/training.py
    src/facet/models/experimental/paper_accurate/d4pm/processor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "skills/facetpy-diagram/assets"))

from facetpy_svg import (  # noqa: E402
    C, HEAD, Diagram, capsule, card, container, decision, edge, eeg_wave,
    node_dot, rounded_top, text, text_width, _geom,
)

OUT_SVG = REPO / "docs/source/_static/diagrams/d4pm_pa_architecture.svg"
OUT_PNG = REPO / "docs/source/_static/diagrams/d4pm_pa_architecture.png"

# --------------------------------------------------------------------------- #
#  Canvas + two-line title block (shared with the rest of the series)
# --------------------------------------------------------------------------- #
H = 4140
d = Diagram(H)

TITLE = "D4PM — Paper-Accurate Edition"
SUB = ("Dual-branch denoising diffusion with joint posterior sampling · "
       "src/facet/models/experimental/paper_accurate/d4pm/")
d.add(text(70, 52, TITLE, size=24, weight=700, fill=C["ink"]),
      f'<rect x="70" y="62" width="{22 + len(TITLE) * 9}" height="3" rx="1.5" '
      f'fill="url(#fp-header)"/>',
      eeg_wave(70 + text_width(TITLE, 24) + 34, 44, 90),
      text(70, 84, SUB, size=13.5, fill=C["slate"]))

# --------------------------------------------------------------------------- #
#  Custom brand elements (shared vocabulary of the series)
# --------------------------------------------------------------------------- #
d.add_defs(
    f'<marker id="fp-skip" markerWidth="12" markerHeight="12" refX="9" refY="5" '
    f'orient="auto"><path d="M1 1 L10 5 L1 9 Z" fill="{C["blue"]}"/></marker>',
    f'<marker id="fp-skip2" markerWidth="12" markerHeight="12" refX="9" refY="5" '
    f'orient="auto"><path d="M1 1 L10 5 L1 9 Z" fill="{C["blue400"]}"/></marker>',
    f'<marker id="fp-skip-start" markerWidth="12" markerHeight="12" refX="1" '
    f'refY="5" orient="auto"><path d="M10 1 L1 5 L10 9 Z" '
    f'fill="{C["blue"]}"/></marker>',
    f'<marker id="fp-loop" markerWidth="14" markerHeight="14" refX="10" refY="6" '
    f'orient="auto"><path d="M1 1 L12 6 L1 11 Z" fill="{C["blue"]}"/></marker>',
)


def skip_edge(points, soft=False):
    """Dashed accent bypass connector — visually distinct from the main chain."""
    col = C["blue400"] if soft else C["blue"]
    mk = "fp-skip2" if soft else "fp-skip"
    pstr = " ".join(f"{px},{py}" for px, py in points)
    return (f'<polyline points="{pstr}" fill="none" stroke="{col}" '
            f'stroke-opacity="0.9" stroke-width="2.2" stroke-dasharray="9 5" '
            f'stroke-linejoin="round" marker-end="url(#{mk})"/>')


def share_link(x1, x2, y, label):
    """Double-headed dashed link: 'these two use the SAME module instance'."""
    line = (f'<path d="M{x1} {y} H{x2}" fill="none" stroke="{C["blue"]}" '
            f'stroke-width="2" stroke-dasharray="7 5" '
            f'marker-end="url(#fp-skip)" marker-start="url(#fp-skip-start)"/>')
    lx, ly = (x1 + x2) / 2, y + 4
    bw = text_width(label, 11) + 18
    chip = (f'<rect x="{lx-bw/2:.1f}" y="{ly-12}" width="{bw:.1f}" height="17" '
            f'rx="5" fill="{C["surface"]}" stroke="{C["blue"]}" '
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


def layer_row(x, y, w, h, txt, accent=None, size=12.5, dx=0):
    """Compact process row — pill(), with a controllable text size."""
    accent = accent or C["blue"]
    svg = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
           f'fill="{C["tint"]}" stroke="{accent}" stroke-opacity="0.55" '
           f'stroke-width="1.4" filter="url(#fp-shadow)"/>'
           f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" fill="{accent}"/>'
           + text(x + w / 2 + 2 + dx, y + h / 2 + 4.5, txt, size=size,
                  weight=500, anchor="middle"))
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


def step_dot(cx, cy, n, dark=True):
    """A connectome dot carrying a step number — the ordered-body marker."""
    fill = "url(#fp-header)" if dark else C["blue200"]
    fg = C["header_fg"] if dark else C["navy_d"]
    return (f'<circle cx="{cx}" cy="{cy}" r="11" fill="{fill}" '
            f'stroke="#ffffff" stroke-opacity="0.85" stroke-width="1.4"/>'
            + text(cx, cy + 4, str(n), size=12, fill=fg, weight=700,
                   anchor="middle"))


def step_row(x, y, w, h, n, txt, size=12.5):
    """Numbered process row — the ordered body of the reverse loop."""
    g = layer_row(x, y, w, h, txt, size=size, dx=14)
    g["svg"] += step_dot(x + 26, y + h / 2, n)
    return g


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
    p.append(text(x0 + gw / 2 + bw / 2, cy + 4, txt, size=10.5,
                  fill=C["navy_d"], weight=700, anchor="middle"))
    return "".join(p)


def backdrop(x, y, w, h, fill, op):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
            f'fill="{fill}" fill-opacity="{op}"/>')


def jdot(x, y):
    """Junction dot where a connector branches."""
    return f'<circle cx="{x}" cy="{y}" r="4.5" fill="{C["blue"]}"/>'


def bar(points):
    """Structural connector segment without an arrowhead (fork / merge bars)."""
    return edge(points, marker_end=None)


def hop_h(x1, x2, y, hops=(), color=None, dashed=True, marker=True):
    """Horizontal accent run from x1→x2 with semicircular hops over crossings."""
    color = color or C["blue"]
    parts = [f"M{x1} {y}"]
    for hx in sorted(hops):
        parts.append(f"L{hx-8} {y} A 8 8 0 0 1 {hx+8} {y}")
    parts.append(f"L{x2} {y}")
    dash = ' stroke-dasharray="9 5"' if dashed else ""
    mk = ' marker-end="url(#fp-skip)"' if marker else ""
    return (f'<path d="{" ".join(parts)}" fill="none" stroke="{color}" '
            f'stroke-opacity="0.9" stroke-width="2.2"{dash}{mk}/>')


def loop_edge(points, label=None):
    """The reverse-loop back-edge: solid, heavy, brand-blue, rounded corners."""
    pstr = " ".join(f"{px},{py}" for px, py in points)
    return (f'<polyline points="{pstr}" fill="none" stroke="{C["blue"]}" '
            f'stroke-opacity="0.95" stroke-width="3" stroke-linejoin="round" '
            f'stroke-linecap="round" marker-end="url(#fp-loop)"/>')


def stack_card(x, y, w, title, lines):
    """A card with two ghost copies behind it — the 'repeated N times' mark."""
    g = card(x, y, w, 0, title, lines)
    h = g["h"]
    ghosts = "".join(
        f'<rect x="{x+o}" y="{y+o}" width="{w}" height="{h}" rx="12" '
        f'fill="{C["surface"]}" stroke="{C["ink"]}" stroke-opacity="0.16" '
        f'stroke-width="1.25" filter="url(#fp-shadow)"/>'
        for o in (18, 9))
    g["svg"] = ghosts + g["svg"]
    return g


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
    """Gradient emphasis node, optionally with a circled operator glyph."""
    cy = y + h / 2
    p = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
         f'fill="url(#fp-header)" filter="url(#fp-shadow)"/>']
    shift = 0
    if glyph:
        gx = x + 34
        shift = 20
        p += [f'<circle cx="{gx}" cy="{cy}" r="11" fill="none" '
              f'stroke="{C["blue200"]}" stroke-width="1.8"/>',
              f'<line x1="{gx-5.5}" y1="{cy}" x2="{gx+5.5}" y2="{cy}" '
              f'stroke="{C["blue200"]}" stroke-width="1.8" '
              f'stroke-linecap="round"/>']
        if glyph == "+":
            p.append(f'<line x1="{gx}" y1="{cy-5.5}" x2="{gx}" y2="{cy+5.5}" '
                     f'stroke="{C["blue200"]}" stroke-width="1.8" '
                     f'stroke-linecap="round"/>')
    p.append(text(x + w / 2 + shift, cy - 4, main, size=14.5,
                  fill=C["header_fg"], weight=700, anchor="middle"))
    p.append(text(x + w / 2 + shift, cy + 15, sub, size=11,
                  fill=C["blue200"], weight=500, anchor="middle"))
    return _geom("".join(p), x, y, w, h)


MID = 500
CX, CW = 40, 920                 # every region container
CR = CX + CW                     # 960
LANE_R = 978                     # outer-right lane: "uses the ε-predictor"

# Region geometry ----------------------------------------------------------- #
R1_Y, R1_H = 292, 1342                       # ε-predictor
LOOP_X, LOOP_Y, LOOP_W, LOOP_H = 56, 850, 876, 534    # × 3 layer loop
R2_Y, R2_H = 1684, 1100                      # training step
R3_Y, R3_H = 2834, 1082                      # inference
RL_X, RL_Y, RL_W, RL_H = 56, 3034, 888, 646  # reverse loop

# Region tints and the two repeat/loop bands go down FIRST, so every connector
# and node draws on top of them.
d.add_edge(
    "".join(f'<rect x="{LOOP_X+o}" y="{LOOP_Y+o}" width="{LOOP_W}" '
            f'height="{LOOP_H}" rx="18" fill="none" stroke="{C["blue"]}" '
            f'stroke-width="1.4" stroke-opacity="{op}" stroke-dasharray="7 5"/>'
            for o, op in ((13, 0.18), (6.5, 0.32))),
    backdrop(LOOP_X, LOOP_Y, LOOP_W, LOOP_H, C["blue400"], 0.13),
    f'<rect x="{LOOP_X}" y="{LOOP_Y}" width="{LOOP_W}" height="{LOOP_H}" '
    f'rx="18" fill="none" stroke="{C["blue"]}" stroke-width="2" '
    f'stroke-opacity="0.75" stroke-dasharray="7 5"/>',
    backdrop(CX, R3_Y, CW, R3_H, C["blue400"], 0.10),
    backdrop(RL_X, RL_Y, RL_W, RL_H, C["ink"], 0.035),
    f'<rect x="{RL_X}" y="{RL_Y}" width="{RL_W}" height="{RL_H}" rx="18" '
    f'fill="none" stroke="{C["blue"]}" stroke-width="2" stroke-opacity="0.7" '
    f'stroke-dasharray="7 5"/>',
)

# =========================================================================== #
#  CALLOUT — the TorchScript caveat, directly under the title
# =========================================================================== #
cav = caveat_card(
    50, 104, 900, "TorchScript export is a stub",
    ["torch.jit.trace hits a trace-stable stub that returns zeros",
     "MultiheadAttention kernel selection breaks trace's sanity check, so "
     "D4PMTrainingModule.forward() short-circuits under tracing.",
     "Real inference runs the state-dict checkpoint through the adapter's "
     "iterative sampler —",
     "the .ts file cannot be used for this model."])

# =========================================================================== #
#  REGION 1 — the ε-predictor (D4PMNoisePredictor)
# =========================================================================== #
r1_frame = container(CX, R1_Y, CW, R1_H, "ε-predictor — D4PMNoisePredictor")
r1_sub = text(58, 338,
              "two instances with INDEPENDENT weights:  predictor (artifact)  ·  "
              "predictor_clean (EEG)",
              size=12.5, fill=C["slate"], italic=True)

# --- band 1: conditioning embedding (side sub-block) + the ×2 stack --------- #
cond_noise = multi_row(76, 356, 250, 48, [
    ("SinusoidalNoiseLevelEmbedding", 11.0, C["ink"], 500),
    ("input: √ᾱ*  (continuous level)", 10.0, C["slate"], None)])
cond_class = multi_row(350, 356, 250, 48, [
    ("ClassEmbedding(z)", 11.5, C["ink"], 500),
    ("num_classes = 1", 10.0, C["slate"], None)])
cond_mlp = multi_row(76, 420, 250, 48, [
    ("embed_mlp", 11.5, C["ink"], 500),
    ("Linear → ReLU → Linear", 10.0, C["slate"], None)])
cond_merge = emph_node(76, 486, 524, 58,
                       "cond_embed = noise_e + class_e", "embed_dim = 128")

d.add_edge(
    edge([(201, 404), (201, 416)]),
    edge([(201, 468), (201, 482)]),
    edge([(475, 404), (475, 482)]),
)

instances = stack_card(634, 356, 304, "× 2 instances", [
    "predictor → artifact branch",
    "predictor_clean → EEG branch",
    "independent weights, no sharing"])

# --- band 2: the two structurally parallel input paths --------------------- #
LB_X, RB_X, BW = 150, 590, 260          # lane B (h_t) and lane C (y)
LB_CX, RB_CX = LB_X + BW / 2, RB_X + BW / 2      # 280 / 720

paths_chip = mini_chip(MID, 572, "two parallel input paths — structural")
in_x = card(LB_X, 600, BW, 0, "h_t", ["noise-perturbed state"])
in_y = card(RB_X, 600, BW, 0, "y", ["conditioning observation"])
conv_x = layer_row(LB_X, 705, BW, 40, "Conv1d 1→64  k=3")
conv_y = layer_row(RB_X, 705, BW, 40, "Conv1d 1→64  k=3")
lin_x = layer_row(LB_X, 769, BW, 40, "Linear 64→128")
lin_y = layer_row(RB_X, 769, BW, 40, "Linear 64→128")
proj_txt = text(MID, 833, "both paths share the same proj_in Linear",
                size=11.5, fill=C["slate"], anchor="middle", italic=True)

d.add_edge(
    edge([(LB_CX, 681), (LB_CX, 701)]), edge([(RB_CX, 681), (RB_CX, 701)]),
    edge([(LB_CX, 745), (LB_CX, 765)]), edge([(RB_CX, 745), (RB_CX, 765)]),
    share_link(LB_X + BW + 4, RB_X - 4, 789, "shared proj_in"),
    # into the layer loop
    edge([(LB_CX, 813), (LB_CX, 890)]), edge([(RB_CX, 813), (RB_CX, 890)]),
)

# --- band 3: the layer loop ------------------------------------------------ #
loop_chip = mini_chip(232, 872, "× 3 layers  ·  n_layers = 3  ·  the paper's ×3")
ml_chip = mini_chip(700, 872, "layers_x and layers_c are separate ModuleLists")

TL = ("TransformerEncoderLayer1D", 10.5, C["slate"], None)
TL2 = ("post-norm · n_heads = 2 · d_ff = 512", 10.0, C["slate"], None)
lay_x = multi_row(LB_X, 894, BW, 66,
                  [("x = layer_x(x)", 12.5, C["ink"], 600), TL, TL2])
lay_c = multi_row(RB_X, 894, BW, 66,
                  [("c = layer_c(c)", 12.5, C["ink"], 600), TL, TL2])

film = emph_node(350, 1000, 300, 66, "DualFiLM",
                 "shared — one instance per depth level", glyph=None)
film = halo(film, C["blue"], dashed=False, pad=6, width=2.4)

film_x = layer_row(LB_X, 1100, BW, 44, "x = film(x, cond_embed)")
film_c = layer_row(RB_X, 1100, BW, 44, "c = film(c, cond_embed)")

merge = emph_node(300, 1188, 400, 66, "x  =  x + c",
                  "fusion happens INSIDE every layer", glyph="+")
merge = halo(merge, C["blue200"], dashed=False, pad=6, width=2.4)

exit_row = layer_row(LB_X, 1314, 700, 46,
                     "next iteration:   layer_x receives the fused x    ·    "
                     "layer_c receives c", size=12)

d.add_edge(
    # the two paths run straight down, clear of the centred DualFiLM node
    edge([(LB_CX, 960), (LB_CX, 1096)]),
    edge([(RB_CX, 960), (RB_CX, 1096)]),
    # the SAME DualFiLM instance fans out to both paths
    skip_edge([(420, 1074), (420, 1084), (340, 1084), (340, 1094)]),
    skip_edge([(580, 1074), (580, 1084), (660, 1084), (660, 1094)]),
    # fusion
    edge([(LB_CX, 1144), (LB_CX, 1170), (390, 1170), (390, 1182)]),
    edge([(RB_CX, 1144), (RB_CX, 1170), (610, 1170), (610, 1182)]),
    edge([(MID, 1260), (MID, 1310)]),
    # c continues on its own path into the next iteration
    skip_edge([(RB_CX, 1170), (890, 1170), (890, 1337), (854, 1337)]),
)
# conditioning embedding → the shared DualFiLM (left corridor, hops the x lane)
d.add_edge(edge([(76, 515), (52, 515), (52, 1033)], marker_end=None),
           hop_h(52, 344, 1033, hops=(LB_CX,)))

# after the last of the 3 layers the fused x leaves the loop for the head
d.add_edge(edge([(292, 1221), (88, 1221), (88, 1510), (132, 1510),
                 (132, 1526)]))

fuse_note = multi_note(110, 1408, 834, [
    "Fusion happens INSIDE every layer — easy to get wrong",
    "x = x + c is applied at each of the 3 depths. The FUSED x carries into the "
    "next layer_x,",
    "while c continues on its own path through layers_c and is never "
    "overwritten by the fusion.",
    "One DualFiLM per depth level modulates BOTH paths with the same γ, ξ "
    "before they are added.",
])

# --- band 4: output head --------------------------------------------------- #
HEAD_Y, HW, HGAP = 1534, 152, 32
HX = [56 + i * (HW + HGAP) for i in range(5)]
h1 = layer_row(HX[0], HEAD_Y, HW, 46, "Linear 128→64", size=12)
h2 = layer_row(HX[1], HEAD_Y, HW, 46, "Conv1d 64→64  k=3", size=11.5)
h3 = layer_row(HX[2], HEAD_Y, HW, 46, "ReLU", size=12)
h4 = layer_row(HX[3], HEAD_Y, HW, 46, "Conv1d 64→1  k=1", size=11.5)
h5 = capsule(HX[4], HEAD_Y, HW, 46, "ε̂")
d.add_edge(*[edge([(HX[i] + HW + 3, HEAD_Y + 23), (HX[i + 1] - 4, HEAD_Y + 23)])
             for i in range(4)])
head_txt = text(MID, 1608, "the paper's (3×1, 1×1) projection head",
                size=11.5, fill=C["slate"], anchor="middle", italic=True)

# =========================================================================== #
#  REGION 2 — training step (D4PMTrainingModule.forward)
# =========================================================================== #
r2_frame = container(CX, R2_Y, CW, R2_H, "Training step")

pack = card(280, 1726, 440, 0, "Dataset pack", [
    "(B, 3, T)  =  [ noisy_y , artifact₀ , clean₀ ]",
    "dual_branch = True   (paper-faithful default)"])

samp_lbl = text(76, 1870, "sample per example:", size=11.5, fill=C["slate"],
                italic=True)
chips = [mini_chip(300, 1866, "t ~ U{0 … 199}"),
         mini_chip(500, 1866, "u ~ U[0,1)"),
         mini_chip(700, 1866, "ε ~ N(0, I)")]

level = card(220, 1906, 560, 0, "√ᾱ*  =  lo + u · (hi − lo)", [
    "lo = √ᾱ_{t−1}        ·        hi = √ᾱ_t",
    "continuous level, not the discrete √ᾱ_t — used for BOTH q_sample",
    "and the conditioning embedding"])
level = halo(level, C["blue"], dashed=False, width=2.4)
level["svg"] += badge(780 - 14, 1906 + 18, "paper component #1")

fwd = emph_node(220, 2074, 560, 62, "h_t  =  √ᾱ* · x₀ + √(1−ᾱ*) · ε",
                "applied to BOTH x₀ with the SAME level and the SAME ε",
                glyph=None)
ht_art = layer_row(180, 2184, 280, 44, "h_t^art  ←  artifact₀")
ht_cln = layer_row(540, 2184, 280, 44, "h_t^clean  ←  clean₀")
share_txt = text(MID, 2262,
                 "shared level and shared ε keep the two marginals consistently "
                 "corrupted for joint sampling",
                 size=11.5, fill=C["slate"], anchor="middle", italic=True)

pred_a = card(120, 2310, 360, 0, "predictor", [
    "(h_t^art, y, √ᾱ*, z)", "→  pred_ε_art"])
pred_a["svg"] += badge(480 - 14, 2310 + 18, "ε-predictor")
pred_c = card(520, 2310, 360, 0, "predictor_clean", [
    "(h_t^clean, y, √ᾱ*, z)", "→  pred_ε_clean"])
pred_c["svg"] += badge(880 - 14, 2310 + 18, "ε-predictor")

cat_row = layer_row(220, 2458, 560, 46,
                    "cat([pred_ε_art, ε, pred_ε_clean, ε])   →   (B, 4, T)")
loss = card(280, 2534, 440, 0, "D4PMEpsilonLoss", [
    "L1 between each (pred, true) ε pair",
    "0.5 · (L_art + L_clean)   —   Eq. 1"])
eval_note = multi_note(56, 2674, 888, [
    "eval mode is deterministic",
    "t is spread across the schedule  ·  u = 0.5 (interval midpoint)  ·  ε = 0",
    "→ the validation loss is comparable across epochs",
])

d.add_edge(
    edge([(MID, 1834), (MID, 1851)]),
    edge([(MID, 1882), (MID, 1902)]),
    edge([(MID, 2040), (MID, 2070)]),
    bar([(MID, 2136), (MID, 2158)]), bar([(320, 2158), (680, 2158)]),
    edge([(320, 2158), (320, 2180)]), edge([(680, 2158), (680, 2180)]),
    edge([(320, 2228), (320, 2244), (200, 2244), (200, 2306)]),
    edge([(680, 2228), (680, 2244), (800, 2244), (800, 2306)]),
    edge([(300, 2414), (300, 2434), (MID, 2434), (MID, 2454)]),
    bar([(700, 2414), (700, 2434), (MID, 2434)]),
    edge([(MID, 2504), (MID, 2530)]),
)

# =========================================================================== #
#  REGION 3 — inference: Algorithm 1
# =========================================================================== #
r3_frame = container(CX, R3_Y, CW, R3_H,
                     "Inference — joint posterior sampling (Algorithm 1)")
r3_frame = halo(r3_frame, C["blue"], dashed=False, pad=5, width=2.2)

init = card(240, 2876, 520, 0, "Init", [
    "y = demeaned noisy epoch",
    "x_T^art ~ N(0, I)        ·        x_T^clean ~ N(0, I)",
    "strided schedule:  linspace(num_steps−1, 0, sample_steps)"])

rl_chip = mini_chip(196, 3056, "reverse loop — one ancestral step per t")

SX, SW = 150, 700
s1 = step_row(SX, 3080, SW, 44, 1,
              "x₀^clean = predict_x0(predictor_clean, x_t^clean, y, t)", size=12)
s2 = step_row(SX, 3144, SW, 44, 2,
              "x₀^art   = predict_x0(predictor, x_t^art, y, t)", size=12)
s3 = card(SX, 3220, SW, 0, "residual = y − (x₀^clean + λ_SNR · x₀^art)",
          ["the mixture constraint  y = x + x'·λ_SNR    (Eq. 2 / Eq. 4)"])
s3 = halo(s3, C["blue"], dashed=False, width=2.4)
s3["svg"] += (badge(SX + SW - 14, 3220 + 18, "the defining idea")
              + step_dot(SX + 18, 3220 + 18, 3, dark=False))
s4a = step_row(SX, 3340, 340, 44, 4, "x̂₀^clean = x₀^clean + λ_dc · residual",
               size=11.5)
s4b = step_row(510, 3340, 340, 44, 4, "x̂₀^art = x₀^art + (1 − λ_dc) · residual",
               size=11.5)
s4_txt = text(MID, 3408, "the residual is split between the two branches",
              size=11.5, fill=C["slate"], anchor="middle", italic=True)
s5 = card(SX, 3428, SW, 0, "ancestral DDPM step for BOTH branches", [
    "x_{t_prev}  =  coef1 · x̂₀  +  coef2 · x_t  +  √(posterior_variance) · z",
    "true ancestral update via posterior_mean_coef1 / coef2   ·   "
    "η = 0 on the final step"])
s5["svg"] += step_dot(SX + 18, 3428 + 18, 5, dark=False)
dec = decision(MID, 3600, 300, 92, "t_prev ≤ 0 ?")
dec["svg"] += step_dot(MID, 3578, 6)

out_art = card(300, 3716, 400, 0, "artifact  =  x̂₀^art",
               ["predicted artifact, per channel-epoch"])
corr = capsule(250, 3836, 500, 46,
               "DeepLearningCorrection:   corrected = noisy − artifact")

d.add_edge(
    edge([(MID, 3003), (MID, 3030)]),
    edge([(MID, 3124), (MID, 3140)]),
    edge([(MID, 3188), (MID, 3213)]),
    bar([(MID, 3308), (MID, 3324)]), bar([(320, 3324), (680, 3324)]),
    edge([(320, 3324), (320, 3336)]), edge([(680, 3324), (680, 3336)]),
    edge([(320, 3384), (320, 3424)]), edge([(680, 3384), (680, 3424)]),
    edge([(MID, 3532), (MID, 3550)]),
    edge([(MID, 3646), (MID, 3712)],
         label=("yes  ·  exit the loop", MID, 3670)),
    edge([(MID, 3797), (MID, 3832)]),
)
# the back-edge — this is a LOOP, not a stray connector
d.add_edge(loop_edge([(346, 3600), (96, 3600), (96, 3102), (146, 3102)]))

# =========================================================================== #
#  The shared ε-predictor lane: Region 1 feeds every call site below
# =========================================================================== #
d.add_edge(
    edge([(944, 1557), (LANE_R, 1557), (LANE_R, 3166)], marker_end=None),
    hop_h(LANE_R, 300, 2290, hops=(800,), marker=False),
    skip_edge([(300, 2290), (300, 2304)]),
    skip_edge([(700, 2290), (700, 2304)]),
    hop_h(LANE_R, 854, 3102),
    hop_h(LANE_R, 854, 3166),
)

cfg = card(CX, 3966, CW, 0, "Config (Niazy proof-fit)", [
    "num_steps 200 · β 1e-4 → 0.02 · feats 64 · d_model 128 · d_ff 512 · "
    "n_heads 2 · n_layers 3",
    "embed_dim 128 · num_classes 1 · norm_first false · dual_branch true · "
    "λ_SNR 1.0",
    "Loss: L1 ε-prediction (Eq. 1) · epoch_samples 512 · channel-wise",
])

# =========================================================================== #
#  Assemble
# =========================================================================== #
d.add(r1_frame, r2_frame, r3_frame)
d.add(cav, r1_sub)
d.add(cond_noise, cond_class, cond_mlp, cond_merge, instances)
d.add(paths_chip, in_x, in_y, conv_x, conv_y, lin_x, lin_y, proj_txt)
d.add(loop_chip, ml_chip, lay_x, lay_c, film, film_x, film_c, merge, exit_row,
      jdot(RB_CX, 1170),
      skip_label_vertical(890, 1250, "c  (unfused)"),
      mini_chip(170, 1010, "cond_embed → DualFiLM", size=11),
      skip_label_vertical(88, 1300, "after the 3rd layer"))
d.add(fuse_note, h1, h2, h3, h4, h5, head_txt)
d.add(pack, samp_lbl, *chips, level, fwd, ht_art, ht_cln, share_txt,
      pred_a, pred_c, cat_row, loss, eval_note)
d.add(init, rl_chip, s1, s2, s3, s4a, s4b, s4_txt, s5, dec, out_art, corr)
d.add(text(348, 3592, "no", size=11.5, fill=C["blue"], weight=700,
           anchor="end"),
      skip_label_vertical(96, 3360, "next t → t_prev"))
d.add(skip_label_vertical(LANE_R, 2010, "uses the ε-predictor above"),
      skip_label_vertical(LANE_R, 2900, "uses the ε-predictor above"),
      jdot(LANE_R, 2290), jdot(LANE_R, 3102))
d.add(cfg)

d.render_png(str(OUT_PNG), svg_path=str(OUT_SVG), width=1280)
print("svg:", OUT_SVG)
print("png:", OUT_PNG)
print("canvas H:", H)
for nm, g in [("cav", cav), ("cond_merge", cond_merge), ("instances", instances),
              ("in_x", in_x), ("lin_x", lin_x), ("lay_x", lay_x),
              ("film", film), ("film_x", film_x), ("merge", merge),
              ("exit_row", exit_row), ("fuse_note", fuse_note), ("h5", h5),
              ("pack", pack), ("level", level), ("fwd", fwd),
              ("pred_a", pred_a), ("cat_row", cat_row), ("loss", loss),
              ("eval_note", eval_note), ("init", init), ("s1", s1),
              ("s3", s3), ("s5", s5), ("dec", dec), ("out_art", out_art),
              ("corr", corr), ("cfg", cfg)]:
    print(f"  {nm:11s} x={g['x']:>5} y={g['y']:>5} w={g['w']:>4} "
          f"h={g['h']:>4} bottom={g['y']+g['h']}")
print("R1:", R1_Y, "->", R1_Y + R1_H)
print("R2:", R2_Y, "->", R2_Y + R2_H)
print("R3:", R3_Y, "->", R3_Y + R3_H)
print("loop band:", LOOP_Y, "->", LOOP_Y + LOOP_H)
print("reverse loop band:", RL_Y, "->", RL_Y + RL_H)

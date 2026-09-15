"""DHCT-GAN (paper-accurate edition) — generator dataflow + adversarial structure.

One diagram, two regions:
  * the generator flows top-to-bottom (stem → 4× CNN-LGTB encoder → bottleneck →
    dual decoder → two gating heads → fusion → exported artifact Y2),
  * the adversarial part is fenced off at the bottom (D1/D2/D3, LSGAN losses).

Verified against src/facet/models/dhct_gan_paper_accurate_edition/training.py.
Shares its custom helpers (skip_edge, skip_label_vertical, multi_note,
layer_row, halo, the two-line title block) with the other diagrams in the series.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "skills/facetpy-diagram/assets"))

from facetpy_svg import (  # noqa: E402
    C, Diagram, capsule, card, container, edge, eeg_wave, text, text_width,
    _geom,
)

OUT_SVG = REPO / "docs/source/_static/diagrams/dhct_gan_pa_architecture.svg"
OUT_PNG = REPO / "docs/source/_static/diagrams/dhct_gan_pa_architecture.png"

# --------------------------------------------------------------------------- #
#  Canvas + two-line title block (same treatment as the sibling diagrams)
# --------------------------------------------------------------------------- #
H = 3110
d = Diagram(H)

TITLE = "DHCT-GAN — Paper-Accurate Edition"
SUB = ("Dual-branch hybrid CNN-Transformer generator + 3 LSGAN discriminators · "
       "src/facet/models/dhct_gan_paper_accurate_edition/")
d.add(text(70, 52, TITLE, size=24, weight=700, fill=C["ink"]),
      f'<rect x="70" y="62" width="{22 + len(TITLE) * 9}" height="3" rx="1.5" '
      f'fill="url(#fp-header)"/>',
      eeg_wave(70 + text_width(TITLE, 24) + 34, 44, 90),
      text(70, 84, SUB, size=13.5, fill=C["slate"]))

# --------------------------------------------------------------------------- #
#  Shared custom elements (reused from build_dpae / build_ic_unet / build_cascaded)
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
    """Folded-corner annotation that carries several lines (heading first)."""
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


def halo(g, color, dashed=False, pad=7, width=2.2):
    """Emphasis ring around a card — marks the fusion node."""
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


def backdrop(x, y, w, h, fill=None, op=0.03):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
            f'fill="{fill or C["ink"]}" fill-opacity="{op}"/>')


def jdot(x, y):
    """Junction dot where a connector branches."""
    return f'<circle cx="{x}" cy="{y}" r="4.5" fill="{C["blue"]}"/>'


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


def plain(points, color=None, width=2.2, dashed=True):
    """Accent polyline without a marker (used for the Xraw vertical run)."""
    color = color or C["blue"]
    pstr = " ".join(f"{px},{py}" for px, py in points)
    dash = ' stroke-dasharray="9 5"' if dashed else ""
    return (f'<polyline points="{pstr}" fill="none" stroke="{color}" '
            f'stroke-opacity="0.9" stroke-width="{width}" '
            f'stroke-linejoin="round"{dash}/>')


def bar(points):
    """Structural connector segment without an arrowhead (fork / merge bars)."""
    return edge(points, marker_end=None)


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


print("helpers ready")

# =========================================================================== #
#  GENERATOR — main column
# =========================================================================== #
MID = 500
SCX = 364                      # centre of the expanded EncoderStage column

inp = card(340, 112, 320, 0, "Noisy Epoch  Xraw",
           ["(B, 1, 512)", "single channel, demeaned"])
stem = card(340, 250, 320, 0, "Preprocessing stem",
            ["2 × Conv1d", "optional AvgPool  (stem_pool = false)"])
d.add_edge(edge([(MID, 219), (MID, 247)]))

# --- Encoder container ----------------------------------------------------- #
ENC_Y, ENC_H = 400, 530
enc_frame = container(90, ENC_Y, 866, ENC_H, "Encoder — 4 × CNN-LGTB stage")

d.add_edge(edge([(MID, 357), (MID, 380), (SCX, 380), (SCX, 449)]))

lbl = text(112, 446, "EncoderStage — 1 of 4, expanded",
           size=11, fill=C["slate"], weight=700)

row_a = layer_row(SCX - 150, 452, 300, 42, "shared CNNBlock  in→out")

# fork bar into the two parallel paths
LPX, RPX = 227, 501
d.add_edge(bar([(SCX, 497), (SCX, 512)]),
           edge([(LPX, 512), (RPX, 512)], marker_end=None,
                label=("two PARALLEL paths", SCX, 517)),
           edge([(LPX, 512), (LPX, 521)]),
           edge([(RPX, 512), (RPX, 521)]))

path_cnn = card(112, 524, 230, 0, "parallel CNN path",
                ["Conv1d k=3", "BatchNorm · LeakyReLU(0.2)"])
path_lgtb = card(386, 524, 230, 0, "LGTB path",
                 ["LGTBStack (n_lgtb = 2)", "num_heads = 4 · see below"])

# merge bar back to the fuse row
d.add_edge(bar([(LPX, 631), (LPX, 646)]),
           bar([(RPX, 631), (RPX, 646)]),
           bar([(LPX, 646), (RPX, 646)]),
           edge([(SCX, 646), (SCX, 655)]))

fuse = layer_row(SCX - 240, 658, 480, 42,
                 "fuse: sum → Conv1d 1×1 · BatchNorm · LeakyReLU(0.2)")

d.add_edge(edge([(SCX, 703), (SCX, 726)]))
pool = layer_row(SCX - 140, 730, 280, 42, "AvgPool1d k=2 s=2")
d.add_edge(edge([(SCX, 775), (SCX, 784)]))
nxt = text(SCX, 796, "→ next stage", size=11, fill=C["slate"], anchor="middle")

repeat = stack_card(650, 446, 280, "× 4 stages",
                    ["16 → 32 → 64 → 128", "same internals per stage",
                     "sequence length halves", "one skip per stage"])

lgtb_note = multi_note(110, 812, 826, [
    "LGTB block — num_heads = 4 · each sub-layer preceded by LayerNorm",
    "1. Local self-attention — split into a FIXED COUNT of 8 blocks, "
    "attend within each, concat  (+residual)",
    "2. Global self-attention — attend over the whole sequence  (+residual)",
    "3. Feedforward — Linear C→2C · GELU · Linear 2C→C  (+residual)",
])

# encoder → bottleneck
d.add_edge(edge([(SCX, 933), (SCX, 950), (MID, 950), (MID, 967)]))

bottle = card(350, 970, 300, 0, "Bottleneck", ["CNNBlock 128→128"])

# --- Dual decoder ---------------------------------------------------------- #
DEC_Y, DEC_H = 1090, 415
dec_frame = container(90, DEC_Y, 866, DEC_H, "Dual decoder branches")

LCX, RCX = 255, 745          # clean / artifact column centres
LX, RX, CW = 140, 630, 230

# bottleneck fans out into both branches
d.add_edge(edge([(MID, 1054), (MID, 1100), (300, 1100), (300, 1147)]),
           edge([(MID, 1054), (MID, 1100), (700, 1100), (700, 1147)]))

dec_clean = card(LX, 1150, CW, 0, "clean_decoder",
                 ["4 × DecoderStage", "separate weights"])
dec_art = card(RX, 1150, CW, 0, "artifact_decoder",
               ["4 × DecoderStage", "separate weights"])

dec_note = multi_note(388, 1150, 224, [
    "DecoderStage =",
    "Upsample ×2 → interpolate",
    "to skip length → concat skip",
    "→ Conv1d 1×1 reduce",
    "→ CNNBlock",
], size=11)

head_clean = layer_row(LX, 1310, CW, 42, "clean_head  Conv1d 1×1")
head_art = layer_row(RX, 1310, CW, 42, "artifact_head  Conv1d 1×1")
d.add_edge(edge([(LCX, 1257), (LCX, 1306)]),
           edge([(RCX, 1257), (RCX, 1306)]))

y1 = card(LX, 1390, CW, 0, "Y1  (clean)", ["(B, 1, 512)"])
y2 = card(RX, 1390, CW, 0, "Y2  (artifact)", ["(B, 1, 512)"])
d.add_edge(edge([(LCX, 1353), (LCX, 1386)]),
           edge([(RCX, 1353), (RCX, 1386)]))

# --- encoder skips: two symmetric dashed branches off the fuse node --------- #
d.add_edge(skip_edge([(607, 679), (966, 679), (966, 1202), (864, 1202)]),
           skip_edge([(121, 679), (72, 679), (72, 1202), (136, 1202)]))
d.add(jdot(607, 679), jdot(121, 679),
      skip_label_vertical(966, 1000, "skips → decoder"),
      skip_label_vertical(72, 1000, "skips → decoder"))

# --- decoder feature taps feeding the gating heads ------------------------- #
d.add_edge(edge([(330, 1257), (330, 1285), (430, 1285), (430, 1593)],
                label=("clean_feat", 430, 1454)),
           edge([(670, 1257), (670, 1285), (570, 1285), (570, 1593)],
                label=("artifact_feat", 570, 1454)))

# --- Gating + fusion ------------------------------------------------------- #
GAT_Y, GAT_H = 1560, 430
gat_frame = container(90, GAT_Y, 866, GAT_H, "Gating & fusion (Eqs. 4–5)")

concat = layer_row(280, 1596, 440, 42,
                   "concat(clean_feat, artifact_feat) → 2 × base channels")

G1X, G2X = 315, 685
d.add_edge(bar([(MID, 1641), (MID, 1656)]),
           bar([(G1X, 1656), (G2X, 1656)]),
           edge([(G1X, 1656), (G1X, 1667)]),
           edge([(G2X, 1656), (G2X, 1667)]))

GATE_LINES = ["Conv1d k=3 · LeakyReLU(0.2)", "Conv1d 1×1 · Tanh"]
gate1 = card(170, 1670, 290, 0, "gate1  (Eq. 4)", GATE_LINES)
gate2 = card(540, 1670, 290, 0, "gate2  (Eq. 4)", GATE_LINES)

m1 = layer_row(235, 1800, 160, 38, "Ymask1")
m2 = layer_row(605, 1800, 160, 38, "Ymask2")
d.add_edge(edge([(G1X, 1777), (G1X, 1796)]),
           edge([(G2X, 1777), (G2X, 1796)]))

fusion = card(190, 1880, 620, 0, "Ypre = Ymask1 · Y1 + Ymask2 · (Xraw − Y2)",
              ["masks are NOT forced to sum to 1"])
fusion = halo(fusion, C["blue"])
d.add_edge(edge([(G1X, 1838), (G1X, 1870)]),
           edge([(G2X, 1838), (G2X, 1870)]))

# --- Y1 / Y2 into the fusion (and onward to the discriminators) ------------ #
d.add_edge(edge([(LX, 1420), (114, 1420), (114, 1900), (178, 1900)]),
           edge([(RX + CW, 1420), (900, 1420), (900, 1900), (822, 1900)]))
d.add(jdot(114, 1900))

# --- Xraw bypass: input → fusion, down the outer left margin --------------- #
d.add_edge(plain([(340, 164), (42, 164), (42, 1940)]))
d.add_edge(hop_h(42, 178, 1940, hops=(114,)))
d.add(jdot(340, 164), skip_label_vertical(42, 700, "Xraw"))

# --- exported inference path ---------------------------------------------- #
d.add_edge(edge([(RX + CW, 1450), (966, 1450), (966, 2010), (740, 2010),
                 (740, 2027)]))
d.add(jdot(966, 2010))

inf = card(540, 2030, 400, 0, "forward() returns Y2 only",
           ["(B, 1, 512) predicted artifact",
            "TorchScript subtraction contract unchanged"],
           stereotype="exported inference path")
inf_note = multi_note(110, 2040, 370, [
    "Inference vs. training",
    "forward() emits Y2 alone — the artifact.",
    "Y1, Ypre and the three discriminators",
    "exist only inside the training loop.",
])

corr = capsule(540, 2170, 400, 46,
               "DeepLearningCorrection:  corrected = noisy − artifact")
d.add_edge(edge([(740, 2137), (740, 2166)]))

# =========================================================================== #
#  ADVERSARIAL TRAINING — fenced-off region
# =========================================================================== #
ADV_Y, ADV_H = 2290, 590
d.add_edge(backdrop(90, ADV_Y, 866, ADV_H, C["blue400"], 0.075))
adv_frame = container(90, ADV_Y, 866, ADV_H,
                      "Adversarial training (training only)")

# three feeds: Ypre straight down the spine, Y2 down the right lane,
# Y1 out along the far-left lane into D1's side.
d.add_edge(edge([(MID, 1971), (MID, 2357)], label=("Ypre", MID, 2274)))
d.add_edge(edge([(966, 2010), (966, 2250), (770, 2250), (770, 2357)],
                label=("Y2", 966, 2150)))
d.add_edge(edge([(114, 1900), (114, 2000), (64, 2000), (64, 2424),
                 (106, 2424)], label=("Y1", 64, 2200)))

DL = ["FeatureDiscriminator", "own private Adam", "lr 1e-4 · betas 0.9/0.999"]
d1 = card(110, 2360, 240, 0, "D1  ← Y1", ["clean"] + DL)
d3 = card(380, 2360, 240, 0, "D3  ← Ypre", ["fused"] + DL)
d2 = card(650, 2360, 240, 0, "D2  ← Y2", ["artifact / noise"] + DL)

disc_note = multi_note(110, 2545, 826, [
    "Shared FeatureDiscriminator",
    "4 × [Conv1d k=3 s=2 p=1 · BatchNorm · LeakyReLU(0.2)]  →  score Conv1d",
    "no sigmoid (LSGAN) · returns score map + intermediate feature maps",
    "base_channels 16 · depth 4 · own private Adam (betas 0.9/0.999, lr 1e-4) "
    "per discriminator",
])

loss = card(110, 2675, 826, 0, "Losses", [
    "Generator:  Loss = Loss1 + Loss2 + Loss3",
    "each Lossᵢ = MSE + λ_feat · L_feat + λ_adv · L_adv",
    "L_adv = mean((D(G(X)) − 1)²)          (LSGAN, Eq. 12)",
    "L_D  = mean(0.5 · D(G(X))² + 0.5 · (D(Y) − 1)²)          (Eq. 13)",
    "L_feat = MSE between intermediate discriminator features of target "
    "vs prediction  (Eq. 11)",
    "λ_feat = 1.0 · λ_adv = 0.1  (not specified in the paper — documented defaults)",
])

# =========================================================================== #
#  CONFIG
# =========================================================================== #
cfg = card(90, 2920, 866, 0, "Config (Niazy proof-fit)", [
    "base_channels 16 · depth 4 · num_heads 4 · n_local_blocks 8 · n_lgtb 2",
    "AdamW lr 1e-3 · weight decay 1e-4 · grad clip 1.0 · batch 64 · 80 max epochs",
    "Reconstruction loss is MSE (paper Eq. 10), not L1",
])

# =========================================================================== #
#  Assemble
# =========================================================================== #
d.add(enc_frame, dec_frame, gat_frame, adv_frame)           # frames first
d.add(inp, stem, lbl, row_a, path_cnn, path_lgtb, fuse, pool, nxt,
      repeat, lgtb_note, bottle)
d.add(dec_clean, dec_art, dec_note, head_clean, head_art, y1, y2)
d.add(concat, gate1, gate2, m1, m2, fusion)
d.add(inf, corr, inf_note)
d.add(d1, d3, d2, disc_note, loss, cfg)

d.render_png(str(OUT_PNG), svg_path=str(OUT_SVG), width=1280)
print("svg:", OUT_SVG)
print("png:", OUT_PNG)
for nm, g in [("inp", inp), ("stem", stem), ("row_a", row_a),
              ("path_cnn", path_cnn), ("repeat", repeat),
              ("lgtb_note", lgtb_note), ("bottle", bottle),
              ("dec_clean", dec_clean), ("dec_note", dec_note),
              ("y1", y1), ("concat", concat), ("gate1", gate1),
              ("fusion", fusion), ("inf", inf), ("d1", d1),
              ("disc_note", disc_note), ("loss", loss), ("cfg", cfg)]:
    print(f"  {nm:10s} x={g['x']:>4} y={g['y']:>4} w={g['w']:>3} "
          f"h={g['h']:>3} bottom={g['y']+g['h']}")

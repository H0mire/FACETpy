"""Conv-TasNet paper-accurate edition — full-run architecture diagram.

The diagram is intentionally code-first: it shows the exact Niazy proof-fit
configuration, the 16-block sequential residual TCN, all-block skip summation,
the shared encoder-latent mask bypass, and one decoder module reused twice.

Verified against
    src/facet/models/conv_tasnet_paper_accurate_edition/training.py
    src/facet/models/conv_tasnet_paper_accurate_edition/
        training_niazy_proof_fit.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "skills/facetpy-diagram/assets"))

from facetpy_svg import (  # noqa: E402
    C,
    Diagram,
    _geom,
    capsule,
    card,
    container,
    edge,
    eeg_wave,
    pill,
    text,
    text_width,
)

OUT_SVG = REPO / "docs/source/_static/diagrams/conv_tasnet_pa_architecture.svg"
OUT_PNG = REPO / "docs/source/_static/diagrams/conv_tasnet_pa_architecture.png"

H = 3000
d = Diagram(H)

TITLE = "Conv-TasNet — Paper-Accurate Edition"
SUB = "Niazy proof-fit defaults · linear encoder · ordered clean/artifact sources"
d.add(
    text(70, 52, TITLE, size=24, weight=700, fill=C["ink"]),
    f'<rect x="70" y="62" width="{22 + len(TITLE) * 9}" height="3" '
    f'rx="1.5" fill="url(#fp-header)"/>',
    eeg_wave(70 + text_width(TITLE, 24) + 34, 44, 90),
    text(70, 84, SUB, size=13.5, fill=C["slate"]),
)

d.add_defs(
    f'<marker id="fp-skip" markerWidth="12" markerHeight="12" refX="9" '
    f'refY="5" orient="auto"><path d="M1 1 L10 5 L1 9 Z" '
    f'fill="{C["blue"]}"/></marker>',
)


def skip_edge(points, *, marker=True, soft=False, width=2.1):
    """Dashed accent path for latent bypasses, residuals, and skip sums."""
    pstr = " ".join(f"{px},{py}" for px, py in points)
    color = C["blue400"] if soft else C["blue"]
    marker_attr = ' marker-end="url(#fp-skip)"' if marker else ""
    return (
        f'<polyline points="{pstr}" fill="none" stroke="{color}" '
        f'stroke-opacity="0.92" stroke-width="{width}" '
        f'stroke-dasharray="8 5" stroke-linejoin="round"{marker_attr}/>'
    )


def layer_row(x, y, w, h, label, *, accent=None, size=12.3):
    """Compact FACETpy process row with a controllable label size."""
    accent = accent or C["blue"]
    svg = (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
        f'fill="{C["tint"]}" stroke="{accent}" stroke-opacity="0.58" '
        f'stroke-width="1.4" filter="url(#fp-shadow)"/>'
        f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" '
        f'fill="{accent}"/>'
        + text(
            x + w / 2 + 2,
            y + h / 2 + 4.5,
            label,
            size=size,
            fill=C["ink"],
            weight=500,
            anchor="middle",
        )
    )
    return _geom(svg, x, y, w, h)


def multi_row(x, y, w, h, rows, *, accent=None):
    """Compact process row with two or more centred lines."""
    accent = accent or C["blue"]
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
        f'fill="{C["tint"]}" stroke="{accent}" stroke-opacity="0.58" '
        f'stroke-width="1.4" filter="url(#fp-shadow)"/>',
        f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" '
        f'fill="{accent}"/>',
    ]
    line_y = y + h / 2 - (len(rows) - 1) * 9 + 4
    for label, size, fill, weight in rows:
        parts.append(
            text(
                x + w / 2 + 2,
                line_y,
                label,
                size=size,
                fill=fill,
                weight=weight,
                anchor="middle",
            )
        )
        line_y += 18
    return _geom("".join(parts), x, y, w, h)


def mini_chip(cx, cy, label, *, size=11, surface=True):
    """Small labelled chip used for exact shapes and reuse annotations."""
    chip_w = text_width(label, size) + 22
    fill = C["surface"] if surface else C["tint"]
    return (
        f'<rect x="{cx-chip_w/2:.1f}" y="{cy-11}" width="{chip_w:.1f}" '
        f'height="22" rx="11" fill="{fill}" stroke="{C["blue"]}" '
        f'stroke-opacity="0.38"/>'
        + text(
            cx,
            cy + 4,
            label,
            size=size,
            fill=C["blue"],
            weight=600,
            anchor="middle",
        )
    )


def block_chip(x, y, number, dilation):
    """One compact schedule chip; arrows between chips are the residual stream."""
    w, h = 82, 44
    svg = (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'fill="{C["surface"]}" stroke="{C["blue"]}" '
        f'stroke-opacity="0.55" stroke-width="1.4" '
        f'filter="url(#fp-shadow)"/>'
        + text(
            x + w / 2,
            y + 18,
            f"d={dilation}",
            size=12.5,
            fill=C["ink"],
            weight=700,
            anchor="middle",
        )
        + text(
            x + w / 2,
            y + 35,
            f"block {number}",
            size=9.7,
            fill=C["slate"],
            anchor="middle",
        )
    )
    return _geom(svg, x, y, w, h)


def operator_node(cx, cy, glyph):
    """Small circled arithmetic operator in the brand palette."""
    return (
        f'<circle cx="{cx}" cy="{cy}" r="15" fill="{C["surface"]}" '
        f'stroke="{C["blue"]}" stroke-width="2" '
        f'filter="url(#fp-shadow)"/>'
        + text(
            cx,
            cy + 5,
            glyph,
            size=19,
            fill=C["blue"],
            weight=700,
            anchor="middle",
        )
    )


def product_node(x, y, w, main, sub):
    """Gradient elementwise-product node for a separated source branch."""
    h = 72
    cx, cy = x + w / 2, y + h / 2
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
        f'fill="url(#fp-header)" filter="url(#fp-shadow)"/>',
        f'<circle cx="{x+34}" cy="{cy}" r="11" fill="none" '
        f'stroke="{C["blue200"]}" stroke-width="1.8"/>',
        text(
            x + 34,
            cy + 5,
            "×",
            size=17,
            fill=C["blue200"],
            weight=700,
            anchor="middle",
        ),
        text(
            cx + 18,
            cy - 4,
            main,
            size=14.5,
            fill=C["header_fg"],
            weight=700,
            anchor="middle",
        ),
        text(
            cx + 18,
            cy + 16,
            sub,
            size=11,
            fill=C["blue200"],
            weight=500,
            anchor="middle",
        ),
    ]
    return _geom("".join(parts), x, y, w, h)


def vertical_chip(x, y, label):
    """Rotated label for a long dashed bypass lane."""
    chip_w = text_width(label, 10.7) + 20
    return (
        f'<g transform="rotate(-90 {x} {y})">'
        f'<rect x="{x-chip_w/2:.1f}" y="{y-10}" width="{chip_w:.1f}" '
        f'height="20" rx="6" fill="{C["surface"]}" '
        f'stroke="{C["blue"]}" stroke-opacity="0.7"/>'
        + text(
            x,
            y + 4,
            label,
            size=10.7,
            fill=C["blue"],
            weight=600,
            anchor="middle",
        )
        + "</g>"
    )


def multi_note(x, y, w, lines, *, size=11.4):
    """Folded-corner code-accuracy note with multiple readable lines."""
    fold = 14
    h = 24 + len(lines) * 18
    parts = [
        f'<path d="M{x} {y} L{x+w-fold} {y} L{x+w} {y+fold} '
        f'L{x+w} {y+h} L{x} {y+h} Z" fill="{C["tint"]}" '
        f'stroke="{C["slate"]}" stroke-opacity="0.45" stroke-width="1"/>',
        f'<path d="M{x+w-fold} {y} L{x+w-fold} {y+fold} '
        f'L{x+w} {y+fold}" fill="none" stroke="{C["slate"]}" '
        f'stroke-opacity="0.45"/>',
    ]
    line_y = y + 22
    for index, label in enumerate(lines):
        parts.append(
            text(
                x + 14,
                line_y,
                label,
                size=size + (0.4 if index == 0 else 0),
                fill=C["blue"] if index == 0 else C["slate"],
                weight=700 if index == 0 else None,
            )
        )
        line_y += 18
    return _geom("".join(parts), x, y, w, h)


MID = 500

# Entry and encoder --------------------------------------------------------- #
inp = card(
    330,
    108,
    340,
    81,
    "Input epoch",
    ["channel-wise demeaned noisy centre", "(B, 1, 512)"],
)
encoder = card(
    250,
    255,
    500,
    0,
    "Linear learned encoder",
    [
        "Conv1d 1→N=256 · k=L=16 · stride=8=L/2",
        "bias=False · linear learned analysis transform",
        "paper-best default: no ReLU after the encoder",
        "shared latent w: (B, 256, 63)",
    ],
)
prelude = card(
    270,
    472,
    460,
    0,
    "Separator prelude",
    [
        "global layer norm over channels + time",
        "bottleneck Conv1d 1×1 · N=256→B=128",
        "residual stream h₀: (B, 128, 63)",
    ],
)

latent_junction_y = 433
d.add_edge(
    edge([(MID, inp["y"] + inp["h"] + 3), (MID, encoder["y"] - 4)]),
    edge(
        [(MID, encoder["y"] + encoder["h"] + 3), (MID, prelude["y"] - 4)]
    ),
)

# The exact same latent w continues into the separator and bypasses it to both
# mask products. The two far-side lanes make the fan-out unambiguous.
d.add_edge(
    skip_edge(
        [(MID - 4, latent_junction_y), (22, latent_junction_y), (22, 2246), (101, 2246)]
    ),
    skip_edge(
        [(MID + 4, latent_junction_y), (978, latent_junction_y), (978, 2246), (899, 2246)]
    ),
)

# TCN schedule -------------------------------------------------------------- #
TCN_Y, TCN_H = 640, 1150
tcn_frame = container(40, TCN_Y, 920, TCN_H, "TCN separator · 2 repeats × 8 blocks")

dilations = (1, 2, 4, 8, 16, 32, 64, 128)
chip_x = [145 + index * 96 for index in range(8)]
repeat_1 = [block_chip(x, 700, index + 1, dilations[index]) for index, x in enumerate(chip_x)]
repeat_2 = [block_chip(x, 800, index + 9, dilations[index]) for index, x in enumerate(chip_x)]

# Main residual stream: all 16 blocks execute sequentially. The long bridge is
# the repeat boundary, not a parallel fork.
d.add_edge(
    edge(
        [
            (MID, prelude["y"] + prelude["h"] + 3),
            (MID, 678),
            (repeat_1[0]["cx"], 678),
            (repeat_1[0]["cx"], repeat_1[0]["y"] - 4),
        ]
    )
)
for row in (repeat_1, repeat_2):
    for left, right in zip(row, row[1:]):
        d.add_edge(
            edge(
                [
                    (left["x"] + left["w"] + 3, left["cy"]),
                    (right["x"] - 4, right["cy"]),
                ]
            )
        )
d.add_edge(
    edge(
        [
            (repeat_1[-1]["x"] + repeat_1[-1]["w"] + 3, repeat_1[-1]["cy"]),
            (930, repeat_1[-1]["cy"]),
            (930, 780),
            (repeat_2[0]["cx"], 780),
            (repeat_2[0]["cx"], repeat_2[0]["y"] - 4),
        ]
    )
)

# Every block contributes one Sc-wide skip tensor. Two local rails (eight
# contributions each) feed the one all-block sum below the representative inset.
for chip in repeat_1:
    d.add_edge(
        skip_edge([(chip["cx"], chip["y"] + chip["h"] + 2), (chip["cx"], 760)], marker=False, soft=True, width=1.6)
    )
for chip in repeat_2:
    d.add_edge(
        skip_edge([(chip["cx"], chip["y"] + chip["h"] + 2), (chip["cx"], 860)], marker=False, soft=True, width=1.6)
    )
d.add_edge(
    skip_edge([(145, 760), (899, 760)], marker=False, soft=True, width=1.8),
    skip_edge([(145, 860), (899, 860)], marker=False, soft=True, width=1.8),
    skip_edge([(145, 760), (67, 760), (67, 1692), (326, 1692)]),
    skip_edge([(899, 860), (933, 860), (933, 1692), (674, 1692)]),
)

# Representative TemporalBlock anatomy ------------------------------------ #
rep = card(
    85,
    915,
    830,
    655,
    "Expanded TemporalBlock(d) — representative anatomy ×16",
)
tb_input = pill(345, 970, 310, 38, "block input hᵢ  ·  (B, 128, 63)")
expand = layer_row(345, 1025, 310, 38, "Conv1d 1×1 expand · B128→H256")
act_1 = layer_row(390, 1075, 220, 38, "PReLU")
norm_1 = layer_row(390, 1125, 220, 38, "global layer norm")
dconv = multi_row(
    345,
    1175,
    310,
    54,
    [
        ("depthwise Conv1d H256→H256", 11.7, C["ink"], 600),
        ("k=3 · dilation=d · same pad · groups=H", 10.4, C["slate"], None),
    ],
    accent=C["blue400"],
)
act_2 = layer_row(390, 1245, 220, 38, "PReLU")
norm_2 = layer_row(390, 1295, 220, 38, "global layer norm")
res_head = layer_row(
    135,
    1350,
    320,
    48,
    "residual head · Conv1d 1×1 H256→B128",
    size=11.1,
)
skip_head = layer_row(
    545,
    1350,
    320,
    48,
    "skip head · Conv1d 1×1 H256→Sc128",
    size=11.2,
)
res_out = pill(145, 1470, 300, 38, "hᵢ₊₁  ·  residual continues")
skip_out = pill(555, 1470, 300, 38, "sᵢ  ·  (B, Sc=128, 63)", accent=C["blue400"])

rep_connectors = (
    edge([(tb_input["cx"], tb_input["y"] + tb_input["h"] + 3), (expand["cx"], expand["y"] - 4)]),
    edge([(expand["cx"], expand["y"] + expand["h"] + 3), (act_1["cx"], act_1["y"] - 4)]),
    edge([(act_1["cx"], act_1["y"] + act_1["h"] + 3), (norm_1["cx"], norm_1["y"] - 4)]),
    edge([(norm_1["cx"], norm_1["y"] + norm_1["h"] + 3), (dconv["cx"], dconv["y"] - 4)]),
    edge([(dconv["cx"], dconv["y"] + dconv["h"] + 3), (act_2["cx"], act_2["y"] - 4)]),
    edge([(act_2["cx"], act_2["y"] + act_2["h"] + 3), (norm_2["cx"], norm_2["y"] - 4)]),
    edge(
        [
            (norm_2["cx"], norm_2["y"] + norm_2["h"] + 3),
            (norm_2["cx"], 1340),
            (res_head["cx"], 1340),
            (res_head["cx"], res_head["y"] - 4),
        ]
    ),
    edge(
        [
            (norm_2["cx"], norm_2["y"] + norm_2["h"] + 3),
            (norm_2["cx"], 1340),
            (skip_head["cx"], 1340),
            (skip_head["cx"], skip_head["y"] - 4),
        ]
    ),
    edge([(res_head["cx"], res_head["y"] + res_head["h"] + 3), (res_head["cx"], 1412)]),
    edge([(295, 1448), (295, res_out["y"] - 4)]),
    edge([(skip_head["cx"], skip_head["y"] + skip_head["h"] + 3), (skip_out["cx"], skip_out["y"] - 4)]),
    skip_edge(
        [
            (tb_input["x"] - 4, tb_input["cy"]),
            (112, tb_input["cy"]),
            (112, 1430),
            (277, 1430),
        ],
        soft=True,
    ),
)

skip_sum = card(
    330,
    1640,
    340,
    81,
    "skip_sum = Σ sᵢ",
    ["all 16 block skip heads", "(B, Sc=128, 63)"],
)

# Mask head and source branches -------------------------------------------- #
mask_head = card(
    250,
    1835,
    500,
    0,
    "Mask head",
    [
        "PReLU",
        "Conv1d 1×1 · Sc128→C×N=2×256=512",
        "Sigmoid",
        "reshape masks → (B, 2, 256, 63)",
    ],
)
mask_clean = card(
    105,
    2040,
    320,
    81,
    "mask 0 · clean EEG",
    ["m_clean", "(B, 256, 63)"],
)
mask_artifact = card(
    575,
    2040,
    320,
    81,
    "mask 1 · gradient artifact",
    ["m_artifact", "(B, 256, 63)"],
)
product_clean = product_node(
    105, 2210, 320, "w × m_clean", "elementwise · (B, 256, 63)"
)
product_artifact = product_node(
    575, 2210, 320, "w × m_artifact", "elementwise · (B, 256, 63)"
)

d.add_edge(
    edge([(skip_sum["cx"], skip_sum["y"] + skip_sum["h"] + 3), (mask_head["cx"], mask_head["y"] - 4)]),
    edge(
        [
            (mask_head["cx"], mask_head["y"] + mask_head["h"] + 3),
            (mask_head["cx"], 2014),
            (mask_clean["cx"], 2014),
            (mask_clean["cx"], mask_clean["y"] - 4),
        ]
    ),
    edge(
        [
            (mask_head["cx"], mask_head["y"] + mask_head["h"] + 3),
            (mask_head["cx"], 2014),
            (mask_artifact["cx"], 2014),
            (mask_artifact["cx"], mask_artifact["y"] - 4),
        ]
    ),
    edge([(mask_clean["cx"], mask_clean["y"] + mask_clean["h"] + 3), (product_clean["cx"], product_clean["y"] - 4)]),
    edge([(mask_artifact["cx"], mask_artifact["y"] + mask_artifact["h"] + 3), (product_artifact["cx"], product_artifact["y"] - 4)]),
)

# One decoder object, two independent calls -------------------------------- #
decoder = card(
    180,
    2360,
    640,
    150,
    "ONE shared decoder — reused for both sources",
    [
        "ConvTranspose1d N=256→1 · k=L=16 · stride=8 · bias=False",
        "same parameter instance called independently for source 0 and source 1",
        "crop/slice decoded[..., :n_samples] → length 512",
    ],
)
out_clean = card(
    105,
    2585,
    320,
    81,
    "source 0 · clean EEG",
    ["decoder(w × m_clean)[..., :512]", "(B, 1, 512)"],
)
out_artifact = card(
    575,
    2585,
    320,
    81,
    "source 1 · gradient artifact",
    ["decoder(w × m_artifact)[..., :512]", "(B, 1, 512)"],
)
ordered_output = capsule(275, 2765, 450, 50, "ordered output  [clean, artifact]  ·  (B, 2, 512)")

d.add_edge(
    edge(
        [
            (product_clean["cx"], product_clean["y"] + product_clean["h"] + 3),
            (product_clean["cx"], 2330),
            (310, 2330),
            (310, decoder["y"] - 4),
        ]
    ),
    edge(
        [
            (product_artifact["cx"], product_artifact["y"] + product_artifact["h"] + 3),
            (product_artifact["cx"], 2330),
            (690, 2330),
            (690, decoder["y"] - 4),
        ]
    ),
    edge([(310, decoder["y"] + decoder["h"] + 3), (310, 2555), (out_clean["cx"], 2555), (out_clean["cx"], out_clean["y"] - 4)]),
    edge([(690, decoder["y"] + decoder["h"] + 3), (690, 2555), (out_artifact["cx"], 2555), (out_artifact["cx"], out_artifact["y"] - 4)]),
    edge([(out_clean["cx"], out_clean["y"] + out_clean["h"] + 3), (out_clean["cx"], 2735), (390, 2735), (390, ordered_output["y"] - 4)]),
    edge([(out_artifact["cx"], out_artifact["y"] + out_artifact["h"] + 3), (out_artifact["cx"], 2735), (610, 2735), (610, ordered_output["y"] - 4)]),
)

training_note = multi_note(
    105,
    2850,
    790,
    [
        "Training note — outside the forward graph",
        "Default loss is ordered-source MSE: clean index 0, artifact index 1; no PIT/permutation branch.",
        "Optional consistency_mse is a beyond-paper source-additivity regularizer.",
    ],
)

# Node layer --------------------------------------------------------------- #
d.add(tcn_frame)
d.add(
    text(60, 680, "sequential residual stream h", size=11.7, fill=C["slate"], italic=True),
    text(60, 726, "repeat 1", size=11.5, fill=C["blue"], weight=700),
    text(60, 826, "repeat 2", size=11.5, fill=C["blue"], weight=700),
    mini_chip(780, 665, "d schedule per repeat: 1, 2, 4, 8, 16, 32, 64, 128"),
)
d.add(*repeat_1, *repeat_2)
d.add(
    text(
        MID,
        893,
        "solid arrows: residual propagation · dashed rails: every block emits a skip tensor",
        size=11,
        fill=C["slate"],
        anchor="middle",
    )
)
d.add(rep)
d.add(*rep_connectors)
d.add(
    tb_input,
    expand,
    act_1,
    norm_1,
    dconv,
    act_2,
    norm_2,
    res_head,
    skip_head,
    operator_node(295, 1430, "+"),
    res_out,
    skip_out,
    vertical_chip(112, 1220, "block-input residual bypass"),
    text(
        MID,
        1545,
        "Per code, the depthwise layer's pointwise mixing is folded into the residual and skip 1×1 heads.",
        size=10.5,
        fill=C["slate"],
        anchor="middle",
    ),
)
d.add(
    vertical_chip(67, 1235, "8 skip outputs · repeat 1"),
    vertical_chip(933, 1235, "8 skip outputs · repeat 2"),
    skip_sum,
)

d.add(inp, encoder, prelude)
d.add(
    f'<circle cx="{MID}" cy="{latent_junction_y}" r="4.8" fill="{C["blue"]}"/>',
    text(
        518,
        425,
        "shared latent w · (B, 256, 63)",
        size=11.3,
        fill=C["blue"],
        weight=600,
    ),
    vertical_chip(22, 1325, "shared w → clean mask product"),
    vertical_chip(978, 1325, "shared w → artifact mask product"),
)

d.add(mask_head, mask_clean, mask_artifact, product_clean, product_artifact)
d.add(
    decoder,
    mini_chip(310, 2488, "call 0 · same weights"),
    mini_chip(690, 2488, "call 1 · same weights"),
    out_clean,
    out_artifact,
    ordered_output,
    training_note,
)

d.render_png(str(OUT_PNG), svg_path=str(OUT_SVG), width=1280)
print("svg:", OUT_SVG)
print("png:", OUT_PNG)
print("canvas:", 1000, "×", H)

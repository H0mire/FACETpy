"""Build the paper-accurate ST-GNN architecture diagram.

The figure is deliberately source-backed: it depicts the full seven-epoch,
30-electrode forward pass, the persistent scaled-Laplacian buffer, both
STConvBlocks, and the centre-epoch artifact returned to FACETpy.
"""
from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(
    0, str(REPO / ".claude/skills/facetpy-diagram/assets")
)

from facetpy_svg import (  # noqa: E402
    C,
    Diagram,
    _geom,
    card,
    card_shell,
    container,
    divider,
    edge,
    eeg_wave,
    node_dot,
    pill,
    rounded_top,
    text,
    text_width,
)


OUT_DIR = REPO / "docs/source/_static/diagrams"
OUT_SVG = OUT_DIR / "st_gnn_pa_architecture.svg"
OUT_PNG = OUT_DIR / "st_gnn_pa_architecture.png"

CANVAS_H = 1340
d = Diagram(CANVAS_H, background=False)


def multiline_text(
    x: float,
    y: float,
    lines: list[str],
    *,
    size: float = 11,
    fill: str | None = None,
    weight: int | None = None,
    anchor: str = "start",
    leading: float = 16,
) -> str:
    """Token-coloured compact text with deterministic line spacing."""
    return "".join(
        text(
            x,
            y + index * leading,
            line,
            size=size,
            fill=fill or C["slate"],
            weight=weight,
            anchor=anchor,
        )
        for index, line in enumerate(lines)
    )


def mini_chip(
    cx: float,
    cy: float,
    label: str,
    *,
    size: float = 10.4,
    accent: str | None = None,
    fill: str | None = None,
    pad: float = 18,
) -> str:
    """Small shape/config chip sized by the shared toolkit estimator."""
    accent = accent or C["blue"]
    chip_w = text_width(label, size) + pad
    return (
        f'<rect x="{cx-chip_w/2:.1f}" y="{cy-10.5}" width="{chip_w:.1f}" '
        f'height="21" rx="10.5" fill="{fill or C["surface"]}" '
        f'stroke="{accent}" stroke-opacity="0.42"/>'
        + text(
            cx,
            cy + 3.8,
            label,
            size=size,
            fill=accent,
            weight=600,
            anchor="middle",
        )
    )


def stage_row(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    *,
    accent: str | None = None,
) -> dict:
    """A branded compact stage with a strong title and explanatory rows."""
    accent = accent or C["blue"]
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'fill="{C["tint"]}" stroke="{accent}" stroke-opacity="0.52" '
        f'stroke-width="1.35" filter="url(#fp-shadow)"/>',
        f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" '
        f'fill="{accent}"/>',
        text(x + 12, y + 20, title, size=11.5, fill=C["ink"], weight=700),
    ]
    parts.append(
        multiline_text(
            x + 12,
            y + 38,
            lines,
            size=9.65,
            fill=C["slate"],
            leading=14,
        )
    )
    return _geom("".join(parts), x, y, w, h)


def block_shell(x: float, y: float, w: float, h: float, title_label: str) -> dict:
    """A standard FACETpy titled card used as an expanded STConvBlock."""
    parts = [
        card_shell(x, y, w, h),
        f'<path d="{rounded_top(x, y, w, 36, 12)}" fill="url(#fp-header)"/>',
        node_dot(x + 18, y + 18),
        text(
            x + 34,
            y + 23,
            title_label,
            size=14.1,
            fill=C["header_fg"],
            weight=600,
        ),
        divider(x, y + 36, w),
    ]
    return _geom("".join(parts), x, y, w, h)


def residual_edge(points: list[tuple[float, float]]) -> str:
    """Dashed accent residual branch, always kept inside its block card."""
    pstr = " ".join(f"{px},{py}" for px, py in points)
    return (
        f'<polyline points="{pstr}" fill="none" stroke="{C["blue"]}" '
        f'stroke-opacity="0.92" stroke-width="1.9" stroke-dasharray="7 5" '
        f'stroke-linejoin="round" marker-end="url(#fp-residual)"/>'
    )


def vertical_label(x: float, y: float, label: str) -> str:
    chip_w = text_width(label, 9.1) + 16
    return (
        f'<g transform="rotate(-90 {x} {y})">'
        f'<rect x="{x-chip_w/2:.1f}" y="{y-8.5}" width="{chip_w:.1f}" '
        f'height="17" rx="5" fill="{C["surface"]}" '
        f'stroke="{C["blue"]}" stroke-opacity="0.5"/>'
        + text(
            x,
            y + 3.2,
            label,
            size=9.1,
            fill=C["blue"],
            weight=600,
            anchor="middle",
        )
        + "</g>"
    )


def epoch_strip(x: float, y: float, slot_w: float = 20, gap: float = 3) -> str:
    """Seven trigger-aligned epochs with index 3/fourth highlighted."""
    parts: list[str] = []
    for index in range(7):
        sx = x + index * (slot_w + gap)
        is_center = index == 3
        fill = "url(#fp-header)" if is_center else C["tint"]
        stroke = C["blue"] if is_center else C["slate"]
        label_fill = C["header_fg"] if is_center else C["slate"]
        parts.extend(
            [
                f'<rect x="{sx}" y="{y}" width="{slot_w}" height="31" rx="6" '
                f'fill="{fill}" stroke="{stroke}" stroke-opacity="0.55"/>',
                text(
                    sx + slot_w / 2,
                    y + 20,
                    str(index),
                    size=9.5,
                    fill=label_fill,
                    weight=700 if is_center else 500,
                    anchor="middle",
                ),
            ]
        )
    center_x = x + 3 * (slot_w + gap) + slot_w / 2
    parts.append(
        text(
            center_x,
            y + 46,
            "CENTER · fourth",
            size=9.4,
            fill=C["blue"],
            weight=700,
            anchor="middle",
        )
    )
    return "".join(parts)


def scalp_graph(cx: float, cy: float, radius: float = 47) -> str:
    """Small unit-sphere/electrode graph motif using only brand tokens."""
    # A representative subset of the 30 nodes keeps the motif readable.
    nodes = [
        (-22, -32), (0, -39), (22, -32), (-35, -15), (-12, -14),
        (12, -14), (35, -15), (-38, 10), (-18, 8), (0, 5), (18, 8),
        (38, 10), (-27, 29), (0, 35), (27, 29),
    ]
    links = [
        (0, 1), (1, 2), (0, 4), (1, 4), (1, 5), (2, 5),
        (3, 4), (4, 5), (5, 6), (3, 7), (4, 8), (4, 9),
        (5, 9), (5, 10), (6, 11), (7, 8), (8, 9), (9, 10),
        (10, 11), (8, 12), (9, 13), (10, 14), (12, 13), (13, 14),
    ]
    parts = [
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{C["tint"]}" '
        f'stroke="{C["blue"]}" stroke-opacity="0.5" stroke-width="1.4"/>',
        f'<path d="M{cx-radius-2} {cy-8} q-8 8 0 16 M{cx+radius+2} {cy-8} '
        f'q8 8 0 16 M{cx-7} {cy-radius} q7 -9 14 0" fill="none" '
        f'stroke="{C["blue"]}" stroke-opacity="0.48" stroke-width="1.3"/>',
    ]
    for a, b in links:
        ax, ay = nodes[a]
        bx, by = nodes[b]
        parts.append(
            f'<line x1="{cx+ax}" y1="{cy+ay}" x2="{cx+bx}" y2="{cy+by}" '
            f'stroke="{C["blue400"]}" stroke-opacity="0.58" stroke-width="1"/>'
        )
    for nx, ny in nodes:
        parts.append(
            f'<circle cx="{cx+nx}" cy="{cy+ny}" r="3.4" '
            f'fill="{C["blue200"]}" stroke="{C["surface"]}" stroke-width="1"/>'
        )
    return "".join(parts)


def subtraction_tail(x: float, y: float, w: float) -> str:
    """Compact outside-model adapter and FACETpy correction tail."""
    cx = x + w / 2
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="137" rx="10" '
        f'fill="{C["tint"]}" stroke="{C["blue"]}" stroke-opacity="0.45"/>',
        text(cx, y + 19, "Outside model", size=10.2, fill=C["blue"], weight=700,
             anchor="middle"),
        multiline_text(
            cx,
            y + 36,
            [
                "each channel:",
                "512 → native centre_len",
                "adapter returns",
                "artifact estimate",
            ],
            size=8.85,
            fill=C["slate"],
            anchor="middle",
            leading=13,
        ),
        f'<line x1="{x+12}" y1="{y+84}" x2="{x+w-12}" y2="{y+84}" '
        f'stroke="{C["ink"]}" stroke-opacity="0.12"/>',
        text(cx, y + 102, "DeepLearningCorrection", size=9.35,
             fill=C["ink"], weight=700, anchor="middle"),
        text(cx, y + 121, "recorded EEG − artifact", size=9.2,
             fill=C["blue"], weight=600, anchor="middle"),
        text(cx, y + 134, "→ corrected EEG", size=9.2,
             fill=C["blue"], weight=600, anchor="middle"),
    ]
    return "".join(parts)


d.add_defs(
    f'<marker id="fp-residual" markerWidth="11" markerHeight="11" '
    f'refX="9" refY="5" orient="auto"><path d="M1 1 L9 5 L1 9 Z" '
    f'fill="{C["blue"]}"/></marker>',
    f'<marker id="fp-buffer-arrow" markerWidth="11" markerHeight="11" '
    f'refX="9" refY="5" orient="auto"><path d="M1 1 L9 5 L1 9" '
    f'fill="none" stroke="{C["blue"]}" stroke-width="1.6"/></marker>',
)


# Title block. A two-line treatment prevents the long subtitle from colliding
# with the signature wave while preserving the fixed-width design contract.
TITLE = "ST-GNN — Paper-Accurate Edition"
SUBTITLE = (
    "7-epoch multichannel EEG · temporal GLUs · Chebyshev electrode mixing · "
    "centre artifact"
)
d.add(
    text(35, 46, TITLE, size=24, fill=C["ink"], weight=700),
    f'<rect x="35" y="57" width="{22 + len(TITLE) * 9}" height="3" '
    f'rx="1.5" fill="url(#fp-header)"/>',
    text(35, 79, SUBTITLE, size=13.2, fill=C["slate"]),
    eeg_wave(846, 43, 105),
)


# -------------------------------------------------------------------------
# LEFT: dataset/adapter context and exact in-forward preparation.
# -------------------------------------------------------------------------
input_card = card(25, 115, 190, 695, "Context + preparation")
d.add(input_card)
d.add(
    text(40, 171, "OUTSIDE MODEL · adapter / dataset", size=8.65,
         fill=C["blue"], weight=700),
    multiline_text(
        40,
        195,
        ["Trigger-aligned full graph context", "7 epochs × all 30 EEG electrodes"],
        size=10.5,
        fill=C["ink"],
        weight=600,
        leading=17,
    ),
    epoch_strip(39, 232),
    mini_chip(120, 305, "input [B,7,30,512]", size=9.7),
    scalp_graph(73, 375, 35),
    multiline_text(
        119,
        352,
        ["30 electrode", "graph nodes", "all channels"],
        size=9.8,
        fill=C["slate"],
        leading=15,
    ),
    multiline_text(
        120,
        432,
        ["full multichannel context", "not channel-wise inference"],
        size=9.9,
        fill=C["blue"],
        weight=700,
        anchor="middle",
        leading=15,
    ),
    divider(39, 470, 162),
    text(40, 496, "MODEL.FORWARD · input preparation", size=8.7,
         fill=C["blue"], weight=700),
    multiline_text(
        40,
        523,
        [
            "For every electrode:",
            "concatenate 7 epochs in time",
            "reshape / permute",
        ],
        size=10.4,
        fill=C["ink"],
        weight=600,
        leading=19,
    ),
    mini_chip(120, 603, "[B,1,30,3584]", size=10.5, fill=C["tint"]),
    multiline_text(
        120,
        635,
        [
            "1 feature plane",
            "30 graph nodes",
            "full 7×512 time axis",
        ],
        size=10.1,
        fill=C["slate"],
        anchor="middle",
        leading=17,
    ),
    mini_chip(120, 714, "prepared tensor", size=9.5),
)
d.add(
    edge([(120, 459), (120, 477)], marker_end="arrow"),
    edge([(120, 683), (120, 701)], marker_end="arrow"),
)


# -------------------------------------------------------------------------
# CENTER: two expanded, sequential STConvBlocks.
# -------------------------------------------------------------------------
stack = container(230, 115, 570, 715, "Two sequential STConvBlocks")
b1 = block_shell(245, 158, 263, 637, "Block 1 · feature width 1→64")
b2 = block_shell(522, 158, 263, 637, "Block 2 · feature width 64→64")
d.add(stack, b1, b2)


def populate_block(x: float, *, block_number: int) -> dict[str, dict]:
    stage_x, stage_w = x + 15, 205
    if block_number == 1:
        input_shape = "input [B,1,30,3584]"
        t1_title = "Temporal GLU · 1→64"
        residual = "1×1 Conv · 1→64"
    else:
        input_shape = "input [B,64,30,3584]"
        t1_title = "Temporal GLU · 64→64"
        residual = "identity residual"

    stages = {
        "t1": stage_row(
            stage_x,
            236,
            stage_w,
            82,
            t1_title,
            [
                "K=3 · symmetric · causal=false",
                "P × sigmoid(Q) · time / electrode",
                "→ [B,64,30,3584]",
            ],
        ),
        "cheb": stage_row(
            stage_x,
            333,
            stage_w,
            96,
            "ChebConv K=3 · 64→16 + ReLU",
            [
                "uses l_tilde · mixes 30 nodes",
                "at every time step · spatial",
                "→ [B,16,30,3584]",
            ],
            accent=C["blue400"],
        ),
        "drop": stage_row(
            stage_x,
            444,
            stage_w,
            58,
            "Dropout · p=0.1",
            ["after ReLU(ChebConv)", "shape [B,16,30,3584]"],
            accent=C["blue400"],
        ),
        "t2": stage_row(
            stage_x,
            517,
            stage_w,
            82,
            "Temporal GLU · 16→64",
            [
                "K=3 · symmetric length-preserving",
                "time modelling per electrode",
                "→ [B,64,30,3584]",
            ],
        ),
        "sum": stage_row(
            stage_x,
            618,
            stage_w,
            76,
            "⊕ sum → Channel LayerNorm",
            ["feature axis · per node / time", "no dropout after LayerNorm"],
        ),
    }
    d.add(
        mini_chip(
            stage_x + stage_w / 2,
            216,
            input_shape,
            size=9.55,
            fill=C["tint"],
        ),
        *stages.values(),
        mini_chip(
            stage_x + stage_w / 2,
            737,
            "output [B,64,30,3584]",
            size=9.5,
            fill=C["tint"],
        ),
    )
    # Main path: strictly solid; residual branch: dashed and internal.
    cx = stage_x + stage_w / 2
    internal_flow = []
    for y1, y2 in [(227, 232), (321, 329), (432, 440), (505, 513), (602, 614),
                   (697, 724)]:
        internal_flow.append(edge([(cx, y1), (cx, y2)]))
    # These paths belong on the node layer: an edge-layer path would be hidden
    # by the expanded block's opaque card shell.
    d.add(
        *internal_flow,
        residual_edge(
            [
                (stage_x + stage_w + 3, 216),
                (x + 246, 216),
                (x + 246, 656),
                (stage_x + stage_w + 3, 656),
            ]
        ),
        vertical_label(x + 245, 438, residual),
    )
    return stages


b1_stages = populate_block(245, block_number=1)
b2_stages = populate_block(522, block_number=2)

# Macro flow remains left→right while anatomy is expanded top→bottom.
d.add_edge(
    edge([(218, 192), (241, 192)]),
    edge([(511, 192), (519, 192)]),
    edge([(788, 192), (807, 192)]),
)
d.add(
    mini_chip(515, 131, "sequential · Block 1 → Block 2", size=10.2,
              fill=C["tint"]),
)


# -------------------------------------------------------------------------
# RIGHT: output head, centre extraction, returned artifact, correction tail.
# -------------------------------------------------------------------------
out_card = card(810, 115, 165, 695, "Output + centre")
d.add(out_card)
d.add(
    text(892.5, 169, "LENGTH-PRESERVING HEAD", size=9.4,
         fill=C["blue"], weight=700, anchor="middle"),
    pill(824, 188, 137, 52, "Conv2d 64→64 · (1×3)"),
    text(892.5, 258, "symmetric pad → ReLU", size=9.5,
         fill=C["slate"], anchor="middle"),
    pill(824, 278, 137, 48, "1×1 Conv · 64→1"),
    mini_chip(892.5, 351, "[B,1,30,3584]", size=9.4, fill=C["tint"]),
    divider(824, 378, 137),
    text(892.5, 404, "CENTRE-EPOCH EXTRACTION", size=9.25,
         fill=C["blue"], weight=700, anchor="middle"),
    epoch_strip(821.5, 423, slot_w=16, gap=2),
    multiline_text(
        892.5,
        489,
        ["slice samples", "1536:2048", "fourth of seven"],
        size=9.9,
        fill=C["ink"],
        weight=600,
        anchor="middle",
        leading=16,
    ),
    mini_chip(892.5, 553, "artifact [B,30,512]", size=9.25,
              fill=C["tint"]),
    subtraction_tail(824, 584, 137),
)
d.add(
    edge([(892.5, 243), (892.5, 274)]),
    edge([(892.5, 329), (892.5, 338)]),
    edge([(892.5, 362), (892.5, 374)]),
    edge([(892.5, 540), (892.5, 542)]),
    edge([(892.5, 566), (892.5, 580)]),
)


# -------------------------------------------------------------------------
# SIDE INPUT: graph construction and persistent l_tilde model buffer.
# -------------------------------------------------------------------------
graph_panel = container(25, 855, 420, 390, "Graph construction · outside forward")
d.add_edge(
    f'<rect x="25" y="855" width="420" height="390" rx="14" '
    f'fill="{C["surface"]}" fill-opacity="0.96" stroke="{C["ink"]}" '
    f'stroke-opacity="{C["border_op"]}" stroke-width="1.25" '
    f'filter="url(#fp-shadow)"/>'
)
d.add(graph_panel)
d.add(
    scalp_graph(92, 954, 48),
    text(92, 1021, "30 electrode nodes", size=9.8, fill=C["blue"],
         weight=700, anchor="middle"),
    multiline_text(
        160,
        902,
        [
            "Fixed Niazy channel order + standard_1005",
            "montage (legacy aliases)",
            "positions → unit sphere → geodesic distances",
            "symmetric spatial k-NN · k=4 · Gaussian weights",
            "self loops",
        ],
        size=10.25,
        fill=C["ink"],
        leading=20,
    ),
    divider(45, 1047, 380),
    multiline_text(
        45,
        1074,
        [
            "Normalized Laplacian  L = I − D⁻¹ᐟ² A D⁻¹ᐟ²",
            "lambda_max ≈ 2  ⇒  L_tilde = L − I",
        ],
        size=10.6,
        fill=C["slate"],
        weight=600,
        leading=21,
    ),
    pill(74, 1132, 322, 48, "persistent buffer · l_tilde [30,30]",
         accent=C["blue400"]),
    multiline_text(
        235,
        1201,
        [
            "built at model instantiation · registered persistent buffer",
            "non-trainable · not a named parameter · not recomputed in forward",
        ],
        size=9.65,
        fill=C["blue"],
        weight=600,
        anchor="middle",
        leading=17,
    ),
)


def buffer_edge(points: list[tuple[float, float]]) -> str:
    pstr = " ".join(f"{px},{py}" for px, py in points)
    return (
        f'<polyline points="{pstr}" fill="none" stroke="{C["blue"]}" '
        f'stroke-opacity="0.8" stroke-width="1.8" stroke-dasharray="4 4" '
        f'stroke-linejoin="round" marker-end="url(#fp-buffer-arrow)"/>'
    )


# These are deliberately the only l_tilde delivery arrows. Their endpoints are
# the ChebConv stages, never the raw EEG/preparation or either Temporal GLU.
d.add_edge(
    buffer_edge(
        [(106, 1128), (38, 1128), (38, 841), (219, 841),
         (219, 374), (256, 374)]
    ),
    buffer_edge(
        [(364, 1128), (438, 1128), (438, 841), (798, 841),
         (798, 374), (775, 374)]
    ),
)
d.add(
    # Short visible deliveries bridge each opaque block shell and terminate
    # immediately outside the corresponding ChebConv stage.
    buffer_edge([(248, 374), (256, 374)]),
    buffer_edge([(782, 374), (746, 374)]),
    mini_chip(219, 841, "l_tilde → ChebConv only", size=9.2,
              accent=C["blue"], fill=C["surface"]),
    mini_chip(798, 841, "l_tilde only", size=9.2,
              accent=C["blue"], fill=C["surface"]),
)


# -------------------------------------------------------------------------
# Accuracy caveat and proof-fit configuration.
# -------------------------------------------------------------------------
d.add(
    card(
        465,
        875,
        510,
        148,
        "Architecture caveat",
        [
            "No U-Net encoder→decoder skips.",
            "Residuals exist only inside STConvBlocks.",
            "Both are used: temporal 3584 samples · spatial 30-node graph.",
        ],
    ),
    card(
        465,
        1045,
        510,
        176,
        "Niazy proof-fit · full-run defaults",
        [
            "hidden=64 · bottleneck=16 · time_kernel=3 · Chebyshev k_order=3",
            "dropout=0.1 · knn_k=4 · causal=false · channel_wise=false",
            "TGLU outputs preserve T=3584 · ChebConv acts per time step",
            "centre_index=3 · centre slice [1536:2048]",
        ],
    ),
)


OUT_DIR.mkdir(parents=True, exist_ok=True)
d.render_png(str(OUT_PNG), svg_path=str(OUT_SVG), width=1280)
print(f"svg: {OUT_SVG}")
print(f"png: {OUT_PNG}")

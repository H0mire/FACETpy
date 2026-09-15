"""Build the SepFormer paper-accurate FACETpy architecture diagram.

The content and configured shapes are sourced from the verified full run in
``sepformer_paper_accurate_edition``.  The composition deliberately separates
the encoder-feature carrier used by mask multiplication from Transformer
residuals: it is a solid data dependency, never an encoder-to-decoder skip.
"""
from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".claude/skills/facetpy-diagram/assets"))

from facetpy_svg import (  # noqa: E402
    C,
    CANVAS_W,
    Diagram,
    _geom,
    card,
    card_shell,
    container,
    divider,
    edge,
    eeg_wave,
    node_dot,
    rounded_top,
    text,
    text_width,
)


OUT_SVG = REPO / "docs/source/_static/diagrams/sepformer_pa_architecture.svg"
OUT_PNG = REPO / "docs/source/_static/diagrams/sepformer_pa_architecture.png"

H = 2160
d = Diagram(H, theme="light")
assert d.width == CANVAS_W == 1000


# ---------------------------------------------------------------------------
# FACETpy-shaped custom building blocks
# ---------------------------------------------------------------------------
d.add_defs(
    f'<marker id="fp-residual" markerWidth="11" markerHeight="11" '
    f'refX="8" refY="5" orient="auto">'
    f'<path d="M1 1 L9 5 L1 9" fill="none" stroke="{C["blue"]}" '
    f'stroke-width="1.4"/></marker>'
)


def multi_note(x, y, w, lines, *, heading=True, size=10.8):
    """Folded-corner note with deterministic multi-line layout."""
    fold = 14
    h = 18 + len(lines) * 18
    parts = [
        f'<path d="M{x} {y} L{x+w-fold} {y} L{x+w} {y+fold} '
        f'L{x+w} {y+h} L{x} {y+h} Z" fill="{C["tint"]}" '
        f'stroke="{C["slate"]}" stroke-opacity="0.45" stroke-width="1"/>',
        f'<path d="M{x+w-fold} {y} L{x+w-fold} {y+fold} L{x+w} {y+fold}" '
        f'fill="none" stroke="{C["slate"]}" stroke-opacity="0.45"/>',
    ]
    line_y = y + 22
    for index, label in enumerate(lines):
        is_heading = heading and index == 0
        parts.append(
            text(
                x + 12,
                line_y,
                label,
                size=size + (0.35 if is_heading else 0),
                fill=C["blue"] if is_heading else C["slate"],
                weight=700 if is_heading else None,
            )
        )
        line_y += 18
    return _geom("".join(parts), x, y, w, h)


def layer_row(x, y, w, h, label, *, sub=None, size=11.1, accent=None):
    """A compact branded process row, optionally with a second line."""
    accent = accent or C["blue"]
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'fill="{C["tint"]}" stroke="{accent}" stroke-opacity="0.55" '
        f'stroke-width="1.35" filter="url(#fp-shadow)"/>',
        f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" '
        f'fill="{accent}"/>',
    ]
    if sub is None:
        parts.append(
            text(
                x + w / 2 + 2,
                y + h / 2 + 4,
                label,
                size=size,
                fill=C["ink"],
                weight=600,
                anchor="middle",
            )
        )
    else:
        parts.extend(
            [
                text(
                    x + w / 2 + 2,
                    y + h / 2 - 4,
                    label,
                    size=size,
                    fill=C["ink"],
                    weight=600,
                    anchor="middle",
                ),
                text(
                    x + w / 2 + 2,
                    y + h / 2 + 14,
                    sub,
                    size=size - 1,
                    fill=C["slate"],
                    anchor="middle",
                ),
            ]
        )
    return _geom("".join(parts), x, y, w, h)


def mini_chip(cx, cy, label, *, size=10.3, fill=None, strong=False):
    """Small labelled capsule for exact shapes, formulas, and callouts."""
    w = text_width(label, size) + 18
    fill = fill or C["surface"]
    return (
        f'<rect x="{cx-w/2:.1f}" y="{cy-10}" width="{w:.1f}" height="20" '
        f'rx="10" fill="{fill}" stroke="{C["blue"]}" '
        f'stroke-opacity="0.38"/>'
        + text(
            cx,
            cy + 3.6,
            label,
            size=size,
            fill=C["blue"],
            weight=700 if strong else 600,
            anchor="middle",
        )
    )


def epoch_strip_card(x, y, w, title, *, output=False):
    """Seven trigger-aligned slots with index 3 highlighted as CENTER."""
    h = 184 if not output else 226
    geom = card(x, y, w, h, title)
    parts = [geom["svg"]]
    if output:
        info = [
            "reshape  [B,7,1,S]",
            "decoded[:,3,:,:] · 4th of 7",
        ]
    else:
        info = [
            "[B,7,1,S] · configured S=512",
            "one EEG channel",
        ]
    for index, label in enumerate(info):
        parts.append(
            text(
                x + w / 2,
                y + 58 + index * 18,
                label,
                size=10.5,
                fill=C["slate"],
                weight=600 if index == 0 else None,
                anchor="middle",
            )
        )
    box_w, gap, box_h = 22, 4, 38
    strip_w = 7 * box_w + 6 * gap
    start_x = x + (w - strip_w) / 2
    box_y = y + 93
    for index in range(7):
        box_x = start_x + index * (box_w + gap)
        if index == 3:
            parts.extend(
                [
                    f'<rect x="{box_x}" y="{box_y}" width="{box_w}" '
                    f'height="{box_h}" rx="7" fill="url(#fp-header)" '
                    f'stroke="{C["blue"]}" stroke-width="1.6"/>',
                    text(
                        box_x + box_w / 2,
                        box_y + 24,
                        "3",
                        size=11,
                        fill=C["header_fg"],
                        weight=700,
                        anchor="middle",
                    ),
                ]
            )
        else:
            parts.extend(
                [
                    f'<rect x="{box_x}" y="{box_y}" width="{box_w}" '
                    f'height="{box_h}" rx="7" fill="{C["tint"]}" '
                    f'stroke="{C["slate"]}" stroke-opacity="0.40"/>',
                    text(
                        box_x + box_w / 2,
                        box_y + 24,
                        str(index),
                        size=10.5,
                        fill=C["slate"],
                        weight=600,
                        anchor="middle",
                    ),
                ]
            )
    parts.append(mini_chip(x + w / 2, y + 149, "slot 3 · CENTER", strong=True))
    if output:
        parts.extend(
            [
                text(
                    x + w / 2,
                    y + 179,
                    "artifact  [B,1,S]",
                    size=11.2,
                    fill=C["ink"],
                    weight=700,
                    anchor="middle",
                ),
                text(
                    x + w / 2,
                    y + 199,
                    "configured [B,1,512]",
                    size=10.4,
                    fill=C["blue"],
                    weight=600,
                    anchor="middle",
                ),
            ]
        )
    return _geom("".join(parts), x, y, w, h)


def residual_path(points, *, marker=True, width=1.45):
    """Dashed accent used only for residual paths inside Transformer stages."""
    pstr = " ".join(f"{px},{py}" for px, py in points)
    marker_attr = ' marker-end="url(#fp-residual)"' if marker else ""
    return (
        f'<polyline points="{pstr}" fill="none" stroke="{C["blue"]}" '
        f'stroke-opacity="0.88" stroke-width="{width}" '
        f'stroke-dasharray="5 4" stroke-linejoin="round"{marker_attr}/>'
    )


def transformer_stage(x, y, w, title, scope, layers, skip_label):
    """Expanded intra/inter stage with per-layer and whole-stack residuals."""
    h = 370
    head = 36
    parts = [
        card_shell(x, y, w, h),
        f'<path d="{rounded_top(x, y, w, head, 12)}" fill="url(#fp-header)"/>',
        node_dot(x + 18, y + head / 2),
        text(
            x + 33,
            y + 23,
            title,
            size=13.2,
            fill=C["header_fg"],
            weight=600,
        ),
        divider(x, y + head, w),
        text(
            x + w / 2,
            y + 57,
            scope,
            size=9.7,
            fill=C["slate"],
            weight=600,
            anchor="middle",
        ),
        mini_chip(
            x + w / 2,
            y + 82,
            f"{layers} pre-norm layers · 8 heads",
            size=9.4,
            fill=C["tint"],
            strong=True,
        ),
        text(
            x + w / 2,
            y + 101,
            "per-layer residuals · dashed",
            size=8.5,
            fill=C["blue"],
            weight=600,
            anchor="middle",
        ),
    ]

    # Two sublayer rows.  Each dashed loop starts at z, stays in the card's
    # side gutter, and returns to the corresponding + node without touching text.
    row_x, row_w = x + 23, w - 46
    row1 = layer_row(row_x, y + 108, row_w, 48, "LN → MHA → +", size=10.4)
    row2 = layer_row(row_x, y + 168, row_w, 48, "LN → FFN 512 → +", size=10.2)
    parts.extend([row1["svg"], row2["svg"]])
    parts.extend(
        [
            residual_path(
                [
                    (row_x + 3, y + 132),
                    (x + 10, y + 132),
                    (x + 10, y + 150),
                    (row_x + row_w - 8, y + 150),
                ]
            ),
            residual_path(
                [
                    (row_x + 3, y + 192),
                    (x + 10, y + 192),
                    (x + 10, y + 210),
                    (row_x + row_w - 8, y + 210),
                ]
            ),
            text(
                x + w / 2,
                y + 236,
                "whole-stack residual",
                size=9.4,
                fill=C["slate"],
                anchor="middle",
            ),
            mini_chip(
                x + w / 2,
                y + 257,
                "g^K(z+PE)+z",
                size=10.1,
                fill=C["tint"],
                strong=True,
            ),
        ]
    )
    projection = layer_row(
        row_x,
        y + 282,
        row_w,
        38,
        "Linear + channel norm",
        size=9.9,
    )
    skip = layer_row(
        row_x,
        y + 330,
        row_w,
        28,
        skip_label,
        size=9.9,
        accent=C["blue400"],
    )
    parts.extend([projection["svg"], skip["svg"]])
    parts.append(
        residual_path(
            [
                (x + w - 13, y + 101),
                (x + w - 7, y + 101),
                (x + w - 7, y + 343),
                (row_x + row_w - 8, y + 343),
            ],
            width=1.65,
        )
    )
    return _geom("".join(parts), x, y, w, h)


def product_node(x, y, w):
    """Elementwise mask product with the two inputs explicitly named."""
    h = 92
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
        f'fill="url(#fp-header)" filter="url(#fp-shadow)"/>',
        f'<circle cx="{x+31}" cy="{y+h/2}" r="12" fill="none" '
        f'stroke="{C["blue200"]}" stroke-width="1.8"/>',
        text(
            x + 31,
            y + h / 2 + 5,
            "×",
            size=18,
            fill=C["blue200"],
            weight=700,
            anchor="middle",
        ),
        text(
            x + 122,
            y + 31,
            "mask × original features",
            size=11.2,
            fill=C["header_fg"],
            weight=700,
            anchor="middle",
        ),
        text(
            x + 122,
            y + 53,
            "elementwise",
            size=9.8,
            fill=C["blue200"],
            weight=600,
            anchor="middle",
        ),
        text(
            x + 122,
            y + 73,
            "masked [B,128,447]",
            size=10.3,
            fill=C["blue200"],
            weight=600,
            anchor="middle",
        ),
    ]
    return _geom("".join(parts), x, y, w, h)


# ---------------------------------------------------------------------------
# Title furniture
# ---------------------------------------------------------------------------
TITLE = "SepFormer — Paper-Accurate Edition"
SUBTITLE = "7-epoch EEG context · dual-path Transformer separator · centre-epoch artifact"
d.add(
    text(70, 52, TITLE, size=24, weight=700, fill=C["ink"]),
    f'<rect x="70" y="62" width="{22 + len(TITLE) * 9}" height="3" '
    f'rx="1.5" fill="url(#fp-header)"/>',
    eeg_wave(70 + text_width(TITLE, 24) + 34, 44, 90),
    text(70, 84, SUBTITLE, size=13.5, fill=C["slate"]),
)


# ---------------------------------------------------------------------------
# Left: adapter, seven-epoch input, flatten, encoder
# ---------------------------------------------------------------------------
adapter_note = multi_note(
    20,
    112,
    215,
    [
        "ADAPTER / DATASET · outside model",
        "trigger-aligned seven-neighbour window",
        "for one EEG channel",
        "window formed outside forward",
    ],
    size=10.2,
)
input_window = epoch_strip_card(20, 210, 215, "Input window")
flatten = card(
    20,
    410,
    215,
    0,
    "Temporal concatenate",
    [
        "model forward begins",
        "[B,7,1,S] → [B,1,7×S]",
        "configured [B,1,3584]",
    ],
)
encoder = card(
    20,
    550,
    215,
    0,
    "Encoder",
    [
        "Conv1d 1→128",
        "K=16 · stride=8 · no padding",
        "ReLU",
        "original features [B,128,447]",
    ],
)


# ---------------------------------------------------------------------------
# Centre: dominant separator and expanded dual-path block
# ---------------------------------------------------------------------------
separator = container(260, 112, 500, 1375, "Separator")
caveat = multi_note(
    280,
    154,
    460,
    [
        "ARCHITECTURE CAVEAT",
        "No U-Net encoder→decoder skips.",
        "Residuals exist only inside the Separator / Transformer stacks.",
        "The original encoder-feature carrier feeds mask ×, not the decoder.",
    ],
    size=10.8,
)
feature_note = multi_note(
    280,
    425,
    460,
    [
        "FEATURE-SPACE CHUNKS",
        "Latent feature chunks — NOT the seven EEG epochs.",
    ],
    size=11.1,
)
projection = card(
    280,
    550,
    205,
    0,
    "Channel projection",
    [
        "Channel LayerNorm",
        "+ Linear / 1×1 Conv",
        "[B,128,447]",
    ],
)
chunking = card(
    510,
    532,
    230,
    0,
    "Feature-space chunking",
    [
        "50% overlap · size 64 · hop 32",
        "one right-pad feature frame",
        "[B,128,13 chunks,64 frames]",
    ],
)

dual_block = container(280, 705, 460, 520, "N=2 Dual-Path Blocks · sequential")
block_shape = mini_chip(
    510,
    745,
    "×2 sequential · INTRA → INTER · shape [B,128,13,64]",
    size=10.1,
    fill=C["tint"],
    strong=True,
)
intra = transformer_stage(
    300,
    767,
    195,
    "Intra-Chunk",
    "local within each 64-frame chunk",
    8,
    "+ block input",
)
inter = transformer_stage(
    525,
    767,
    195,
    "Inter-Chunk",
    "across the 13 chunks",
    4,
    "+ intra result",
)
block_output_shape = mini_chip(
    510,
    1194,
    "each block preserves [B,128,13,64]",
    size=10.4,
    fill=C["surface"],
)

pre_ola = layer_row(
    280,
    1250,
    210,
    72,
    "PReLU + Linear / 1×1 Conv",
    sub="on chunks · before overlap-add",
    size=10.7,
)
ola = layer_row(
    520,
    1250,
    220,
    72,
    "Overlap-Add",
    sub="→ [B,128,447]",
    size=11.4,
)
feed_forward = card(
    280,
    1350,
    285,
    0,
    "Two-layer FeedForward + ReLU",
    [
        "1×1 Conv 128→128 · ReLU",
        "1×1 Conv 128→128",
    ],
)
mask = card(
    590,
    1350,
    150,
    0,
    "Artifact mask",
    [
        "ReLU · N_s=1",
        "[B,128,447]",
    ],
)


# ---------------------------------------------------------------------------
# Right: product, decoder, center slice, pipeline tail
# ---------------------------------------------------------------------------
product = product_node(780, 1360, 200)
decoder = card(
    780,
    1505,
    200,
    0,
    "Decoder",
    [
        "ConvTranspose1d 128→1",
        "K=16 · stride=8 · no padding",
        "[B,1,7×S]",
        "configured [B,1,3584]",
    ],
)
centre_output = epoch_strip_card(780, 1680, 200, "Centre-epoch output", output=True)

pipeline_tail = container(515, 1930, 465, 180, "Pipeline tail · outside model forward")
adapter_tail = card(
    535,
    1970,
    190,
    0,
    "Adapter tail",
    [
        "may remove prediction mean",
        "resample S→native",
        "centre_len",
    ],
)
correction = card(
    745,
    1970,
    215,
    0,
    "FACETpy correction",
    [
        "DeepLearningCorrection",
        "measured EEG − artifact",
        "→ corrected EEG",
    ],
)


# Primary dataflow.  Card endpoints stop short so marker heads remain visible.
d.add_edge(
    edge([(127.5, adapter_note["y"] + adapter_note["h"] + 3), (127.5, input_window["y"] - 4)]),
    edge([(127.5, input_window["y"] + input_window["h"] + 3), (127.5, flatten["y"] - 4)]),
    edge([(127.5, flatten["y"] + flatten["h"] + 3), (127.5, encoder["y"] - 4)]),
    edge([(encoder["x"] + encoder["w"] + 3, 625), (projection["x"] - 4, 625)]),
    edge([(projection["x"] + projection["w"] + 3, 615), (chunking["x"] - 4, 615)]),
    edge(
        [
            (625, chunking["y"] + chunking["h"] + 3),
            (625, 684),
            (510, 684),
            (510, dual_block["y"] - 4),
        ]
    ),
    edge([(intra["x"] + intra["w"] + 3, 952), (inter["x"] - 4, 952)]),
    edge(
        [
            (510, dual_block["y"] + dual_block["h"] + 3),
            (510, 1238),
            (385, 1238),
            (385, pre_ola["y"] - 4),
        ]
    ),
    edge([(pre_ola["x"] + pre_ola["w"] + 3, 1286), (ola["x"] - 4, 1286)]),
    edge([(630, ola["y"] + ola["h"] + 3), (630, 1330), (422.5, 1330), (422.5, feed_forward["y"] - 4)]),
    edge([(feed_forward["x"] + feed_forward["w"] + 3, 1402), (mask["x"] - 4, 1402)]),
    edge([(mask["x"] + mask["w"] + 3, 1402), (product["x"] - 4, 1402)], label=("mask", 758, 1390)),
    edge([(880, product["y"] + product["h"] + 3), (880, decoder["y"] - 4)]),
    edge([(880, decoder["y"] + decoder["h"] + 3), (880, centre_output["y"] - 4)]),
    edge(
        [
            (880, centre_output["y"] + centre_output["h"] + 3),
            (880, 1918),
            (500, 1918),
            (500, 2022),
            (adapter_tail["x"] - 4, 2022),
        ],
        label=("centre slice", 700, 1918),
    ),
    edge([(adapter_tail["x"] + adapter_tail["w"] + 3, 2022), (correction["x"] - 4, 2022)]),
)


# The encoder fan-out is a solid carrier/data dependency.  It routes outside
# and below the Separator, then rises only to the multiplication node.  It does
# not touch the decoder; only dashed paths in the diagram are residuals above.
d.add_edge(
    edge(
        [
            (encoder["x"] + encoder["w"] + 3, 625),
            (248, 625),
            (248, 1515),
            (765, 1515),
            (765, 1435),
            (product["x"] - 4, 1435),
        ],
        label=("ORIGINAL features for mask × — not a U-Net skip", 506, 1505),
    )
)


# Junction makes the two solid encoder-feature destinations explicit.
d.add(
    f'<circle cx="238" cy="625" r="4.2" fill="{C["surface"]}" '
    f'stroke="{C["ink"]}" stroke-width="1.5"/>',
    separator,
    caveat,
    feature_note,
    projection,
    chunking,
    dual_block,
    block_shape,
    intra,
    inter,
    block_output_shape,
    pre_ola,
    ola,
    feed_forward,
    mask,
    adapter_note,
    input_window,
    flatten,
    encoder,
    product,
    decoder,
    centre_output,
    pipeline_tail,
    adapter_tail,
    correction,
)


def main() -> None:
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    d.save(OUT_SVG)
    d.render_png(OUT_PNG, svg_path=OUT_SVG, width=1280)


if __name__ == "__main__":
    main()

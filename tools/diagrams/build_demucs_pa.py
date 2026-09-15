"""Demucs — Paper-Accurate Edition: configured full-run architecture diagram.

The diagram follows the exact model and adapter path configured by
``training_niazy_proof_fit.yaml``.  In particular, it makes the two unrelated
resampling operations explicit and shows the configured decoder's 6996-sample
output being right-padded (not cropped) to the 7168-sample upsampled target.

Verified against
    src/facet/models/experimental/paper_accurate/demucs/training.py
    src/facet/models/experimental/paper_accurate/demucs/processor.py
    src/facet/models/experimental/paper_accurate/demucs/training_niazy_proof_fit.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "skills/facetpy-diagram/assets"))

from facetpy_svg import (  # noqa: E402
    C,
    CANVAS_W,
    Diagram,
    _geom,
    card,
    capsule,
    container,
    edge,
    eeg_wave,
    text,
    text_width,
)

OUT_SVG = REPO / "docs/source/_static/diagrams/demucs_pa_architecture.svg"
OUT_PNG = REPO / "docs/source/_static/diagrams/demucs_pa_architecture.png"

H = 3130
d = Diagram(H, theme="light")
assert d.width == CANVAS_W == 1000

TITLE = "Demucs — Paper-Accurate Edition"
SUBTITLE = (
    "7-epoch time-domain U-Net · 2-layer BiLSTM · summed skips · "
    "artifact center slice"
)
d.add(
    text(70, 52, TITLE, size=24, weight=700, fill=C["ink"]),
    f'<rect x="70" y="62" width="{22 + len(TITLE) * 9}" height="3" '
    f'rx="1.5" fill="url(#fp-header)"/>',
    eeg_wave(70 + text_width(TITLE, 24) + 34, 44, 90),
    text(70, 84, SUBTITLE, size=13.5, fill=C["slate"]),
)


# --------------------------------------------------------------------------- #
#  Shared custom elements used by the paper-accurate diagram series
# --------------------------------------------------------------------------- #
d.add_defs(
    f'<marker id="fp-skip" markerWidth="12" markerHeight="12" refX="9" '
    f'refY="5" orient="auto"><path d="M1 1 L10 5 L1 9 Z" '
    f'fill="{C["blue"]}"/></marker>'
)


def layer_row(x, y, w, h, label, *, accent=None, size=12.5):
    """Compact branded process row with a controllable text size."""
    accent = accent or C["blue"]
    svg = (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
        f'fill="{C["tint"]}" stroke="{accent}" stroke-opacity="0.55" '
        f'stroke-width="1.4" filter="url(#fp-shadow)"/>'
        f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" '
        f'fill="{accent}"/>'
        + text(
            x + w / 2 + 2,
            y + h / 2 + 4.5,
            label,
            size=size,
            weight=500,
            anchor="middle",
        )
    )
    return _geom(svg, x, y, w, h)


def multi_row(x, y, w, h, rows, *, accent=None):
    """A branded process row carrying multiple centred text lines."""
    accent = accent or C["blue"]
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
        f'fill="{C["tint"]}" stroke="{accent}" stroke-opacity="0.55" '
        f'stroke-width="1.4" filter="url(#fp-shadow)"/>'
        f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" '
        f'fill="{accent}"/>'
    ]
    line_y = y + h / 2 - (len(rows) - 1) * 10 + 4.5
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
        line_y += 20
    return _geom("".join(parts), x, y, w, h)


def multi_note(x, y, w, lines, *, size=11.1):
    """Folded-corner note with one heading and several short body lines."""
    fold = 14
    h = 16 + len(lines) * 18 + 8
    parts = [
        f'<path d="M{x} {y} L{x+w-fold} {y} L{x+w} {y+fold} '
        f'L{x+w} {y+h} L{x} {y+h} Z" fill="{C["tint"]}" '
        f'stroke="{C["slate"]}" stroke-opacity="0.45" stroke-width="1"/>',
        f'<path d="M{x+w-fold} {y} L{x+w-fold} {y+fold} L{x+w} {y+fold}" '
        f'fill="none" stroke="{C["slate"]}" stroke-opacity="0.45"/>',
    ]
    line_y = y + 21
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


def mini_chip(cx, cy, label, *, size=10.8):
    """Small inline chip for a configuration fact."""
    width = text_width(label, size) + 22
    return (
        f'<rect x="{cx-width/2:.1f}" y="{cy-11}" width="{width:.1f}" '
        f'height="22" rx="11" fill="{C["surface"]}" stroke="{C["blue"]}" '
        f'stroke-opacity="0.35"/>'
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


def epoch_strip_card(x, y, w, title, *, output=False):
    """Seven epoch slots with the centre (index 3) highlighted."""
    h = 174
    geom = card(x, y, w, h, title)
    box_w, gap, box_h = 76, 14, 42
    strip_w = 7 * box_w + 6 * gap
    start_x = x + (w - strip_w) / 2
    box_y = y + 74
    parts = [geom["svg"]]
    intro = (
        "model prediction over 7 × 512-sample slots"
        if output
        else "one EEG channel · 7 trigger-aligned epochs · 512 samples each"
    )
    parts.append(
        text(x + w / 2, y + 58, intro, size=12, fill=C["slate"], anchor="middle")
    )
    for index in range(7):
        box_x = start_x + index * (box_w + gap)
        if index == 3:
            parts.append(
                f'<rect x="{box_x}" y="{box_y}" width="{box_w}" '
                f'height="{box_h}" rx="9" fill="url(#fp-header)" '
                f'stroke="{C["blue"]}" stroke-width="1.7"/>'
            )
            parts.append(
                text(
                    box_x + box_w / 2,
                    box_y + 26,
                    "3 · CENTER",
                    size=10.5,
                    fill=C["header_fg"],
                    weight=700,
                    anchor="middle",
                )
            )
        else:
            parts.append(
                f'<rect x="{box_x}" y="{box_y}" width="{box_w}" '
                f'height="{box_h}" rx="9" fill="{C["tint"]}" '
                f'stroke="{C["slate"]}" stroke-opacity="0.45" '
                f'stroke-width="1.2"/>'
            )
            parts.append(
                text(
                    box_x + box_w / 2,
                    box_y + 26,
                    str(index),
                    size=12.5,
                    fill=C["slate"],
                    weight=600,
                    anchor="middle",
                )
            )
    footer_label = (
        "prediction[3×512 : 4×512] = prediction[1536:2048]  →  (512,)"
        if output
        else "center epoch index 3 is the adapter's eventual output slot"
    )
    parts.append(
        text(
            x + w / 2,
            y + 151,
            footer_label,
            size=11.5,
            fill=C["blue"] if output else C["slate"],
            weight=600 if output else None,
            anchor="middle",
        )
    )
    geom["svg"] = "".join(parts)
    return geom


def merge_node(cx, cy):
    """Circled plus: every Demucs skip merge is addition, never concat."""
    return (
        f'<circle cx="{cx}" cy="{cy}" r="14" fill="url(#fp-header)" '
        f'stroke="{C["blue200"]}" stroke-width="1.5" '
        f'filter="url(#fp-shadow)"/>'
        f'<line x1="{cx-6}" y1="{cy}" x2="{cx+6}" y2="{cy}" '
        f'stroke="{C["header_fg"]}" stroke-width="2" stroke-linecap="round"/>'
        f'<line x1="{cx}" y1="{cy-6}" x2="{cx}" y2="{cy+6}" '
        f'stroke="{C["header_fg"]}" stroke-width="2" stroke-linecap="round"/>'
    )


def skip_edge(points, label, lx, ly):
    """Dashed blue U-Net skip with its exact centre-crop/sum operation."""
    point_string = " ".join(f"{px},{py}" for px, py in points)
    width = text_width(label, 10.5) + 18
    return (
        f'<polyline points="{point_string}" fill="none" stroke="{C["blue"]}" '
        f'stroke-opacity="0.9" stroke-width="2.1" stroke-dasharray="8 5" '
        f'stroke-linejoin="round" marker-end="url(#fp-skip)"/>'
        f'<rect x="{lx-width/2:.1f}" y="{ly-11}" width="{width:.1f}" '
        f'height="19" rx="5" fill="{C["surface"]}" stroke="{C["blue"]}" '
        f'stroke-opacity="0.65"/>'
        + text(
            lx,
            ly + 2.5,
            label,
            size=10.5,
            fill=C["blue"],
            weight=600,
            anchor="middle",
        )
    )


# --------------------------------------------------------------------------- #
#  Input and adapter preparation
# --------------------------------------------------------------------------- #
input_frame = container(40, 108, 920, 312, "Input / adapter preparation")
epochs_in = epoch_strip_card(70, 142, 860, "Trigger-aligned context")
flatten = multi_row(
    250,
    340,
    500,
    58,
    [
        ("flatten 7 epochs → one waveform; demean_input = true", 12.2, C["ink"], 500),
        ("model input x:  (B, 1, 3584)", 12.2, C["blue"], 700),
    ],
)

upsample = multi_row(
    220,
    452,
    560,
    66,
    [
        ("_Resampler(factor=2) · zero-stuff + sinc/Kaiser FIR", 12.2, C["ink"], 600),
        ("(B,1,3584) → (B,1,7168) · factor=1 disables", 11.7, C["blue"], 600),
    ],
    accent=C["blue400"],
)
upsample["svg"] += mini_chip(706, 466, "FULL RUN ×2", size=10.2)


# --------------------------------------------------------------------------- #
#  Configured Demucs U-Net — encoder down, decoder up
# --------------------------------------------------------------------------- #
unet_frame = container(
    40,
    548,
    920,
    1412,
    "Configured Demucs U-Net · depth 4 · K=8 · stride=4 · summed skips",
)
valid = multi_row(
    220,
    582,
    560,
    58,
    [
        ("_run_unet: valid_length(7168) = 7168 · no initial pad", 12.2, C["ink"], 600),
        ("arbitrary lengths: right-pad first, restore final length afterward", 11.3, C["slate"], None),
    ],
    accent=C["blue400"],
)

enc_anatomy = card(
    66,
    664,
    412,
    0,
    "Encoder block anatomy",
    [
        "Conv1d K=8, S=4, pad=0 → ReLU",
        "Conv1d K=1, C→2C → GLU (back to C)",
        "initial_channels=64 · depth=4 · no batch norm",
    ],
)
dec_anatomy = card(
    522,
    664,
    412,
    0,
    "Decoder block anatomy",
    [
        "Conv1d K=3, pad=1, C→2C → GLU",
        "ConvTranspose1d K=8, S=4 → ReLU",
        "final stage: Identity (linear signed output)",
    ],
)
enc_anatomy["svg"] += mini_chip(398, 682, "auto_depth: 4 → 4", size=9.8)

ENC_X, ENC_W = 66, 314
DEC_X, DEC_W = 650, 284
MERGE_X = 610
STAGE_Y = (830, 1065, 1300, 1535)

encoders = [
    card(ENC_X, STAGE_Y[0], ENC_W, 0, "E1 · 1 → 64", [
        "Conv1d K8/S4 → ReLU",
        "1×1 Conv 64→128 → GLU (→64)",
        "output  (B,64,1791)",
    ]),
    card(ENC_X, STAGE_Y[1], ENC_W, 0, "E2 · 64 → 128", [
        "Conv1d K8/S4 → ReLU",
        "1×1 Conv 128→256 → GLU (→128)",
        "output  (B,128,446)",
    ]),
    card(ENC_X, STAGE_Y[2], ENC_W, 0, "E3 · 128 → 256", [
        "Conv1d K8/S4 → ReLU",
        "1×1 Conv 256→512 → GLU (→256)",
        "output  (B,256,110)",
    ]),
    card(ENC_X, STAGE_Y[3], ENC_W, 0, "E4 · 256 → 512", [
        "Conv1d K8/S4 → ReLU",
        "1×1 Conv 512→1024 → GLU (→512)",
        "output  (B,512,26)",
    ]),
]

# D4 is opposite E1 at the top; D1 is opposite E4 at the bottom.
decoders = [
    card(DEC_X, STAGE_Y[3], DEC_W, 0, "D1 · 512 → 256", [
        "K3/pad1: 512→1024 → GLU",
        "ConvTranspose K8/S4 → ReLU",
        "output  (B,256,108)",
    ]),
    card(DEC_X, STAGE_Y[2], DEC_W, 0, "D2 · 256 → 128", [
        "K3/pad1: 256→512 → GLU",
        "ConvTranspose K8/S4 → ReLU",
        "output  (B,128,436)",
    ]),
    card(DEC_X, STAGE_Y[1], DEC_W, 0, "D3 · 128 → 64", [
        "K3/pad1: 128→256 → GLU",
        "ConvTranspose K8/S4 → ReLU",
        "output  (B,64,1748)",
    ]),
    card(DEC_X, STAGE_Y[0], DEC_W, 0, "D4 · 64 → 1", [
        "K3/pad1: 64→128 → GLU",
        "ConvTranspose K8/S4 → Identity",
        "linear output  (B,1,6996)",
    ]),
]

bottleneck = card(
    220,
    1738,
    560,
    0,
    "Bottleneck · 2-layer bidirectional LSTM",
    [
        "permute: (B,512,26) → (T,B,C) = (26,B,512)",
        "input_size = hidden_size = 512 · num_layers = 2",
        "raw bidirectional output: (26,B,1024)",
        "Linear 1024→512; permute back → (B,512,26)",
        "then summed with center-cropped E4 at pre-D1 +",
    ],
)

stage_centers = [geom["cy"] for geom in encoders]
merge_nodes = [merge_node(MERGE_X, cy) for cy in stage_centers]


# --------------------------------------------------------------------------- #
#  Internal length restore and model-internal downsampling
# --------------------------------------------------------------------------- #
restore_frame = container(40, 1996, 920, 368, "Internal output length + paper 2× restore")
restore = card(
    205,
    2040,
    590,
    0,
    "_center_crop_1d(out, 7168)",
    [
        "raw D4 output: (B,1,6996) — shorter than target",
        "RIGHT-PADS 172 zeros → (B,1,7168) for this exact path",
        "helper crops if longer; right-pads if shorter",
    ],
)
downsample = card(
    205,
    2202,
    590,
    0,
    "_Resampler.downsample · factor 2",
    [
        "sinc/Kaiser low-pass FIR, then decimate ÷2 → (B,1,3584)",
        "_center_crop_1d(..., 3584): exact original-length restore",
        "full artifact context  (B,1,3584)",
    ],
)


# --------------------------------------------------------------------------- #
#  Adapter tail — centre slice and a separate native-length resample
# --------------------------------------------------------------------------- #
tail_frame = container(40, 2408, 920, 682, "Pipeline adapter tail · native trigger timing")
model_out = capsule(250, 2450, 430, 46, "MODEL ARTIFACT CONTEXT  ·  (B,1,3584)")
shift_note = multi_note(
    700,
    2438,
    235,
    [
        "Adapter option — not forward()",
        "n_shifts = 1 by default",
        ">1: roll → run → inverse-roll",
        "then average before slicing",
    ],
    size=10.2,
)
epochs_out = epoch_strip_card(100, 2548, 800, "Center-epoch slice · radius = 3", output=True)
remove_mean = layer_row(
    250,
    2754,
    500,
    46,
    "optional prediction-mean removal  ·  shape stays (512,)",
    accent=C["blue400"],
    size=11.8,
)
native_resample = card(
    190,
    2834,
    620,
    0,
    "Native center-epoch resampling · adapter",
    [
        "_resample_1d: 512 → center_len",
        "center_len is the native trigger-to-trigger epoch length",
        "distinct from the model-internal 2× sinc/Kaiser trick",
    ],
)
artifact_out = capsule(
    190,
    3012,
    620,
    48,
    "ARTIFACT  ·  insert into estimated_artifacts / return",
)
d.add(
    text(
        500,
        3080,
        "native center epoch artifact estimate  ·  shape (center_len,)",
        size=11.2,
        fill=C["slate"],
        anchor="middle",
    )
)


# --------------------------------------------------------------------------- #
#  Connectors — solid main dataflow, dashed accent U-Net skips
# --------------------------------------------------------------------------- #
ENC_CX = encoders[0]["cx"]

d.add_edge(
    edge([(500, epochs_in["y"] + epochs_in["h"] + 3), (500, flatten["y"] - 4)]),
    edge([(500, flatten["y"] + flatten["h"] + 3), (500, upsample["y"] - 4)]),
    edge([(500, upsample["y"] + upsample["h"] + 3), (500, valid["y"] - 4)]),
    edge([
        (500, valid["y"] + valid["h"] + 3),
        (500, 808),
        (ENC_CX, 808),
        (ENC_CX, encoders[0]["y"] - 4),
    ]),
)

# Encoder descent.
for upper, lower in zip(encoders, encoders[1:]):
    d.add_edge(
        edge([
            (ENC_CX, upper["y"] + upper["h"] + 3),
            (ENC_CX, lower["y"] - 4),
        ])
    )
d.add_edge(
    edge([
        (ENC_CX, encoders[-1]["y"] + encoders[-1]["h"] + 3),
        (ENC_CX, 1704),
        (bottleneck["cx"], 1704),
        (bottleneck["cx"], bottleneck["y"] - 4),
    ])
)

# Bottleneck output rises to the pre-D1 sum in the clear central corridor.
d.add_edge(
    edge(
        [
            (bottleneck["x"] + bottleneck["w"] - 60, bottleneck["y"] - 3),
            (MERGE_X, bottleneck["y"] - 3),
            (MERGE_X, stage_centers[3] + 17),
        ],
        marker_end=None,
    ),
    edge(
        [(MERGE_X + 17, stage_centers[3]), (DEC_X - 4, stage_centers[3])]
    ),
)

# Decoder ascent: D1→pre-D2, D2→pre-D3, D3→pre-D4.
for decoder_index in range(3):
    current = decoders[decoder_index]
    target_row = 2 - decoder_index
    target_y = stage_centers[target_row]
    d.add_edge(
        edge(
            [
                (current["cx"], current["y"] - 3),
                (MERGE_X, current["y"] - 3),
                (MERGE_X, target_y + 17),
            ],
            marker_end=None,
        ),
        edge([(MERGE_X + 17, target_y), (DEC_X - 4, target_y)]),
    )

# Dashed skips land just outside each merge so their markers remain visible.
skip_specs = (
    (0, "crop 1791→1748 · SUM"),
    (1, "crop 446→436 · SUM"),
    (2, "crop 110→108 · SUM"),
    (3, "crop 26→26 · SUM"),
)
for row_index, label in skip_specs:
    cy = stage_centers[row_index]
    d.add_edge(
        skip_edge(
            [(encoders[row_index]["x"] + encoders[row_index]["w"] + 3, cy),
             (MERGE_X - 20, cy)],
            label,
            492,
            cy - 3,
        )
    )

# D4 exits upward, then uses the outside-right lane to reach the restore tail.
d.add_edge(
    edge(
        [
            (decoders[3]["cx"], decoders[3]["y"] - 3),
            (972, decoders[3]["y"] - 3),
            (972, 2018),
            (restore["cx"], 2018),
            (restore["cx"], restore["y"] - 4),
        ],
        label=("raw (B,1,6996)", 884, 2018),
    ),
    edge([
        (restore["cx"], restore["y"] + restore["h"] + 3),
        (restore["cx"], downsample["y"] - 4),
    ]),
    edge([
        (downsample["cx"], downsample["y"] + downsample["h"] + 3),
        (downsample["cx"], model_out["y"] - 4),
    ]),
    edge([
        (model_out["cx"], model_out["y"] + model_out["h"] + 3),
        (model_out["cx"], epochs_out["y"] - 4),
    ]),
    edge([
        (epochs_out["cx"], epochs_out["y"] + epochs_out["h"] + 3),
        (epochs_out["cx"], remove_mean["y"] - 4),
    ]),
    edge([
        (remove_mean["cx"], remove_mean["y"] + remove_mean["h"] + 3),
        (remove_mean["cx"], native_resample["y"] - 4),
    ]),
    edge([
        (native_resample["cx"], native_resample["y"] + native_resample["h"] + 3),
        (native_resample["cx"], artifact_out["y"] - 4),
    ]),
)


# Containers first on the node layer, followed by the cards and merge glyphs.
d.add(input_frame, unet_frame, restore_frame, tail_frame)
d.add(epochs_in, flatten, upsample, valid, enc_anatomy, dec_anatomy)
d.add(*encoders, *decoders, bottleneck)
d.add(*merge_nodes)
d.add(
    text(404, 815, "ENCODER ↓", size=11, fill=C["blue"], weight=700, anchor="end"),
    text(646, 815, "↑ DECODER", size=11, fill=C["blue"], weight=700),
    text(
        604,
        1929,
        "+ at every pre-decoder merge = elementwise SUM, never concatenation",
        size=11.2,
        fill=C["blue"],
        weight=600,
        anchor="middle",
    ),
)
d.add(restore, downsample, model_out, shift_note, epochs_out, remove_mean)
d.add(native_resample, artifact_out)

d.render_png(str(OUT_PNG), svg_path=str(OUT_SVG), width=1280)
print("svg:", OUT_SVG)
print("png:", OUT_PNG)
print("canvas:", CANVAS_W, "x", H)
print("configured encoder shapes:", ["1791", "446", "110", "26"])
print("configured decoder shapes:", ["108", "436", "1748", "6996"])

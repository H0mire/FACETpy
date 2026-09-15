"""Build the ViT Spectrogram MAE paper-accurate FACETpy diagram.

The figure distinguishes inference data flow from the masked-centre training
objective.  In particular, only the 80 visible patches enter the encoder; the
single shared learned mask token is introduced for the first time in the
lightweight decoder.  The cached noisy phase is a side carrier used only for
waveform synthesis.
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


OUT_SVG = REPO / "docs/source/_static/diagrams/vit_spectrogram_pa_architecture.svg"
OUT_PNG = REPO / "docs/source/_static/diagrams/vit_spectrogram_pa_architecture.png"

H = 1850
d = Diagram(H, theme="light")
assert d.width == CANVAS_W == 1000


def titled_panel(x, y, w, h, title, lines=(), *, title_size=13.0):
    """Compact branded card with explicit, deterministic line placement."""
    parts = [
        card_shell(x, y, w, h),
        f'<path d="{rounded_top(x, y, w, 36, 12)}" fill="url(#fp-header)"/>',
        node_dot(x + 18, y + 18),
        text(x + 33, y + 23, title, size=title_size, fill=C["header_fg"], weight=600),
        divider(x, y + 36, w),
    ]
    for index, line in enumerate(lines):
        parts.append(
            text(
                x + 13,
                y + 58 + index * 19,
                line,
                size=10.5,
                fill=C["slate"],
                weight=600 if index == 0 else None,
            )
        )
    return _geom("".join(parts), x, y, w, h)


def layer_row(x, y, w, h, label, sub=None, *, accent=None, size=10.7):
    """One or two line stage row using the FACETpy process treatment."""
    accent = accent or C["blue"]
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" '
        f'fill="{C["tint"]}" stroke="{accent}" stroke-opacity="0.55" '
        f'stroke-width="1.3" filter="url(#fp-shadow)"/>',
        f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" fill="{accent}"/>',
    ]
    if sub is None:
        parts.append(text(x + w / 2 + 2, y + h / 2 + 4, label, size=size,
                          fill=C["ink"], weight=600, anchor="middle"))
    else:
        parts.extend(
            [
                text(x + w / 2 + 2, y + h / 2 - 4, label, size=size,
                     fill=C["ink"], weight=600, anchor="middle"),
                text(x + w / 2 + 2, y + h / 2 + 13, sub, size=size - 1,
                     fill=C["slate"], anchor="middle"),
            ]
        )
    return _geom("".join(parts), x, y, w, h)


def chip(cx, cy, label, *, fill=None, accent=None, strong=False, size=9.8):
    """Auto-sized shape/formula chip."""
    fill = fill or C["surface"]
    accent = accent or C["blue"]
    w = text_width(label, size) + 17
    return (
        f'<rect x="{cx-w/2:.1f}" y="{cy-10}" width="{w:.1f}" height="20" '
        f'rx="10" fill="{fill}" stroke="{accent}" stroke-opacity="0.46"/>'
        + text(cx, cy + 3.4, label, size=size, fill=accent,
               weight=700 if strong else 600, anchor="middle")
    )


def multi_note(x, y, w, lines, *, accent=False, size=10.4):
    """Folded note that supports multiple lines without clipping."""
    fold = 14
    h = 17 + len(lines) * 18
    parts = [
        f'<path d="M{x} {y} L{x+w-fold} {y} L{x+w} {y+fold} L{x+w} {y+h} '
        f'L{x} {y+h} Z" fill="{C["tint"]}" stroke="{C["slate"]}" '
        f'stroke-opacity="0.45"/>',
        f'<path d="M{x+w-fold} {y} L{x+w-fold} {y+fold} L{x+w} {y+fold}" '
        f'fill="none" stroke="{C["slate"]}" stroke-opacity="0.45"/>',
    ]
    for index, line in enumerate(lines):
        parts.append(text(x + 11, y + 21 + index * 18, line, size=size,
                          fill=C["blue"] if accent and index == 0 else C["slate"],
                          weight=700 if index == 0 else None))
    return _geom("".join(parts), x, y, w, h)


def epoch_card(x, y, w):
    """Seven trigger-aligned input slots with centre slot 3 highlighted."""
    h = 214
    parts = [
        card_shell(x, y, w, h),
        f'<path d="{rounded_top(x, y, w, 36, 12)}" fill="url(#fp-header)"/>',
        node_dot(x + 18, y + 18),
        text(x + 33, y + 13, "«outside model»", size=8.6,
             fill=C["header_fg"], italic=True, opacity=0.72),
        text(x + 33, y + 29, "Trigger-aligned input", size=12.3,
             fill=C["header_fg"], weight=600),
        divider(x, y + 36, w),
    ]
    parts.extend(
        [
            text(x + w / 2, y + 60, "dataset / adapter", size=10.7,
                 fill=C["blue"], weight=700, anchor="middle"),
            text(x + w / 2, y + 80, "[B,7,1,S] · S=512", size=10.5,
                 fill=C["slate"], weight=600, anchor="middle"),
            text(x + w / 2, y + 98, "one EEG channel", size=10.2,
                 fill=C["slate"], anchor="middle"),
        ]
    )
    bw, gap, bh = 17, 3, 35
    strip_w = 7 * bw + 6 * gap
    sx = x + (w - strip_w) / 2
    by = y + 115
    for index in range(7):
        bx = sx + index * (bw + gap)
        chosen = index == 3
        parts.append(
            f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="5" '
            f'fill="{"url(#fp-header)" if chosen else C["tint"]}" '
            f'stroke="{C["blue"]}" stroke-opacity="{1 if chosen else 0.32}"/>'
        )
        parts.append(text(bx + bw / 2, by + 22, str(index), size=9.8,
                          fill=C["header_fg"] if chosen else C["slate"],
                          weight=700 if chosen else 600, anchor="middle"))
    parts.extend(
        [
            chip(x + w / 2, y + 169, "slot 3 · 4th · CENTRE", strong=True),
            text(x + w / 2, y + 198, "then concatenate / reshape", size=10.2,
                 fill=C["slate"], anchor="middle"),
        ]
    )
    return _geom("".join(parts), x, y, w, h)


def patch_grid_card(x, y, w):
    """Exact 8×14 patch grid with structural columns 5–8 masked."""
    h = 286
    parts = [
        card_shell(x, y, w, h),
        f'<path d="{rounded_top(x, y, w, 36, 12)}" fill="url(#fp-header)"/>',
        node_dot(x + 18, y + 18),
        text(x + 33, y + 23, "Structural patch mask", size=12.2,
             fill=C["header_fg"], weight=600),
        divider(x, y + 36, w),
    ]
    parts.extend(
        [
            text(x + w / 2, y + 58, "4 freq × 16 time · 64 px", size=10.3,
                 fill=C["slate"], weight=600, anchor="middle"),
            text(x + w / 2, y + 77, "grid 8 × 14 = 112", size=10.3,
                 fill=C["blue"], weight=700, anchor="middle"),
        ]
    )
    cell_w, cell_h, gap = 7.1, 11, 1.5
    grid_w = 14 * cell_w + 13 * gap
    grid_h = 8 * cell_h + 7 * gap
    gx = x + (w - grid_w) / 2
    gy = y + 94
    for row in range(8):
        for col in range(14):
            masked = 5 <= col <= 8
            fill = C["blue"] if masked else C["surface"]
            parts.append(
                f'<rect x="{gx+col*(cell_w+gap):.1f}" y="{gy+row*(cell_h+gap):.1f}" '
                f'width="{cell_w}" height="{cell_h}" rx="1.4" fill="{fill}" '
                f'fill-opacity="{0.88 if masked else 1}" stroke="{C["blue"]}" '
                f'stroke-opacity="{0.75 if masked else 0.28}" stroke-width="0.7"/>'
            )
    parts.extend(
        [
            text(x + w / 2, gy + grid_h + 20, "cols 5–8 · all 8 freq rows", size=9.8,
                 fill=C["blue"], weight=700, anchor="middle"),
            text(x + w / 2, gy + grid_h + 39, "32 masked · 80 visible", size=10.2,
                 fill=C["ink"], weight=700, anchor="middle"),
            text(x + w / 2, gy + grid_h + 57, "centre + 1-patch margin", size=9.7,
                 fill=C["slate"], anchor="middle"),
            text(x + w / 2, gy + grid_h + 78, "NOT uniform random 75%", size=10,
                 fill=C["neg"], weight=700, anchor="middle"),
        ]
    )
    return _geom("".join(parts), x, y, w, h)


def transformer_card(x, y, w, h, *, encoder):
    """Expanded encoder/decoder card with contained PreNorm residual blocks."""
    if encoder:
        title = "VISIBLE-ONLY ViT encoder"
        badge = "6× · width 192"
        intro = [
            ("Patch embed", "Linear 64→192 + encoder 2D sin–cos pos"),
            ("Input LN", "visible tokens only · dropout 0"),
        ]
        block_lines = [
            "x + MSA(LN(x)) · 6 heads",
            "x + MLP/GELU(LN(x))",
            "MLP 192→768→192",
        ]
        tail = ("Final LN", "output [B,80,192]")
        accent = C["blue"]
    else:
        title = "Lightweight MAE decoder"
        badge = "2× · width 96"
        intro = [
            ("Project visible", "192→96 · scatter 80 into 112 slots"),
            ("Insert mask token", "ONE SHARED learned token × 32 positions"),
            ("Decoder positions", "separate fixed 2D sin–cos · all 112"),
        ]
        block_lines = [
            "x + MSA(LN(x)) · 4 heads",
            "x + MLP/GELU(LN(x))",
            "MLP 96→384→96",
        ]
        tail = ("Decoder LN + Linear", "96→64 · [B,112,64]")
        accent = C["blue400"]

    parts = [
        card_shell(x, y, w, h),
        f'<path d="{rounded_top(x, y, w, 36, 12)}" fill="url(#fp-header)"/>',
        node_dot(x + 18, y + 18),
        text(x + 33, y + 23, title, size=12.3 if not encoder else 12.6,
             fill=C["header_fg"], weight=600),
        divider(x, y + 36, w),
        chip(x + w / 2, y + 54, badge, strong=True, fill=C["tint"]),
    ]
    ry = y + 75
    for label, sub in intro:
        parts.append(layer_row(x + 12, ry, w - 24, 48, label, sub,
                               accent=accent, size=10.4)["svg"])
        ry += 58

    block_h = 112
    parts.extend(
        [
            f'<rect x="{x+12}" y="{ry}" width="{w-24}" height="{block_h}" rx="10" '
            f'fill="{C["surface"]}" stroke="{accent}" stroke-width="1.6" '
            f'stroke-opacity="0.72" filter="url(#fp-shadow)"/>',
            f'<rect x="{x+12}" y="{ry}" width="{w-24}" height="25" rx="10" '
            f'fill="{C["tint"]}"/>',
            text(x + w / 2, ry + 17, "PreNorm residual Transformer block",
                 size=10.3, fill=C["blue"], weight=700, anchor="middle"),
        ]
    )
    for index, line in enumerate(block_lines):
        parts.append(text(x + 24, ry + 44 + index * 20, line, size=10.3,
                          fill=C["ink"] if index < 2 else C["slate"],
                          weight=600 if index < 2 else None))
    parts.append(text(x + w - 20, ry + 101, "residuals contained", size=8.9,
                      fill=C["blue"], anchor="end"))
    ry += block_h + 12
    parts.append(layer_row(x + 12, ry, w - 24, 48, tail[0], tail[1],
                           accent=accent, size=10.4)["svg"])
    return _geom("".join(parts), x, y, w, h)


# ---------------------------------------------------------------------------
# Canvas furniture and subsystem frames
# ---------------------------------------------------------------------------
d.add(
    text(36, 42, "ViT Spectrogram MAE — Paper-Accurate Edition", size=24,
         fill=C["ink"], weight=700),
    text(36, 67,
         "structural centre masking · visible-only ViT encoder · lightweight MAE decoder · clean EEG",
         size=13.4, fill=C["slate"]),
    f'<rect x="36" y="76" width="430" height="3" rx="1.5" fill="url(#fp-header)"/>',
    eeg_wave(850, 51, 105),
)

analysis_frame = container(22, 100, 956, 360, "1 · analysis + structural mask")
mae_frame = container(22, 482, 956, 655, "2 · visible-only MAE")
recon_frame = container(22, 1159, 956, 345, "3 · inference reconstruction")
training_frame = container(22, 1526, 458, 190, "training objective · side branch")
tail_frame = container(500, 1526, 478, 190, "outside-model FACETpy tail")
d.add(analysis_frame, mae_frame, recon_frame, training_frame, tail_frame)


# ---------------------------------------------------------------------------
# 1 · Input, STFT, magnitude/phase split, exact structural mask
# ---------------------------------------------------------------------------
epochs = epoch_card(40, 129, 166)
reshape = titled_panel(224, 155, 132, 136, "Model input", [
    "concatenate / reshape",
    "[B,3584]",
    "7 × 512 samples",
])
stft = titled_panel(374, 129, 184, 214, "Hann STFT", [
    "n_fft 64 · hop 16",
    "center=True",
    "complex [B,33,225]",
    "crop → [B,32,224]",
])
split = titled_panel(576, 129, 188, 214, "Magnitude / phase split", [
    "MAIN: log1p(|Z|)",
    "2D time–frequency image",
    "SIDE: angle(Z)",
    "cached ORIGINAL phase",
])
grid = patch_grid_card(782, 129, 176)
d.add(epochs, reshape, stft, split, grid)

d.add_edge(
    edge([(epochs["x"] + epochs["w"] + 3, 230), (reshape["x"] - 3, 230)]),
    edge([(reshape["x"] + reshape["w"] + 3, 223), (stft["x"] - 3, 223)]),
    edge([(stft["x"] + stft["w"] + 3, 223), (split["x"] - 3, 223)]),
    edge([(split["x"] + split["w"] + 3, 223), (grid["x"] - 3, 223)]),
)
d.add(
    multi_note(224, 375, 540, [
        "extra random regularization: OFF / follow-up, not wired",
        "The implemented mask is deterministic structure across time columns 5–8.",
    ], accent=True),
)


# ---------------------------------------------------------------------------
# 2 · Split, visible-only encoder, decoder-only mask-token insertion
# ---------------------------------------------------------------------------
visible = titled_panel(43, 547, 172, 144, "Visible originals", [
    "80 patches [B,80,64]",
    "original noisy log-mag",
    "ONLY branch into encoder",
])
withheld = titled_panel(43, 905, 172, 144, "Masked positions", [
    "32 centre positions",
    "withheld from encoder",
    "position indices only",
])
encoder = transformer_card(252, 530, 255, 548, encoder=True)
decoder = transformer_card(548, 530, 274, 548, encoder=False)
prediction = titled_panel(846, 690, 112, 208, "Prediction", [
    "all patches",
    "[B,112,64]",
    "select masked",
    "[B,32,64]",
], title_size=11.7)
d.add(visible, withheld, encoder, decoder, prediction)

# Structural split descends from grid into both branches; visible branch alone
# crosses the prominent encoder boundary.  Masked positions bypass underneath.
d.add_edge(
    edge([(grid["x"] - 3, 360), (770, 360), (770, 518), (129, 518),
          (129, visible["y"] - 3)], label=("structural split", 505, 510)),
    edge([(129, 518), (129, withheld["y"] - 3)], marker_end="arrow"),
    edge([(visible["x"] + visible["w"] + 3, 685), (encoder["x"] - 3, 685)]),
    edge([(encoder["x"] + encoder["w"] + 3, 805), (decoder["x"] - 3, 805)]),
    edge([(withheld["x"] + withheld["w"] + 3, 985), (230, 985), (230, 1100),
          (530, 1100), (530, 665), (decoder["x"] - 3, 665)],
         label=("32 indices bypass encoder", 380, 1122)),
    edge([(decoder["x"] + decoder["w"] + 3, 805), (prediction["x"] - 3, 805)]),
)

d.add(
    multi_note(43, 718, 172, [
        "ENCODER SEES",
        "NO MASK TOKENS",
        "Masked values and tokens",
        "stay outside this boundary.",
    ], accent=True, size=10.1),
    multi_note(548, 1081, 410, [
        "MASK TOKEN FIRST OCCURS HERE — DECODER ONLY",
        "One shared learned token is reused at each of the 32 masked positions.",
    ], accent=True, size=10.3),
    chip(379, 1048, "80 encoded visible tokens · [B,80,192]", strong=True),
)


# ---------------------------------------------------------------------------
# 3 · Inference merge, phase reuse, inverse transform, clean centre epoch
# ---------------------------------------------------------------------------
orig_visible = titled_panel(43, 1214, 163, 148, "Keep visible", [
    "80 ORIGINAL noisy",
    "log-mag patches",
    "unchanged",
])
insert_masked = titled_panel(225, 1214, 163, 148, "Insert masked", [
    "32 predicted",
    "centre patches only",
    "inference merge",
])
unpatch = titled_panel(407, 1214, 162, 166, "Unpatchify", [
    "clean / inpainted",
    "log-mag [B,32,224]",
    "expm1 · clamp ≥ 0",
])
polar = titled_panel(588, 1214, 162, 166, "Polar synthesis", [
    "predicted magnitude",
    "+ cached noisy phase",
    "pad → [B,33,225]",
])
istft = titled_panel(769, 1214, 112, 166, "Hann iSTFT", [
    "same window",
    "length 3584",
], title_size=11.8)
clean = titled_panel(899, 1214, 75, 222, "CLEAN", [
    "EEG",
    "slot 3",
    "1536:2048",
    "[B,1,S]",
    "S=512",
], title_size=10.7)
d.add(orig_visible, insert_masked, unpatch, polar, istft, clean)

d.add_edge(
    edge([(visible["x"] - 3, 660), (14, 660), (14, 1200), (129, 1200),
          (129, orig_visible["y"] - 3)]),
    edge([(902, prediction["y"] + prediction["h"] + 3), (902, 1168), (306, 1168), (306, insert_masked["y"] - 3)],
         label=("masked prediction only", 605, 1155)),
    edge([(orig_visible["x"] + orig_visible["w"] + 3, 1300), (insert_masked["x"] - 3, 1300)]),
    edge([(insert_masked["x"] + insert_masked["w"] + 3, 1300), (unpatch["x"] - 3, 1300)]),
    edge([(unpatch["x"] + unpatch["w"] + 3, 1300), (polar["x"] - 3, 1300)]),
    edge([(polar["x"] + polar["w"] + 3, 1300), (istft["x"] - 3, 1300)]),
    edge([(istft["x"] + istft["w"] + 3, 1300), (clean["x"] - 3, 1300)]),
)

# Cached ORIGINAL/noisy phase: a labelled side carrier from STFT/split to polar
# only.  It deliberately never touches patchification, the encoder, or decoder.
d.add_edge(
    edge([(split["x"] + split["w"] + 3, 330), (774, 330), (774, 432),
          (834, 432), (834, 1194), (669, 1194), (669, polar["y"] - 3)],
         label=("cached ORIGINAL / noisy phase · never enters ViT", 828, 1179)),
)
d.add(
    multi_note(43, 1395, 707, [
        "EEG CAVEAT · magnitude-only prediction reuses noisy input phase",
        "Phase is not reconstructed; waveform synthesis combines predicted magnitude with cached noisy phase.",
    ], accent=True, size=10.5),
    chip(816, 1417, "centre slice 1536:2048", strong=True),
)


# ---------------------------------------------------------------------------
# Training-only objective and the outside-model FACETpy adapter/correction tail
# ---------------------------------------------------------------------------
train_pred = layer_row(44, 1570, 126, 54, "masked prediction", "[B,32,64]")
train_target = layer_row(44, 1640, 126, 54, "masked target", "optional patch norm ON")
train_loss = titled_panel(208, 1583, 247, 96, "Masked-centre MSE only", [
    "compare only 32 structural centre patches",
    "not an all-patch reconstruction loss",
], title_size=12.1)
d.add(train_pred, train_target, train_loss)
d.add_edge(
    edge([(170, 1597), (205, 1597)], marker_end="arrow"),
    edge([(170, 1667), (190, 1667), (190, 1650), (205, 1650)], marker_end="arrow"),
)

resample = layer_row(520, 1574, 129, 56, "resample clean", "S → native centre_len")
artifact = layer_row(675, 1574, 129, 56, "derive artifact", "noisy centre − clean")
subtract = layer_row(830, 1574, 128, 56, "FACETpy correction", "subtract artifact")
d.add(resample, artifact, subtract)
d.add_edge(
    edge([(clean["cx"], clean["y"] + clean["h"] + 3), (clean["cx"], 1512),
          (584, 1512), (584, resample["y"] - 3)], label=("model output = clean", 772, 1498)),
    edge([(resample["x"] + resample["w"] + 3, 1602), (artifact["x"] - 3, 1602)]),
    edge([(artifact["x"] + artifact["w"] + 3, 1602), (subtract["x"] - 3, 1602)]),
)
d.add(
    multi_note(520, 1650, 438, [
        "The adapter derives the artifact; the neural model predicts clean EEG.",
        "FACETpy then subtracts that derived artifact from the noisy centre epoch.",
    ], accent=True, size=10.2),
)


# Compact source/config legend and the built-in footer complete the canvas.
d.add(
    multi_note(22, 1742, 956, [
        "Verified full-run contract",
        "B=batch · 7 epochs · one channel · S=512 · STFT crop 32×224 · 112 patches · 80 visible / 32 masked",
        "Encoder: 6 blocks, d=192, 6 heads, MLP=768, dropout 0  ·  Decoder: 2 blocks, d=96, 4 heads, MLP=384",
    ], accent=True, size=10.5),
)


OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
d.render_png(str(OUT_PNG), svg_path=str(OUT_SVG), width=1280)

"""Build detailed diagrams for the selected FACETpy thesis variants.

The diagrams are deliberately derived from the deployment implementations and
the selected Run-8 grid records, rather than from paper abstracts.  Each profile describes the network blocks; the catalog supplies its selected
configuration, artifact identity and canonical measured result.  It writes editable SVG plus PNG previews and a provenance manifest.

Run: ``uv run python tools/diagrams/build_selected_variants.py``
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from facetpy_svg import C, Diagram, capsule, edge, pill, text  # noqa: E402

OUT = ROOT / "output/thesis_figures/selected_variants"
W, H = 1000, 760


def box(d: Diagram, x: float, y: float, w: float, title: str, detail: str = "", terminal=False):
    """A compact two-line FACETpy block, reusable across the selected series."""
    h = 74
    node = (capsule if terminal else pill)(x, y, w, h, "")
    node["svg"] = re.sub(r"<text\\b.*?</text>", "", node["svg"])
    fill = C["header_fg"] if terminal else C["ink"]
    node["svg"] += text(
        x + w / 2,
        y + 31,
        title,
        size=min(15.5, (w - 18) / max(len(title) * 0.57, 1)),
        weight=650,
        anchor="middle",
        fill=fill,
    )
    if detail:
        node["svg"] += text(
            x + w / 2, y + 54, detail, size=11.2, anchor="middle", fill=C["header_fg"] if terminal else C["slate"]
        )
    d.add(node)


def wire(d: Diagram, pts, *, dashed=False):
    fragment = edge(pts)
    if dashed:
        fragment = fragment.replace("<polyline ", '<polyline stroke-dasharray="7 6" ')
    d.add_edge(fragment)


def title(d: Diagram, name: str, selection: str, summary: str):
    d.add(
        text(48, 43, name, size=25, weight=700, fill=C["ink"]),
        f'<rect x="48" y="53" width="250" height="3" rx="1.5" fill="url(#fp-header)"/>',
        text(48, 81, selection, size=12.5, weight=650, fill=C["blue"]),
        text(48, 104, summary, size=12.1, fill=C["slate"]),
    )


def footer(d: Diagram, config: str, result: str, note: str = ""):
    y = 653
    d.add(
        f'<rect x="48" y="{y}" width="904" height="72" rx="12" fill="{C["tint"]}" stroke="{C["slate"]}" stroke-opacity="0.28"/>'
    )
    d.add(text(65, y + 25, "Selected configuration", size=11.5, weight=700, fill=C["slate"]))
    d.add(text(65, y + 47, config, size=12.2, fill=C["ink"]))
    d.add(text(572, y + 25, "Observed result", size=11.5, weight=700, fill=C["slate"]))
    d.add(text(572, y + 47, result, size=12.2, fill=C["ink"]))
    if note:
        d.add(text(65, y + 66, note, size=10.5, fill=C["slate"]))


def common(name, selection, summary, blocks, config, result, *, skips=(), note=""):
    d = Diagram(H, grid=False, footer_text=None, background=True)
    title(d, name, selection, summary)
    # Main flow runs across two rows.  Each profile supplies coordinates and verified labels.
    for item in blocks:
        box(d, *item)
    for a, b in zip(blocks, blocks[1:]):
        if a[1] == b[1] and abs((a[0] + a[2]) - b[0]) < 100:
            wire(d, [(a[0] + a[2] + 4, a[1] + 37), (b[0] - 4, b[1] + 37)])
    for pts, dashed in skips:
        wire(d, pts, dashed=dashed)
    footer(d, config, result, note)
    return d


PROFILES = [
    dict(
        slug="dpae_selected",
        name="Dual-Pathway Autoencoder (DPAE)",
        summary="Local dilated and global wide-kernel pathways fuse before a transposed-convolution decoder.",
        blocks=[
            (48, 160, 142, "Noisy centre", "1 × 512", True),
            (220, 160, 150, "Local path", "dilations 1·2·4·8"),
            (400, 160, 150, "Global path", "kernels 15·11·7"),
            (580, 160, 128, "Concat + 1×1", "fusion"),
            (738, 160, 160, "Decoder", "2× deconv + head"),
            (770, 315, 128, "Artifact", "1 × 512", True),
        ],
        skips=[([(295, 234), (295, 270), (818, 270), (818, 311)], True), ([(655, 234), (655, 270)], False)],
    ),
    dict(
        slug="cascaded_dae_selected",
        name="Cascaded Denoising Autoencoder",
        summary="Two fully connected denoisers estimate an artifact, form a residual, then refine it.",
        blocks=[
            (48, 160, 142, "Noisy centre", "1 × 512", True),
            (220, 160, 170, "DAE stage 1", "512→128→512"),
            (435, 160, 72, "−", "residual", True),
            (550, 160, 170, "DAE stage 2", "512→128→512"),
            (770, 160, 72, "+", "sum", True),
            (860, 160, 92, "Artifact", "1 × 512", True),
        ],
        skips=[([(305, 234), (305, 300), (806, 300), (806, 234)], True)],
    ),
    dict(
        slug="cascaded_context_dae_selected",
        name="Cascaded Context DAE",
        summary="Stage one maps seven epochs to the centre artifact; stage two refines the centre residual.",
        blocks=[
            (48, 160, 142, "Context", "7 × 512", True),
            (220, 160, 170, "Context DAE 1", "3584→512"),
            (435, 160, 72, "−", "centre residual", True),
            (550, 160, 170, "Context DAE 2", "3584→512"),
            (770, 160, 72, "+", "sum", True),
            (860, 160, 92, "Artifact", "1 × 512", True),
        ],
        skips=[
            ([(120, 234), (120, 300), (300, 300), (300, 234)], True),
            ([(305, 234), (305, 300), (806, 300), (806, 234)], True),
        ],
    ),
    dict(
        slug="conv_tasnet_selected",
        name="Conv-TasNet",
        summary="A learned encoder, dilated temporal-convolution separator, masks, and shared decoder separate two sources.",
        blocks=[
            (48, 160, 142, "Noisy centre", "1 × 512", True),
            (220, 160, 130, "Encoder", "256 filters, k=16"),
            (382, 160, 132, "1×1 bottleneck", "128 channels"),
            (548, 160, 164, "TCN separator", "2 × 8 blocks"),
            (746, 160, 138, "Two masks", "sigmoid"),
            (765, 315, 138, "Decoder", "transpose conv"),
            (850, 470, 102, "Artifact", "source 1", True),
        ],
        skips=[
            ([(285, 234), (285, 278), (834, 278), (834, 311)], True),
            ([(815, 234), (815, 278), (835, 278), (835, 311)], False),
            ([(834, 389), (834, 465)], False),
        ],
    ),
    dict(
        slug="demucs_selected",
        name="Demucs",
        summary="Four convolutional encoder/decoder levels, summed skip routes, and a two-layer BiLSTM process seven epochs jointly.",
        blocks=[
            (48, 160, 142, "Context", "7 × 512", True),
            (220, 160, 130, "Encoder 1–4", "k=8, stride 4"),
            (385, 160, 132, "GLU stages", "channels 64→512"),
            (552, 160, 128, "BiLSTM", "2 layers"),
            (714, 160, 142, "Projection", "1024→512"),
            (740, 315, 150, "Decoder 4–1", "GLU + transpose conv"),
            (840, 470, 112, "Artifact", "centre 512", True),
        ],
        skips=[
            ([(286, 234), (286, 278), (815, 278), (815, 311)], True),
            ([(450, 234), (450, 278)], True),
            ([(815, 389), (815, 465)], False),
        ],
    ),
    dict(
        slug="sepformer_selected",
        name="SepFormer",
        summary="Encoded context is chunked; dual-path Transformer blocks alternate intra- and inter-chunk attention before masking.",
        blocks=[
            (48, 160, 142, "Context", "7 × 512", True),
            (220, 160, 130, "Encoder", "128 channels"),
            (382, 160, 125, "Segment", "chunks 64"),
            (540, 160, 154, "Intra Transformer", "4 layers, 4 heads"),
            (728, 160, 154, "Inter Transformer", "4 layers, 4 heads"),
            (765, 315, 138, "Mask + decoder", "ReLU + transposed conv"),
            (850, 470, 102, "Artifact", "centre 512", True),
        ],
        skips=[
            ([(447, 234), (447, 278), (610, 278), (610, 234)], True),
            ([(805, 234), (805, 278), (834, 278), (834, 311)], True),
            ([(834, 389), (834, 465)], False),
        ],
    ),
    dict(
        slug="denoise_mamba_selected",
        name="DenoiseMamba",
        summary="Projected signal passes through convolutional selective state-space blocks with residual routes before an artifact head.",
        blocks=[
            (48, 160, 142, "Context", "7 × 512", True),
            (220, 160, 128, "Input projection", "Conv1d"),
            (382, 160, 140, "Mamba block ×N", "in-proj + gate"),
            (560, 160, 132, "Depthwise conv", "local mixing"),
            (730, 160, 132, "Selective SSM", "parallel scan"),
            (770, 315, 128, "Output projection", "Conv1d head"),
            (850, 470, 102, "Artifact", "centre 512", True),
        ],
        skips=[
            ([(452, 234), (452, 278), (635, 278), (635, 234)], True),
            ([(795, 234), (795, 278), (834, 278), (834, 311)], True),
            ([(834, 389), (834, 465)], False),
        ],
    ),
    dict(
        slug="d4pm_selected",
        name="D4PM",
        summary="A conditional diffusion model iterates a Transformer noise predictor through reverse denoising steps.",
        blocks=[
            (48, 160, 142, "Noisy centre", "1 × 512", True),
            (220, 160, 124, "Noise level", "sinusoidal embed"),
            (376, 160, 134, "x / condition", "two Conv1d stems"),
            (542, 160, 154, "Transformer ×2", "2 heads, d=128"),
            (730, 160, 132, "ε head", "FiLM conditioned"),
            (765, 315, 138, "Reverse update", "t = 200…0"),
            (850, 470, 102, "Artifact", "1 × 512", True),
        ],
        skips=[
            ([(282, 234), (282, 278), (618, 278), (618, 234)], True),
            ([(834, 234), (834, 278), (834, 311)], False),
            ([(834, 389), (834, 465)], False),
            ([(835, 465), (950, 465), (950, 125), (734, 125)], True),
        ],
    ),
    dict(
        slug="nested_gan_selected",
        name="Nested GAN",
        summary="A centre-epoch spectral Restormer estimates an initial artifact; a temporal U-Net refines it from the residual context.",
        blocks=[
            (48, 160, 142, "Context", "7 × 512", True),
            (220, 160, 130, "Centre select", "epoch 4"),
            (382, 160, 128, "STFT", "64 / 16 / 64"),
            (542, 160, 154, "Restormer ×4", "48 ch, 4 heads"),
            (730, 160, 132, "iSTFT artifact", "centre estimate"),
            (765, 315, 138, "Residual U-Net", "3 down / 3 up"),
            (850, 470, 102, "Artifact", "sum", True),
        ],
        skips=[
            ([(118, 234), (118, 278), (834, 278), (834, 311)], True),
            ([(795, 234), (795, 278), (834, 278), (834, 311)], False),
            ([(834, 389), (834, 465)], False),
        ],
    ),
    dict(
        slug="vit_spectrogram_selected",
        name="ViT-Spectrogram",
        summary="STFT magnitude is patch-embedded, processed by a masked Vision Transformer, and applied as a complex spectrum mask.",
        blocks=[
            (48, 160, 142, "Noisy centre", "1 × 512", True),
            (220, 160, 128, "STFT", "64 / 16 / 64"),
            (382, 160, 132, "Log magnitude", "32 × 224"),
            (548, 160, 148, "4×16 patches", "tokens + positions"),
            (730, 160, 132, "ViT encoder ×6", "d=192, 6 heads"),
            (760, 315, 142, "Complex mask + iSTFT", "centre crop"),
            (850, 470, 102, "Clean", "1 × 512", True),
        ],
        skips=[([(286, 234), (286, 278), (832, 278), (832, 311)], True), ([(834, 389), (834, 465)], False)],
    ),
    dict(
        slug="dhct_gan_selected",
        name="DHCT-GAN",
        summary="A convolutional U-Net uses local/global attention in encoder stages and separate clean/artifact decoder paths with a gate.",
        blocks=[
            (48, 160, 142, "Noisy centre", "1 × 512", True),
            (220, 160, 128, "CNN stem", "7×1 then 3×1"),
            (382, 160, 142, "Encoder ×4", "CNN + local/global attn"),
            (556, 160, 132, "Bottleneck", "depth 4"),
            (696, 160, 112, "Clean decoder", "skip fusion"),
            (828, 160, 124, "Artifact decoder", "skip fusion"),
            (790, 315, 132, "Gate + heads", "1×1 outputs"),
            (850, 470, 102, "Artifact", "1 × 512", True),
        ],
        skips=[
            ([(450, 234), (450, 278), (780, 278), (780, 311)], True),
            ([(615, 234), (615, 278), (895, 278), (895, 311)], True),
            ([(752, 234), (752, 278), (856, 278), (856, 311)], False),
            ([(890, 234), (890, 311)], False),
            ([(856, 389), (856, 465)], False),
        ],
        note="The incomplete Run 8 grid is not used for selected-variant claims.",
    ),
    dict(
        slug="ic_unet_selected",
        name="IC-U-Net",
        summary="ICA projects 30-channel context into components; a 1-D U-Net predicts clean components before inverse ICA and centre extraction.",
        blocks=[
            (48, 160, 142, "Context", "30 × 7 × 512", True),
            (220, 160, 128, "ICA", "fit components"),
            (382, 160, 132, "DoubleConv", "64 channels"),
            (548, 160, 130, "Down ×3", "pool + DoubleConv"),
            (714, 160, 122, "Up ×3", "skips + upconv"),
            (860, 160, 92, "Inverse ICA", "clean full"),
            (820, 315, 132, "Centre crop", "clean→artifact"),
            (850, 470, 102, "Artifact", "30 × 512", True),
        ],
        skips=[
            ([(448, 234), (448, 278), (775, 278), (775, 234)], True),
            ([(614, 234), (614, 278), (780, 278), (780, 234)], True),
            ([(884, 234), (884, 311)], False),
            ([(884, 389), (884, 465)], False),
        ],
        note="Diagnostic only: not retained as a valid correction result.",
    ),
    dict(
        slug="st_gnn_selected",
        name="ST-GNN",
        summary="Temporal GLU and Chebyshev graph convolution operate on a 30-electrode graph, then a 1×1 head extracts the centre epoch.",
        blocks=[
            (48, 160, 142, "Context", "30 × 7 × 512", True),
            (220, 160, 128, "Temporal GLU", "kernel 3"),
            (382, 160, 142, "ChebConv", "K=3; kNN graph"),
            (556, 160, 128, "ST-Conv ×2", "residual blocks"),
            (720, 160, 132, "Temporal GLU", "kernel 3"),
            (860, 160, 92, "1×1 head", "full sequence"),
            (820, 315, 132, "Centre crop", "epoch 4"),
            (850, 470, 102, "Artifact", "30 × 512", True),
        ],
        skips=[
            ([(452, 234), (452, 278), (620, 278), (620, 234)], True),
            ([(620, 234), (620, 278), (780, 278), (780, 234)], True),
            ([(884, 234), (884, 311)], False),
            ([(884, 389), (884, 465)], False),
        ],
        note="Diagnostic only: low residual is invalidated by the discontinuity check.",
    ),
]


# The figure's settings and result labels come from the catalog and canonical
# tables. The network topology below is the sole editable drawing source.
SELECTED = {
    "dpae": "deployment_dpae",
    "cascaded_dae": "deployment_cascaded_dae",
    "cascaded_context_dae": "deployment_cascaded_context_dae",
    "conv_tasnet": "deployment_conv_tasnet",
    "demucs": "run8_demucs_lr0_001_ic64_sisdr0_s42",
    "sepformer": "deployment_sepformer",
    "denoise_mamba": "deployment_denoise_mamba",
    "d4pm": "training_run7_d4pm",
    "nested_gan": "run8_nested_gan_lr0_00015_ch48_sisdr1_s42",
    "vit_spectrogram": "run8_vit_spectrogram_lr0_001_dim192_sisdr1_s42",
    "dhct_gan": "run8_dhct_gan_lr0_0001_bc8_sisdr0_s42",
    "ic_unet": "deployment_ic_unet",
    "st_gnn": "deployment_st_gnn",
}


def selected_profiles():
    import csv, copy, yaml

    sys.path.insert(0, str(ROOT))
    from masterthesis_guide.reproduce import load_catalog

    c = load_catalog()
    with (ROOT / c["results"]["table_phase2_selected_variant_results"]["path"]).open() as f:
        phase2 = {r["arm"]: r for r in csv.DictReader(f)}
    with (ROOT / c["results"]["table_phase3_before_after"]["path"]).open() as f:
        phase3 = {r["family"]: r for r in csv.DictReader(f)}
    profiles = copy.deepcopy(PROFILES)
    for p in profiles:
        family = p["slug"].removesuffix("_selected")
        eid = SELECTED[family]
        e = c["experiments"][eid]
        cfg = yaml.safe_load((ROOT / e["config"]).read_text())
        p["selection"] = eid
        p["sources"] = [
            e["config"],
            c["models"][e["model"]]["module"].replace("facet.", "src/facet/").replace(".", "/") + "/training.py",
        ]
        training = cfg.get("training", {})
        p["config"] = (
            f"lr {training.get('learning_rate', 'recorded')} · batch {training.get('batch_size', 'recorded')} · see resolved config"
        )
        row = phase2.get("dhct_gan_context_7_epoch" if family == "dhct_gan" else f"{family}_deployment")
        p["result"] = (
            f"Recorded pipeline: {float(row['ga_rest_uv']):.3f} µV" if row else "No recorded deployment result"
        )
        if family in {"demucs", "nested_gan", "vit_spectrogram"}:
            label = {"demucs": "Demucs", "nested_gan": "Nested GAN", "vit_spectrogram": "ViT-Spectrogram"}[family]
            p["result"] = f"Selected grid: {float(phase3[label]['best_grid_ga_rest_uv']):.3f} µV"
        p["note"] = "Invalid pipeline waveform; retain as a negative result." if family in {"ic_unet", "st_gnn"} else ""
        if family == "dhct_gan":
            p["summary"] = (
                "The selected epoch-axis wrapper passes seven concatenated epochs through the CNN/Transformer network."
            )
            p["blocks"][0] = (*p["blocks"][0][:3], "Context", "7 × 512", True)
            p["note"] = "Recovered seven-context trial; the DHCT-GAN grid was incomplete."
        if family == "denoise_mamba":
            p["blocks"][0] = (*p["blocks"][0][:3], "Noisy centre", "1 × 512", True)
        if family == "vit_spectrogram":
            p["blocks"][0] = (*p["blocks"][0][:3], "Context", "7 × 512", True)
            p["blocks"][-1] = (*p["blocks"][-1][:3], "Artifact", "noisy − clean", True)
            p["note"] = "The core reconstructs clean EEG; the deployment wrapper returns its complement."
        if family == "d4pm":
            p["note"] = "Training configuration retained; no valid Phase-2 deployment score is available."
        yield p


def render(profile):
    d = common(
        profile["name"],
        profile["selection"],
        profile["summary"],
        profile["blocks"],
        profile["config"],
        profile["result"],
        skips=profile.get("skips", ()),
        note=profile.get("note", ""),
    )
    svg_path = OUT / f"{profile['slug']}.svg"
    png_path = OUT / f"{profile['slug']}.png"
    svg = d.render_svg().replace('width="1000" height="760"', 'width="180mm" height="136.8mm"', 1)
    svg_path.write_text(svg)
    subprocess.run(["rsvg-convert", "-w", "2700", str(svg_path), "-o", str(png_path)], check=True)
    return svg_path, png_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"builder": str(Path(__file__).relative_to(ROOT)), "series": "selected thesis variants", "diagrams": []}
    for p in selected_profiles():
        svg, png = render(p)
        manifest["diagrams"].append(
            {
                "slug": p["slug"],
                "title": p["name"],
                "selection": p["selection"],
                "configuration": p["config"],
                "observed_result": p["result"],
                "svg": svg.name,
                "png": png.name,
                "sources": p["sources"],
                "sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (svg, png)},
            }
        )
    (OUT / "provenance_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(PROFILES)} selected-variant diagrams to {OUT}")


if __name__ == "__main__":
    main()

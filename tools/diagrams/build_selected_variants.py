"""Build detailed diagrams for the selected FACETpy thesis variants.

The diagrams are deliberately derived from the deployment implementations and
the selected Run-8 grid records, rather than from paper abstracts.  This file
extends the existing ``tools/diagrams/build_run7_compact_*.py`` visual language
with a denser, source-oriented block sequence for the variants used in the
thesis.  It writes editable SVG plus PNG previews and a provenance manifest.

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
sys.path.insert(0, str(ROOT / "skills/facetpy-diagram/assets"))
from facetpy_svg import C, Diagram, capsule, edge, pill, text  # noqa: E402

OUT = ROOT / "output/thesis_figures/selected_variants"
W, H = 1000, 760


def box(d: Diagram, x: float, y: float, w: float, title: str, detail: str = "", terminal=False):
    """A compact two-line FACETpy block, reusable across the selected series."""
    h = 74
    node = (capsule if terminal else pill)(x, y, w, h, "")
    node["svg"] = re.sub(r"<text\\b.*?</text>", "", node["svg"])
    fill = C["header_fg"] if terminal else C["ink"]
    node["svg"] += text(x + w / 2, y + 31, title, size=15.5, weight=650, anchor="middle", fill=fill)
    if detail:
        node["svg"] += text(x + w / 2, y + 54, detail, size=11.2, anchor="middle", fill=C["header_fg"] if terminal else C["slate"])
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
    d.add(f'<rect x="48" y="{y}" width="904" height="72" rx="12" fill="{C["tint"]}" stroke="{C["slate"]}" stroke-opacity="0.28"/>')
    d.add(text(65, y + 25, "Selected configuration", size=11.5, weight=700, fill=C["slate"]))
    d.add(text(65, y + 47, config, size=12.2, fill=C["ink"]))
    d.add(text(572, y + 25, "Observed result", size=11.5, weight=700, fill=C["slate"]))
    d.add(text(572, y + 47, result, size=12.2, fill=C["ink"]))
    if note:
        d.add(text(65, y + 66, note, size=10.5, fill=C["slate"]))


def common(name, selection, summary, blocks, config, result, *, skips=(), note=""):
    d = Diagram(H, grid=False, footer_text=None)
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
        slug="dpae_selected", name="Dual-Pathway Autoencoder (DPAE)", selection="Run 7 deployment edition", summary="Local dilated and global wide-kernel pathways fuse before a transposed-convolution decoder.",
        blocks=[(48,160,142,"Noisy centre", "1 × 512", True),(220,160,150,"Local path", "dilations 1·2·4·8"),(400,160,150,"Global path", "kernels 15·11·7"),(580,160,128,"Concat + 1×1", "fusion"),(738,160,160,"Decoder", "2× deconv + head"),(770,315,128,"Artifact", "1 × 512", True)],
        skips=[([(295,234),(295,270),(818,270),(818,311)], True), ([(655,234),(655,270)], False)],
        config="base filters 32 · latent filters 128 · artifact target", result="Run 7 pipeline: 4.13 µV GA residual", sources=["src/facet/models/masterthesis/dpae/deployment/training.py","src/facet/models/masterthesis/dpae/deployment/training_niazy_proof_fit_seed45.yaml"]),
    dict(
        slug="cascaded_dae_selected", name="Cascaded Denoising Autoencoder", selection="Run 7 deployment edition", summary="Two fully connected denoisers estimate an artifact, form a residual, then refine it.",
        blocks=[(48,160,142,"Noisy centre", "1 × 512", True),(220,160,170,"DAE stage 1", "512→128→512"),(435,160,72,"−", "residual", True),(550,160,170,"DAE stage 2", "512→128→512"),(770,160,72,"+", "sum", True),(860,160,92,"Artifact", "1 × 512", True)],
        skips=[([(305,234),(305,300),(806,300),(806,234)], True)],
        config="dropout 0.2 · batch 128 · artifact target", result="Run 7 pipeline: 4.05 µV GA residual", sources=["src/facet/models/masterthesis/cascaded_dae/deployment/training.py","src/facet/models/masterthesis/cascaded_dae/deployment/training_niazy_proof_fit_seed45.yaml"]),
    dict(
        slug="cascaded_context_dae_selected", name="Cascaded Context DAE", selection="Run 7 deployment edition", summary="Stage one maps seven epochs to the centre artifact; stage two refines the centre residual.",
        blocks=[(48,160,142,"Context", "7 × 512", True),(220,160,170,"Context DAE 1", "3584→512"),(435,160,72,"−", "centre residual", True),(550,160,170,"Context DAE 2", "3584→512"),(770,160,72,"+", "sum", True),(860,160,92,"Artifact", "1 × 512", True)],
        skips=[([(120,234),(120,300),(300,300),(300,234)], True), ([(305,234),(305,300),(806,300),(806,234)], True)],
        config="7-epoch context · dropout 0.2 · artifact target", result="Run 7 pipeline: 3.08 µV GA residual", sources=["src/facet/models/masterthesis/cascaded_context_dae/deployment/training.py","src/facet/models/masterthesis/cascaded_context_dae/deployment/training_niazy_proof_fit_seed45.yaml"]),
    dict(
        slug="conv_tasnet_selected", name="Conv-TasNet", selection="Run 7 deployment edition", summary="A learned encoder, dilated temporal-convolution separator, masks, and shared decoder separate two sources.",
        blocks=[(48,160,142,"Noisy centre", "1 × 512", True),(220,160,130,"Encoder", "256 filters, k=16"),(382,160,132,"1×1 bottleneck", "128 channels"),(548,160,164,"TCN separator", "2 × 8 blocks"),(746,160,138,"Two masks", "sigmoid"),(765,315,138,"Decoder", "transpose conv"),(850,470,102,"Artifact", "source 1", True)],
        skips=[([(285,234),(285,278),(834,278),(834,311)], True), ([(815,234),(815,278),(835,278),(835,311)], False), ([(834,389),(834,465)], False)],
        config="n_sources 2 · masks sigmoid · artifact source index 1", result="Run 7 pipeline: 3.39 µV GA residual", sources=["src/facet/models/masterthesis/conv_tasnet/deployment/training.py","src/facet/models/masterthesis/conv_tasnet/deployment/training_niazy_proof_fit_seed45.yaml"]),
    dict(
        slug="demucs_selected", name="Demucs", selection="Run 8 selected grid configuration", summary="Four convolutional encoder/decoder levels, summed skip routes, and a two-layer BiLSTM process seven epochs jointly.",
        blocks=[(48,160,142,"Context", "7 × 512", True),(220,160,130,"Encoder 1–4", "k=8, stride 4"),(385,160,132,"GLU stages", "channels 64→512"),(552,160,128,"BiLSTM", "2 layers"),(714,160,142,"Projection", "1024→512"),(740,315,150,"Decoder 4–1", "GLU + transpose conv"),(840,470,112,"Artifact", "centre 512", True)],
        skips=[([(286,234),(286,278),(815,278),(815,311)], True), ([(450,234),(450,278)], True), ([(815,389),(815,465)], False)],
        config="lr 1e−3 · initial channels 64 · SI-SDR weight 0", result="Run 8: 1.89 µV GA residual", sources=["src/facet/models/masterthesis/demucs/deployment/training.py","src/facet/models/masterthesis/demucs/deployment/training_niazy_proof_fit.yaml","output/run8_fetched/gridpod4/grids/run8/grid_demucs_screen.json"]),
    dict(
        slug="sepformer_selected", name="SepFormer", selection="Run 7 deployment edition", summary="Encoded context is chunked; dual-path Transformer blocks alternate intra- and inter-chunk attention before masking.",
        blocks=[(48,160,142,"Context", "7 × 512", True),(220,160,130,"Encoder", "128 channels"),(382,160,125,"Segment", "chunks 64"),(540,160,154,"Intra Transformer", "4 layers, 4 heads"),(728,160,154,"Inter Transformer", "4 layers, 4 heads"),(765,315,138,"Mask + decoder", "ReLU + transposed conv"),(850,470,102,"Artifact", "centre 512", True)],
        skips=[([(447,234),(447,278),(610,278),(610,234)], True), ([(805,234),(805,278),(834,278),(834,311)], True), ([(834,389),(834,465)], False)],
        config="2 dual-path blocks · dFF 256 · 7-epoch context", result="Run 7 pipeline: 2.43 µV GA residual", sources=["src/facet/models/masterthesis/sepformer/deployment/training.py","src/facet/models/masterthesis/sepformer/deployment/training_niazy_proof_fit_seed45.yaml"]),
    dict(
        slug="denoise_mamba_selected", name="DenoiseMamba", selection="Run 7 deployment edition", summary="Projected signal passes through convolutional selective state-space blocks with residual routes before an artifact head.",
        blocks=[(48,160,142,"Context", "7 × 512", True),(220,160,128,"Input projection", "Conv1d"),(382,160,140,"Mamba block ×N", "in-proj + gate"),(560,160,132,"Depthwise conv", "local mixing"),(730,160,132,"Selective SSM", "parallel scan"),(770,315,128,"Output projection", "Conv1d head"),(850,470,102,"Artifact", "centre 512", True)],
        skips=[([(452,234),(452,278),(635,278),(635,234)], True), ([(795,234),(795,278),(834,278),(834,311)], True), ([(834,389),(834,465)], False)],
        config="7-epoch context configuration · artifact target", result="Run 7 pipeline: 4.08 µV GA residual", sources=["src/facet/models/masterthesis/denoise_mamba/deployment/training.py","src/facet/models/masterthesis/denoise_mamba/deployment/training_niazy_proof_fit_ctx_epochs_seed45.yaml"]),
    dict(
        slug="d4pm_selected", name="D4PM", selection="Run 7 deployment edition", summary="A conditional diffusion model iterates a Transformer noise predictor through reverse denoising steps.",
        blocks=[(48,160,142,"Noisy centre", "1 × 512", True),(220,160,124,"Noise level", "sinusoidal embed"),(376,160,134,"x / condition", "two Conv1d stems"),(542,160,154,"Transformer ×2", "2 heads, d=128"),(730,160,132,"ε head", "FiLM conditioned"),(765,315,138,"Reverse update", "t = 200…0"),(850,470,102,"Artifact", "1 × 512", True)],
        skips=[([(282,234),(282,278),(618,278),(618,234)], True), ([(834,234),(834,278),(834,311)], False), ([(834,389),(834,465)], False), ([(835,465),(950,465),(950,125),(734,125)], True)],
        config="200 diffusion steps · d_model 128 · waveform fraction 0.25", result="Run 7 pipeline: 2.82 µV GA residual", sources=["src/facet/models/masterthesis/d4pm/deployment/training.py","src/facet/models/masterthesis/d4pm/deployment/training_niazy_proof_fit_seed45.yaml"]),
    dict(
        slug="nested_gan_selected", name="Nested GAN", selection="Run 8 selected grid configuration", summary="A centre-epoch spectral Restormer estimates an initial artifact; a temporal U-Net refines it from the residual context.",
        blocks=[(48,160,142,"Context", "7 × 512", True),(220,160,130,"Centre select", "epoch 4"),(382,160,128,"STFT", "64 / 16 / 64"),(542,160,154,"Restormer ×4", "48 ch, 4 heads"),(730,160,132,"iSTFT artifact", "centre estimate"),(765,315,138,"Residual U-Net", "3 down / 3 up"),(850,470,102,"Artifact", "sum", True)],
        skips=[([(118,234),(118,278),(834,278),(834,311)], True), ([(795,234),(795,278),(834,278),(834,311)], False), ([(834,389),(834,465)], False)],
        config="lr 1.5e−4 · inner channels 48 · SI-SDR weight 1", result="Run 8: 0.974 µV GA residual", sources=["src/facet/models/masterthesis/nested_gan/deployment/training.py","src/facet/models/masterthesis/nested_gan/deployment/training_niazy_proof_fit.yaml","output/run8_fetched/gridpod1/grids/run8/grid_nested_gan_screen.json"]),
    dict(
        slug="vit_spectrogram_selected", name="ViT-Spectrogram", selection="Run 8 selected grid configuration", summary="STFT magnitude is patch-embedded, processed by a masked Vision Transformer, and applied as a complex spectrum mask.",
        blocks=[(48,160,142,"Noisy centre", "1 × 512", True),(220,160,128,"STFT", "64 / 16 / 64"),(382,160,132,"Log magnitude", "32 × 224"),(548,160,148,"4×16 patches", "tokens + positions"),(730,160,132,"ViT encoder ×6", "d=192, 6 heads"),(760,315,142,"Complex mask + iSTFT", "centre crop"),(850,470,102,"Clean", "1 × 512", True)],
        skips=[([(286,234),(286,278),(832,278),(832,311)], True), ([(834,389),(834,465)], False)],
        config="lr 1e−3 · embed dim 192 · SI-SDR weight 1", result="Run 8: 2.222 µV GA residual", sources=["src/facet/models/masterthesis/vit_spectrogram/deployment/training.py","src/facet/models/masterthesis/vit_spectrogram/deployment/training_niazy_proof_fit.yaml","output/run8_fetched/gridpod2/grids/run8/grid_vit_spectrogram_screen.json"]),
    dict(
        slug="dhct_gan_selected", name="DHCT-GAN", selection="Run 7 deployment edition · Run 8 search incomplete", summary="A convolutional U-Net uses local/global attention in encoder stages and separate clean/artifact decoder paths with a gate.",
        blocks=[(48,160,142,"Noisy centre", "1 × 512", True),(220,160,128,"CNN stem", "7×1 then 3×1"),(382,160,142,"Encoder ×4", "CNN + local/global attn"),(556,160,132,"Bottleneck", "depth 4"),(696,160,112,"Clean decoder", "skip fusion"),(828,160,124,"Artifact decoder", "skip fusion"),(790,315,132,"Gate + heads", "1×1 outputs"),(850,470,102,"Artifact", "1 × 512", True)],
        skips=[([(450,234),(450,278),(780,278),(780,311)], True), ([(615,234),(615,278),(895,278),(895,311)], True), ([(752,234),(752,278),(856,278),(856,311)], False), ([(890,234),(890,311)], False), ([(856,389),(856,465)], False)],
        config="base channels 16 · depth 4 · heads 4 · Run 8: 2/27 only", result="Run 7: 2.44 µV GA residual", note="The incomplete Run 8 grid is not used for selected-variant claims.", sources=["src/facet/models/masterthesis/dhct_gan/deployment/training.py","src/facet/models/masterthesis/dhct_gan/deployment/training_niazy_proof_fit_seed45.yaml","output/run8_fetched/gridpod3/grids/run8/grid_dhct_gan_screen.json"]),
    dict(
        slug="ic_unet_selected", name="IC-U-Net", selection="Run 7 multichannel deployment edition", summary="ICA projects 30-channel context into components; a 1-D U-Net predicts clean components before inverse ICA and centre extraction.",
        blocks=[(48,160,142,"Context", "30 × 7 × 512", True),(220,160,128,"ICA", "fit components"),(382,160,132,"DoubleConv", "64 channels"),(548,160,130,"Down ×3", "pool + DoubleConv"),(714,160,122,"Up ×3", "skips + upconv"),(860,160,92,"Inverse ICA", "clean full"),(820,315,132,"Centre crop", "clean→artifact"),(850,470,102,"Artifact", "30 × 512", True)],
        skips=[([(448,234),(448,278),(775,278),(775,234)], True), ([(614,234),(614,278),(780,278),(780,234)], True), ([(884,234),(884,311)], False), ([(884,389),(884,465)], False)],
        config="30 channels · 7 epochs · ICA + base channels 64", result="Run 7: 4.56 µV GA residual; output discontinuity", note="Diagnostic only: not retained as a valid correction result.", sources=["src/facet/models/masterthesis/ic_unet/deployment/training.py","src/facet/models/masterthesis/ic_unet/deployment/training_niazy_proof_fit_seed45.yaml"]),
    dict(
        slug="st_gnn_selected", name="ST-GNN", selection="Run 7 multichannel deployment edition", summary="Temporal GLU and Chebyshev graph convolution operate on a 30-electrode graph, then a 1×1 head extracts the centre epoch.",
        blocks=[(48,160,142,"Context", "30 × 7 × 512", True),(220,160,128,"Temporal GLU", "kernel 3"),(382,160,142,"ChebConv", "K=3; kNN graph"),(556,160,128,"ST-Conv ×2", "residual blocks"),(720,160,132,"Temporal GLU", "kernel 3"),(860,160,92,"1×1 head", "full sequence"),(820,315,132,"Centre crop", "epoch 4"),(850,470,102,"Artifact", "30 × 512", True)],
        skips=[([(452,234),(452,278),(620,278),(620,234)], True), ([(620,234),(620,278),(780,278),(780,234)], True), ([(884,234),(884,311)], False), ([(884,389),(884,465)], False)],
        config="hidden channels 16 · Chebyshev order 3 · kNN=4", result="Run 7: 0.57 µV GA residual; staircase output", note="Diagnostic only: low residual is invalidated by the discontinuity check.", sources=["src/facet/models/masterthesis/st_gnn/deployment/training.py","src/facet/models/masterthesis/st_gnn/deployment/training_niazy_proof_fit_seed45.yaml"]),
]


def render(profile):
    d = common(profile["name"], profile["selection"], profile["summary"], profile["blocks"], profile["config"], profile["result"], skips=profile.get("skips", ()), note=profile.get("note", ""))
    svg_path = OUT / f"{profile['slug']}.svg"
    png_path = OUT / f"{profile['slug']}.png"
    svg = d.render_svg().replace('width="1000" height="760"', 'width="180mm" height="136.8mm"', 1)
    svg_path.write_text(svg)
    subprocess.run(["rsvg-convert", "-w", "2700", str(svg_path), "-o", str(png_path)], check=True)
    return svg_path, png_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"builder": str(Path(__file__).relative_to(ROOT)), "series": "selected thesis variants", "diagrams": []}
    captions = ["# Selected FACETpy variants — captions\n"]
    for p in PROFILES:
        svg, png = render(p)
        captions.append(f"- **{p['name']}.** {p['summary']} Selected variant: {p['selection']}.\n")
        manifest["diagrams"].append({
            "slug": p["slug"], "title": p["name"], "selection": p["selection"], "configuration": p["config"], "observed_result": p["result"],
            "svg": svg.name, "png": png.name, "sources": p["sources"],
            "sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (svg, png)},
        })
    (OUT / "captions.md").write_text("\n".join(captions))
    (OUT / "provenance_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(PROFILES)} selected-variant diagrams to {OUT}")


if __name__ == "__main__":
    main()

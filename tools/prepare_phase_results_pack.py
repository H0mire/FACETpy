#!/usr/bin/env python3
"""Create a compact, thesis-ready results pack arranged by project phase."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "thesis_results_by_phase"
FIG = ROOT / "output" / "results_evidence_pack" / "delivery" / "thesis_ready_figures"
TAB = ROOT / "output" / "results_evidence_pack" / "delivery" / "thesis_ready_tables"


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def copy(source: Path, target: Path) -> None:
    if source.exists():
        shutil.copy2(source, target)


def svg_to_png(svg: Path) -> None:
    subprocess.run(["rsvg-convert", "-o", str(svg.with_suffix(".png")), "-w", "1800", str(svg)], check=True)


def save_svg(path: Path, body: str, width: int, height: int) -> None:
    text = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>text{{font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;fill:#172033}} .title{{font-size:26px;font-weight:700}} .label{{font-size:15px}} .small{{font-size:12px;fill:#4b5563}} .axis{{stroke:#64748b;stroke-width:1}} .grid{{stroke:#d8dee8;stroke-width:1}} </style>
<rect width="100%" height="100%" fill="white"/>{body}</svg>'''
    path.write_text(text, encoding="utf-8")
    svg_to_png(path)


def phase0() -> None:
    out = OUT / "phase_0_legacy_feasibility"
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {"arm": "Legacy AAS reference", "corrected_rms_uv": 26.03, "ga_residual_uv": 0.14, "relative_to_aas": 1.0},
        {"arm": "Legacy FC-DAE cascade", "corrected_rms_uv": 25.05, "ga_residual_uv": 0.53, "relative_to_aas": 3.8},
    ]
    write_csv(out / "table_phase0_legacy_aas_comparison.csv", list(rows[0]), rows)
    baseline, dae = rows
    body = f'''<text x="70" y="55" class="title">Phase 0 — Legacy FC-DAE versus AAS</text>
<text x="70" y="86" class="label">Residual gradient artifact; lower is better</text>
<line x1="130" y1="430" x2="880" y2="430" class="axis"/>
<line x1="130" y1="120" x2="130" y2="430" class="axis"/>
<text x="95" y="435" class="small">0</text><text x="78" y="125" class="small">0.65</text>
<rect x="260" y="363" width="180" height="67" fill="#4C78A8" rx="4"/><rect x="590" y="177" width="180" height="253" fill="#E45756" rx="4"/>
<text x="350" y="348" text-anchor="middle" class="label">0.14 µV</text><text x="680" y="162" text-anchor="middle" class="label">0.53 µV</text>
<text x="350" y="462" text-anchor="middle" class="label">Legacy AAS</text><text x="680" y="462" text-anchor="middle" class="label">Legacy FC-DAE</text>'''
    save_svg(out / "figure_phase0_legacy_aas_comparison.svg", body, 960, 510)
    write_text(out / "caption_phase0_legacy_aas_comparison.txt", "Figure X. Residual gradient-artifact amplitude of the legacy cascaded FC-DAE and its AAS reference in FACETpy 0.1.0. Lower values indicate less residual artifact.")
    write_text(out / "scope_note.txt", "Use this as a feasibility result only. The legacy target and evaluation originated from the same recording; the comparison does not establish generalization or superiority over AAS.")


def phase1() -> None:
    out = OUT / "phase_1_unified_holdout"
    out.mkdir(parents=True, exist_ok=True)
    copy(TAB / "table_5_4_family_level_ranking.csv", out / "table_phase1_unified_holdout_ranking.csv")
    copy(TAB / "table_5_2c_inference_cost.csv", out / "table_phase1_inference_cost.csv")
    copy(FIG / "figure_5_4_family_level_ranking.png", out / "figure_phase1_unified_snr_ranking.png")
    copy(FIG / "figure_5_7_quality_cost_pareto.png", out / "figure_phase1_quality_cost_pareto.png")
    write_text(out / "captions.txt", "Figure X. Model-family ranking on the unified holdout by clean-signal SNR improvement.\n\nFigure Y. Quality–cost comparison on the unified holdout; inference cost was measured under a common CPU protocol.")
    write_text(out / "scope_note.txt", "All models were compared on the same 166 context windows across 30 channels. This is an adaptation benchmark on an AAS-derived proof-of-fit dataset, not a generalization study.")


def phase2() -> None:
    out = OUT / "phase_2_pipeline_deployment"
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "docs" / "research" / "run_7_pipeline_results.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    # Preserve the original common Run-7 table verbatim, then provide a
    # thesis-selection table that resolves later provenance findings.  DPAE
    # remains the original measured Run-7 deployment arm because its later
    # seven-epoch weights are unavailable.  The two older DHCT deployment
    # arms are superseded by the exported seven-epoch context model for which
    # a pipeline screen result was recovered from the Run-8 output archive.
    write_csv(out / "table_phase2_pipeline_results.csv", list(rows[0]), rows)
    selected = [
        r for r in rows
        if r["arm"] not in {"dhct_gan_deployment", "dhct_gan_v2_deployment"}
    ]
    selected.append({
        "arm": "dhct_gan_context_7_epoch",
        "rms_uv": "",
        "ga_rest_uv": "3.7179",
        "naht_ratio": "1.3111",
        "median_sample_step_uv": "",
        "step_channel": "",
        "n_harmonics": "9",
        "epoch_rate_hz": "7.0137",
        "x_reference": "",
        "hinweis": "7-epoch context export; first completed grid point (2/27 grid points completed), not a grid winner",
    })
    write_csv(out / "table_phase2_selected_variant_results.csv", list(rows[0]), selected)
    write_text(out / "selected_variants_note.txt", "Selected-variant view: DPAE retains the measured Run-7 deployment edition because the later seven-epoch checkpoint is unavailable. DHCT-GAN uses the recovered seven-epoch context export (base channels 8, learning rate 1e-4, SI-SDR weight 0); its result is a valid pipeline measurement but not a tuned grid-search winner, because the grid stopped after 2 of 27 planned points. The original common Run-7 table remains in table_phase2_pipeline_results.csv for provenance.")
    keep = [r for r in selected if r["arm"] not in {"uncorrected"}]
    keep.sort(key=lambda r: float(r["ga_rest_uv"]))
    labels = [r["arm"].replace("_deployment", "") for r in keep]
    ga = [float(r["ga_rest_uv"]) for r in keep]
    seam = [float(r["naht_ratio"]) for r in keep]
    colors = ["#E45756" if "st_gnn" in x or "ic_unet" in x else "#4C78A8" for x in labels]
    left, right, y0, step = 250, 960, 110, 33
    max_ga, max_seam = max(ga), max(seam)
    body = '<text x="70" y="42" class="title">Phase 2 — End-to-end pipeline results</text><text x="90" y="78" class="label">Residual gradient artifact (µV)</text><text x="830" y="78" class="label">Epoch-boundary step ratio</text>'
    for i, (label, g, s, color) in enumerate(zip(labels, ga, seam, colors)):
        y = y0 + i * step
        body += f'<text x="235" y="{y+15}" text-anchor="end" class="small">{label}</text><rect x="{left}" y="{y}" width="{620*g/max_ga:.1f}" height="22" fill="{color}"/><text x="{left+625*g/max_ga:.1f}" y="{y+15}" class="small">{g:.2f}</text><rect x="{right}" y="{y}" width="{340*s/max_seam:.1f}" height="22" fill="{color}"/><text x="{right+345*s/max_seam:.1f}" y="{y+15}" class="small">{s:.2f}</text>'
    body += '<line x1="254" y1="94" x2="254" y2="610" stroke="#222" stroke-dasharray="5,4"/><text x="257" y="95" class="small">FARM</text><line x1="1230" y1="94" x2="1230" y2="610" stroke="#E45756" stroke-dasharray="5,4"/><text x="1233" y="95" class="small">review threshold</text>'
    save_svg(out / "figure_phase2_pipeline_results.svg", body, 1350, 650)
    copy(FIG / "figure_5_10_failure_mode_examples.png", out / "figure_phase2_failure_mode_examples.png")
    write_text(out / "captions.txt", "Figure X. Selected end-to-end pipeline results. Lower residual artifact is preferable; red bars identify outputs requiring visual rejection because of discontinuities. DPAE retains its measured Run-7 deployment edition; DHCT-GAN is represented by its recovered seven-epoch context export, which is a completed pipeline measurement but not a completed grid-search optimum.\n\nFigure Y. Representative failure modes demonstrating why numerical residual metrics were interpreted together with waveform inspection.")
    write_text(out / "scope_note.txt", "Phase 2 reports end-to-end correction on the EDF pipeline. The AAS-derived proof-fit target makes FARM a structural upper bound; no learned model reached FARM. The selected-variant table separates the original shared Run-7 table from later recovered model artifacts; do not describe the DHCT-GAN context result as a grid-search winner.")


def load_grid(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, [r for r in data["zeilen"] if r.get("status") == "ok" and float(r.get("naht_ratio", 0)) <= 2.5]


def phase3() -> None:
    out = OUT / "phase_3_grid_search"
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        "Nested GAN": ROOT / "output/run8_fetched/gridpod1/grids/run8/grid_nested_gan_screen.json",
        "ViT-Spectrogram": ROOT / "output/run8_fetched/gridpod2/grids/run8/grid_vit_spectrogram_screen.json",
        "Demucs": ROOT / "output/run8_fetched/gridpod4/grids/run8/grid_demucs_screen.json",
    }
    phase2 = {"Nested GAN": 1.403, "ViT-Spectrogram": 1.432, "Demucs": 3.819}
    all_rows, best_rows = [], []
    grids = []
    for family, path in sources.items():
        meta, valid = load_grid(path)
        for row in meta["zeilen"]:
            all_rows.append({"family": family, **row})
        best = min(valid, key=lambda r: float(r["ga_rest_uv"]))
        best_rows.append({"family": family, "phase2_ga_rest_uv": phase2[family], "best_grid_ga_rest_uv": best["ga_rest_uv"], "learning_rate": best["lr"], "capacity": best["kapazitaet"], "si_sdr_weight": best["si_sdr"], "valid_grid_points": len(valid)})
        grids.append((family, meta, valid))
    best_rows.append({"family": "DHCT-GAN", "phase2_ga_rest_uv": 2.442, "best_grid_ga_rest_uv": "not reported", "learning_rate": "search stopped", "capacity": "2/27 points", "si_sdr_weight": "not reported", "valid_grid_points": 2})
    write_csv(out / "table_phase3_grid_runs.csv", sorted({k for r in all_rows for k in r}), all_rows)
    write_csv(out / "table_phase3_before_after.csv", list(best_rows[0]), best_rows)
    shown, max_value = best_rows[:3], max(float(r["phase2_ga_rest_uv"]) for r in best_rows[:3])
    body = '<text x="70" y="45" class="title">Phase 3 — Before and after grid search</text><text x="70" y="76" class="label">Residual gradient artifact (µV); lower is better</text><line x1="90" y1="430" x2="1000" y2="430" class="axis"/>'
    for i, row in enumerate(shown):
        x = 180 + 290*i; p2, p3 = float(row["phase2_ga_rest_uv"]), float(row["best_grid_ga_rest_uv"])
        h2, h3 = 290*p2/max_value, 290*p3/max_value
        body += f'<rect x="{x}" y="{430-h2:.1f}" width="78" height="{h2:.1f}" fill="#A0A0A0"/><rect x="{x+88}" y="{430-h3:.1f}" width="78" height="{h3:.1f}" fill="#4C78A8"/><text x="{x+39}" y="{420-h2:.1f}" text-anchor="middle" class="small">{p2:.2f}</text><text x="{x+127}" y="{420-h3:.1f}" text-anchor="middle" class="small">{p3:.2f}</text><text x="{x+83}" y="462" text-anchor="middle" class="label">{row["family"]}</text>'
    body += '<rect x="680" y="88" width="16" height="16" fill="#A0A0A0"/><text x="704" y="101" class="small">Phase 2</text><rect x="800" y="88" width="16" height="16" fill="#4C78A8"/><text x="824" y="101" class="small">best Phase 3 point</text>'
    save_svg(out / "figure_phase3_before_after_tuning.svg", body, 1080, 510)
    # Lower residual artifact is preferable: encode improvement with a
    # semantically positive green and progressively worse results with yellow,
    # orange, and red.
    palette = ["#16803C", "#7CCB5E", "#F2D65C", "#EE984A", "#C74440"]
    body = '<text x="50" y="40" class="title">Phase 3 — Grid-search landscapes</text><text x="50" y="67" class="small">Each cell: best valid residual artifact (µV) across SI-SDR weights; lower is better.</text>'
    for panel, (family, meta, valid) in enumerate(grids):
        ox, oy, cell = 65 + panel*450, 140, 100
        lrs, caps = meta["gitterwerte"]["lr"], meta["gitterwerte"]["kapazitaet"]
        vals = []
        for cap in caps:
            for lr in lrs:
                a = [float(r["ga_rest_uv"]) for r in valid if r["kapazitaet"] == cap and r["lr"] == lr]
                vals.extend(a)
        lo, hi = min(vals), max(vals)
        body += f'<text x="{ox+130}" y="110" text-anchor="middle" class="label">{family}</text>'
        for i, cap in enumerate(caps):
            body += f'<text x="{ox-10}" y="{oy+i*cell+55}" text-anchor="end" class="small">{cap}</text>'
            for j, lr in enumerate(lrs):
                a = [float(r["ga_rest_uv"]) for r in valid if r["kapazitaet"] == cap and r["lr"] == lr]
                if a:
                    v = min(a); idx = int(round((v-lo)/(hi-lo or 1)*(len(palette)-1))); color=palette[idx]
                    # The global SVG text rule supplies a dark default.  Set the
                    # annotation colour inline so it reliably overrides that
                    # rule after rasterisation.  The two light Viridis tones
                    # need dark text; the three dark tones need white text.
                    text_color = "#172033" if idx in (1, 2) else "#FFFFFF"
                    body += f'<rect x="{ox+j*cell}" y="{oy+i*cell}" width="{cell-3}" height="{cell-3}" fill="{color}"/><text x="{ox+j*cell+48}" y="{oy+i*cell+55}" text-anchor="middle" style="font-size:16px;font-family:Arial;fill:{text_color}">{v:.2f}</text>'
                else: body += f'<rect x="{ox+j*cell}" y="{oy+i*cell}" width="{cell-3}" height="{cell-3}" fill="#e5e7eb"/>'
        for j, lr in enumerate(lrs): body += f'<text x="{ox+j*cell+48}" y="{oy+3*cell+23}" text-anchor="middle" class="small">{lr:g}</text>'
        capacity_label = {"Nested GAN": "inner channels", "ViT-Spectrogram": "embedding dimension", "Demucs": "initial channels"}[family]
        body += f'<text x="{ox+130}" y="{oy+3*cell+50}" text-anchor="middle" class="small">learning rate →</text><text x="{ox-45}" y="{oy+145}" text-anchor="middle" transform="rotate(-90 {ox-45} {oy+145})" class="small">{capacity_label}</text>'
    save_svg(out / "figure_phase3_grid_heatmaps.svg", body, 1450, 560)
    # Full three-dimensional view: rows represent SI-SDR weight, columns the
    # model family, and each small heatmap retains learning rate × width.
    # This makes the third searched axis visible instead of collapsing it.
    full_palette = ["#16803C", "#7CCB5E", "#F2D65C", "#EE984A", "#C74440"]
    body = ('<text x="50" y="40" class="title">Phase 3 — Full grid-search landscapes</text>'
            '<text x="50" y="67" class="small">Residual gradient artifact (µV); lower is better. Green = lower residual, red = higher residual, within each model family.</text>')
    panel_w, panel_h, cell, x0, y0 = 390, 210, 52, 150, 135
    for col, (family, meta, valid) in enumerate(grids):
        lrs, caps = meta["gitterwerte"]["lr"], meta["gitterwerte"]["kapazitaet"]
        # Hold the colour scale fixed within each model family so the three
        # SI-SDR rows can be compared visually.
        family_vals = [float(r["ga_rest_uv"]) for r in valid]
        family_lo, family_hi = min(family_vals), max(family_vals)
        family_best = family_lo
        capacity_label = {"Nested GAN": "inner channels", "ViT-Spectrogram": "embedding dimension", "Demucs": "initial channels"}[family]
        body += f'<text x="{x0 + col*panel_w + 100}" y="105" text-anchor="middle" class="label">{family}</text>'
        for row, sisdr in enumerate((0.0, 1.0, 3.0)):
            ox, oy = x0 + col*panel_w, y0 + row*panel_h
            if col == 0:
                body += f'<text x="75" y="{oy+82}" text-anchor="end" class="small">SI-SDR = {sisdr:g}</text>'
            for i, cap in enumerate(caps):
                body += f'<text x="{ox-12}" y="{oy+i*cell+32}" text-anchor="end" class="small">{cap}</text>'
                for j, lr in enumerate(lrs):
                    point = [float(r["ga_rest_uv"]) for r in valid if r["kapazitaet"] == cap and r["lr"] == lr and float(r["si_sdr"]) == sisdr]
                    if point:
                        v = point[0]
                        idx = int(round((v-family_lo)/(family_hi-family_lo or 1)*(len(full_palette)-1)))
                        color = full_palette[idx]
                        text_color = "#172033" if idx in (1, 2) else "#FFFFFF"
                        is_best = abs(v - family_best) < 1e-12
                        outline = ' stroke="#0F172A" stroke-width="4"' if is_best else ''
                        body += f'<rect x="{ox+j*cell}" y="{oy+i*cell}" width="{cell-2}" height="{cell-2}" fill="{color}"{outline}/><text x="{ox+j*cell+25}" y="{oy+i*cell+32}" text-anchor="middle" style="font-size:12px;font-family:Arial;fill:{text_color}">{v:.2f}</text>'
            for j, lr in enumerate(lrs):
                body += f'<text x="{ox+j*cell+25}" y="{oy+3*cell+18}" text-anchor="middle" class="small">{lr:g}</text>'
            body += f'<text x="{ox+76}" y="{oy+3*cell+40}" text-anchor="middle" class="small">learning rate →</text>'
            if row == 1:
                body += f'<text x="{ox-42}" y="{oy+80}" text-anchor="middle" transform="rotate(-90 {ox-42} {oy+80})" class="small">{capacity_label}</text>'
    body += '<rect x="1010" y="24" width="16" height="16" fill="none" stroke="#0F172A" stroke-width="3"/><text x="1035" y="37" class="small">best valid configuration per model</text>'
    save_svg(out / "figure_phase3_grid_heatmaps_full.svg", body, 1300, 800)
    write_text(out / "captions.txt", "Figure X. End-to-end residual gradient artifact before Phase-3 tuning and at the best valid grid point. FARM is shown as the reference line.\n\nFigure Y. Grid-search landscape for three completed model families. Each cell shows the best valid residual artifact across the third grid axis, SI-SDR weight; lower is better.\n\nDHCT-GAN is not plotted as a completed grid because only two of the planned 27 points were run.")
    write_text(out / "scope_note.txt", "Phase 3 used one seed per grid point and selected the best validation checkpoint. Grid conclusions therefore show task-specific tuning evidence, not a multi-seed confirmation study.")


def phase3_wega() -> None:
    """Create the corresponding 3×3 view for the Weg-A clean-target search."""
    out = OUT / "phase_3_wega_grid_search"
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        "Nested GAN": ROOT / "output/run8_fetched/wegapod/grids/wega/grid_nested_gan_screen.json",
        "ViT-Spectrogram": ROOT / "output/run8_fetched/wegapod/grids/wega/grid_vit_spectrogram_screen.json",
        "Demucs": ROOT / "output/run8_fetched/wegapod/grids/wega/grid_demucs_screen.json",
    }
    grids = []
    rows = []
    for family, path in sources.items():
        meta = json.loads(path.read_text(encoding="utf-8"))
        valid = [r for r in meta["zeilen"] if r.get("status") == "ok"]
        grids.append((family, meta, valid))
        rows.extend({"family": family, **r} for r in meta["zeilen"])
    write_csv(out / "table_phase3_wega_grid_runs.csv", sorted({k for r in rows for k in r}), rows)

    palette = ["#16803C", "#7CCB5E", "#F2D65C", "#EE984A", "#C74440"]
    body = ('<text x="50" y="40" class="title">Phase 3 — Weg-A grid-search landscapes</text>'
            '<text x="50" y="67" class="small">Clean-target error (µV); lower is better. Green = lower error, red = higher error, within each model family.</text>')
    panel_w, panel_h, cell, x0, y0 = 390, 210, 52, 150, 135
    labels = {"Nested GAN": "inner channels", "ViT-Spectrogram": "embedding dimension", "Demucs": "initial channels"}
    for col, (family, meta, valid) in enumerate(grids):
        lrs, caps = meta["gitterwerte"]["lr"], meta["gitterwerte"]["kapazitaet"]
        vals = [float(r["err_uv"]) for r in valid]
        lo, hi, best = min(vals), max(vals), min(vals)
        body += f'<text x="{x0 + col*panel_w + 100}" y="105" text-anchor="middle" class="label">{family}</text>'
        for row, sisdr in enumerate((0.0, 1.0, 3.0)):
            ox, oy = x0 + col*panel_w, y0 + row*panel_h
            if col == 0:
                body += f'<text x="75" y="{oy+82}" text-anchor="end" class="small">SI-SDR = {sisdr:g}</text>'
            for i, cap in enumerate(caps):
                body += f'<text x="{ox-12}" y="{oy+i*cell+32}" text-anchor="end" class="small">{cap}</text>'
                for j, lr in enumerate(lrs):
                    point = [float(r["err_uv"]) for r in valid if r["kapazitaet"] == cap and r["lr"] == lr and float(r["si_sdr"]) == sisdr]
                    if point:
                        v = point[0]
                        idx = int(round((v-lo)/(hi-lo or 1)*(len(palette)-1)))
                        color = palette[idx]
                        text_color = "#172033" if idx in (1, 2) else "#FFFFFF"
                        outline = ' stroke="#0F172A" stroke-width="4"' if abs(v - best) < 1e-12 else ''
                        body += f'<rect x="{ox+j*cell}" y="{oy+i*cell}" width="{cell-2}" height="{cell-2}" fill="{color}"{outline}/><text x="{ox+j*cell+25}" y="{oy+i*cell+32}" text-anchor="middle" style="font-size:12px;font-family:Arial;fill:{text_color}">{v:.2f}</text>'
                    else:
                        body += f'<rect x="{ox+j*cell}" y="{oy+i*cell}" width="{cell-2}" height="{cell-2}" fill="#E5E7EB"/>'
            for j, lr in enumerate(lrs):
                body += f'<text x="{ox+j*cell+25}" y="{oy+3*cell+18}" text-anchor="middle" class="small">{lr:g}</text>'
            body += f'<text x="{ox+76}" y="{oy+3*cell+40}" text-anchor="middle" class="small">learning rate →</text>'
            if row == 1:
                body += f'<text x="{ox-42}" y="{oy+80}" text-anchor="middle" transform="rotate(-90 {ox-42} {oy+80})" class="small">{labels[family]}</text>'
    body += '<rect x="1010" y="24" width="16" height="16" fill="none" stroke="#0F172A" stroke-width="3"/><text x="1035" y="37" class="small">best valid configuration per model</text>'
    save_svg(out / "figure_phase3_wega_grid_heatmaps.svg", body, 1300, 800)
    write_text(out / "caption.txt", "Weg-A grid-search landscapes across learning rate, model capacity, and SI-SDR weight. Each cell shows clean-target error after evaluation; green indicates lower error and outlined cells mark the best valid configuration per model.")
    write_text(out / "scope_note.txt", "Weg-A values are clean-target errors, not residual gradient-artifact values from the Phase-2 EDF pipeline. Empty cells are runs that were rejected or unavailable.")


def index() -> None:
    write_text(OUT / "README.md", """
# Results pack arranged by experimental phase

Every phase folder contains directly insertable PNG figures, CSV tables, captions, and a scope note.

| Folder | Core result | Use in thesis |
|---|---|---|
| `phase_0_legacy_feasibility` | Legacy FC-DAE versus its AAS reference | Short proof-of-fit opening result |
| `phase_1_unified_holdout` | Unified SNR ranking and common inference cost | Cross-family benchmark |
| `phase_2_pipeline_deployment` | End-to-end pipeline correction and failure diagnostics | Paper-aligned deployment result |
| `phase_3_grid_search` | Tuning before/after and grid landscapes | Hyperparameter sensitivity and selected candidates |

Do not merge numbers across phases without preserving the phase-specific scope note.
""")


def main() -> None:
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    phase0(); phase1(); phase2(); phase3(); phase3_wega(); index()
    print(OUT)


if __name__ == "__main__":
    main()

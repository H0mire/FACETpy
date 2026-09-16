#!/usr/bin/env python3
"""Build the FACETpy independent clean-reference dataset diagram."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLKIT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLKIT_DIR))

from facetpy_svg import (  # noqa: E402
    CANVAS_W,
    C,
    Diagram,
    capsule,
    card,
    container,
    edge,
    pill,
    text,
)

(ROOT / "output/thesis_figures/framework").mkdir(parents=True, exist_ok=True)

TITLE = "FACETpy Independent Clean-Reference Dataset"
SUBTITLE = "Independent clean, known IED labels, and builder-level leakage controls"
HEIGHT = 1990
SVG_PATH = (ROOT / "output/thesis_figures/framework").joinpath("facetpy_clean_reference_dataset.svg")
PNG_PATH = (ROOT / "output/thesis_figures/framework").joinpath("facetpy_clean_reference_dataset.png")


def build() -> Diagram:
    diagram = Diagram(HEIGHT, title=TITLE, subtitle="", background=False)
    assert diagram.width == CANVAS_W == 1000
    diagram.add(text(70, 83, SUBTITLE, size=15, fill=C["slate"]))

    # Three source lanes remain visibly separate until the known-event injection
    # and additive mixture band. This prevents the artifact template from being
    # misread as the source of the independent clean target.
    artifact_lane = container(25, 110, 350, 585, "1 · Artifact source")
    clean_lane = container(395, 110, 280, 585, "2 · Clean source")
    ied_lane = container(695, 110, 280, 585, "3 · Known IED source")
    diagram.add(artifact_lane, clean_lane, ied_lane)

    artifact_source = card(
        50,
        155,
        300,
        80,
        "Niazy artifact source",
        ["in-scanner gradient artifact"],
    )
    template = card(
        50,
        260,
        300,
        127,
        "FARM + PCA/OBS template",
        ["FARM template estimate", "OBS: 4 components", "OBS hp_freq: 300 Hz"],
    )
    template_artifact = pill(
        65,
        412,
        270,
        42,
        "template-removable artifact",
    )
    augmentation = card(
        50,
        480,
        300,
        127,
        "Failure-mode augmentation",
        ["amplitude + timing jitter", "motion amplitude modulation", "46 Hz helium-pump line"],
    )
    enriched_artifact = capsule(85, 632, 230, 42, "enriched artifact")

    pretrigger = card(
        420,
        155,
        230,
        150,
        "Niazy pre-trigger",
        [
            "isolate before first trigger",
            "1 Hz high-pass",
            "50 Hz notch + harmonics",
            "no amplitude renormalization",
        ],
    )
    external_clean = card(
        420,
        340,
        230,
        90,
        "External clean EEG",
        ["real GA-free alternative"],
    )
    prepared_clean = capsule(420, 505, 230, 44, "real GA-free clean EEG")

    vepiset = card(
        720,
        155,
        230,
        90,
        "VEPISET real IEDs",
        ["expert-annotated ! markers"],
    )
    isolate_ied = card(
        720,
        280,
        230,
        127,
        "Isolate event",
        ["edge-baseline removal", "raised-cosine edge taper", "retain transient"],
    )
    preserve_ied = card(
        720,
        445,
        230,
        127,
        "Preserve event identity",
        ["native amplitude", "native channel topography", "marker offset retained"],
    )
    isolated_ied = capsule(720, 632, 230, 42, "isolated real IED")

    diagram.add(
        artifact_source,
        template,
        template_artifact,
        augmentation,
        enriched_artifact,
        pretrigger,
        external_clean,
        prepared_clean,
        vepiset,
        isolate_ied,
        preserve_ied,
        isolated_ied,
    )
    diagram.add_edge(
        edge([(200, 238), (200, 257)]),
        edge([(200, 390), (200, 409)]),
        edge([(200, 457), (200, 477)]),
        edge([(200, 610), (200, 629)]),
        edge([(535, 308), (407, 308), (407, 480), (510, 480), (510, 502)]),
        edge(
            [(535, 433), (650, 433), (650, 480), (560, 480), (560, 502)],
            label=("OR", 535, 480),
        ),
        edge([(835, 248), (835, 277)]),
        edge([(835, 410), (835, 442)]),
        edge([(835, 575), (835, 629)]),
    )

    # Known-event injection is a construction operation, not a second detector.
    mix_frame = container(
        30,
        735,
        940,
        220,
        "Known-event injection and additive mixture",
    )
    injection = card(
        60,
        785,
        360,
        127,
        "Inject at known marker",
        [
            "prepared clean + isolated real IED",
            "preserve native amplitude + topography",
            "label comes from injection · no re-detection",
        ],
    )
    clean_with_ied = pill(460, 825, 200, 48, "clean + known IED")
    mixture = card(
        700,
        785,
        245,
        127,
        "Additive mixture",
        ["noisy = clean + artifact", "IED remains in clean target", "artifact is independently added"],
    )
    diagram.add(mix_frame, injection, clean_with_ied, mixture)
    diagram.add_edge(
        edge([(535, 552), (535, 715), (180, 715), (180, 782)]),
        edge([(835, 677), (835, 723), (310, 723), (310, 782)]),
        edge([(423, 849), (457, 849)]),
        edge([(663, 849), (697, 849)]),
        edge(
            [(200, 677), (200, 705), (14, 705), (14, 935), (822, 935), (822, 915)],
            label=("+ enriched artifact", 555, 927),
        ),
    )

    # Shared example contract: one target-channel/center-epoch example with
    # explicit spatial context and a crop guard around every resampled epoch.
    packaging_frame = container(30, 985, 940, 235, "Spatio-temporal packaging")
    identity = card(
        55,
        1035,
        240,
        127,
        "Example identity",
        ["one target channel c", "one center epoch e", "example = (c, e)"],
    )
    geometry = card(
        325,
        1035,
        395,
        127,
        "Stored context geometry",
        [
            "7 epochs × (target + k geodesic montage neighbors)",
            "target channel stored first",
            "k = 2 or 6 by experiment",
        ],
    )
    guard = card(
        750,
        1035,
        195,
        127,
        "576 samples",
        ["512-sample core", "32-sample left guard", "32-sample right guard"],
    )
    diagram.add(packaging_frame, identity, geometry, guard)
    diagram.add_edge(
        edge([(822, 915), (822, 967), (175, 967), (175, 1032)]),
        edge([(298, 1098), (322, 1098)]),
        edge([(723, 1098), (747, 1098)]),
    )

    # The split is made while the builder still knows temporal overlap and clean
    # tiling provenance; the saved assignment is consumed unchanged downstream.
    split_frame = container(30, 1250, 940, 205, "Leakage-free builder-level split")
    contiguous = card(
        50,
        1300,
        205,
        104,
        "Contiguous time split",
        ["ordered center epochs", "train → validation"],
    )
    seam_guard = card(
        280,
        1300,
        205,
        104,
        "Context seam guard",
        ["drop straddling contexts", "no shared 7-epoch window"],
    )
    disjoint = card(
        510,
        1300,
        205,
        104,
        "Disjoint clean tiling",
        ["separate source partitions", "tile within split only"],
    )
    stored_split = card(
        740,
        1300,
        205,
        104,
        "Stored assignment",
        ["example_split", "0 train · 1 validation"],
    )
    diagram.add(split_frame, contiguous, seam_guard, disjoint, stored_split)
    diagram.add_edge(
        edge([(500, 1223), (500, 1235), (152, 1235), (152, 1297)]),
        edge([(258, 1352), (277, 1352)]),
        edge([(488, 1352), (507, 1352)]),
        edge([(718, 1352), (737, 1352)]),
    )

    # Storage is component-wise. The loader rebuilds noisy per item so online
    # augmentation cannot break the clean + artifact invariant.
    npz_frame = container(30, 1485, 940, 220, "NPZ contract")
    stored_arrays = card(
        55,
        1535,
        600,
        136,
        "Stored arrays",
        [
            "clean_context · artifact_context · artifact_context_template",
            "clean_center · artifact_center · artifact_center_template",
            "spike_labels · example_split",
            "channel / epoch / neighbor indices + source provenance",
        ],
    )
    rebuild = card(
        700,
        1535,
        245,
        127,
        "Per-item reconstruction",
        ["noisy is not stored", "rebuild noisy = clean + artifact", "then crop the guards"],
    )
    diagram.add(npz_frame, stored_arrays, rebuild)
    diagram.add_edge(
        edge([(842, 1407), (842, 1468), (355, 1468), (355, 1532)]),
        edge([(658, 1603), (697, 1603)]),
    )

    # Both supported objectives are shown as explicit input/target contracts.
    learning_frame = container(30, 1735, 940, 210, "Learning contracts")
    direct = card(
        60,
        1785,
        400,
        112,
        "Direct formulation",
        ["input: noisy", "target: artifact OR clean"],
    )
    residual = card(
        540,
        1785,
        400,
        127,
        "Residual formulation",
        ["input: noisy − template", "target: artifact − template", "template = FARM-removable part"],
    )
    diagram.add(learning_frame, direct, residual)
    diagram.add_edge(
        edge([(822, 1665), (822, 1718), (260, 1718), (260, 1782)]),
        edge([(822, 1718), (740, 1718), (740, 1782)]),
    )

    return diagram


if __name__ == "__main__":
    output = build()
    output.render_png(str(PNG_PATH), svg_path=str(SVG_PATH), width=1280)
    print(SVG_PATH)
    print(PNG_PATH)

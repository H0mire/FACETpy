#!/usr/bin/env python3
"""Build the FACETpy training-to-correction architecture diagram."""

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
    note,
    pill,
    text,
)

(ROOT / "output/thesis_figures/framework").mkdir(parents=True, exist_ok=True)

TITLE = "From experiment specification to FACETpy correction"
SUBTITLE = "Training harness, GPU fleet, and direct versus residual deployment"
HEIGHT = 1435
SVG_PATH = (ROOT / "output/thesis_figures/framework").joinpath("facetpy_training_to_deployment.svg")
PNG_PATH = (ROOT / "output/thesis_figures/framework").joinpath("facetpy_training_to_deployment.png")


def build() -> Diagram:
    # The separate subtitle line keeps the long title row airy and preserves the
    # signature EEG trace at the right edge.
    diagram = Diagram(HEIGHT, title=TITLE, subtitle="", background=False)
    assert diagram.width == CANVAS_W == 1000

    # Add all frames first so cards and labels remain above their faint fills.
    orchestration = container(35, 108, 930, 305, "Experiment orchestration")
    harness = container(35, 435, 930, 345, "facet-train")
    correction = container(20, 810, 960, 590, "FACETpy correction paths")
    common = container(50, 850, 900, 105, "Common preprocessing")
    direct_lane = container(50, 975, 900, 135, "Direct models · most families")
    residual_lane = container(50, 1130, 900, 150, "Residual Demucs cascade only")
    shared_cleanup = container(50, 1300, 900, 90, "Shared final cleanup")
    diagram.add(
        orchestration,
        harness,
        correction,
        common,
        direct_lane,
        residual_lane,
        shared_cleanup,
    )
    diagram.add(text(70, 83, SUBTITLE, size=15, fill=C["slate"]))

    # A. Experiment orchestration: the local queue dispatches independent jobs
    # to a pool of single-GPU workers. This is task parallelism, not one model
    # distributed across GPUs.
    model_worktree = card(70, 145, 220, 92, "Model worktree", ["architecture-specific code"])
    yaml_config = card(
        70,
        280,
        220,
        110,
        "YAML configuration",
        ["model · data · training", "callbacks · export"],
    )
    fleet_queue = card(355, 205, 245, 96, "Local fleet queue", ["dispatches independent jobs"])
    gpu_pool = card(
        650,
        140,
        290,
        150,
        "RunPod GPU pool",
        ["single-GPU workers", "2 initially · 6 in Run 7", "one job per GPU"],
    )
    parallelism_note = note(360, 340, 300, 42, "task parallelism · not distributed training")
    diagram.add(
        model_worktree,
        yaml_config,
        fleet_queue,
        gpu_pool,
        parallelism_note,
    )
    diagram.add_edge(
        edge([(293, 191), (320, 191), (320, 238), (352, 238)]),
        edge([(293, 335), (325, 335), (325, 270), (352, 270)]),
        edge([(603, 253), (625, 253), (625, 215), (647, 215)]),
        edge([(477.5, 304), (477.5, 337)]),
    )

    # Each independent GPU job enters the same facet-train dataset factory.
    diagram.add_edge(edge([(795, 293), (795, 420), (145, 420), (145, 482)]))

    # B. facet-train harness and retained outputs.
    dataset_factory = pill(60, 485, 170, 52, "dataset factory")
    model_loss = pill(255, 485, 170, 52, "model + loss")
    trainable_wrapper = pill(450, 485, 170, 52, "trainable wrapper")
    trainer_callbacks = pill(645, 485, 200, 52, "Trainer + callbacks")
    diagram.add(dataset_factory, model_loss, trainable_wrapper, trainer_callbacks)
    diagram.add_edge(
        edge([(233, 511), (252, 511)]),
        edge([(428, 511), (447, 511)]),
        edge([(623, 511), (642, 511)]),
    )

    output_trunk = edge([(745, 540), (745, 595), (185, 595), (765, 595)], marker_end=None)
    checkpoints = card(75, 630, 220, 70, "Checkpoints / logs")
    exported_model = card(365, 630, 220, 70, "Exported model")
    inference_config = card(655, 630, 220, 70, "Inference config")
    diagram.add_edge(
        output_trunk,
        edge([(185, 595), (185, 627)]),
        edge([(475, 595), (475, 627)]),
        edge([(765, 595), (765, 627)]),
    )
    diagram.add(checkpoints, exported_model, inference_config)

    handoff = pill(
        600,
        720,
        320,
        42,
        "deployment hand-off · model + inference config",
    )
    diagram.add(handoff)
    diagram.add_edge(
        edge([(475, 703), (475, 710), (700, 710), (700, 717)]),
        edge([(765, 703), (765, 717)]),
    )

    # C. Common FACETpy preprocessing shared by both learned correction paths.
    load_crop = pill(70, 887, 120, 48, "Load + crop")
    filtering = pill(215, 887, 150, 48, "Filter + triggers")
    align = pill(390, 887, 100, 48, "Align")
    aligned_eeg = pill(515, 887, 240, 48, "Aligned contaminated EEG")
    diagram.add(load_crop, filtering, align, aligned_eeg)
    diagram.add_edge(
        edge([(193, 911), (212, 911)]),
        edge([(368, 911), (387, 911)]),
        edge([(493, 911), (512, 911)]),
    )

    # The aligned signal fans out at x=300. Keeping that data rail on the left
    # leaves a dedicated right-side route for the learned model/config hand-off.
    diagram.add_edge(
        edge(
            [(635, 938), (635, 962), (300, 962), (300, 1160)],
            marker_end=None,
        ),
        edge([(300, 1049), (367, 1049)]),
        edge([(300, 1160), (215, 1160), (215, 1167)]),
    )

    # Direct path: most model families operate immediately on aligned,
    # contaminated EEG. No FARM or template PCA/OBS precedes the learned stage.
    direct_correction = pill(370, 1020, 240, 58, "DeepLearningCorrection")
    diagram.add(direct_correction)
    diagram.add_edge(
        edge(
            [(613, 1049), (900, 1049)],
            marker_end=None,
            label=("artifact or clean estimate", 745, 1041),
        )
    )

    # Residual Demucs path: FARM + PCA/OBS recreate the primary template stage
    # used in training before the learned model estimates the residual artifact.
    farm = pill(165, 1170, 100, 56, "FARM")
    pca_obs = pill(300, 1170, 125, 56, "PCA/OBS")
    residual_correction = pill(465, 1170, 245, 56, "DeepLearningCorrection")
    template_note = note(
        550,
        1136,
        350,
        28,
        "FARM + PCA/OBS reproduce the training template stage",
    )
    diagram.add(farm, pca_obs, residual_correction, template_note)
    diagram.add_edge(
        edge([(268, 1198), (297, 1198)]),
        edge([(428, 1198), (462, 1198)]),
        edge(
            [(713, 1198), (900, 1198)],
            marker_end=None,
            label=("learned residual artifact", 806, 1190),
        ),
    )

    # The exported model and inference config deliberately feed either learned
    # stage. Separate outer routes keep them away from every card and label.
    diagram.add_edge(
        edge(
            [(763, 765), (990, 765), (990, 1008), (490, 1008), (490, 1017)],
            label=("model + config", 845, 1000),
        ),
        edge(
            [(757, 765), (10, 765), (10, 1260), (587.5, 1260), (587.5, 1229)],
            label=("model + config", 475, 1252),
        ),
    )

    # Both branches merge into the same final cleanup sequence.
    cleanup_pca = pill(260, 1332, 145, 48, "Cleanup PCA")
    downsample = pill(445, 1332, 220, 48, "Downsample + low-pass")
    corrected = capsule(705, 1332, 180, 48, "Corrected EEG")
    diagram.add(cleanup_pca, downsample, corrected)
    diagram.add_edge(
        edge([(900, 1049), (900, 1290), (332.5, 1290), (332.5, 1329)]),
        edge([(408, 1356), (442, 1356)]),
        edge([(668, 1356), (702, 1356)]),
    )

    return diagram


if __name__ == "__main__":
    output = build()
    output.render_png(str(PNG_PATH), svg_path=str(SVG_PATH), width=1280)
    print(SVG_PATH)
    print(PNG_PATH)

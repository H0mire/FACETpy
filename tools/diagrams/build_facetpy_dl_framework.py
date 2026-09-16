#!/usr/bin/env python3
"""Build the FACETpy deep-learning framework architecture diagram."""

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
    decision,
    edge,
    note,
    pill,
    text,
)

(ROOT / "output/thesis_figures/framework").mkdir(parents=True, exist_ok=True)

TITLE = "FACETpy Deep-Learning Framework"
SUBTITLE = "Configuration-driven training and pipeline-ready inference"
HEIGHT = 1715
SVG_PATH = (ROOT / "output/thesis_figures/framework").joinpath("facetpy_dl_framework.svg")
PNG_PATH = (ROOT / "output/thesis_figures/framework").joinpath("facetpy_dl_framework.png")


def build() -> Diagram:
    """Compose the architecture from the shared FACETpy SVG primitives."""
    # Keep the long subtitle on a dedicated row so the title EEG trace remains
    # visible inside the fixed-width canvas.
    diagram = Diagram(HEIGHT, title=TITLE, subtitle="", background=False)
    assert diagram.width == CANVAS_W == 1000

    # Grouping frames are registered first so cards and pills remain on top.
    configuration = container(25, 105, 950, 320, "Entry + configuration")
    training = container(25, 460, 680, 725, "Training side")
    artifacts = container(720, 460, 255, 725, "Run artifacts")
    deployment = container(735, 735, 225, 400, "Deployment hand-off")
    inference = container(15, 1215, 970, 440, "FACETpy pipeline inference")
    diagram.add(configuration, training, artifacts, deployment, inference)
    diagram.add(text(70, 83, SUBTITLE, size=15, fill=C["slate"]))

    # ------------------------------------------------------------------
    # Entry and configuration band.
    # ------------------------------------------------------------------
    cli = capsule(55, 145, 270, 44, "facet-train fit --config run.yaml")
    config = card(355, 125, 590, 145, "TrainingCLIConfig")
    config_model = pill(375, 180, 120, 34, "model")
    config_data = pill(505, 180, 110, 34, "data")
    config_training = pill(625, 180, 120, 34, "training")
    config_export = pill(755, 180, 170, 34, "export + inference")
    resolution = card(
        55,
        235,
        275,
        155,
        "Resolution + validation",
        [
            "framework + target/output contract",
            "resolve module:function factories",
            "persist resolved JSON / YAML",
        ],
    )
    resolved_spec = card(
        355,
        290,
        590,
        110,
        "Resolved run specification",
        [
            "validated contract · dynamic factories resolved",
            "resolved JSON / YAML frozen for the run",
        ],
    )
    diagram.add(
        cli,
        config,
        config_model,
        config_data,
        config_training,
        config_export,
        resolution,
        resolved_spec,
    )
    diagram.add_edge(
        edge([(328, 167), (352, 167)], label=("loads", 340, 153)),
        edge([(650, 273), (650, 282), (333, 282)], label=("validate + resolve", 500, 282)),
        edge([(333, 350), (352, 350)]),
    )

    # A single training inlet keeps the fan-out legible while retaining the
    # four clearly separated source configuration groups above.
    training_config = pill(220, 490, 300, 34, "data · factories · training settings")
    diagram.add(training_config)
    diagram.add_edge(
        edge(
            [(650, 403), (650, 445), (370, 445), (370, 487)],
            label=("resolved configuration", 500, 445),
        ),
        edge(
            [(850, 403), (970, 403), (970, 1082), (936, 1082)],
            label=("export + inference contract", 882, 445),
        ),
    )

    # ------------------------------------------------------------------
    # Training side.
    # ------------------------------------------------------------------
    data_construction = card(50, 550, 310, 270, "Data construction")
    context_route = pill(70, 600, 270, 40, "context_factory → EEGArtifactDataset")
    augmentation = note(100, 646, 210, 30, "optional augmentation")
    route_or = decision(205, 696, 64, 30, "OR")
    custom_route = pill(70, 718, 270, 40, "dataset_factory → custom dataset")
    deterministic_split = pill(70, 770, 270, 34, "deterministic train / validation splits")

    factories = card(
        385,
        550,
        295,
        230,
        "Factory construction",
        [
            "model factory · loss factory",
            "optimizer + scheduler (optional)",
            "injected metadata:",
            "channels · chunk · sampling rate",
            "input / target shapes",
        ],
    )

    wrapper = card(
        50,
        850,
        630,
        160,
        "TrainableModelWrapper",
        stereotype="training-side only",
    )
    pytorch_wrapper = pill(65, 898, 180, 34, "PyTorchModelWrapper")
    tensorflow_wrapper = pill(255, 898, 190, 34, "TensorFlowModelWrapper")
    custom_wrapper = pill(455, 898, 210, 34, "custom / adversarial wrapper")

    trainer = card(
        50,
        1040,
        300,
        130,
        "Trainer",
        [
            "epoch / batch loop + TrainingState",
            "calls wrapper train / eval steps",
            "dispatches callback hooks",
        ],
        stereotype="framework-agnostic",
    )
    callbacks = card(
        385,
        1035,
        295,
        130,
        "Callback set",
        [
            "checkpoint · early stopping",
            "JSONL metrics · loss plot",
            "prediction samples",
        ],
    )

    diagram.add(
        data_construction,
        context_route,
        augmentation,
        route_or,
        custom_route,
        deterministic_split,
        factories,
        wrapper,
        pytorch_wrapper,
        tensorflow_wrapper,
        custom_wrapper,
        text(
            365,
            962,
            "tensor conversion · train_step · eval_step",
            size=12.5,
            anchor="middle",
        ),
        text(
            365,
            988,
            "optimizer / scheduler · checkpoint I/O",
            size=12.5,
            anchor="middle",
        ),
        trainer,
        callbacks,
    )

    # Config fan-out, batch flow, wrapper construction, and callback hooks.
    diagram.add_edge(
        edge([(300, 527), (300, 547)], label=("data", 277, 542)),
        edge([(450, 527), (450, 547)], label=("factories", 487, 542)),
        edge([(523, 507), (695, 507), (695, 1025), (200, 1025), (200, 1037)]),
        edge(
            [(205, 823), (205, 830), (15, 830), (15, 1096), (47, 1096)],
            label=("batches", 110, 830),
        ),
        edge([(532, 783), (532, 847)], label=("builds", 560, 820)),
        edge(
            [(280, 1037), (280, 1013)],
            label=("train_step / eval_step", 280, 1027),
        ),
        edge(
            [(353, 1096), (382, 1096)],
        ),
    )

    # ------------------------------------------------------------------
    # Run artifacts and the deliberate deployment hand-off.
    # ------------------------------------------------------------------
    evidence = card(
        745,
        515,
        205,
        175,
        "Evidence / restart state",
        [
            "checkpoints",
            "resolved config",
            "JSONL metric logs",
            "loss + prediction plots",
            "run summary",
        ],
    )
    exported = card(
        750,
        785,
        195,
        95,
        "Exported model",
        ["TorchScript or Keras"],
    )
    inference_config = card(
        750,
        905,
        195,
        115,
        "Inference config",
        ["persisted model contract"],
    )
    handoff = capsule(765, 1060, 168, 44, "model + config only")
    diagram.add(evidence, exported, inference_config, handoff)
    diagram.add_edge(
        edge(
            [(683, 1100), (710, 1100), (710, 625), (742, 625)],
            label=("writes artifacts", 690, 850),
        ),
        edge([(948, 832), (965, 832), (965, 1042), (850, 1042), (850, 1057)]),
        edge([(847, 1023), (847, 1057)]),
    )

    # ------------------------------------------------------------------
    # FACETpy pipeline inference.
    # ------------------------------------------------------------------
    registry = card(
        40,
        1260,
        220,
        160,
        "Model registry",
        ["DeepLearningModelRegistry", "resolves named adapter"],
    )
    adapter = card(
        280,
        1260,
        250,
        160,
        "DeepLearningModelAdapter",
        [
            "+ persisted DeepLearningModelSpec",
            "PyTorch · TensorFlow · ONNX",
            "NumPy · custom family adapter",
        ],
        stereotype="inference-side only",
    )
    mapping = card(
        550,
        1260,
        205,
        160,
        "Validation + mapping",
        [
            "runtime + checkpoint validation",
            "demeaning",
            "chunk / channel / context",
            "tensor mapping",
        ],
    )
    prediction = card(
        775,
        1260,
        185,
        160,
        "predict()",
        ["DeepLearningPrediction", "artifact · clean · both"],
    )

    processing_context = card(
        40,
        1490,
        210,
        105,
        "ProcessingContext",
        ["aligned EEG"],
    )
    correction = card(
        300,
        1475,
        310,
        150,
        "DeepLearningCorrection",
        [
            "subtract artifact estimate",
            "or replace with clean estimate",
            "store run metadata",
        ],
    )
    corrected = capsule(670, 1518, 135, 48, "corrected EEG")
    downstream = card(
        830,
        1490,
        130,
        105,
        "Continue",
        ["next FACETpy", "processors"],
    )

    diagram.add(
        registry,
        adapter,
        mapping,
        prediction,
        processing_context,
        correction,
        corrected,
        downstream,
    )

    # The only cross-boundary deployment edge starts at the paired hand-off.
    diagram.add_edge(
        edge(
            [(849, 1107), (849, 1200), (150, 1200), (150, 1257)],
            label=("exported model + inference config", 520, 1200),
        ),
        edge([(263, 1340), (277, 1340)]),
        edge([(533, 1340), (547, 1340)]),
        edge([(758, 1340), (772, 1340)]),
        edge(
            [(455, 1472), (455, 1440), (150, 1440), (150, 1423)],
            label=("resolve + predict", 300, 1440),
        ),
        edge(
            [(867, 1423), (867, 1455), (550, 1455), (550, 1472)],
            label=("prediction contract", 750, 1455),
        ),
        edge([(253, 1542), (297, 1542)]),
        edge([(613, 1550), (667, 1550)]),
        edge([(808, 1542), (827, 1542)]),
    )

    return diagram


if __name__ == "__main__":
    output = build()
    output.render_png(str(PNG_PATH), svg_path=str(SVG_PATH), width=1280)
    print(SVG_PATH)
    print(PNG_PATH)

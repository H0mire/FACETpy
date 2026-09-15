#!/usr/bin/env python3
"""Build the FACETpy AAS-derived proof-fit dataset diagram."""

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

TITLE = "FACETpy AAS-Derived Proof-Fit Dataset"
SUBTITLE = "A same-recording surrogate path for bounded learnability evidence"
HEIGHT = 790
SVG_PATH = (ROOT / "output/thesis_figures/framework").joinpath("facetpy_proof_fit_dataset.svg")
PNG_PATH = (ROOT / "output/thesis_figures/framework").joinpath("facetpy_proof_fit_dataset.png")


def build() -> Diagram:
    diagram = Diagram(HEIGHT, title=TITLE, subtitle="", background=False)
    assert diagram.width == CANVAS_W == 1000
    diagram.add(text(70, 83, SUBTITLE, size=15, fill=C["slate"]))

    # The interpretation note comes before the mechanics so the figure cannot be
    # read as an independent-ground-truth construction.
    scope = card(
        45,
        108,
        910,
        105,
        "Interpretation boundary",
        [
            "SAME NIAZY RECORDING supplies both surrogate-clean and artifact terms",
            "AAS-ceiling evidence · proof of fit only · NOT independent clean ground truth",
        ],
    )
    diagram.add(scope)

    source = card(
        45,
        285,
        160,
        96,
        "Niazy recording",
        ["raw in-scanner EEG", "shared source"],
    )

    bundle_frame = container(245, 245, 340, 235, "AAS artifact bundle")
    corrected = card(
        270,
        290,
        290,
        75,
        "AAS-corrected surrogate",
        ["clean-context source"],
    )
    artifact = card(
        270,
        390,
        290,
        75,
        "AAS-estimated artifact",
        ["artifact-context source"],
    )

    epochs = card(
        640,
        285,
        315,
        112,
        "Canonical epochs",
        ["trigger-to-trigger boundaries", "bandlimited resample → 512 samples"],
    )
    reconstruction = pill(
        630,
        425,
        325,
        43,
        "noisy = corrected surrogate + estimated artifact",
    )

    context = card(
        45,
        555,
        260,
        106,
        "Spatio-temporal context",
        ["7 consecutive epochs", "30 EEG channels"],
    )
    target = card(
        370,
        555,
        235,
        106,
        "Supervised target",
        ["center epoch", "artifact waveform"],
    )
    output = card(
        670,
        535,
        285,
        146,
        "Proof-fit NPZ",
        [
            "output/niazy_proof_fit_context_512/",
            "niazy_proof_fit_context_dataset.npz",
            "surrogate pairs + center targets",
        ],
    )

    # Frames first, then nodes; connectors stay beneath every card.
    diagram.add(bundle_frame)
    diagram.add(
        source,
        corrected,
        artifact,
        epochs,
        reconstruction,
        context,
        target,
        output,
    )

    # One source fans into the two bundle products. Both products then share the
    # trigger grid and are packed into the same seven-epoch examples.
    diagram.add_edge(
        edge([(208, 333), (230, 333), (230, 328), (267, 328)]),
        edge([(208, 333), (230, 333), (230, 428), (267, 428)]),
        edge([(563, 328), (610, 328), (610, 321), (637, 321)]),
        edge([(563, 428), (610, 428), (610, 361), (637, 361)]),
        edge([(798, 400), (798, 422)]),
        edge([(798, 471), (798, 505), (175, 505), (175, 552)]),
        edge([(308, 608), (367, 608)]),
        edge([(608, 608), (667, 608)]),
    )

    return diagram


if __name__ == "__main__":
    output = build()
    output.render_png(str(PNG_PATH), svg_path=str(SVG_PATH), width=1280)
    print(SVG_PATH)
    print(PNG_PATH)

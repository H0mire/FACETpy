"""Run the Phase-0 architecture in the current pipeline (adapted execution)."""
from __future__ import annotations

import argparse
from pathlib import Path

from facet.correction import DeepLearningCorrection
from facet.models.masterthesis import pipeline
from facet.models.masterthesis.legacy_cascaded_dae import LegacyDLAdapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    model = LegacyDLAdapter(args.checkpoint, device=args.device)
    model._load()
    result = pipeline.build(args.edf, correctors=[DeepLearningCorrection(model=model)],
                            name="phase0_adapted").run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.get_raw().save(args.out, overwrite=False)


if __name__ == "__main__":
    main()

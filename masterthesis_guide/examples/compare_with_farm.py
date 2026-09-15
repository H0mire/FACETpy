"""Produce matched FARM and selected-model pipeline outputs for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from facet.correction import DeepLearningCorrection
from facet.models.masterthesis import pipeline
from masterthesis_guide.reproduce import adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    model = adapter(args.experiment, device=args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, correctors in (("farm", pipeline.farm()), (args.experiment, [DeepLearningCorrection(model=model)])):
        result = pipeline.build(args.edf, correctors=correctors, name=name).run()
        if not result.success:
            raise RuntimeError(result.error)
        result.get_raw().save(args.out_dir / f"{name}_raw.fif", overwrite=False)


if __name__ == "__main__":
    main()

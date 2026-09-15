"""Run one catalogued model in the recorded FACETpy pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from facet.correction import DeepLearningCorrection
from facet.models.masterthesis import pipeline
from masterthesis_guide.reproduce import adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Stable experiment ID from the catalog")
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output MNE FIF file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--without-pca", action="store_true")
    args = parser.parse_args(argv)
    if not args.edf.is_file():
        parser.error(f"Input recording is missing: {args.edf}")
    model = adapter(args.experiment, device=args.device)
    # Resolve the model before running the more expensive recording preprocessing.
    if hasattr(model, "_load_model"):
        model._load_model()
    else:
        model._load()
    result = pipeline.build(args.edf, correctors=[DeepLearningCorrection(model=model)],
                            include_pca=not args.without_pca, name=args.experiment).run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.get_raw().save(args.out, overwrite=False)


if __name__ == "__main__":
    main()

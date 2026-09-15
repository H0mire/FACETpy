"""Evaluate a selected model using the recorded Phase-1 holdout protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from facet.models.masterthesis.adapters import predict_from_context
from masterthesis_guide.reproduce import ROOT, adapter, data_path, load_catalog
from facet.evaluation.thesis_metrics import compute_metrics
from masterthesis_guide.reproduce import load_holdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Phase-1 holdout experiment ID")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    catalog = load_catalog()
    experiment = catalog["experiments"][args.experiment]
    if experiment["phase"] not in (1, 2) or not experiment["artifacts"]:
        parser.error(
            "Select a Phase-1 or Phase-2 neural model with an available artifact. This command always uses the unified-holdout protocol."
        )
    indices_record = json.loads((ROOT / catalog["datasets"]["proof_fit"]["split"]).read_text())
    indices = np.asarray(indices_record["indices"], dtype=int)
    holdout = load_holdout(data_path("proof_fit", catalog, data_root=args.data_root), indices)
    model_adapter = adapter(args.experiment, catalog, device=args.device)
    model, _ = model_adapter._load_model()
    # These names and the metric call are checked against the retained evaluator.
    prediction = predict_from_context(model_adapter.packing, model, holdout["noisy_context"], device=args.device)
    metrics = compute_metrics(
        holdout["noisy_center"],
        holdout["clean_center"],
        holdout["artifact_center"],
        prediction,
        sfreq_hz=holdout["sfreq"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()

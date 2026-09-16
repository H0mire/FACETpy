"""Compare deployed artifact predictors on identical, locked Weg-A examples.

The original Phase-3 model and the retrained model differ in training data and
objective weights. This is a checkpoint comparison, not a spike-loss ablation.
Amplitude metrics include underlying EEG; they do not isolate injected spikes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from facet.training.deployment_data import build_weg_a_packed_dataset
from facet.training.spike_metrics import compute_spike_metrics, compute_spike_metrics_per_example


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = yaml.safe_load(args.config.read_text())
    # The original Demucs trace embeds CPU LSTM hidden-state tensors.
    baseline_device = args.baseline_device or (
        "cpu" if ".demucs.deployment." in config["model"]["factory"] else args.device
    )
    dataset = build_weg_a_packed_dataset(
        args.dataset,
        packing=config["data"]["kwargs"]["packing"],
        include_spike=True,
        max_shift=0,
        background_mix_prob=0.0,
    )
    with np.load(args.dataset, allow_pickle=True) as archive:
        split = archive["example_split"]
    if dataset.n_channels != 1 or len(dataset) != len(split):
        raise ValueError("This comparison requires the locked single-channel dataset")
    indices = np.flatnonzero(split == 2)
    if not len(indices):
        raise ValueError("No locked holdout examples (example_split == 2)")
    factory_module, factory_name = config["model"]["factory"].split(":")
    model = getattr(importlib.import_module(factory_module), factory_name)(**config["model"]["kwargs"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    models = {
        "phase3": torch.jit.load(str(args.baseline), map_location=baseline_device).eval(),
        "spike_aware": model.to(args.device).eval(),
    }
    predictions = {name: [] for name in models}
    targets, masks, inputs = [], [], []
    with torch.inference_mode():
        for offset in range(0, len(indices), args.batch_size):
            pairs = [dataset[int(i)] for i in indices[offset : offset + args.batch_size]]
            x = torch.from_numpy(np.stack([pair[0] for pair in pairs])).to(args.device)
            y = np.stack([pair[1] for pair in pairs])
            clean, noisy, mask = y[:, 1, 0], y[:, 2, 0], y[:, 3, 0]
            targets.append(clean)
            masks.append(mask)
            inputs.append(noisy)
            for name, predictor in models.items():
                model_input = x.to(baseline_device) if name == "phase3" else x
                artifact = predictor(model_input).detach().cpu().numpy()
                if artifact.shape != (len(pairs), 1, dataset.epoch_samples):
                    raise ValueError(f"Unexpected {name} artifact shape: {artifact.shape}")
                corrected = noisy - artifact[:, 0]
                if not np.isfinite(corrected).all():
                    raise ValueError(f"Nonfinite predictions from {name}")
                predictions[name].append(corrected)
    target = np.concatenate(targets)
    mask = np.concatenate(masks)
    predictions = {name: np.concatenate(parts) for name, parts in predictions.items()}
    predictions["uncorrected"] = np.concatenate(inputs)
    # Match the existing evaluation protocol's native trigger duration.
    effective_sfreq = dataset.epoch_samples / 0.194
    neighborhood = int(round(0.050 * effective_sfreq))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Locked Weg-A checkpoint comparison; not an isolated loss ablation",
        "split": 2,
        "n_examples": len(indices),
        "epoch_duration_s": 0.194,
        "device": args.device,
        "baseline_device": baseline_device,
        "neighborhood_ms": 50,
        "spike_dilate_ms": 0,
        "dataset_sha256": sha256(args.dataset),
        "baseline": str(args.baseline),
        "baseline_sha256": sha256(args.baseline),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": config,
        "metrics": {},
    }
    for name, prediction in predictions.items():
        metrics = compute_spike_metrics(prediction, target, mask, neighborhood_samples=neighborhood)
        metrics["rmse_uv"] = float(np.sqrt(np.mean((prediction - target) ** 2)) * 1e6)
        report["metrics"][name] = metrics
        per_example = compute_spike_metrics_per_example(
            prediction,
            target,
            mask,
            neighborhood_samples=neighborhood,
        )
        per_example["dataset_index"] = indices[per_example["example_index"]]
        with (args.output_dir / f"{name}_per_example.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(per_example)
            writer.writerows(zip(*per_example.values(), strict=True))
    np.savez_compressed(
        args.output_dir / "predictions.npz", dataset_index=indices, target=target, spike_mask=mask, **predictions
    )
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()

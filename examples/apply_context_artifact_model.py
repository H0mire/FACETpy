"""Apply a trained multi-epoch artifact model to an EEG-fMRI recording.

The model is expected to be a TorchScript export produced by
``facet-train fit`` from ``src/facet/models/demo01/training.yaml``.
It consumes seven consecutive trigger-defined epochs and predicts the artifact
for the center epoch only.

Example:
    uv run python examples/apply_context_artifact_model.py \
        --checkpoint training_output/sevenepochcontextartifactnet_20260429_204945/exports/seven_epoch_context_artifact_net.ts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np

from facet import (
    DownSample,
    DropChannels,
    EpochContextDeepLearningCorrection,
    HighPassFilter,
    LowPassFilter,
    TriggerAligner,
    TriggerDetector,
    UpSample,
    load,
)

DEFAULT_INPUT = Path("./examples/datasets/NiazyFMRI.edf")
DEFAULT_OUTPUT_DIR = Path("./output/context_artifact_model_inference")
DEFAULT_NON_EEG_CHANNELS = ["EKG", "EMG", "EOG", "ECG"]


def _artifact_raw_from_context(context) -> mne.io.BaseRaw:
    estimated_noise = context.get_estimated_noise()
    if estimated_noise is None:
        raise RuntimeError("The context does not contain a predicted artifact estimate.")

    raw = context.get_raw()
    artifact_raw = mne.io.RawArray(estimated_noise.astype(np.float32), raw.info.copy(), verbose=False)
    artifact_raw.set_meas_date(raw.info["meas_date"])
    return artifact_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input EEG-fMRI recording.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="TorchScript artifact model export.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for inference outputs.")
    parser.add_argument("--trigger-regex", default=r"\b1\b", help="Trigger event regex.")
    parser.add_argument("--context-epochs", type=int, default=7, help="Odd number of epochs consumed by the model.")
    parser.add_argument(
        "--epoch-samples",
        type=int,
        default=292,
        help="Fixed model epoch length. Use the value from the training dataset metadata.",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device used for TorchScript inference.")
    parser.add_argument("--upsample", type=int, default=10, help="Temporary upsampling factor for trigger alignment.")
    parser.add_argument(
        "--keep-input-mean",
        action="store_true",
        help="Do not remove the per-epoch input mean before model inference.",
    )
    parser.add_argument(
        "--keep-prediction-mean",
        action="store_true",
        help="Do not remove the per-epoch mean from predicted artifacts before subtraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context = load(str(args.input), preload=True, artifact_to_trigger_offset=-0.005)

    context = (
        context
        | DropChannels(channels=DEFAULT_NON_EEG_CHANNELS)
        | TriggerDetector(regex=args.trigger_regex)
        | HighPassFilter(freq=1.0)
        | UpSample(factor=args.upsample)
        | TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False)
        | DownSample(factor=args.upsample)
        | EpochContextDeepLearningCorrection(
            checkpoint_path=args.checkpoint,
            context_epochs=args.context_epochs,
            epoch_samples=args.epoch_samples,
            device=args.device,
            demean_input=not args.keep_input_mean,
            remove_prediction_mean=not args.keep_prediction_mean,
        )
        | LowPassFilter(freq=70.0)
    )

    corrected_path = args.output_dir / "context_dl_corrected_raw.fif"
    artifact_path = args.output_dir / "context_dl_predicted_artifact_raw.fif"

    context.get_raw().save(corrected_path, overwrite=True, verbose=False)
    _artifact_raw_from_context(context).save(artifact_path, overwrite=True, verbose=False)

    run = context.metadata.custom["epoch_context_deep_learning_runs"][-1]
    print("Saved context deep-learning inference outputs:")
    print(f"  corrected raw : {corrected_path}")
    print(f"  artifact raw  : {artifact_path}")
    print(f"  checkpoint    : {args.checkpoint}")
    print(f"  channels      : {len(run['channels'])}")
    print(f"  center epochs : {run['corrected_epochs']}")
    print(f"  epoch samples : {run['epoch_samples']}")
    print(
        "  native lengths: "
        f"{run['epoch_length_min']} / {run['epoch_length_median']:.1f} / {run['epoch_length_max']} samples"
    )


if __name__ == "__main__":
    main()

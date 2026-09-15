"""Generate matched pipeline arms for residual and spike-preservation evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from facet.correction import DeepLearningCorrection
from facet.models.masterthesis import pipeline
from masterthesis_guide.reproduce import adapter, load_catalog, selected_artifact, sha256

WINDOW_START_S, WINDOW_STOP_S = 25.0, 160.0


def save_arm(result, path: Path, elapsed: float) -> None:
    """Preserve the original arm format: EEG microvolts and window-local triggers."""
    if not result.success:
        raise RuntimeError(result.error)
    raw = result.get_raw()
    sfreq = float(raw.info["sfreq"])
    first, last = int(WINDOW_START_S * sfreq), int(WINDOW_STOP_S * sfreq)
    if raw.n_times < last:
        raise ValueError("The recording does not cover the complete 25–160 s evaluation window")
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude=[])
    triggers = np.asarray(result.context.get_triggers(), dtype=np.int64)
    triggers = triggers[(triggers >= first) & (triggers < last)] - first
    np.savez_compressed(
        path,
        data=raw.get_data(picks=picks)[:, first:last].astype(np.float32) * 1e6,
        ch_names=np.asarray([raw.ch_names[i] for i in picks]),
        sfreq=sfreq,
        elapsed=elapsed,
        triggers=triggers,
        window_s=np.asarray([WINDOW_START_S, WINDOW_STOP_S]),
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", default=[], help="Catalog ID; repeat for multiple models")
    parser.add_argument("--baseline", action="append", choices=["farm", "uncorrected"], default=[])
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-fif", action="store_true", help="Also save the full corrected recording for spectra")
    args = parser.parse_args(argv)
    if not args.experiment and not args.baseline:
        parser.error("Select an experiment or baseline")
    catalog = load_catalog()
    jobs = [(name, pipeline.farm() if name == "farm" else [], None) for name in args.baseline]
    for eid in args.experiment:
        aid, _ = selected_artifact(eid, catalog, device=args.device)
        model = adapter(eid, catalog, device=args.device)
        jobs.append((eid, [DeepLearningCorrection(model=model)], aid))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, correctors, aid in jobs:
        output = args.out_dir / f"{name}.npz"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite a previous arm: {output}")
        start = time.perf_counter()
        result = pipeline.build(args.edf, correctors=correctors, name=name).run()
        save_arm(result, output, time.perf_counter() - start)
        if args.save_fif:
            result.get_raw().save(args.out_dir / f"{name}_raw.fif", overwrite=False)
        metadata = {
            "experiment": name,
            "artifact": aid,
            "device": args.device,
            "input_sha256": sha256(args.edf),
            "protocol": "pipeline",
            "window_s": [WINDOW_START_S, WINDOW_STOP_S],
            "artifact_sha256": catalog["artifacts"][aid]["sha256"] if aid else None,
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(output, flush=True)


if __name__ == "__main__":
    main()

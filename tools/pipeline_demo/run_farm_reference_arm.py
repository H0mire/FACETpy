"""Export matched FARM and uncorrected reference arms for a given EDF input."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "pipeline_demo"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import mne
    import reference_chain

    args.out.mkdir(parents=True, exist_ok=True)
    for name, correctors, include_pca in [
        ("uncorrected", [], False),
        ("farm", reference_chain.farm(), True),
    ]:
        t0 = time.perf_counter()
        result = reference_chain.build(args.edf, correctors=correctors,
                                       include_pca=include_pca, name=name).run()
        if not result.success:
            raise RuntimeError(f"{name}: {result.error}")
        raw = result.context.get_raw()
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude=[])
        sfreq = float(raw.info["sfreq"])
        i0, i1 = int(25 * sfreq), int(160 * sfreq)
        data = raw.get_data(picks=picks)[:, i0:i1].astype(np.float32) * 1e6
        triggers = np.asarray(result.context.get_triggers() if result.context.has_triggers() else [], dtype=np.int64)
        triggers = triggers[(triggers >= i0) & (triggers < i1)] - i0
        np.savez_compressed(args.out / f"{name}.npz", data=data,
                            ch_names=np.asarray([raw.ch_names[i] for i in picks], dtype=object),
                            sfreq=sfreq, elapsed=time.perf_counter() - t0,
                            triggers=triggers, window_s=np.asarray([25.0, 160.0]))
        print(f"{name}: complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

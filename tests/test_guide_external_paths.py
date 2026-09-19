"""Guide tools must accept data outside the repository and retain its origin."""

import json
import subprocess
import sys
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run_tool(relative, *args):
    subprocess.run(
        [sys.executable, str(ROOT / relative), *map(str, args)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_spike_injection_with_external_edf(tmp_path):
    source = tmp_path / "source.edf"
    output = tmp_path / "injected.edf"
    time = np.arange(1000) / 500
    values = np.stack([np.sin(2 * np.pi * 8 * time), np.cos(2 * np.pi * 9 * time)]) * 1e-6
    raw = mne.io.RawArray(values, mne.create_info(["Fp1", "Fp2"], 500, "eeg"), verbose=False)
    mne.export.export_raw(source, raw, fmt="edf", verbose=False)
    original = source.read_bytes()

    run_tool("tools/pipeline_demo/inject_spikes.py", "--input", source, "--out", output, "--at", "1.0")

    truth = json.loads(output.with_suffix(".truth.json").read_text())
    assert (ROOT / truth["input"]).resolve() == source.resolve()
    assert len(truth["spikes"]) == 1
    assert truth["n_channels_injected"] == 2
    injected = mne.io.read_raw_edf(output, preload=True, verbose=False).get_data()
    assert np.max(np.abs(injected - values)) > 50e-6
    assert source.read_bytes() == original


def test_single_channel_derivation_with_external_metadata(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "derived.npz"
    context = np.arange(2 * 3 * 2 * 8, dtype=np.float32).reshape(2, 3, 2, 8)
    split = np.array([0, 2])
    np.savez(
        source,
        artifact_context=context,
        artifact_context_template=context + 1,
        clean_context=context + 2,
        neighbor_channel_indices=np.array([[4, 5], [7, 8]]),
        target_channel_index=np.array([4, 7]),
        k_neighbors=np.array([1]),
        example_split=split,
    )
    metadata = source.with_name("source_metadata.json")
    metadata.write_text(json.dumps({"k_neighbors": 1, "input_shape": [3, 2, 8]}))
    original, original_meta = source.read_bytes(), metadata.read_bytes()

    run_tool("tools/dataset_building/derive_single_channel_weg_a.py", "--input", source, "--output", output)

    with np.load(output) as derived:
        np.testing.assert_array_equal(derived["artifact_context"], context[:, :, :1])
        np.testing.assert_array_equal(derived["example_split"], split)
    result = json.loads(output.with_name("derived_metadata.json").read_text())
    assert result["input_shape"] == [3, 1, 8]
    assert result["k_neighbors"] == 0
    assert (ROOT / result["derived_from"]).resolve() == source.resolve()
    assert source.read_bytes() == original
    assert metadata.read_bytes() == original_meta

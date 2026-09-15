"""Measure saved pipeline arms using the recorded residual and seam protocol.

Inputs contain microvolts and triggers relative to the 25–160 s window.
Residuals are evaluated from 4.5 to 135 s inside that window. The comb score
measures periodic artifact; the separate seam score detects discontinuities.
Original CSV field names are retained for comparison with recorded evidence."""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
from pathlib import Path
import mne
import numpy as np
from facet.core import ProcessingContext, ProcessingMetadata
from facet.evaluation import EpochSeamStepCalculator, GradientArtifactResidualCalculator
REPO = Path(__file__).resolve().parents[2]
T0_IN_WINDOW, T1_IN_WINDOW = (4.5, 135.0)
WINDOW_OFFSET_S = 25.0
SEAM_SUSPECT = 1.8
SEAM_BROKEN = 2.5

def measure(path: Path, t0: float, t1: float, step_channel: str) -> dict:
    a = np.load(path, allow_pickle=True)
    names = [str(x) for x in a['ch_names']]
    sfreq = float(a['sfreq'])
    raw = mne.io.RawArray(a['data'].astype(np.float64) * 1e-06, mne.create_info(names, sfreq, 'eeg'), verbose=False)
    ctx = ProcessingContext(raw=raw, metadata=ProcessingMetadata(triggers=list(map(int, a['triggers']))))
    comb = GradientArtifactResidualCalculator(tmin=t0, tmax=t1).execute(ctx).metadata.custom['gradient_artifact_residual']
    seam = EpochSeamStepCalculator(tmin=t0, tmax=t1).execute(ctx).metadata.custom['epoch_seam_step']
    ch = step_channel if step_channel in names else names[0]
    diffs = np.abs(np.diff(a['data'][names.index(ch)][int(t0 * sfreq):int(t1 * sfreq)].astype(np.float64)))
    return {'arm': path.stem, 'rms_uv': round(comb['rms_uv'], 3), 'ga_rest_uv': round(comb['comb_rms_uv'], 3), 'naht_ratio': round(seam['ratio'], 3), 'median_sample_step_uv': round(float(np.median(diffs)), 4), 'step_channel': ch, 'n_harmonics': comb['n_harmonics'], 'epoch_rate_hz': round(comb['epoch_rate_hz'], 4)}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reference', default='farm', help='Reference.')
    ap.add_argument('--channel', default='Fp1', help='Channel.')
    ap.add_argument('--start', type=float, default=T0_IN_WINDOW)
    ap.add_argument('--stop', type=float, default=T1_IN_WINDOW)
    args = ap.parse_args()
    paths = sorted((Path(p) for p in glob.glob(str(args.arm_dir / '*.npz'))))
    if not paths:
        raise SystemExit(f'No arms found in {args.arm_dir} — run tools/pipeline_demo/run_arms.py first')
    rows = [measure(p, args.start, args.stop, args.channel) for p in paths]
    ref = next((r for r in rows if r['arm'] == args.reference), None)
    if ref is None:
        raise SystemExit(f"Reference arm {args.reference!r} is missing; available: {[r['arm'] for r in rows]}")
    for r in rows:
        r['x_reference'] = round(r['ga_rest_uv'] / ref['ga_rest_uv'], 2)
        r['hinweis'] = 'Invalid seam; inspect waveform' if r['naht_ratio'] >= SEAM_BROKEN else 'Suspicious seam' if r['naht_ratio'] >= SEAM_SUSPECT else ''
    rows.sort(key=lambda r: r['ga_rest_uv'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {'arm_dir': os.path.relpath(args.arm_dir, REPO), 'window_s': [WINDOW_OFFSET_S + args.start, WINDOW_OFFSET_S + args.stop], 'reference_arm': args.reference, 'step_channel': args.channel, 'metrics': {'ga_rest_uv': 'GradientArtifactResidualCalculator.comb_rms_uv, fmax 70 Hz', 'naht_ratio': 'EpochSeamStepCalculator.ratio', 'median_sample_step_uv': 'Median absolute adjacent-sample difference on the selected channel.'}, 'caveat': 'The comb residual depends on the time window and measures only periodic residuals. Interpret it together with the seam score and waveform.'}
    meta_path = args.out.with_suffix('.meta.json')
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"{'Arm':32s}{'RMS':>8s}{'GA-Rest':>9s}{'×ref':>7s}{'Naht':>7s}  Note")
    for r in rows:
        print(f"{r['arm']:32s}{r['rms_uv']:8.2f}{r['ga_rest_uv']:9.2f}{r['x_reference']:7.1f}{r['naht_ratio']:7.2f}  {r['hinweis']}")
    print(f'\nWritten: {args.out}\n             {meta_path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

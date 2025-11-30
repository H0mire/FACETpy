import mne
from scipy.io import loadmat
import numpy as np


def load_mat_to_mne(mat_path, sfreq=500.0):
    """
    Load EEG data from a .mat file and create an MNE Raw object.

    Parameters
    ----------
    mat_path : str
        Path to the .mat file containing 'eeg_data' array (in µV).
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object with EEG data (in volts).
    """
    # Load the .mat file
    mat = loadmat(mat_path)
    eeg = mat['eeg_data']  # shape (n_ch, n_samples), in µV

    # Build an MNE RawArray in volts (MNE expects volts)
    ch_names = [f"EEG{i+1}" for i in range(eeg.shape[0])]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg / 1e6, info)

    return raw


def filter_eeg(raw):
    """
    Apply notch and band-pass filtering to EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object with EEG data (in volts).

    Returns
    -------
    raw : mne.io.Raw
        Filtered MNE Raw object.
    """
    # Notch + band-pass
    raw.notch_filter([50, 100], picks="eeg", verbose=False)
    raw.filter(1.0, 70.0, picks="eeg", verbose=False)

    return raw


def parse_spike_times(mat_or_path, label_marker="!", markers_alt=None, debug=False):
    """Extract spike times (seconds) from a .mat structure.

    Accepts either a loaded dict (scipy.io.loadmat) or a path.

    Parameters
    ----------
    mat_or_path : dict | str
        Loaded mat dict or path to .mat file containing 'events'.
    label_marker : str
        Primary marker indicating a spike (default '!').
    markers_alt : list[str] | None
        Optional list of additional marker strings to accept.
    debug : bool
        If True, prints diagnostic information about event parsing.

    Returns
    -------
    spike_sec : list[float]
        Sorted unique spike onset times in seconds.
    """
    if isinstance(mat_or_path, str):
        try:
            mat_dict = loadmat(mat_or_path)
        except Exception as e:
            if debug:
                print(f"Failed to load mat file '{mat_or_path}': {e}")
            return []
    else:
        mat_dict = mat_or_path

    events = mat_dict.get('events')
    if events is None:
        if debug:
            print("No 'events' key found. Available keys:", [k for k in mat_dict.keys() if not k.startswith('__')])
        return []

    spike_sec = []
    accepted_markers = {label_marker}
    if markers_alt:
        accepted_markers.update(markers_alt)

    # events expected shape (n_events, 3) with time string in first col, label in third
    for i, row in enumerate(events):
        try:
            if row.shape[0] < 3:
                continue
            time_raw = row[0]
            label_raw = row[2]
            # Normalize label
            label = label_raw.strip() if isinstance(label_raw, str) else str(label_raw).strip()
            if label in accepted_markers:
                # Normalize time string: MATLAB may pad with spaces
                t_str = time_raw.strip() if isinstance(time_raw, str) else str(time_raw).strip()
                t_val = float(t_str)
                spike_sec.append(t_val)
        except Exception as e:
            if debug and i < 10:
                print(f"Row {i} parse error: {e}")
            continue

    # Deduplicate & sort
    if spike_sec:
        spike_sec = sorted(set(spike_sec))

    if debug:
        unique_labels = sorted({(row[2].strip() if isinstance(row[2], str) else str(row[2]).strip()) for row in events})
        print(f"Parsed {len(spike_sec)} spike times. Unique labels present: {unique_labels}")
        if not spike_sec:
            print("Warning: No spike markers matched; check label_marker argument.")

    return spike_sec
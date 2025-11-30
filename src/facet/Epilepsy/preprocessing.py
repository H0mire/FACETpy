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

    # Try to find channel names
    ch_names = None
    possible_keys = ['channels', 'channel_labels', 'labels', 'chan_names']
    for key in possible_keys:
        if key in mat:
            try:
                # Handle different ways strings might be stored in .mat
                vals = mat[key]
                if vals.shape[0] == eeg.shape[0]: # Column vector or list matching channels
                    ch_names = [str(v[0]) if isinstance(v, np.ndarray) else str(v) for v in vals]
                    # Clean up formatting (e.g. remove extra quotes or spaces)
                    ch_names = [n.strip().replace("'", "").replace('"', "") for n in ch_names]
                    break
                elif vals.shape[1] == eeg.shape[0]: # Row vector
                    ch_names = [str(v) for v in vals[0]]
                    ch_names = [n.strip().replace("'", "").replace('"', "") for n in ch_names]
                    break
            except Exception as e:
                print(f"Found key '{key}' but failed to parse channel names: {e}")

    if ch_names is None:
        # Check if it matches the known 29-channel dataset (vepiset)
        if eeg.shape[0] == 29:
            print("Detected 29 channels, assuming vepiset dataset layout.")
            ch_names = [
                'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz',
                'PG1', 'PG2', 'A1', 'A2',
                'ECG1', 'ECG2',
                'EMG1', 'EMG2', 'EMG3', 'EMG4'
            ]
            # Define types
            # First 19 are standard EEG
            # Next 4 are auricular (PG1, PG2, A1, A2). 
            # We set them to 'misc' because PG1/PG2 often have overlapping positions in standard montages,
            # causing MNE plotting to crash. Also, we typically want ICA on scalp EEG only.
            ch_types = ['eeg'] * 19 + ['misc'] * 4 + ['ecg'] * 2 + ['emg'] * 4
        else:
            # Fallback to generic names
            ch_names = [f"EEG{i+1}" for i in range(eeg.shape[0])]
            ch_types = 'eeg'
    else:
        ch_types = 'eeg'

    # Build an MNE RawArray in volts (MNE expects volts)
    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg / 1e6, info)

    # Try to set a standard montage if names look standard
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        # Only set if enough channels match
        # MNE's set_montage will raise an error or warning if channels don't match, 
        # but we can use on_missing='ignore' or 'warn'
        raw.set_montage(montage, on_missing='ignore')
    except Exception as e:
        print(f"Could not set standard montage: {e}")

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


def prepare_eeg_data(mat_path, sfreq=500.0):
    """
    Load, filter, and parse EEG data for the pipeline.

    Parameters
    ----------
    mat_path : str
        Path to the .mat file.
    sfreq : float
        Sampling frequency.

    Returns
    -------
    raw : mne.io.Raw
        Standard filtered raw object (1-70Hz).
    raw_ica : mne.io.Raw
        ICA filtered raw object (1-100Hz).
    spike_sec : list
        List of spike times in seconds.
    """
    raw = load_mat_to_mne(mat_path, sfreq=sfreq)
    raw = filter_eeg(raw)
    raw_ica = raw.copy().filter(1., 100., picks='eeg', verbose=False)

    mat = loadmat(mat_path)
    spike_sec = parse_spike_times(mat, label_marker="!")

    return raw, raw_ica, spike_sec
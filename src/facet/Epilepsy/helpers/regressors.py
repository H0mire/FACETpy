import numpy as np
from scipy.signal import resample, fftconvolve
from mne.filter import filter_data


def _double_gamma_hrf(t, peak_time=6.0, undershoot=16.0, ratio=6.0):
    """Double-gamma HRF (canonical, peaking at ~6s)."""
    t = np.array(t, dtype=float)
    h1 = (t / peak_time)**2 * np.exp(-t / peak_time)
    h2 = (t / undershoot)**2 * np.exp(-t / undershoot) / ratio
    hrf = h1 - h2
    hrf[hrf < 0] = 0
    hrf /= hrf.max() if hrf.max() > 0 else 1
    return hrf


# ======================= Grouiller 2011 — topography-based regressor =======================

def _build_epileptic_map(raw, spike_sec, half_win_s=0.15, band=(1., 30.)):
    """Build the epileptic voltage map from averaged spike epochs.

    Grouiller 2011: "spikes were averaged and the EEG voltage map
    corresponding to the peak of the global field power of the average
    spike was considered as the epileptic map."

    Parameters
    ----------
    raw : mne.io.Raw
        EEG data (outside-scanner or in-scanner long-term recording).
    spike_sec : array-like
        Spike times in seconds.
    half_win_s : float
        Half-window around each spike (seconds).
    band : tuple
        Band-pass filter range. Paper uses 1–30 Hz.

    Returns
    -------
    epileptic_map : ndarray, shape (n_channels,)
        Voltage vector at GFP peak of the averaged spike, divided by its
        GFP (unit-norm topography).
    """
    import mne

    sf = raw.info['sfreq']
    hw = int(round(half_win_s * sf))
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')

    # Band-pass filter a copy to 1–30 Hz (paper spec)
    raw_filt = raw.copy().filter(band[0], band[1], picks='eeg', verbose=False)
    data = raw_filt.get_data(picks=eeg_picks)  # (n_ch, n_times)

    # Epoch and average
    epochs = []
    for t in spike_sec:
        center = int(round(t * sf))
        start, stop = center - hw, center + hw
        if 0 <= start and stop <= data.shape[1]:
            epochs.append(data[:, start:stop])

    if len(epochs) < 3:
        raise ValueError(f"Only {len(epochs)} valid spike epochs — need ≥3 for a stable map.")

    avg = np.mean(epochs, axis=0)  # (n_ch, win_len)

    # GFP at each time point: spatial std across channels
    gfp = np.std(avg, axis=0)
    peak_idx = np.argmax(gfp)

    # Epileptic map = voltage vector at GFP peak, normalised by its GFP
    epileptic_map = avg[:, peak_idx]
    gfp_peak = gfp[peak_idx]
    if gfp_peak > 0:
        epileptic_map = epileptic_map / gfp_peak

    return epileptic_map


def _compute_spatial_correlation_timecourse(raw, epileptic_map, band=(1., 30.)):
    """Compute continuous spatial correlation between epileptic map and EEG.

    Grouiller 2011: "For each time frame of the intra-MRI EEG, we
    calculated the spatial correlation with the epileptic map template.
    Correlation is based only on topographic comparison by dividing the
    maps by the global field power with no consideration of the polarity
    (the absolute value of the correlation is used)."

    "The time course of the square of the correlation coefficient
    quantifying the presence of the epileptic map was convolved with the
    canonical haemodynamic response function."

    Parameters
    ----------
    raw : mne.io.Raw
        In-scanner EEG (artefact-corrected).
    epileptic_map : ndarray, shape (n_channels,)
        GFP-normalised epileptic voltage map.
    band : tuple
        Band-pass filter range (paper: 1–30 Hz).

    Returns
    -------
    corr_sq : ndarray, shape (n_times,)
        Squared absolute spatial correlation at each time frame.
    """
    import mne

    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')

    raw_filt = raw.copy().filter(band[0], band[1], picks='eeg', verbose=False)
    data = raw_filt.get_data(picks=eeg_picks)  # (n_ch, n_times)

    n_ch, n_times = data.shape

    # GFP at each time frame
    gfp = np.std(data, axis=0)  # (n_times,)
    gfp[gfp == 0] = 1e-12  # avoid division by zero

    # Normalise each time frame by its GFP
    data_norm = data / gfp[np.newaxis, :]  # (n_ch, n_times)

    # Spatial correlation: dot product of normalised map with normalised data
    # epileptic_map is already GFP-normalised
    # Pearson correlation across channels for each time frame:
    map_centered = epileptic_map - epileptic_map.mean()
    map_norm = np.linalg.norm(map_centered)
    if map_norm == 0:
        return np.zeros(n_times)

    corr = np.zeros(n_times)
    for t_idx in range(n_times):
        frame = data_norm[:, t_idx]
        frame_centered = frame - frame.mean()
        frame_norm = np.linalg.norm(frame_centered)
        if frame_norm > 0:
            corr[t_idx] = np.dot(map_centered, frame_centered) / (map_norm * frame_norm)

    # Polarity-invariant + square (paper: "|correlation|² ")
    corr_sq = corr ** 2  # squaring already removes sign

    return corr_sq


def build_grouiller_regressor(raw, spike_sec, half_win_s=0.15, tr=2.5, band=(1., 30.)):
    """Build Grouiller 2011 topography-based fMRI regressor.

    Grouiller 2011: "We built patient-specific EEG maps of the epileptic
    activity derived from spikes detected in the long-term clinical
    video-EEG monitoring. We then calculated the strength of the presence
    of these maps in the EEG recorded during functional MRI as a function
    of time using correlation. Finally, we used this correlation
    coefficient to inform functional MRI analysis."

    Steps:
    1. Build epileptic voltage map from averaged spikes (GFP-peak snapshot)
    2. Compute spatial correlation of this map with every EEG time frame
    3. Square the correlation (polarity-invariant, emphasises high values)
    4. Convolve with canonical HRF
    5. Downsample to fMRI TR

    Parameters
    ----------
    raw : mne.io.Raw
        In-scanner EEG data (artefact-corrected, with spike annotations).
    spike_sec : array-like
        Spike times in seconds (from long-term EEG or in-scanner markings).
    half_win_s : float
        Half-window for spike epoching (seconds).
    tr : float
        fMRI repetition time (seconds).
    band : tuple
        Band-pass for both map construction and correlation (paper: 1–30 Hz).

    Returns
    -------
    regressor : ndarray
        Regressor time course sampled at TR.
    epileptic_map : ndarray
        The epileptic voltage map used (for diagnostics / plotting).
    """
    sf = raw.info['sfreq']
    total_duration = raw.n_times / sf

    # Step 1: build epileptic voltage map
    epileptic_map = _build_epileptic_map(raw, spike_sec, half_win_s=half_win_s, band=band)

    # Step 2–3: continuous spatial correlation (squared)
    corr_sq = _compute_spatial_correlation_timecourse(raw, epileptic_map, band=band)

    # Step 4: convolve with canonical HRF
    hrf_len = int(20 * sf)  # 20s kernel at EEG sampling rate
    hrf_t = np.arange(hrf_len) / sf
    hrf = _double_gamma_hrf(hrf_t)
    convolved = fftconvolve(corr_sq, hrf)[:len(corr_sq)]

    # Step 5: downsample to TR
    n_tr = int(np.floor(total_duration / tr))
    t_eeg = np.arange(len(convolved)) / sf
    tr_times = np.arange(n_tr) * tr
    regressor = np.interp(tr_times, t_eeg, convolved)

    return regressor, epileptic_map

# Ebrahimzadeh et al. 2021 HRF regressor generation
# ======================= HRF convolution =======================
def hrf_kernel(t, peak=6, undershoot=16):
    """Simple double-gamma HRF kernel."""
    h1 = (t / peak) ** 2 * np.exp(-t / peak)
    h2 = 0.1 * (t / undershoot) ** 2 * np.exp(-t / undershoot)
    return h1 - h2

def generate_hrf_regressors(component_tc, sfreq, peaks_s=[3, 5, 7, 9], tr=2.5):
    """Generate HRF-convolved regressors for component timecourse, resampled to fMRI TR."""
    t = np.arange(0, 20, 1/sfreq)  # 20s kernel
    total_duration_sec = len(component_tc) / sfreq
    n_tr = int(np.floor(total_duration_sec / tr))
    regressors = {}
    for peak in peaks_s:
        kernel = hrf_kernel(t, peak=peak)
        regressor_full = np.convolve(component_tc, kernel, mode='same')
        # Resample to TR
        regressor_resampled = resample(regressor_full, n_tr)
        regressors[f'{peak}s'] = regressor_resampled
    return regressors


# Grouiller et al.  HRF regressor generation

def compute_and_attach_ica_regressors(detection, sfreq, tr):
    """
    Compute ICA regressors and attach them to the detection object.
    
    Parameters
    ----------
    detection : TemplateICADetection
        The detection result object.
    sfreq : float
        Sampling frequency.
    tr : float
        fMRI Repetition Time.
        
    Returns
    -------
    detection : TemplateICADetection
        Updated detection object with regressors attached.
    """
    if tr is not None and len(detection.accepted_components) > 0:
        # Use the first accepted component
        # detection.component_timecourses contains the timecourses of accepted components in order
        if detection.component_timecourses and len(detection.component_timecourses) > 0:
            ica_source = detection.component_timecourses[0]
            
            regressors_dict = generate_hrf_regressors(
                component_tc=ica_source,
                sfreq=sfreq,
                tr=tr
            )
            
            detection.regressor_ica = regressors_dict.get('5s')
            detection.regressors_ica_all = regressors_dict
            
    return detection

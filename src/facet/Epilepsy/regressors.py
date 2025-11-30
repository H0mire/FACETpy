import numpy as np
from scipy.signal import resample, fftconvolve

def _double_gamma_hrf(t, peak_time=6.0, undershoot=16.0, ratio=6.0):
    """Double-gamma HRF (canonical, peaking at ~6s)."""
    t = np.array(t, dtype=float)
    h1 = (t / peak_time)**2 * np.exp(-t / peak_time)
    h2 = (t / undershoot)**2 * np.exp(-t / undershoot) / ratio
    hrf = h1 - h2
    hrf[hrf < 0] = 0
    hrf /= hrf.max() if hrf.max() > 0 else 1
    return hrf

def build_grouiller_regressor(
    spike_times_sec,
    total_duration_sec,
    tr=2.5,
    hrf_model="canonical",
    dt=0.01,  # High-res for convolution
):
    """Build Grouiller-style regressor: stick at spikes, convolve with HRF, resample to TR."""
    if not spike_times_sec or total_duration_sec <= 0:
        return np.zeros(int(total_duration_sec / tr))

    # High-res time axis
    t_high = np.arange(0, total_duration_sec, dt)
    stick = np.zeros_like(t_high)
    for s in spike_times_sec:
        idx = int(round(s / dt))
        if 0 <= idx < len(stick):
            stick[idx] = 1.0

    # Convolve with HRF
    hrf_len = int(20 / dt)  # 20s decay
    hrf_t = np.arange(hrf_len) * dt
    hrf = _double_gamma_hrf(hrf_t)
    conv = fftconvolve(stick, hrf)[:len(stick)]

    # Resample to TR
    n_tr = int(np.floor(total_duration_sec / tr))
    tr_times = np.arange(n_tr) * tr
    regressor = np.interp(tr_times, t_high, conv)
    return regressor

# Ebrahimzadeh et al. 2021 HRF regressor generation
# ======================= HRF convolution =======================
def hrf_kernel(t, peak=6, undershoot=16):
    """Simple double-gamma HRF kernel."""
    h1 = (t / peak) ** 2 * np.exp(-t / peak)
    h2 = 0.1 * (t / undershoot) ** 2 * np.exp(-t / undershoot)
    return h1 - h2

def generate_hrf_regressors(component_tc, sfreq, peaks_s=[3, 5, 6, 7, 9], tr=2.5):
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
            
            detection.regressor_ica = regressors_dict.get('6s')
            detection.regressors_ica_all = regressors_dict
            
    return detection

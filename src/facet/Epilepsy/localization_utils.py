import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from nilearn.glm.first_level import glover_hrf
import mne

def make_spike_regressor(peaks, raw, hrf_length_s=32.0, tr=1.0):
    """
    Build an HRF‐convolved regressor from ICA‐detected spike times.

    Parameters
    ----------
    peaks : array-like of int
        Sample indices (in the EEG time‐series) of detected spikes.
    raw : mne.io.Raw
        Your filtered, µV‐scaled MNE Raw object.
    hrf_length_s : float
        Length of the HRF impulse response (seconds).
    tr : float
        Time resolution (seconds) for the HRF. If you're doing EEG‐only
        GLM at the same sampling rate put tr=1/ raw.info['sfreq'], or
        if you're feeding this to an fMRI GLM set tr = your fMRI TR.

    Returns
    -------
    regressor : 1-D ndarray, shape (n_times,)
        Continuous HRF‐convolved regressor (same length as raw.n_times).
    """
    sfreq = raw.info['sfreq']
    n_times = raw.n_times

    # 1) binary event train
    event = np.zeros(n_times, dtype=float)
    event[peaks] = 1.0

    # 2) get a canonical HRF sampled at `tr` seconds
    #    glover_hrf returns at temporal resolution `tr` by default
    hrf_at_tr = glover_hrf(tr=tr, oversampling=1, time_length=hrf_length_s)

    # 3) up‐ or down‐sample the HRF to EEG sampling rate:
    #    if tr == 1/sfreq, then hrf_at_tr is already at EEG rate.
    if abs(tr - 1.0/sfreq) > 1e-6:
        # interpolate to match EEG dt
        
        n_hrf_samples = int(np.round(hrf_length_s * sfreq)) + 1
        hrf = resample(hrf_at_tr, n_hrf_samples)
    else:
        hrf = hrf_at_tr

    # 4) convolve and trim back to original length
    reg = np.convolve(event, hrf)[:n_times]

    # 5) optionally z‐score or demean
    reg = (reg - reg.mean()) / (reg.std() + 1e-12)
    
    # 1) report
    times_sec = peaks / raw.info['sfreq']
    print(f"Detected {len(peaks)} ICA spikes at samples → seconds:\n",
        list(zip(peaks.tolist(), np.round(times_sec, 3).tolist())))

    # 2) build  regressor manually here ,can inspect its length
    n_samps = raw.n_times
    tr      = 1.0 / raw.info['sfreq']
    frame_times = np.arange(n_samps) * tr      # in seconds

    # make a delta train
    delta = np.zeros_like(frame_times)
    delta[peaks] = 1

    # fetch (or build) your HRF at step tr
    hrf   = glover_hrf(tr, time_length=32.0)   # or your canonical_hrf
    # convolve (mode='full') then trim back to length
    reg   = np.convolve(delta, hrf)[:n_samps]

    plt.figure(figsize=(6,3))
    plt.plot(frame_times, reg, label='HRF‐convolved regressor')
    plt.scatter(times_sec, np.ones_like(times_sec)*reg.max()*0.8,
                color='C1', marker='v', label='ICA spike events')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.title('Spike regressor & original detected events')
    plt.show()

    print(f"Regressor shape: {reg.shape}, time span: 0→{frame_times[-1]:.1f}s")
    return reg




def localize_evoked(evoked, inv, lambda2=1.0/9.0, method="dSPM",
                    subject=None, hemi="both", subjects_dir=None):
    """
    Apply an inverse operator to an Evoked, return & plot an stc.

    Parameters
    ----------
    evoked : mne.Evoked
    inv    : mne.inverse_operator
    lambda2: float     regularization (1/SNR²)
    method : str       "MNE" | "dSPM" | "sLORETA"
    subject, hemi, subjects_dir : for plotting

    Returns
    -------
    stc : mne.SourceEstimate
    """
    stc = mne.minimum_norm.apply_inverse(evoked, inv,
                                          lambda2=lambda2,
                                          method=method)
    brain = stc.plot(subject=subject,
                     hemi=hemi,
                     subjects_dir=subjects_dir,
                     time_unit="s",
                     initial_time=0.0,
                     clim="auto")
    return stc


def make_spike_evoked(raw, spike_sec, tmin=-0.05, tmax=0.05):
    """
    Extract little epochs around each annotated spike time (in seconds)
    and average them into a single Evoked object.
    """
    sfreq = raw.info['sfreq']
    events = np.array([
        [int(t * sfreq), 0, 1]
        for t in spike_sec
        if 0 < t + tmin < raw.times[-1] and 0 < t + tmax < raw.times[-1]
    ], int)
    # create an MNE Annotations and Epochs
    annot = mne.Annotations(onset=spike_sec, duration=0.0, description=['IED']*len(spike_sec))
    raw_annot = raw.copy().set_annotations(annot)
    epochs = mne.Epochs(raw_annot,
                        events=events,
                        event_id=1,
                        tmin=tmin,
                        tmax=tmax,
                        baseline=(None, 0),
                        preload=True)
    evoked = epochs.average()
    return evoked

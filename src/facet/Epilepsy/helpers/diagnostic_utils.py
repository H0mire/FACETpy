import matplotlib.pyplot as plt
import numpy as np
import os

def list_top_spike_channels(raw, spike_sec, half_win_s=0.10, top_n=5):
    """
    Identify the top N EEG channels by average peak-to-peak amplitude around annotated spikes.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG data, assumed in microvolts (µV).
    spike_sec : list of float
        Spike annotation times in seconds.
    half_win_s : float
        Half-window length in seconds for P–P calculation (default ±0.10 s).
    top_n : int
        Number of top channels to return (default 5).

    Returns
    -------
    top_channels : list of tuples
        List of (index, name, mean_pp) for the top N channels, sorted descending.
    """
    sfreq = raw.info['sfreq']
    half_samples = int(round(half_win_s * sfreq))
    idxs = (np.array(spike_sec) * sfreq).astype(int)

    n_ch = raw.info['nchan']
    data = raw.get_data()  # shape (n_ch, n_times), µV

    mean_pp = np.zeros(n_ch)
    for ch in range(n_ch):
        pps = []
        for idx in idxs:
            start, stop = idx - half_samples, idx + half_samples
            if 0 <= start and stop <= data.shape[1]:
                segment = data[ch, start:stop]
                pps.append(np.ptp(segment))
        mean_pp[ch] = np.mean(pps) if pps else 0

    order = np.argsort(mean_pp)[::-1]
    top = [(int(ch), raw.ch_names[ch], float(mean_pp[ch])) for ch in order[:top_n]]
    return top




def plot_match_with_template(raw, channel, match_times, template, sfreq, window=0.3):
    """
    Plot signal segments around matches with spike template overlayed.
    """
    signal = raw.get_data(picks=[channel])[0]
    half_win = int((window / 2) * sfreq)
    template_len = len(template)

    for match_time in match_times[:5]:
        center = int(match_time * sfreq)
        start = center - half_win
        end = center + half_win

        if start < 0 or end > len(signal):
            continue

        segment = signal[start:end]
        seg_time = np.arange(start, end) / sfreq

        segment_norm = normalize_signal(segment)
        template_norm = normalize_signal(template)

        overlay = np.zeros_like(segment_norm)
        temp_start = (len(segment_norm) - template_len) // 2
        overlay[temp_start:temp_start + template_len] = template_norm

        plt.figure(figsize=(10, 3))
        plt.plot(seg_time, segment_norm, label=f"{channel} segment", alpha=0.7)
        plt.plot(seg_time, overlay, label="Template", linestyle='--')
        plt.title(f"Visual Match at {match_time:.2f}s in {channel}")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_detection_components(detection, component_indices=None, top_k=None, max_plot_seconds=30,
                              save_figs=False, outdir=None, show_figs=True, manual_spike_times=None):
    """Plot components from a returned TemplateICADetection without recomputing ICA.

    Parameters
    ----------
    detection : TemplateICADetection
        The object returned by `select_components_template_ica` (contains `ica`,
        `accepted_components`, `component_timecourses`, `template_z` and `raw`).
    component_indices : list[int], optional
        Specific component indices (absolute ICA indices) to plot. Must be a subset of
        `detection.accepted_components`.
    top_k : int, optional
        If provided and `component_indices` is None, plot only the first `top_k`
        components from `detection.accepted_components` (order as returned).
    max_plot_seconds : int
        Seconds of continuous data to plot from the start.

    manual_spike_times : list[float], optional
        Manual spike annotation times (seconds) to show on the continuous plot and
        to use for averaging if refined times are unavailable.

    This function uses the stored `detection.ica` and `detection.raw` to obtain the
    full source matrix, avoiding any need to rerun ICA or the pipeline.
    """
    if detection is None:
        raise ValueError('detection must be provided')

    # Determine which components to show
    accepted = detection.accepted_components or []
    if component_indices is not None:
        # validate
        sel = [int(i) for i in component_indices if i in accepted]
    elif top_k is not None:
        sel = accepted[:top_k]
    else:
        sel = accepted

    if not sel:
        print('No accepted components available to plot.')
        return

    # Prefer to use the fitted ICA and raw to get full S (n_comp x n_times)
    if getattr(detection, 'ica', None) is not None and getattr(detection, 'raw', None) is not None:
        try:
            S = detection.ica.get_sources(detection.raw).get_data()
            raw = detection.raw
            template = detection.template_z if hasattr(detection, 'template_z') else None
            spikes = detection.refined_times if hasattr(detection, 'refined_times') else None
            # Use refined times for epochs when available, else fall back to manual
            spikes_for_epochs = spikes if (spikes is not None and len(spikes) > 0) else manual_spike_times
            plot_ica_components_timecourses(raw, ica=detection.ica, S=S, component_indices=sel,
                                            template_z=template, spike_times=spikes_for_epochs,
                                            max_plot_seconds=max_plot_seconds, save_figs=save_figs, outdir=outdir, show_figs=show_figs,
                                            manual_spike_times=manual_spike_times)
            return
        except Exception:
            # fallback to using stored timecourses
            pass

    # Fallback: use stored component_timecourses (these are only accepted components)
    if getattr(detection, 'component_timecourses', None) is not None:
        # Build a small S matrix where row order corresponds to accepted_components
        comp_tcs = detection.component_timecourses
        # map sel indices to positions inside accepted_components
        idx_map = {comp: i for i, comp in enumerate(detection.accepted_components)}
        rows = []
        for comp in sel:
            pos = idx_map.get(comp, None)
            if pos is None:
                continue
            rows.append(comp_tcs[pos])
        if not rows:
            print('No matching component timecourses found in detection object.')
            return
        S_small = np.vstack(rows)
        # Create a fake raw.times array if raw not present
        if getattr(detection, 'raw', None) is not None:
            raw = detection.raw
        else:
            # Make a minimal times array
            n_times = S_small.shape[1]
            class _FakeRaw:
                pass
            raw = _FakeRaw()
            raw.times = np.arange(n_times) / 500.0
            raw.info = {'sfreq': 500.0}

        spikes_attr = getattr(detection, 'refined_times', None)
        spikes_for_epochs = spikes_attr if (spikes_attr is not None and len(spikes_attr) > 0) else manual_spike_times
        plot_ica_components_timecourses(raw, ica=None, S=S_small, component_indices=list(range(S_small.shape[0])),
                template_z=getattr(detection, 'template_z', None),
                spike_times=spikes_for_epochs,
                max_plot_seconds=max_plot_seconds, save_figs=save_figs, outdir=outdir, show_figs=show_figs,
                manual_spike_times=manual_spike_times)
        return

    print('Detection object did not contain ICA or component timecourses to plot.')



def normalize_signal(sig):
    """Zero-mean, unit-variance normalization."""
    return (sig - np.mean(sig)) / np.std(sig)



def plot_ica_with_annotations(raw, ica_trace, peaks_ica, spike_sec,
                              match_tol_s=0.1, tmin=0, tmax=None ):
    """
    raw         : mne.io.Raw    (just to get sfreq and times)
    ica_trace   : 1D array (n_times,) your chosen ICA component (µV)
    peaks_ica   : 1D int array    sample indices where you ran find_peaks()
    spike_sec   : list of floats  manual annotation times in seconds
    match_tol_s : float           tolerance window for a “hit”
    tmin,tmax   : floats          time range to plot (in seconds)
    """
    sf = raw.info['sfreq']
    times = np.arange(ica_trace.size) / sf
    if tmax is None:
        tmax = times[-1]
    # convert peaks to times
    peaks_t = peaks_ica / sf

    # full overview
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(times, ica_trace, color='C0', lw=0.6, label='ICA trace (µV)')
    ax.scatter(peaks_t, np.interp(peaks_t, times, ica_trace),
               marker='v', color='C1', label='Detected peaks')
    ax.vlines(spike_sec, ica_trace.min(), ica_trace.max(),
              color='C3', linestyle='--', alpha=0.7, label='Annotations')
    ax.set_xlim(tmin, tmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('ICA component vs annotated spikes & detected events')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # now zoom in on the first few annotations
    for s in spike_sec[:5]:
        window = match_tol_s * 5  # e.g. 5× your tolerance
        sel = (times >= s-window) & (times <= s+window)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(times[sel], ica_trace[sel], color='C0', lw=1)
        # detected peaks in window
        sel_peaks = peaks_t[(peaks_t >= s-window) & (peaks_t <= s+window)]
        ax.scatter(sel_peaks,
                   np.interp(sel_peaks, times, ica_trace),
                   marker='v', color='C1')
        ax.axvline(s, color='C3', linestyle='--', label='Annotation')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('µV')
        ax.set_title(f'Zoom around manual spike @ {s:.2f}s')
        ax.legend()
        plt.tight_layout()
        plt.show()


def plot_ica_components_timecourses(raw, ica=None, S=None, component_indices=None,
                                     template_z=None, spike_times=None, max_plot_seconds=30,
                                     save_figs=False, outdir=None, show_figs=True, manual_spike_times=None):
    """Plot ICA component continuous timecourses and averaged epochs.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG object (for sfreq and times).
    ica : mne.preprocessing.ICA, optional
        Fitted MNE ICA object. If provided, `S` may be omitted.
    S : ndarray, shape (n_components, n_times), optional
        Precomputed ICA source matrix. Used if `ica` is not provided.
    component_indices : list[int]
        Indices of components to plot.
    template_z : ndarray, optional
        Z-scored spike template to overlay on averaged epochs.
    spike_times : list[float], optional
        Spike annotation times (seconds) used to compute average epochs.
    max_plot_seconds : int
        Number of seconds from the start to display for continuous plot.
    manual_spike_times : list[float], optional
        Manual spike annotation times (seconds) to plot as a reference.
    """
    if component_indices is None or len(component_indices) == 0:
        return

    sf = raw.info['sfreq']
    times = raw.times

    if S is None:
        if ica is None:
            raise ValueError('Either ica or S must be provided')
        S = ica.get_sources(raw).get_data()

    n_times = S.shape[1]
    
    # Determine plot range: if manual spikes exist, center around the first one
    start_samp = 0
    if manual_spike_times and len(manual_spike_times) > 0:
        first_spike = manual_spike_times[0]
        # Try to center the 30s window around the first spike
        # e.g. start 10s before it
        start_sec = max(0, first_spike - 10)
        start_samp = int(start_sec * sf)
    
    end_samp = int(min(n_times, start_samp + max_plot_seconds * sf))
    
    # If the window is too short (end of file), shift back
    if end_samp - start_samp < max_plot_seconds * sf and n_times > max_plot_seconds * sf:
        start_samp = int(n_times - max_plot_seconds * sf)
        end_samp = n_times

    for idx in component_indices:
        comp = S[idx]
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)

        # Continuous segment
        t_seg = times[start_samp:end_samp]
        axes[0].plot(t_seg, comp[start_samp:end_samp], color='C0', linewidth=0.8)
        axes[0].set_title(f'ICA component {idx} — continuous ({max_plot_seconds}s window)')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude (a.u.)')

        # Mark manual spike times within the segment
        if manual_spike_times is not None:
            for s in manual_spike_times:
                if t_seg[0] <= s <= t_seg[-1]:
                    axes[0].axvline(s, color='g', linestyle='-', alpha=0.8, label='Manual' if 'Manual' not in [l.get_label() for l in axes[0].lines] else "")

        # Mark refined spike times within the segment
        if spike_times is not None:
            for s in spike_times:
                if t_seg[0] <= s <= t_seg[-1]:
                    axes[0].axvline(s, color='r', linestyle='--', alpha=0.6, label='Refined' if 'Refined' not in [l.get_label() for l in axes[0].lines] else "")
        
        if manual_spike_times is not None or spike_times is not None:
            axes[0].legend(loc='upper right')

        # Average epoch around spikes
        if spike_times is not None and template_z is not None:
            L = len(template_z)
            half = L // 2
            epochs = []
            for s in spike_times:
                samp = int(round(s * sf))
                start = samp - half
                end = start + L
                if start < 0 or end > len(comp):
                    continue
                epochs.append(comp[start:end])
            if len(epochs) > 0:
                epochs = np.vstack(epochs)
                mean_epoch = epochs.mean(axis=0)
                t_epoch = (np.arange(len(mean_epoch)) - half) / sf
                axes[1].plot(t_epoch, mean_epoch, label='Component mean', color='C1')
                # overlay scaled template
                tpl = template_z.copy()
                if np.std(tpl) > 0:
                    tpl = (tpl - tpl.mean()) / (tpl.std() + 1e-12)
                    tpl = tpl * (np.std(mean_epoch) * 0.8)
                axes[1].plot(t_epoch, tpl, label='Template (scaled)', color='k', linestyle='--')
                axes[1].axvline(0, color='r', linestyle=':', label='Spike')
                axes[1].set_title(f'ICA component {idx} — averaged epoch (n={len(epochs)})')
                axes[1].set_xlabel('Time (s)')
                axes[1].legend()
            else:
                axes[1].text(0.5, 0.5, 'No full epochs available for averaging', ha='center')
                axes[1].set_title(f'ICA component {idx} — averaged epoch (n=0)')
        else:
            axes[1].text(0.5, 0.5, 'Template or spike times not provided', ha='center')
            axes[1].set_title(f'ICA component {idx} — averaged epoch')

        # Save or show
        if save_figs:
            if outdir is None:
                outdir = os.path.join(os.getcwd(), 'results', 'figures')
            os.makedirs(outdir, exist_ok=True)
            fname = f"ica_component_{idx}.png"
            path = os.path.join(outdir, fname)
            fig.savefig(path, dpi=150)
            plt.close(fig)
        elif show_figs:
            plt.show()
        else:
            plt.close(fig)



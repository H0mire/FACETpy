"""Computation and scalp-geometry helpers for single-subject evaluation.

Pure move of the metric/geometry functions that previously lived in
``evaluate_subject.py`` (now focused on reporting + orchestration).  These are
the same functions, imported back by ``evaluate_subject`` — no behavior changed.
"""

from typing import Optional

import mne
import numpy as np

from facet.Epilepsy.helpers.regressors import _build_epileptic_map
from facet.Epilepsy.evaluation.config import HALF_WIN_S


def _eeg_channel_names(detection) -> list:
    """EEG channel names (in epileptic-map order) from the detection's raw."""
    raw = getattr(detection, "raw", None) if detection is not None else None
    if raw is None:
        return []
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude="bads")
    return [raw.ch_names[i] for i in eeg_picks]


def template_channel_info(rec: "SubjectRecord") -> tuple[Optional[str], Optional[str]]:
    """Name and MNE type of the channel the IED template was built from.

    The type lets us confirm the template came from a scalp EEG channel (not an
    ECG/EMG/ear electrode).
    """
    det = rec.detection
    raw = getattr(det, "raw", None) if det is not None else None
    idx = rec.ica_selection_stats.get("template_channel")
    if raw is None or idx is None or not (0 <= idx < len(raw.ch_names)):
        return None, None
    name = raw.ch_names[idx]
    ch_type = raw.get_channel_types(picks=[idx])[0]
    return name, ch_type


def grouiller_peak_channel(rec: "SubjectRecord") -> Optional[str]:
    """Channel with the largest |value| in the epileptic map."""
    emap = rec.epileptic_map
    if emap is None or len(emap) == 0 or not rec.channel_names:
        return None
    peak_idx = int(np.argmax(np.abs(emap)))
    if peak_idx < len(rec.channel_names):
        return rec.channel_names[peak_idx]
    return None


def grouiller_focality(rec: "SubjectRecord") -> float:
    """Focality = max(|map|) / median(|map|).  Higher = more focal."""
    emap = rec.epileptic_map
    if emap is None or len(emap) == 0:
        return np.nan
    a = np.abs(emap)
    med = float(np.median(a))
    return float(np.max(a) / med) if med > 0 else np.nan


def _spike_topographies(raw, spike_sec, half_win_s=HALF_WIN_S, band=(1., 30.)):
    """Unit-norm scalp-voltage topography at each spike's GFP peak.

    Uses ONLY the raw EEG and the annotation times — no template, no ICA, no
    detection — so the resulting metric is *orthogonal* to the timing
    information the annotations provided (it never re-detects anything).

    For every annotated spike the EEG is epoched around the mark, the global
    field power (GFP, spatial std across channels) peak of that epoch is found,
    and the common-average-referenced voltage vector at that instant is taken
    and unit-normalised — exactly the field pattern Grouiller's epileptic map
    is built from, but kept per-spike instead of averaged.
    """
    sf = raw.info["sfreq"]
    hw = int(round(half_win_s * sf))
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude="bads")
    data = (raw.copy()
            .filter(band[0], band[1], picks="eeg", verbose=False)
            .get_data(picks=eeg_picks))          # (n_ch, n_times)
    n_times = data.shape[1]
    topos = []
    for t in spike_sec:
        center = int(round(t * sf))
        start, stop = center - hw, center + hw
        if start < 0 or stop > n_times:
            continue
        seg = data[:, start:stop]                 # (n_ch, win)
        peak = int(np.argmax(np.std(seg, axis=0)))
        v = seg[:, peak].astype(float)
        v = v - v.mean()                          # common-average reference
        nrm = np.linalg.norm(v)
        if nrm > 0:
            topos.append(v / nrm)
    return np.asarray(topos)                       # (n_valid, n_ch)


def spike_field_consistency(rec: "SubjectRecord") -> dict:
    """Non-circular EEG-level validation of the annotated spikes.

    The former detection metric was circular: the annotated spikes build the
    IED template and were then re-detected by that same template.  This metric
    instead asks a question the annotations did NOT directly answer — do the
    marked events share ONE focal cortical generator? — using only the raw EEG
    topography at each annotated time.

    Rationale: interictal spikes from a single focus produce a stable scalp
    field pattern.  If the per-spike topographies are mutually consistent, the
    annotations are electrophysiologically genuine, focal and localisable at
    EEG level; if they scatter, they are heterogeneous or noisy.  No template,
    ICA or detection is involved, so the result cannot be inflated by the
    circularity that undermined the timing-based sensitivity.

    Metrics
    -------
    spatial_n_spikes_used : int
        Annotated spikes with a valid topography.
    spatial_topo_consistency_mean / _median / _min : float
        Pairwise Pearson correlation between per-spike topographies (1 =
        identical field = single focus).  Mean is the headline number.
    spatial_mean_map_focality : float
        max(|mean topography|) / median(|mean topography|); higher = more focal.
    """
    det = rec.detection
    raw = getattr(det, "raw", None) if det is not None else None
    ann_times = (det.original_spike_sec
                 if det is not None and det.original_spike_sec else [])

    base = {
        "spatial_n_spikes_used": 0,
        "spatial_topo_consistency_mean": np.nan,
        "spatial_topo_consistency_median": np.nan,
        "spatial_topo_consistency_min": np.nan,
        "spatial_mean_map_focality": np.nan,
    }
    if raw is None or len(ann_times) < 2:
        return base

    topos = _spike_topographies(raw, ann_times)
    if topos.shape[0] < 2:
        return base

    # Topographies are already mean-centred and unit-norm, so their dot product
    # IS the Pearson correlation across channels.
    corr = topos @ topos.T
    iu = np.triu_indices(corr.shape[0], k=1)
    pair_r = corr[iu]

    mean_map = topos.mean(axis=0)
    a = np.abs(mean_map)
    med = float(np.median(a))
    focality = float(np.max(a) / med) if med > 0 else np.nan

    return {
        "spatial_n_spikes_used": int(topos.shape[0]),
        "spatial_topo_consistency_mean": float(np.mean(pair_r)),
        "spatial_topo_consistency_median": float(np.median(pair_r)),
        "spatial_topo_consistency_min": float(np.min(pair_r)),
        "spatial_mean_map_focality": focality,
    }


def _region_map_from_raw(raw) -> dict:
    """Map each EEG channel name to a region from its *real* head position.

    Region is read from the montage coordinates that also draw the topomap
    (MNE head frame: +x = right, +y = anterior), not from the electrode's
    spelling. The head is split into terciles of the actual electrode cloud, so
    the boundaries are data-driven rather than hand-picked:

    - left / mid / right  from the x tercile,
    - anterior / central / posterior  from the y tercile.

    Returns ``{}`` when no montage is available, letting callers fall back to
    the name-based parser.
    """
    if raw is None:
        return {}
    montage = raw.get_montage()
    if montage is None:
        return {}
    ch_pos = montage.get_positions().get("ch_pos") or {}
    picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude="bads")
    names, xs, ys = [], [], []
    for i in picks:
        p = ch_pos.get(raw.ch_names[i])
        if p is None or not np.all(np.isfinite(p)):
            continue
        names.append(raw.ch_names[i])
        xs.append(float(p[0]))
        ys.append(float(p[1]))
    if len(names) < 3:
        return {}
    xs, ys = np.asarray(xs), np.asarray(ys)
    x_lo, x_hi = np.quantile(xs, [1 / 3, 2 / 3])
    y_lo, y_hi = np.quantile(ys, [1 / 3, 2 / 3])
    out = {}
    for name, x, y in zip(names, xs, ys):
        lr = "right" if x >= x_hi else ("left" if x <= x_lo else "mid")
        ap = "anterior" if y >= y_hi else ("posterior" if y <= y_lo else "central")
        out[name] = f"{lr}-{ap}"
    return out


# Direct channel-name → clinical-lobe lookup, matching the VEPISET dataset's
# folder taxonomy (Frontal / Centro-Parietal / Temporal / Occipital) exactly,
# instead of the coordinate-tercile grid used by ``_region_map_from_raw``.
# "Generalized" has no single site and is therefore not assigned here.
_CLINICAL_LOBE_BY_CHANNEL = {
    "Fp1": "Frontal", "Fp2": "Frontal", "F3": "Frontal", "F4": "Frontal",
    "Fz": "Frontal",
    # F7/F8 are the standard anterior-temporal spike electrodes clinically.
    "F7": "Temporal", "F8": "Temporal",
    "C3": "Centro-Parietal", "C4": "Centro-Parietal", "Cz": "Centro-Parietal",
    "P3": "Centro-Parietal", "P4": "Centro-Parietal", "Pz": "Centro-Parietal",
    "T3": "Temporal", "T4": "Temporal", "T5": "Temporal", "T6": "Temporal",
    "O1": "Occipital", "O2": "Occipital",
}


def _clinical_lobe_of(ch_name: Optional[str]) -> Optional[str]:
    """Clinical lobe (Frontal/Centro-Parietal/Temporal/Occipital) of a channel."""
    if ch_name is None:
        return None
    return _CLINICAL_LOBE_BY_CHANNEL.get(str(ch_name))


def _map_peak_channel(emap, channel_names) -> Optional[str]:
    """Channel name at the largest |value| of a scalp map."""
    if emap is None or len(emap) == 0 or not channel_names:
        return None
    idx = int(np.argmax(np.abs(np.asarray(emap))))
    return channel_names[idx] if idx < len(channel_names) else None


def _spatial_map_corr(map_a, names_a, map_b, names_b) -> Optional[float]:
    """|Pearson r| between two scalp maps over their shared channels.

    Compares the *whole topography* (value at every common electrode), not just
    the peak. Absolute value because a spike map's overall polarity is
    arbitrary (a focus and its sign-flip are the same map). Returns ``None`` if
    fewer than 3 channels overlap.
    """
    if map_a is None or map_b is None or not names_a or not names_b:
        return None
    a = np.asarray(map_a, float)
    b = np.asarray(map_b, float)
    idx_b = {n: i for i, n in enumerate(names_b)}
    va, vb = [], []
    for i, n in enumerate(names_a):
        j = idx_b.get(n)
        if j is not None and i < len(a) and j < len(b):
            va.append(a[i]); vb.append(b[j])
    if len(va) < 3:
        return None
    va, vb = np.asarray(va), np.asarray(vb)
    if np.std(va) == 0 or np.std(vb) == 0:
        return None
    return float(abs(np.corrcoef(va, vb)[0, 1]))


def summarize_cross_method_concordance(rec: "SubjectRecord") -> dict:
    """Do the independently-derived foci agree on a location? (non-circular)

    Three scalp foci, each obtained a *different* way, are compared:

    - ``grouiller_peak_channel`` — peak of the averaged RAW-spike map
      (Grouiller: direct scalp topography of the marked spikes).
    - ``ica_peak_channel`` — peak of the scalp *voltage field* reconstructed
      from the single accepted ICA component whose timecourse best matches the
      IED template (highest median per-window correlation). It is built with the
      same averaged-spike / GFP-peak machinery as the grouiller and fused maps,
      so all three foci sit on the same voltage scale and polarity convention.
      (The raw ICA mixing pattern is instead a normalised, bipolar spatial
      filter whose ``argmax(|·|)`` can land on the return-current pole and gives
      a wrong lobe — see below.)
    - ``fused_peak_channel`` — peak of the artifact-cleaned averaged-spike map
      (the fused map).

    The annotations only supplied *timing*, never *location*, so agreement of
    these independently-derived locations is a genuine, non-circular
    localisation check. Three views are reported: (1) a coarse region match of
    each map's peak electrode from a geometric left/mid/right x
    anterior/central/posterior grid (data-driven terciles of the real montage
    coordinates), (2) a clinical-lobe match (Frontal / Centro-Parietal /
    Temporal / Occipital) from a direct channel-name lookup, matching the
    VEPISET dataset's folder-level labels, and (3) a graded ``|Pearson r|``
    between the *whole* maps (grouiller/fused/ica pairwise), which quantifies
    how similar the full topographies are, not just the peak.
    """
    det = rec.detection
    grouiller_peak = _map_peak_channel(rec.epileptic_map, rec.channel_names)
    fused_peak = _map_peak_channel(rec.fused_epileptic_map, rec.channel_names)

    # Accepted ICA component most representative of the IED timecourse (highest
    # median per-window template correlation). Localised as a scalp VOLTAGE
    # field (not the raw mixing pattern) so it is comparable to grouiller/fused.
    ica_peak = None
    ica_pattern = None
    ica_names = None
    ica = getattr(det, "ica", None) if det is not None else None
    raw = getattr(det, "raw", None) if det is not None else None
    if ica is not None and raw is not None and rec.accepted_indices:
        n_comp = ica.get_components().shape[1]
        # Standalone TCCC localises a SINGLE representative component so it is a
        # genuinely different reconstruction from the fused map (which uses ALL
        # final components). In spatial-gated mode the representative is the
        # highest-|r| survivor (or the spatial fallback); baseline runs (no gate)
        # keep the full accepted set. Reconstructing a voltage field (not the raw
        # bipolar mixing pattern) keeps TCCC on the same scale as grouiller/fused
        # and avoids argmax landing on a return-current pole.
        sg = rec.spatial_gate
        if sg is not None:
            rep = sg.get("spatial_tccc_representative_component")
            sel = [rep] if rep is not None else list(rec.accepted_indices)
        else:
            sel = list(rec.accepted_indices)
        tccc_indices = [i for i in sel if i < n_comp]
        if tccc_indices:
            try:
                recon_ic = ica.apply(
                    raw.copy(), include=list(tccc_indices), verbose=False)
                ica_pattern = _build_epileptic_map(
                    recon_ic, det.refined_times,
                    half_win_s=HALF_WIN_S, band=(1., 30.))
                ica_names = _eeg_channel_names(det)
                ica_peak = _map_peak_channel(ica_pattern, ica_names)
            except Exception:
                # Fall back to the representative component's raw mixing pattern.
                comps = ica.get_components()
                ica_pattern = comps[:, tccc_indices[0]]
                ica_names = ica.ch_names
                pk = int(np.argmax(np.abs(ica_pattern)))
                ica_peak = ica_names[pk] if pk < len(ica_names) else None

    # Coordinate-based region lookup only (data-driven from the montage).
    coord_region = _region_map_from_raw(getattr(det, "raw", None))
    def region_of(ch: Optional[str]) -> Optional[str]:
        return coord_region.get(ch) if ch else None

    g_r, i_r, f_r = (region_of(grouiller_peak),
                     region_of(ica_peak),
                     region_of(fused_peak))
    avail = [r for r in (g_r, i_r, f_r) if r and r != "other"]
    top = max((avail.count(r) for r in set(avail)), default=0)
    all_agree = len(avail) >= 2 and top == len(avail)

    # Clinical-lobe lookup (Frontal/Centro-Parietal/Temporal/Occipital), for
    # direct comparison against the VEPISET dataset's folder-level labels.
    g_l, i_l, f_l = (_clinical_lobe_of(grouiller_peak),
                     _clinical_lobe_of(ica_peak),
                     _clinical_lobe_of(fused_peak))
    avail_l = [r for r in (g_l, i_l, f_l) if r]
    top_l = max((avail_l.count(r) for r in set(avail_l)), default=0)
    all_agree_l = len(avail_l) >= 2 and top_l == len(avail_l)

    # Quantitative agreement: |Pearson r| between the whole scalp maps.
    ch = rec.channel_names
    r_gf = _spatial_map_corr(rec.epileptic_map, ch, rec.fused_epileptic_map, ch)
    r_gi = _spatial_map_corr(rec.epileptic_map, ch, ica_pattern, ica_names)
    r_fi = _spatial_map_corr(rec.fused_epileptic_map, ch, ica_pattern, ica_names)

    # With a single accepted component the fused map IS that component, so
    # fused-ica is a tautological 1.0 and grouiller-ica merely duplicates
    # grouiller-fused. Blank both so the ICA correlations don't fake agreement;
    # the mean then rests on grouiller-fused, the only independent number.
    single_component = len(rec.accepted_indices or []) < 2
    if single_component:
        r_gi = None
        r_fi = None

    corrs = [r for r in (r_gf, r_gi, r_fi) if r is not None]
    map_corr_mean = float(np.mean(corrs)) if corrs else None

    return {
        "grouiller_peak_channel": grouiller_peak,
        "ica_peak_channel": ica_peak,
        "fused_peak_channel": fused_peak,
        "grouiller_region": g_r,
        "ica_region": i_r,
        "fused_region": f_r,
        "n_methods_agreeing": int(top),
        "concordance_all_agree": bool(all_agree),
        "grouiller_lobe": g_l,
        "ica_lobe": i_l,
        "fused_lobe": f_l,
        "n_methods_agreeing_lobe": int(top_l),
        "concordance_all_agree_lobe": bool(all_agree_l),
        "map_corr_grouiller_fused": r_gf,
        "map_corr_grouiller_ica": r_gi,
        "map_corr_fused_ica": r_fi,
        "map_corr_mean": map_corr_mean,
        "map_corr_ica_degenerate": bool(single_component),
    }

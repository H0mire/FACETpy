"""Trigger Alignment Processors Module

Processors for aligning trigger positions to artifact onsets in EEG-fMRI
recordings using cross-correlation techniques.
"""

import mne
import numpy as np
from loguru import logger
from scipy.signal import filtfilt, firls

from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor
from ..helpers.crosscorr import crosscorrelation
from .resampling import UpSample


def _get_pre_post_samples(
    context: ProcessingContext,
    artifact_length: int,
) -> tuple[int, int]:
    """Derive pre/post trigger sample lengths using metadata.

    Parameters
    ----------
    context : ProcessingContext
        Current processing context carrying metadata window hints.
    artifact_length : int
        Total artifact window length in samples.

    Returns
    -------
    tuple of (int, int)
        ``(pre_samples, post_samples)`` clamped to ``[0, artifact_length]``.

    Raises
    ------
    ProcessorValidationError
        If ``artifact_length`` is not positive.
    """
    sfreq = context.get_raw().info["sfreq"]
    metadata = context.metadata

    if artifact_length is None or artifact_length <= 0:
        raise ProcessorValidationError("Artifact length must be positive for alignment")

    if metadata.pre_trigger_samples is not None:
        pre = int(max(0, min(metadata.pre_trigger_samples, artifact_length)))
    else:
        offset_samples = int(round(metadata.artifact_to_trigger_offset * sfreq))
        pre = int(max(0, min(-offset_samples, artifact_length))) if offset_samples < 0 else 0

    if metadata.post_trigger_samples is not None:
        post = int(max(0, min(metadata.post_trigger_samples, artifact_length)))
    else:
        post = int(max(artifact_length - pre, 0))

    if pre + post == 0:
        post = artifact_length
    if pre + post < artifact_length:
        post = artifact_length - pre

    return pre, post


def _extract_epoch_with_padding(
    data: np.ndarray,
    start_idx: int,
    length: int,
    total_length: int,
) -> np.ndarray:
    """Extract a window from data, padding with edge values at boundaries.

    Parameters
    ----------
    data : np.ndarray
        1-D signal array.
    start_idx : int
        First sample index of the desired window (may be negative).
    length : int
        Number of samples to extract.
    total_length : int
        Total length of ``data`` in samples.

    Returns
    -------
    np.ndarray
        Extracted (and possibly padded) segment of shape ``(length,)``.
    """
    end_idx = start_idx + length

    pad_left = max(0, -start_idx)
    pad_right = max(0, end_idx - total_length)

    valid_start = max(0, start_idx)
    valid_end = min(total_length, end_idx)

    segment = data[valid_start:valid_end]
    if pad_left or pad_right:
        segment = np.pad(segment, (pad_left, pad_right), mode="edge")

    if len(segment) != length:
        segment = np.zeros(length, dtype=data.dtype) if len(segment) == 0 else np.resize(segment, length)

    return segment


@register_processor
class TriggerAligner(Processor):
    """Align triggers to artifact positions using cross-correlation.

    Refines trigger positions by finding the lag that maximises cross-
    correlation with a reference artifact epoch. The search can optionally
    operate on temporarily upsampled data for sub-sample precision.

    Parameters
    ----------
    ref_trigger_index : int, optional
        Index of the reference trigger to use as template (default: 0).
    ref_channel : int, optional
        Reference channel index. Uses the first EEG channel when ``None``
        (default: None).
    search_window : int, optional
        Search window in samples. Derived from the upsampling factor when
        ``None`` (default: None).
    save_to_annotations : bool, optional
        If ``True``, save aligned triggers as raw annotations (default: False).
    upsample_for_alignment : bool, optional
        Temporarily upsample data before alignment for sub-sample accuracy
        (default: True).
    """

    name = "trigger_aligner"
    description = "Align triggers using cross-correlation"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = True
    channel_wise = True
    run_once = True

    def __init__(
        self,
        ref_trigger_index: int = 0,
        ref_channel: int | None = None,
        search_window: int | None = None,
        save_to_annotations: bool = False,
        upsample_for_alignment: bool = True,
    ) -> None:
        self.ref_trigger_index = ref_trigger_index
        self.ref_channel = ref_channel
        self.search_window = search_window
        self.save_to_annotations = save_to_annotations
        self.upsample_for_alignment = upsample_for_alignment
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError("Artifact length not set. Run TriggerDetector first.")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers().copy()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info["sfreq"]
        upsampling_factor = context.metadata.upsampling_factor

        # --- LOG ---
        logger.info("Aligning {} triggers using cross-correlation", len(triggers))

        # --- COMPUTE ---
        ref_channel = self._pick_ref_channel(raw)
        search_window = self.search_window if self.search_window is not None else 3 * upsampling_factor

        working_raw, working_triggers, did_upsample = self._prepare_working_data(
            context, raw, triggers, upsampling_factor, sfreq
        )
        tmin, tmax = self._compute_epoch_bounds(context, working_raw, artifact_length, upsampling_factor, did_upsample)
        aligned_triggers = self._align_all_triggers(
            working_raw, working_triggers, tmin, tmax, ref_channel, search_window
        )

        if did_upsample:
            aligned_triggers = (aligned_triggers / upsampling_factor).astype(int)

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        new_metadata.triggers = aligned_triggers
        if len(aligned_triggers) > 1:
            new_metadata.artifact_length = self._recalc_artifact_length(aligned_triggers, new_metadata.volume_gaps)

        if self.save_to_annotations:
            raw_copy = raw.copy()
            raw_copy.set_annotations(
                mne.Annotations(
                    onset=aligned_triggers / sfreq,
                    duration=np.zeros(len(aligned_triggers)),
                    description=["Aligned_Trigger"] * len(aligned_triggers),
                )
            )
            return context.with_raw(raw_copy).with_metadata(new_metadata)

        # --- RETURN ---
        return context.with_metadata(new_metadata)

    def _pick_ref_channel(self, raw: mne.io.Raw) -> int:
        if self.ref_channel is not None:
            return self.ref_channel
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        return int(eeg_channels[0]) if len(eeg_channels) > 0 else 0

    def _prepare_working_data(
        self,
        context: ProcessingContext,
        raw: mne.io.Raw,
        triggers: np.ndarray,
        upsampling_factor: int,
        sfreq: float,
    ) -> tuple[mne.io.Raw, np.ndarray, bool]:
        """Return working raw and triggers, upsampling temporarily if needed.

        Parameters
        ----------
        context : ProcessingContext
            Current processing context.
        raw : mne.io.Raw
            Current raw object.
        triggers : np.ndarray
            Current trigger positions.
        upsampling_factor : int
            Factor to use when upsampling.
        sfreq : float
            Current sampling frequency.

        Returns
        -------
        tuple of (mne.io.Raw, np.ndarray, bool)
            Working raw, working triggers, and whether upsampling was applied.
        """
        if self.upsample_for_alignment and sfreq == context.get_raw_original().info["sfreq"]:
            logger.debug("Temporarily upsampling for alignment")
            temp_context = UpSample(factor=upsampling_factor).execute(context)
            return temp_context.get_raw(), temp_context.get_triggers(), True
        return raw, triggers, False

    def _compute_epoch_bounds(
        self,
        context: ProcessingContext,
        working_raw: mne.io.Raw,
        artifact_length: int,
        upsampling_factor: int,
        did_upsample: bool,
    ) -> tuple[int, int]:
        """Compute tmin/tmax epoch bounds in working-raw sample space.

        Parameters
        ----------
        context : ProcessingContext
            Current processing context.
        working_raw : mne.io.Raw
            Raw object in which alignment is performed.
        artifact_length : int
            Artifact length in original-sample space.
        upsampling_factor : int
            Applied upsampling factor.
        did_upsample : bool
            Whether working_raw is upsampled.

        Returns
        -------
        tuple of (int, int)
            ``(tmin, tmax)`` offsets relative to each trigger sample.
        """
        tmin = int(context.metadata.artifact_to_trigger_offset * working_raw.info["sfreq"])
        tmax = tmin + (artifact_length * upsampling_factor if did_upsample else artifact_length)
        return tmin, tmax

    def _align_all_triggers(
        self,
        working_raw: mne.io.Raw,
        working_triggers: np.ndarray,
        tmin: int,
        tmax: int,
        ref_channel: int,
        search_window: int,
    ) -> np.ndarray:
        """Compute aligned trigger positions for all triggers.

        Parameters
        ----------
        working_raw : mne.io.Raw
            Raw object used for cross-correlation.
        working_triggers : np.ndarray
            Trigger positions in working-raw sample space.
        tmin : int
            Start offset from trigger to artifact onset.
        tmax : int
            End offset from trigger to artifact end.
        ref_channel : int
            Channel index used for cross-correlation.
        search_window : int
            Search radius in samples.

        Returns
        -------
        np.ndarray
            Aligned trigger positions.
        """
        ref_data = working_raw.get_data(picks=[ref_channel])[0]
        ref_trigger = working_triggers[self.ref_trigger_index]
        ref_artifact = ref_data[ref_trigger + tmin : ref_trigger + tmax]

        logger.debug("Using trigger {} as reference", self.ref_trigger_index)
        logger.debug("Reference artifact shape: {}", ref_artifact.shape)

        aligned_triggers = working_triggers.copy()
        for i, trigger in enumerate(working_triggers):
            if i == self.ref_trigger_index:
                continue
            current_artifact = ref_data[trigger + tmin : trigger + tmax + search_window]
            corr = crosscorrelation(current_artifact, ref_artifact, search_window)
            shift = int(np.argmax(corr)) - search_window
            aligned_triggers[i] = trigger + shift

        logger.info("Aligned {} triggers", len(aligned_triggers))
        return aligned_triggers

    def _recalc_artifact_length(self, aligned_triggers: np.ndarray, volume_gaps: bool) -> int:
        """Recalculate artifact length from aligned trigger spacings.

        Parameters
        ----------
        aligned_triggers : np.ndarray
            Aligned trigger positions.
        volume_gaps : bool
            Whether volume-level gaps are present in the trigger sequence.

        Returns
        -------
        int
            Updated artifact length estimate in samples.
        """
        trigger_diffs = np.diff(aligned_triggers)
        if volume_gaps:
            mean_val = np.mean([np.median(trigger_diffs), np.max(trigger_diffs)])
            slice_diffs = trigger_diffs[trigger_diffs < mean_val]
            return int(np.max(slice_diffs))
        return int(np.max(trigger_diffs))


@register_processor
class SliceAligner(TriggerAligner):
    """Align slice triggers on already-upsampled data.

    Identical to :class:`TriggerAligner` but skips the internal temporary
    upsampling step, assuming the data has already been upsampled upstream.

    Parameters
    ----------
    ref_trigger_index : int, optional
        Index of the reference trigger (default: 0).
    ref_channel : int, optional
        Reference channel index (default: None).
    search_window : int, optional
        Search window in samples (default: None).
    save_to_annotations : bool, optional
        Save aligned triggers as annotations (default: False).
    """

    name = "slice_aligner"
    description = "Align slice triggers on upsampled data"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = True
    channel_wise = True
    run_once = True

    def __init__(
        self,
        ref_trigger_index: int = 0,
        ref_channel: int | None = None,
        search_window: int | None = None,
        save_to_annotations: bool = False,
    ) -> None:
        super().__init__(
            ref_trigger_index=ref_trigger_index,
            ref_channel=ref_channel,
            search_window=search_window,
            save_to_annotations=save_to_annotations,
            upsample_for_alignment=False,
        )


@register_processor
class SubsampleAligner(Processor):
    """Refine trigger positions at subsample precision after upsampling.

    Mirrors the intent of the MATLAB RAAlignSubSample step. For each trigger
    a search segment is extracted, cross-correlated against a reference epoch,
    and the trigger is shifted by the lag that maximises the correlation.

    Three modes are available via the ``mode`` parameter:

    - ``"fast"`` (default): sub-sample alignment via parabolic interpolation of
      the cross-correlation peak (~0.05 sample error). The fractional shift is
      baked into the raw data with FFT (sinc) phase shifting; triggers stay at
      their integer positions. Negligible extra cost over ``"legacy"``.
    - ``"quality"``: sub-sample alignment via binary search over the true
      alignment objective on the FFT-shifted data — the MATLAB FACET
      ``AlignSubSample`` approach. Essentially exact (~1e-3 sample error) at the
      cost of ~``interpolation_iters`` IFFTs per epoch (roughly 10x ``"fast"``).
    - ``"legacy"``: whole-sample (integer) alignment. The cross-correlation
      peak is located at integer resolution and the correction is applied
      either by moving the triggers (``apply_to_raw=False``) or by rolling the
      raw data segments (``apply_to_raw=True``). The original, well-tested
      behaviour, kept for safety / backwards compatibility.

    Both sub-sample modes remove residual sub-sample misalignment, the dominant
    source of residual artifact after AAS. (Naive FFT-upsampling of the
    correlation vector was evaluated and rejected: on the normalised,
    non-periodic correlation it is *less* accurate than the ``"fast"``
    parabolic fit.)

    Parameters
    ----------
    ref_trigger_index : int, optional
        Index of the reference trigger used as the alignment template
        (default: 0).
    ref_channel : int, optional
        Reference channel index. Uses the first EEG channel when ``None``
        (default: None).
    search_window : int, optional
        Search radius in samples. Defaults to twice the upsampling factor.
    mode : {"fast", "quality", "legacy"}, optional
        Alignment mode (default: ``"fast"``). See the class docstring.
    apply_to_raw : bool, optional
        ``"legacy"`` mode only: if ``True``, roll raw data segments by the
        computed shifts instead of moving the triggers (default: False).
        Ignored by ``"fast"``/``"quality"`` (which always write to the data).
    interpolation_iters : int, optional
        Binary-search iterations for ``mode="quality"`` (default: 15, matching
        MATLAB FACET).
    ssa_hp_freq : float or None, optional
        High-pass cutoff in Hz applied to the reference channel **before**
        estimating the sub-sample shift, mirroring MATLAB FACET
        ``AlignSubSample`` (``SSAHPFrequency``, default 300 Hz). High-passing
        makes the alignment lock onto the high-frequency gradient-artifact
        edges instead of low-frequency drift / neural signal. The filter feeds
        the shift *estimation* only — the computed shift is still applied to the
        unfiltered raw data. ``None`` or ``0`` disables it. Only used by the
        ``"fast"``/``"quality"`` modes; ``"legacy"`` is never filtered
        (default: 300.0).
    """

    name = "subsample_aligner"
    description = "Refine trigger alignment at subsample precision"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = True
    channel_wise = True
    # Class default; overridden per-instance in __init__. Only "legacy" is truly
    # run-once (it moves the shared trigger metadata, which propagates to every
    # channel). The fractional modes bake the shift into each channel's DATA, so
    # in channel-sequential execution they must run for EVERY channel — otherwise
    # only the first channel gets aligned.
    run_once = True

    def __init__(
        self,
        ref_trigger_index: int = 0,
        ref_channel: int | None = None,
        search_window: int | None = None,
        mode: str = "fast",
        apply_to_raw: bool = False,
        interpolation_iters: int = 15,
        ssa_hp_freq: float | None = 300.0,
    ) -> None:
        if mode not in ("legacy", "fast", "quality"):
            raise ValueError(f"mode must be 'legacy', 'fast' or 'quality', got {mode!r}")
        if ssa_hp_freq is not None and ssa_hp_freq < 0:
            raise ValueError(f"ssa_hp_freq must be None or >= 0, got {ssa_hp_freq!r}")
        self.ref_trigger_index = ref_trigger_index
        self.ref_channel = ref_channel
        self.search_window = search_window
        self.mode = mode
        # "legacy" moves the (shared) trigger metadata once; the fractional modes
        # write per-channel data and must therefore run for every channel under
        # channel-sequential execution (estimate + apply per channel).
        self.run_once = mode == "legacy"
        self.apply_to_raw = apply_to_raw
        self.interpolation_iters = max(1, int(interpolation_iters))
        self.ssa_hp_freq = ssa_hp_freq
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError("Artifact length not set. Run TriggerDetector before SubsampleAligner.")
        if self.ref_trigger_index >= len(context.get_triggers()):
            raise ProcessorValidationError(
                f"Reference trigger index {self.ref_trigger_index} is out of range "
                f"for {len(context.get_triggers())} triggers"
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers().copy()
        artifact_length = context.get_artifact_length()
        n_samples = raw.n_times
        upsampling_factor = max(1, context.metadata.upsampling_factor)

        # --- LOG ---
        logger.info("Refining trigger alignment with subsample precision")

        # --- COMPUTE ---
        pre_samples, post_samples = _get_pre_post_samples(context, artifact_length)
        window_length = pre_samples + post_samples
        search_radius = self._resolve_search_radius(upsampling_factor)
        ref_channel = self._pick_ref_channel(raw)
        ref_signal = raw.get_data(picks=[ref_channel])[0]

        fractional = self.mode != "legacy"

        # SSA high-pass (MATLAB AlignSubSample): high-pass the reference channel
        # before estimating the sub-sample shift so the alignment locks onto the
        # high-frequency artifact edges, not low-frequency drift / neural signal.
        # This feeds the shift ESTIMATION only — the computed shift is applied to
        # the unfiltered raw data below. ``"legacy"`` is intentionally left
        # unfiltered (unchanged original behaviour).
        ssa_hp_applied = None
        estimation_signal = ref_signal
        if fractional:
            estimation_signal, ssa_hp_applied = self._highpass_for_estimation(ref_signal, raw.info["sfreq"])

        # Extract the reference epoch over the SAME extended window the per-epoch
        # search segments use (front/back padded by ``search_radius``). If the
        # reference were the inner window only, ``crosscorrelation`` would pad
        # the shorter array at its end, displacing the zero-lag point by
        # ``search_radius`` and adding a constant +search_radius bias to every
        # computed shift (a zero-offset artifact then yields shift=search_radius
        # instead of 0).
        ref_epoch = _extract_epoch_with_padding(
            estimation_signal,
            triggers[self.ref_trigger_index] - pre_samples - search_radius,
            window_length + 2 * search_radius,
            n_samples,
        )
        shifts = self._compute_shifts(
            estimation_signal,
            triggers,
            ref_epoch,
            pre_samples,
            window_length,
            search_radius,
            n_samples,
            fractional=fractional,
        )

        # --- BUILD RESULT ---
        if fractional:
            applied_to = "raw_fractional"
        elif self.apply_to_raw:
            applied_to = "raw"
        else:
            applied_to = "triggers"

        new_metadata = context.metadata.copy()
        new_metadata.custom.setdefault("subsample_alignment", {}).update(
            {
                "shifts": shifts.tolist(),
                "ref_trigger_index": self.ref_trigger_index,
                "search_window": search_radius,
                "mode": self.mode,
                "applied_to": applied_to,
                "ssa_hp_freq": ssa_hp_applied,
            }
        )
        if new_metadata.pre_trigger_samples is None:
            new_metadata.pre_trigger_samples = pre_samples
        if new_metadata.post_trigger_samples is None:
            new_metadata.post_trigger_samples = post_samples

        # ``shift`` is the offset of the artifact relative to its nominal
        # trigger (positive = artifact sits later). The correction can be
        # applied EITHER by moving the triggers OR by moving the data — never
        # both, otherwise the artifact is displaced twice.
        if fractional:
            # "fast"/"quality": bake the sub-sample shift into the data via FFT
            # phase shifting (like MATLAB AlignSubSample). Triggers stay integer.
            logger.debug("Applying fractional subsample shifts to raw data (mode={})", self.mode)
            new_metadata.triggers = triggers
            if np.any(shifts):
                raw_copy = self._apply_fractional_shifts_to_raw(
                    raw, triggers, shifts, pre_samples, post_samples, n_samples
                )
                return context.with_raw(raw_copy).with_metadata(new_metadata)
            return context.with_metadata(new_metadata)

        if self.apply_to_raw and np.any(shifts):
            # Bake the correction into the data: roll each segment by -shift so
            # the artifact moves back onto the nominal trigger position. The
            # triggers therefore stay at their original positions.
            logger.debug("Applying subsample shifts to raw data segments")
            new_metadata.triggers = triggers
            raw_copy = self._apply_shifts_to_raw(raw, triggers, shifts, pre_samples, post_samples, n_samples)
            return context.with_raw(raw_copy).with_metadata(new_metadata)

        # Default: move the triggers to the measured artifact positions; the
        # raw data is left untouched.
        new_metadata.triggers = np.clip(triggers + shifts, 0, n_samples - 1).astype(int)

        # --- RETURN ---
        return context.with_metadata(new_metadata)

    def _resolve_search_radius(self, upsampling_factor: int) -> int:
        """Return the effective search radius in samples.

        Parameters
        ----------
        upsampling_factor : int
            Current upsampling factor.

        Returns
        -------
        int
            Search radius (≥ 1).
        """
        if self.search_window is not None:
            return int(max(1, self.search_window))
        return int(max(1, upsampling_factor * 2))

    def _pick_ref_channel(self, raw: mne.io.Raw) -> int:
        """Return the reference channel index for cross-correlation.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object whose info is used for channel picking.

        Returns
        -------
        int
            Channel index.
        """
        if self.ref_channel is not None:
            return self.ref_channel
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        return int(eeg_channels[0]) if len(eeg_channels) else 0

    def _design_ssa_highpass(self, sfreq: float) -> np.ndarray | None:
        """Design the SSA high-pass FIR filter (MATLAB ``AlignSubSample``).

        Replicates MATLAB FACET: a fixed-order (100, i.e. 101 taps) ``firls``
        linear-phase high-pass with a ±10 % transition band around
        ``ssa_hp_freq``. The fixed order keeps the filter short (~101 taps)
        regardless of sampling rate, so it never triggers the ``filtfilt``
        padlen blow-up that a cutoff-derived order would on upsampled data.

        Parameters
        ----------
        sfreq : float
            (Upsampled) sampling frequency in Hz.

        Returns
        -------
        np.ndarray or None
            FIR filter weights, or None when high-passing is disabled or the
            requested cutoff is not realisable at this sampling rate.
        """
        cutoff = self.ssa_hp_freq
        if cutoff is None or cutoff <= 0:
            return None

        nyq = 0.5 * sfreq
        lo = (cutoff * 0.9) / nyq
        hi = (cutoff * 1.1) / nyq
        if not 0 < lo < hi < 1:
            logger.warning(
                "SSA high-pass cutoff {} Hz not realisable at {} Hz (band {:.3f}-{:.3f}); disabling SSA high-pass",
                cutoff,
                sfreq,
                lo,
                hi,
            )
            return None

        # MATLAB AlignSubSample uses firls(100, ...) -> 101 taps (fixed order).
        try:
            return firls(101, [0.0, lo, hi, 1.0], [0.0, 0.0, 1.0, 1.0])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("SSA high-pass design failed ({}); disabling SSA high-pass", exc)
            return None

    def _highpass_for_estimation(self, signal: np.ndarray, sfreq: float) -> tuple[np.ndarray, float | None]:
        """High-pass the reference signal for shift estimation.

        Applies a zero-phase ``filtfilt`` with the SSA high-pass weights. Falls
        back to the unfiltered signal (with a debug log) when the filter is
        disabled or the signal is shorter than the required padding, so short
        test recordings never crash.

        Parameters
        ----------
        signal : np.ndarray
            1-D reference channel signal.
        sfreq : float
            (Upsampled) sampling frequency in Hz.

        Returns
        -------
        tuple of (np.ndarray, float or None)
            ``(estimation_signal, applied_cutoff)``. ``applied_cutoff`` is the
            cutoff in Hz when filtering was applied, else None.
        """
        weights = self._design_ssa_highpass(sfreq)
        if weights is None:
            return signal, None

        padlen = 3 * (len(weights) - 1)
        if len(signal) <= padlen:
            logger.debug(
                "Reference signal too short ({} samples) for SSA high-pass padlen ({}); skipping SSA high-pass",
                len(signal),
                padlen,
            )
            return signal, None

        filtered = filtfilt(weights, 1.0, signal)
        return filtered, float(self.ssa_hp_freq)

    def _compute_shifts(
        self,
        ref_signal: np.ndarray,
        triggers: np.ndarray,
        ref_epoch: np.ndarray,
        pre_samples: int,
        window_length: int,
        search_radius: int,
        n_samples: int,
        fractional: bool = False,
    ) -> np.ndarray:
        """Compute per-trigger subsample alignment shifts.

        Parameters
        ----------
        ref_signal : np.ndarray
            1-D reference channel signal.
        triggers : np.ndarray
            Original trigger positions.
        ref_epoch : np.ndarray
            Reference epoch used as the cross-correlation template.
        pre_samples : int
            Samples before the trigger forming the epoch start.
        window_length : int
            Total epoch length (pre + post) in samples.
        search_radius : int
            Search radius in samples.
        n_samples : int
            Total number of samples in the recording.
        fractional : bool, optional
            If ``True``, refine each peak to fractional-sample resolution via
            FFT-upsampling of the correlation vector; otherwise return integer
            shifts (default: False).

        Returns
        -------
        np.ndarray
            Shift for each trigger, shape ``(n_triggers,)``. Integer dtype when
            ``fractional=False``, float dtype otherwise.
        """
        shifts = np.zeros(len(triggers), dtype=float if fractional else int)
        for idx, trigger in enumerate(triggers):
            if idx == self.ref_trigger_index:
                continue
            segment = _extract_epoch_with_padding(
                ref_signal,
                trigger - pre_samples - search_radius,
                window_length + 2 * search_radius,
                n_samples,
            )
            corr = np.nan_to_num(crosscorrelation(segment, ref_epoch, search_radius))
            integer_shift = int(np.argmax(corr) - search_radius)
            if not fractional:
                shifts[idx] = integer_shift
            elif self.mode == "fast":
                shifts[idx] = self._refine_shift_parabolic(corr, search_radius)
            else:  # "quality"
                shifts[idx] = self._refine_shift_matlab(segment, ref_epoch, integer_shift)
        return shifts

    @staticmethod
    def _refine_shift_parabolic(corr: np.ndarray, search_radius: int) -> float:
        """Sub-sample peak via 3-point parabolic interpolation.

        Fast (~0.05 sample error). Bias-bounded but not exact; use
        ``mode="quality"`` for higher fidelity.

        Parameters
        ----------
        corr : np.ndarray
            Correlation values over lags ``-search_radius .. +search_radius``.
        search_radius : int
            Search radius (zero-lag centre).

        Returns
        -------
        float
            Sub-sample shift in samples.
        """
        k = int(np.argmax(corr))
        if k <= 0 or k >= len(corr) - 1:
            return float(k - search_radius)
        a, b, c = corr[k - 1], corr[k], corr[k + 1]
        denom = a - 2 * b + c
        delta = 0.0 if denom == 0 else 0.5 * (a - c) / denom
        return (k - search_radius) + float(delta)

    def _refine_shift_matlab(self, segment: np.ndarray, ref_epoch: np.ndarray, integer_shift: int) -> float:
        """Sub-sample shift via binary search on the true alignment objective.

        Mirrors MATLAB FACET ``AlignSubSample``: the segment is FFT-shifted by
        a fractional amount and compared to the reference; a binary search over
        the shift maximises their (mean-removed) cross-product. Essentially
        exact at the cost of ``interpolation_iters`` IFFTs per epoch.

        Parameters
        ----------
        segment : np.ndarray
            1-D search segment around the current trigger.
        ref_epoch : np.ndarray
            1-D reference epoch (alignment target), same length as ``segment``.
        integer_shift : int
            Integer-resolution shift used to bracket the search.

        Returns
        -------
        float
            Sub-sample shift in samples.
        """
        n = len(segment)
        spectrum = np.fft.fft(segment)
        freqs = np.fft.fftfreq(n)
        ref0 = ref_epoch - ref_epoch.mean()

        def neg_objective(shift: float) -> float:
            # Shift the segment by -shift (undo the artifact's displacement)
            shifted = np.real(np.fft.ifft(spectrum * np.exp(2j * np.pi * freqs * shift)))
            shifted = shifted - shifted.mean()
            return -float(np.dot(shifted, ref0))

        left, mid, right = integer_shift - 1.0, float(integer_shift), integer_shift + 1.0
        c_left, c_mid, c_right = neg_objective(left), neg_objective(mid), neg_objective(right)
        for _ in range(self.interpolation_iters):
            if c_left < c_right:
                right, c_right = mid, c_mid
            else:
                left, c_left = mid, c_mid
            mid = 0.5 * (left + right)
            c_mid = neg_objective(mid)
        return mid

    @staticmethod
    def _fractional_shift(segment: np.ndarray, shift: float) -> np.ndarray:
        """Shift a (channels x time) segment by a fractional number of samples.

        Uses ideal band-limited (sinc) interpolation via FFT phase rotation,
        matching the MATLAB FACET ``AlignSubSample`` resampling. Positive
        ``shift`` delays the signal (moves it right).

        Parameters
        ----------
        segment : np.ndarray
            2-D array of shape ``(n_channels, n_times)``.
        shift : float
            Shift in samples (may be fractional).

        Returns
        -------
        np.ndarray
            The shifted segment, same shape as the input.
        """
        n = segment.shape[-1]
        freqs = np.fft.fftfreq(n)
        phase = np.exp(-2j * np.pi * freqs * shift)
        return np.real(np.fft.ifft(np.fft.fft(segment, axis=-1) * phase, axis=-1))

    def _apply_fractional_shifts_to_raw(
        self,
        raw: mne.io.Raw,
        triggers: np.ndarray,
        shifts: np.ndarray,
        pre_samples: int,
        post_samples: int,
        n_samples: int,
    ) -> mne.io.Raw:
        """Apply fractional shifts to raw data segments via FFT phase shifting.

        Each window is shifted by ``-shift`` so the artifact (sitting ``shift``
        samples after the nominal trigger) moves back onto the trigger. The
        shift is performed on an edge-padded copy of the window to keep the
        circular FFT wrap-around out of the artifact region; only the inner
        window is written back.

        Parameters
        ----------
        raw : mne.io.Raw
            Source raw object (copied internally).
        triggers : np.ndarray
            Original (integer) trigger positions.
        shifts : np.ndarray
            Per-trigger fractional shift values.
        pre_samples : int
            Samples before trigger forming each window start.
        post_samples : int
            Samples after trigger forming each window end.
        n_samples : int
            Total recording length in samples.

        Returns
        -------
        mne.io.Raw
            New raw object with fractionally-shifted segments.
        """
        raw_copy = raw.copy()
        data = raw_copy.get_data()
        n_channels = data.shape[0]
        max_shift = float(np.max(np.abs(shifts))) if len(shifts) else 0.0
        pad = 8 + int(np.ceil(max_shift))

        for idx, shift in enumerate(shifts):
            if shift == 0:
                continue
            window_start = max(0, triggers[idx] - pre_samples)
            window_end = min(n_samples, triggers[idx] + post_samples)
            length = window_end - window_start
            if length <= 0:
                continue
            # Edge-padded extension so the circular shift does not wrap real
            # samples into the artifact region.
            extended = np.empty((n_channels, length + 2 * pad), dtype=data.dtype)
            for ch in range(n_channels):
                extended[ch] = _extract_epoch_with_padding(data[ch], window_start - pad, length + 2 * pad, n_samples)
            shifted = self._fractional_shift(extended, -float(shift))
            data[:, window_start:window_end] = shifted[:, pad : pad + length]

        raw_copy._data[:] = data
        return raw_copy

    def _apply_shifts_to_raw(
        self,
        raw: mne.io.Raw,
        triggers: np.ndarray,
        shifts: np.ndarray,
        pre_samples: int,
        post_samples: int,
        n_samples: int,
    ) -> mne.io.Raw:
        """Apply computed shifts to raw data segments by rolling them.

        Parameters
        ----------
        raw : mne.io.Raw
            Source raw object (will be copied internally).
        triggers : np.ndarray
            Original trigger positions.
        shifts : np.ndarray
            Per-trigger shift values.
        pre_samples : int
            Samples before trigger forming each window start.
        post_samples : int
            Samples after trigger forming each window end.
        n_samples : int
            Total recording length in samples.

        Returns
        -------
        mne.io.Raw
            New raw object with rolled segments.
        """
        raw_copy = raw.copy()
        # Direct _data access avoids a redundant full-array copy on large datasets
        data = raw_copy.get_data()
        for idx, shift in enumerate(shifts):
            if shift == 0:
                continue
            trigger = triggers[idx]
            window_start = max(0, trigger - pre_samples)
            window_end = min(n_samples, trigger + post_samples)
            segment = data[:, window_start:window_end]
            # Roll by -shift: the artifact sits `shift` samples after the
            # nominal trigger, so shifting the segment LEFT by `shift` brings it
            # back onto the trigger position (matches MATLAB AlignSubSample,
            # which shifts the data to the reference rather than moving triggers).
            data[:, window_start:window_end] = np.roll(segment, -shift, axis=1)
        raw_copy._data[:] = data
        return raw_copy

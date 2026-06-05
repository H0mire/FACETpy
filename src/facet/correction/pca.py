"""
PCA-based artifact correction processor.
"""

import mne
import numpy as np
from loguru import logger
from scipy.signal import filtfilt, firls, firwin

from ..console import processor_progress
from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor
from ..helpers.utils import split_vector


@register_processor
class PCACorrection(Processor):
    """Remove fMRI artifacts from EEG data using Principal Component Analysis.

    Splits the acquisition window into trigger-aligned epochs, applies PCA to
    each epoch, reconstructs the data using the retained components, and
    subtracts the reconstruction from the original signal.  The subtracted
    portion is treated as the artifact estimate.

    The number of retained components can be controlled precisely:

    - An integer keeps exactly that many components.
    - A float in (0, 1) retains enough components to explain that fraction of
      the total variance (e.g. 0.95 → 95 %).
    - ``"auto"`` uses MATLAB FACET OBS heuristics.
    - 0 skips PCA for all channels.

    Parameters
    ----------
    n_components : int or float or str
        Number of PCA components to retain (int) or variance fraction to
        retain (float in (0, 1)). Use ``"auto"`` for MATLAB-like OBS auto
        selection. Default: 0.95.
    hp_freq : float, optional
        High-pass cutoff frequency in Hz applied before PCA. None skips
        filtering (default: None).
    hp_filter_weights : np.ndarray, optional
        Pre-computed filter weights; overrides ``hp_freq`` when provided.
    exclude_channels : list, optional
        Channel indices to skip during PCA (default: empty list).
    """

    name = "pca_correction"
    description = "PCA-based artifact removal"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = True
    parallel_safe = True
    channel_wise = True

    TH_SLOPE = 2.0
    TH_CUMVAR = 80.0
    TH_VAREXP = 5.0

    def __init__(
        self,
        n_components: int | float | str = 0.95,
        hp_freq: float | None = None,
        hp_filter_weights: np.ndarray | None = None,
        exclude_channels: list | None = None,
    ) -> None:
        self.n_components = n_components
        self.hp_freq = hp_freq
        self.hp_filter_weights = hp_filter_weights
        self.exclude_channels = exclude_channels or []
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError("Artifact length not set. Run TriggerDetector first.")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

        # --- LOG ---
        logger.info("Applying PCA artifact correction to {} channels", len(eeg_channels))

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found, skipping PCA")
            return context

        # --- COMPUTE ---
        hp_weights = self._resolve_hp_weights(raw.info["sfreq"])
        s_acq_start, s_acq_end = self._get_acquisition_window(context)

        # The HP filter and acquisition window are identical for every channel,
        # so decide ONCE (and log once) whether the filter can be applied. This
        # surfaces an over-long filter as a clear warning instead of letting
        # ``filtfilt`` raise inside the per-channel ``except`` and silently
        # disabling the entire correction.
        if hp_weights is not None and (s_acq_end - s_acq_start) <= len(hp_weights):
            logger.warning(
                "OBS high-pass filter ({} taps) is longer than the acquisition "
                "window ({} samples); proceeding without high-pass filtering",
                len(hp_weights),
                s_acq_end - s_acq_start,
            )
            hp_weights = None

        # Direct _data access avoids a full array copy on large datasets
        estimated_artifacts = np.zeros(raw._data.shape)

        with processor_progress(
            total=len(eeg_channels) or None,
            message="PCA artifact correction",
        ) as progress:
            for idx, ch_idx in enumerate(eeg_channels):
                ch_name = raw.ch_names[ch_idx]
                status_prefix = f"{idx + 1}/{len(eeg_channels)} • {ch_name}"

                if ch_idx in self.exclude_channels:
                    progress.advance(1, message=f"{status_prefix} (excluded)")
                    continue

                if self.n_components == 0:
                    progress.advance(1, message=f"{status_prefix} (disabled)")
                    continue

                try:
                    fitted_artifact = self._calc_fitted_artifact(
                        raw._data[ch_idx],
                        triggers,
                        artifact_length,
                        s_acq_start,
                        s_acq_end,
                        hp_weights,
                    )
                    # Match MATLAB FACET (FACET.m:1266-1267):
                    #   RANoiseAcq = RANoiseAcq + fitted_res
                    #   RAEEGAcq   = RAEEGAcq   - fitted_res
                    raw._data[ch_idx][s_acq_start:s_acq_end] -= fitted_artifact
                    estimated_artifacts[ch_idx][s_acq_start:s_acq_end] += fitted_artifact
                    progress.advance(1, message=status_prefix)
                except Exception as exc:
                    logger.error("PCA failed for channel {}: {}", ch_name, exc)
                    progress.advance(1, message=f"{status_prefix} (error)")

        # --- NOISE ---
        new_ctx = context.with_raw(raw)
        new_ctx.accumulate_noise(estimated_artifacts)

        # --- RETURN ---
        logger.info("PCA correction completed")
        return new_ctx

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------

    def _resolve_hp_weights(self, sfreq: float) -> np.ndarray | None:
        """Return the appropriate high-pass filter weights.

        Parameters
        ----------
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        np.ndarray or None
            Filter weights for use with ``scipy.signal.filtfilt``, or None
            when high-pass filtering is disabled.
        """
        if self.hp_filter_weights is not None:
            return self.hp_filter_weights
        if self.hp_freq is not None and self.hp_freq > 0:
            return self._create_hp_filter(sfreq)
        return None

    def _calc_fitted_artifact(
        self,
        ch_data: np.ndarray,
        triggers: np.ndarray,
        artifact_length: int,
        s_acq_start: int,
        s_acq_end: int,
        hp_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Calculate the OBS-fitted artifact estimate for a single channel.

        Parameters
        ----------
        ch_data : np.ndarray
            Full time series for one channel.
        triggers : np.ndarray
            Trigger sample positions.
        artifact_length : int
            Artifact length in samples.
        s_acq_start : int
            Acquisition window start sample.
        s_acq_end : int
            Acquisition window end sample.
        hp_weights : np.ndarray or None
            High-pass filter weights (or None to skip filtering).

        Returns
        -------
        np.ndarray
            Fitted artifact (OBS reconstruction) of length
            ``s_acq_end - s_acq_start``, to be subtracted from the EEG.
        """
        ch_data_acq = ch_data[s_acq_start:s_acq_end]

        ch_data_filtered = self._apply_hp_filter(ch_data_acq, hp_weights) if hp_weights is not None else ch_data_acq

        adjusted_triggers = triggers - s_acq_start
        # Small offset prevents epoch boundaries from sitting exactly on the trigger
        offset = int(artifact_length * 0.1)

        epochs = split_vector(ch_data_filtered, adjusted_triggers + offset, artifact_length)
        artifact_per_epoch = self._calc_pca(epochs)

        fitted_artifact = np.zeros(len(ch_data_acq))
        for i, trigger in enumerate(adjusted_triggers):
            start_pos = trigger + offset
            end_pos = start_pos + artifact_length
            if start_pos < 0:
                continue
            if end_pos > len(ch_data_acq):
                epoch_length = len(ch_data_acq) - start_pos
                if epoch_length <= 0:
                    continue
                fitted_artifact[start_pos:] = artifact_per_epoch[i, :epoch_length]
            else:
                fitted_artifact[start_pos:end_pos] = artifact_per_epoch[i, :]

        return fitted_artifact

    def _calc_pca(self, epochs: np.ndarray) -> np.ndarray:
        """Apply PCA to epochs and return the OBS-fitted artifact estimate.

        Mirrors the MATLAB FACET OBS routine (``DoPCA.m`` + ``FitOBS.m`` +
        ``FACET.m:1252-1267``): epochs are mean-centered per column, SVD yields
        the optimal basis set, each epoch is projected onto the top components
        (rank-k reconstruction = LSQ fit), and the reconstruction itself is
        returned as the artifact estimate to be subtracted from the EEG.

        Parameters
        ----------
        epochs : np.ndarray
            Epoch matrix of shape (n_epochs, n_times).

        Returns
        -------
        np.ndarray
            Per-epoch OBS reconstruction (artifact estimate) of shape
            (n_epochs, n_times).
        """
        epochs_t = epochs.T

        col_var = np.var(epochs_t, axis=0)
        variance_threshold = 1e-12
        valid_mask = col_var > variance_threshold

        if np.count_nonzero(valid_mask) < 2:
            return np.zeros_like(epochs)

        X_valid = epochs_t[:, valid_mask]
        # MATLAB FACET uses ``detrend('constant')`` — column-wise mean removal
        # only. z-Score scaling would distort the variance ranking of the
        # singular values and break the OBS auto-selection heuristic.
        mean_valid = np.mean(X_valid, axis=0)
        X_centered = X_valid - mean_valid

        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            logger.warning("PCA SVD failed ({}); skipping channel", exc)
            return np.zeros_like(epochs)

        max_components = min(X_centered.shape[0], X_centered.shape[1])
        if max_components <= 1:
            return np.zeros_like(epochs)

        n_components = self._select_n_components(S, max_components, X_centered.shape[0])

        U_reduced = U[:, :n_components]
        S_reduced = S[:n_components]
        Vt_reduced = Vt[:n_components, :]
        # Rank-k reconstruction in the original (mean-restored) space. This is
        # the OBS-fitted artifact estimate — equivalent to MATLAB FACET's
        # ``fitted_res = papc * (pinv(papc) * epoch)``. The broadcasting form
        # ``U * S`` avoids materialising a ``diag(S)`` matrix.
        #
        # ``np.errstate`` suppresses a spurious "divide by zero encountered in
        # matmul" FP flag numpy can raise for this product (it performs no
        # division). Harmless under numpy's default "warn" mode but fatal under
        # ``-W error`` / ``filterwarnings=error``.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            artifact_valid = (U_reduced * S_reduced) @ Vt_reduced + mean_valid

        artifact_full = np.zeros_like(epochs_t)
        artifact_full[:, valid_mask] = artifact_valid

        return artifact_full.T

    def _select_n_components(self, singular_values: np.ndarray, max_components: int, n_samples: int) -> int:
        """Determine the number of PCA components to retain.

        Parameters
        ----------
        singular_values : np.ndarray
            Singular values from the SVD decomposition.
        max_components : int
            Upper bound on the number of components.
        n_samples : int
            Number of samples (time points) used in the SVD.

        Returns
        -------
        int
            Number of components to retain (≥ 1).
        """
        if isinstance(self.n_components, int):
            return max(1, min(self.n_components, max_components))

        if isinstance(self.n_components, str):
            if self.n_components.lower() != "auto":
                raise ValueError("n_components as string must be 'auto'.")

            explained_var = (singular_values**2) / (n_samples - 1)
            explained_pct = 100.0 * (explained_var / np.sum(explained_var))
            n = self._select_n_components_auto(explained_pct)
            return max(1, min(int(n), max_components))

        if not 0 < self.n_components < 1:
            raise ValueError("n_components as float must be between 0 and 1.")

        explained_var = (singular_values**2) / (n_samples - 1)
        explained_ratio = np.cumsum(explained_var) / np.sum(explained_var)
        n = np.searchsorted(explained_ratio, self.n_components) + 1
        return max(1, min(int(n), max_components))

    def _select_n_components_auto(self, explained_pct: np.ndarray) -> int:
        """Select component count using MATLAB FACET OBS auto heuristics."""
        if len(explained_pct) == 0:
            return 1

        d_oev = np.where(np.abs(np.diff(explained_pct)) < self.TH_SLOPE)[0] + 1
        slope_pc = 1
        if len(d_oev) > 0:
            if len(d_oev) >= 4:
                dd_oev = np.diff(d_oev)
                run_pos = None
                for i in range(len(dd_oev) - 2):
                    if dd_oev[i] == 1 and dd_oev[i + 1] == 1 and dd_oev[i + 2] == 1:
                        run_pos = i
                        break
                idx = run_pos if run_pos is not None else 0
                slope_pc = int(max(d_oev[idx] - 1, 1))
            else:
                slope_pc = int(max(d_oev[0] - 1, 1))

        cumvar = np.cumsum(explained_pct)
        tmp = np.where(cumvar > self.TH_CUMVAR)[0]
        cumvar_pc = int(tmp[0] + 1) if len(tmp) > 0 else len(explained_pct)

        tmp = np.where(explained_pct < self.TH_VAREXP)[0]
        varexp_pc = int(max(tmp[0], 1)) if len(tmp) > 0 else len(explained_pct)

        pcs = int(np.floor(np.mean([slope_pc, cumvar_pc, varexp_pc])))
        return max(1, pcs)

    def _create_hp_filter(self, sfreq: float) -> np.ndarray:
        """Create FIR high-pass filter weights.

        Replicates the MATLAB FACET OBS high-pass (``FACET.m:878-888``): a
        linear-phase ``firls`` filter whose **order is derived from the cutoff
        and a ±10 Hz transition band**::

            filtorder = round(1.2 * sfreq / (OBSHPFrequency - 10))   # made even
            f = [0, (hp-10)/nyq, (hp+10)/nyq, 1];   a = [0 0 1 1]

        This keeps the filter short (~23 taps at 300 Hz / 5 kHz), so it never
        triggers the ``filtfilt`` padlen blow-up that the previous
        ``numtaps = max(101, 2*sfreq)`` design caused (~10001 taps at 1 Hz).
        The transition-band formula is only defined for cutoffs above its 10 Hz
        half-width; for ``hp_freq <= 10`` it falls back to a windowed-sinc
        high-pass (which inherently needs a long filter at this sampling rate —
        the call site caps the padding so it still cannot raise).

        Parameters
        ----------
        sfreq : float
            Sampling frequency in Hz (the current, possibly upsampled, rate).

        Returns
        -------
        np.ndarray
            FIR filter taps for use with ``scipy.signal.filtfilt``.
        """
        nyq = 0.5 * sfreq
        cutoff = self.hp_freq

        if cutoff > 10:
            lo = (cutoff - 10) / nyq
            hi = (cutoff + 10) / nyq
            if 0 < lo < hi < 1:
                # MATLAB derives an even filter ORDER; firls taps = order + 1.
                filtorder = int(round(1.2 * sfreq / (cutoff - 10)))
                if filtorder % 2 != 0:
                    filtorder += 1
                numtaps = max(filtorder + 1, 3)
                return firls(numtaps, [0.0, lo, hi, 1.0], [0.0, 0.0, 1.0, 1.0])

        # Low cutoff (≤ 10 Hz): MATLAB's transition-band formula is undefined.
        normalized_cutoff = cutoff / nyq
        numtaps = max(101, int(2 * sfreq))
        if numtaps % 2 == 0:
            numtaps += 1
        return firwin(numtaps, normalized_cutoff, pass_zero=False)

    @staticmethod
    def _apply_hp_filter(signal: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Zero-phase high-pass with padding capped to the signal length.

        ``scipy.signal.filtfilt`` raises when the signal is shorter than its
        default ``padlen = 3*(len(weights)-1)``. Capping the padlen keeps a long
        FIR (low-cutoff fallback) from crashing on short acquisition windows.

        Parameters
        ----------
        signal : np.ndarray
            1-D signal to filter.
        weights : np.ndarray
            FIR high-pass weights.

        Returns
        -------
        np.ndarray
            Zero-phase filtered signal (same length as ``signal``).
        """
        padlen = min(3 * (len(weights) - 1), len(signal) - 1)
        return filtfilt(weights, 1.0, signal, padlen=max(padlen, 0))

    def _get_acquisition_window(self, context: ProcessingContext) -> tuple:
        """Return the start and end sample indices of the acquisition window.

        Parameters
        ----------
        context : ProcessingContext
            Current processing context.

        Returns
        -------
        tuple
            ``(s_acq_start, s_acq_end)`` as integers.
        """
        raw = context.get_raw()
        triggers = context.get_triggers()

        if len(triggers) == 0:
            return 0, raw.n_times

        artifact_length = context.get_artifact_length()
        if artifact_length is None:
            return 0, raw.n_times

        trigger_min = int(np.min(triggers))
        trigger_max = int(np.max(triggers))
        s_acq_start = max(0, trigger_min - artifact_length)
        s_acq_end = min(raw.n_times, trigger_max + artifact_length)
        if s_acq_end <= s_acq_start:
            return 0, raw.n_times
        return s_acq_start, s_acq_end

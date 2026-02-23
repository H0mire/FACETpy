"""Volume-gap artifact correction processor."""

from __future__ import annotations

import mne
import numpy as np
from loguru import logger

from ..console import processor_progress
from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor


@register_processor
class VolumeArtifactCorrection(Processor):
    """Correct volume-transition artifacts around slice-trigger gaps.

    MATLAB FACET's ``RARemoveVolumeArtifact`` subtracts a transition artifact
    from the slice before and after each detected volume gap, then linearly
    interpolates the gap itself. This processor ports that behavior.

    Parameters
    ----------
    template_count : int
        Number of neighboring slices used to form pre/post templates
        (default: 5).
    weighting_position : float
        Relative location of the logistic midpoint inside one artifact epoch
        (default: 0.8).
    weighting_slope : float
        Logistic slope used for transition weighting (default: 20.0).
    """

    name = "volume_artifact_correction"
    description = "Correct transition artifacts around volume gaps"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = True
    parallel_safe = True
    channel_wise = True

    def __init__(
        self,
        template_count: int = 5,
        weighting_position: float = 0.8,
        weighting_slope: float = 20.0,
    ) -> None:
        self.template_count = int(template_count)
        self.weighting_position = float(weighting_position)
        self.weighting_slope = float(weighting_slope)
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)

        if context.get_artifact_length() is None or context.get_artifact_length() <= 1:
            raise ProcessorValidationError("Artifact length must be > 1 for volume artifact correction.")
        if self.template_count < 1:
            raise ProcessorValidationError(f"template_count must be >= 1, got {self.template_count}")
        if not (0.0 <= self.weighting_position <= 1.0):
            raise ProcessorValidationError(f"weighting_position must be in [0, 1], got {self.weighting_position}")
        if self.weighting_slope <= 0:
            raise ProcessorValidationError(f"weighting_slope must be positive, got {self.weighting_slope}")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        if not context.metadata.volume_gaps:
            logger.info("VolumeArtifactCorrection skipped: context metadata indicates no volume gaps.")
            return context

        triggers = context.get_triggers()
        gap_pre_indices = self._find_volume_gap_pre_indices(triggers)
        if gap_pre_indices.size == 0:
            logger.info("VolumeArtifactCorrection skipped: no large trigger-distance gaps found.")
            return context

        raw = context.get_raw().copy()
        artifact_length = int(context.get_artifact_length())
        pre_samples, post_samples = self._resolve_pre_post_samples(context, artifact_length)
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

        # --- LOG ---
        logger.info(
            "Applying volume artifact correction: {} channels, {} volume gaps",
            len(eeg_channels),
            len(gap_pre_indices),
        )

        # --- COMPUTE ---
        weight = self._logistic_weight(artifact_length)
        corrected_pairs = 0
        interpolated_gaps = 0

        with processor_progress(
            total=len(eeg_channels) or None,
            message="Volume artifact correction",
        ) as progress:
            for ch_pos, ch_idx in enumerate(eeg_channels):
                ch_name = raw.ch_names[ch_idx]
                pairs, gaps = self._correct_channel(
                    ch_data=raw._data[ch_idx],
                    triggers=triggers,
                    gap_pre_indices=gap_pre_indices,
                    pre_samples=pre_samples,
                    post_samples=post_samples,
                    weight=weight,
                )
                corrected_pairs += pairs
                interpolated_gaps += gaps
                progress.advance(1, message=f"{ch_pos + 1}/{len(eeg_channels)} • {ch_name}")

        # --- RETURN ---
        logger.info(
            "Volume artifact correction complete: {} corrected slice-pairs, {} interpolated gaps",
            corrected_pairs,
            interpolated_gaps,
        )
        return context.with_raw(raw)

    def _find_volume_gap_pre_indices(self, triggers: np.ndarray) -> np.ndarray:
        """Return indices of triggers directly before a volume gap."""
        if len(triggers) < 2:
            return np.array([], dtype=int)
        diffs = np.diff(triggers)
        middle = float(np.mean([np.min(diffs), np.max(diffs)]))
        return np.where(diffs > middle)[0].astype(int)

    def _resolve_pre_post_samples(self, context: ProcessingContext, artifact_length: int) -> tuple[int, int]:
        """Derive pre/post trigger sample counts for one artifact epoch."""
        metadata = context.metadata
        sfreq = context.get_sfreq()
        max_pre = max(0, artifact_length - 1)

        if metadata.pre_trigger_samples is not None:
            pre_samples = int(max(0, min(metadata.pre_trigger_samples, max_pre)))
        else:
            offset_samples = int(round(metadata.artifact_to_trigger_offset * sfreq))
            pre_samples = int(max(0, min(-offset_samples, max_pre))) if offset_samples < 0 else 0

        max_post = max(0, artifact_length - pre_samples - 1)
        if metadata.post_trigger_samples is not None:
            post_samples = int(max(0, min(metadata.post_trigger_samples, max_post)))
        else:
            post_samples = int(max_post)

        if pre_samples + post_samples + 1 < artifact_length:
            post_samples = artifact_length - pre_samples - 1
        return pre_samples, post_samples

    def _logistic_weight(self, artifact_length: int) -> np.ndarray:
        """Build MATLAB-style logistic transition weights for one epoch."""
        x = np.arange(1, artifact_length + 1, dtype=float) / float(artifact_length)
        return 1.0 / (1.0 + np.exp(-self.weighting_slope * (x - self.weighting_position)))

    def _correct_channel(
        self,
        ch_data: np.ndarray,
        triggers: np.ndarray,
        gap_pre_indices: np.ndarray,
        pre_samples: int,
        post_samples: int,
        weight: np.ndarray,
    ) -> tuple[int, int]:
        """Correct all detected volume gaps for one EEG channel."""
        corrected_pairs = 0
        interpolated_gaps = 0
        epoch_len = pre_samples + post_samples + 1
        n_times = ch_data.shape[0]

        for gap_pre_idx in gap_pre_indices:
            trig_pre_idx = int(gap_pre_idx)
            trig_post_idx = trig_pre_idx + 1

            pre_start = int(triggers[trig_pre_idx] - pre_samples)
            pre_stop = pre_start + epoch_len
            post_start = int(triggers[trig_post_idx] - pre_samples)
            post_stop = post_start + epoch_len

            if pre_start < 0 or post_start < 0 or pre_stop > n_times or post_stop > n_times:
                continue

            prev_indices = np.arange(trig_pre_idx - self.template_count, trig_pre_idx, dtype=int)
            next_indices = np.arange(trig_post_idx + 1, trig_post_idx + 1 + self.template_count, dtype=int)
            if prev_indices[0] < 0 or next_indices[-1] >= len(triggers):
                continue

            data_pre = ch_data[pre_start:pre_stop].copy()
            data_post = ch_data[post_start:post_stop].copy()

            template_pre = self._mean_template(ch_data, triggers[prev_indices], pre_samples, epoch_len)
            template_post = self._mean_template(ch_data, triggers[next_indices], pre_samples, epoch_len)
            if template_pre is None or template_post is None:
                continue

            vol_art_pre = (data_pre - template_pre) * weight
            vol_art_post = (data_post - template_post) * weight[::-1]

            corrected_pre = data_pre - vol_art_pre
            corrected_post = data_post - vol_art_post

            ch_data[pre_start:pre_stop] = corrected_pre
            ch_data[post_start:post_stop] = corrected_post
            corrected_pairs += 1

            gap_start = int(triggers[trig_pre_idx] + post_samples + 1)
            gap_end = int(triggers[trig_post_idx] - pre_samples - 1)
            if gap_end >= gap_start:
                gap_len = gap_end - gap_start + 1
                gap_values = corrected_pre[-1] + (
                    (np.arange(1, gap_len + 1, dtype=float) / float(gap_len + 1))
                    * (corrected_post[0] - corrected_pre[-1])
                )
                ch_data[gap_start : gap_end + 1] = gap_values
                interpolated_gaps += 1

        return corrected_pairs, interpolated_gaps

    def _mean_template(
        self,
        ch_data: np.ndarray,
        template_triggers: np.ndarray,
        pre_samples: int,
        epoch_len: int,
    ) -> np.ndarray | None:
        """Average valid template epochs for one side of a volume gap."""
        segments = []
        n_times = ch_data.shape[0]

        for trigger in template_triggers:
            start = int(trigger - pre_samples)
            stop = start + epoch_len
            if start < 0 or stop > n_times:
                continue
            segments.append(ch_data[start:stop])

        if len(segments) == 0:
            return None
        return np.mean(np.vstack(segments), axis=0)


# Alias for backwards compatibility
RemoveVolumeArtifactCorrection = VolumeArtifactCorrection

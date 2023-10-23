import numpy as np

class Evaluation_Framework:
    def __init__(self):
        self._eeg_to_evaluate = None
        self._eeg_raw_without_artifacts = None
        return

    def init_with_correction(self, correction_framework):
        triggers = correction_framework._triggers
        raw = correction_framework.get_raw_eeg().copy()
        self._eeg_to_evaluate = self._crop(raw = raw, 
            tmin=raw.times[triggers[0]],
            tmax=min(
                raw.times[-1],
                raw.times[triggers[-1]]
                + (
                    raw.times[triggers[-1]]
                    - raw.times[triggers[len(triggers) - 2]]
                )
            )
        )
        print(self._eeg_to_evaluate)
        self._eeg_raw_without_artifacts = self._cutout(raw=raw, 
            tmin=raw.times[triggers[0]],
            tmax=min(
                raw.times[-1],
                raw.times[triggers[-1]]
                + (
                    raw.times[triggers[-1]]
                    - raw.times[triggers[len(triggers) - 2]]
                )
          )
        )
        print(self._eeg_raw_without_artifacts)


    def _crop(self, raw,  tmin, tmax):
        return raw.copy().crop(tmin=tmin, tmax=tmax)

    def _cutout(self,raw, tmin, tmax):
        if self._eeg_raw_without_artifacts is None:
            print("Please set EEG dataset before removing interval.")
            return

        # Der erste Teil des Datensatzes, vor tmin
        first_part = raw.copy().crop(tmax=tmin)

        # Der zweite Teil des Datensatzes, nach tmax
        second_part = raw.copy().crop(tmin=tmax)

        # Die beiden Teile wieder zusammenfügen
        first_part.append(second_part)
        return first_part
    
    def set_to_evaluate(self, to_evaluate):
        self._eeg_to_evaluate = to_evaluate
    def set_raw_without_artifacts(self, raw_without_artifacts):
        self._eeg_raw_without_artifacts= raw_without_artifacts
    def calculate_SNR(self):
        if self._eeg_to_evaluate is None or self._eeg_raw_without_artifacts is None:
            print("Please set both EEG datasets and crop the EEG to evaluate before calculating SNR.")
            return

        # Extracting the data
        data_to_evaluate = self._eeg_to_evaluate.get_data()
        data_without_artifacts = self._eeg_raw_without_artifacts.get_data()

        # Calculate power of the signal
        power_corrected = np.var(data_to_evaluate, axis=1)
        power_without = np.var(data_without_artifacts, axis=1)

        # Calculate power of the residual (noise)
        power_residual = power_corrected - power_without

        # Calculate SNR
        snr = power_without / power_residual

        return snr
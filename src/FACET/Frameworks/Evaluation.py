import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

class Evaluation_Framework:
    def __init__(self):
        self._eeg_to_evaluate = None
        self._eeg_raw_without_artifacts = None
        self._dataset_list_to_evaluate = []
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

    def add_to_evaluate(self, mne_raw,start_time=0, end_time=None):
        if not end_time:
            end_time=0 #TODO: Determine end_time by the last sample of mne_raw eeg data.

        cropped_mne_raw = self._crop(raw=mne_raw,tmin=start_time, tmax=end_time)
        ref_mne_raw = self._crop(raw=mne_raw, tmin=0, tmax=start_time-1)
        artifact_raw_reference_raw_pair = {"raw":cropped_mne_raw,"ref":ref_mne_raw}

        self._dataset_list_to_evaluate.append(artifact_raw_reference_raw_pair)

        return

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
    def evaluate(self, plot=True, measures=[]):
        results=[]
        if "SNR" in measures:
            results.append({"Measure":"SNR","Values":self.evaluate_SNR(),"Unit":"dB"})
        if "RMS" in measures:
            results.append({"Measure":"RMS","Values":self.evaluate_rms(),"Unit":"uV"})
        if "RMS2" in measures:
            results.append({"Measure":"RMS","Values":self.evaluate_SNR(),"Unit":"uV"})
        if "MEDIAN" in measures:
            results.append({"Measure":"MEDIAN","Values":self.evaluate_SNR(),"Unit":"uV"})
        if plot:
            self.plot(results)
        return results
    #Plot all results with matplotlib
    def plot(self, results):

        # Bestimme die Anzahl der Subplots basierend auf der Anzahl der Measures
        num_subplots = len(results)

        # Erstellen Sie Subplots mit 1 Reihe und so vielen Spalten wie es Measures gibt
        fig, axs = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))

        # Wenn nur ein Measure vorhanden ist, wird axs nicht als Liste zurückgegeben
        if num_subplots == 1:
            axs = [axs]

        # Füllen Sie jeden Subplot
        for ax, result in zip(axs, results):
            bars = ax.bar(range(len(result["Values"])), result["Values"])
            ax.set_title(result["Measure"])
            ax.set_xlabel('Text')
            ax.set_ylabel(result["Measure"] + ' in ' + (result['Unit'] if result['Unit'] else ''))
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, 0, round(yval, 2),
                        ha='center', va='bottom', fontsize=8, rotation='vertical', color='blue')

        # Anzeigen des gesamten Fensters mit allen Subplots
        plt.tight_layout()  # Verwendet, um sicherzustellen, dass die Subplots nicht überlappen
        plt.show()

        return 0
    def evaluate_rms(self):
        return 0
    def evaluate_rms2(self):
        return 0
    def evaluate_median(self):
        return 0
    def evaluate_SNR(self):
        """
        Calculates the SNR of the EEG datasets.

        Returns:
            list: SNR values for each dataset.
        """
        if not self._dataset_list_to_evaluate:
            print("Please set both EEG datasets and crop the EEG to evaluate before calculating SNR.")
            return
        results = []
        for mnepair in self._dataset_list_to_evaluate:
            # Extracting the data
            data_to_evaluate = mnepair["raw"].get_data()
            data_reference = mnepair["ref"].get_data()

            # Calculate power of the signal
            power_corrected = np.var(data_to_evaluate, axis=1)
            power_without = np.var(data_reference, axis=1)

            # Calculate power of the residual (noise)
            power_residual = power_corrected - power_without

            # Calculate SNR
            snr = np.abs(power_without / power_residual)

            results.append(np.mean(snr))        

        return results
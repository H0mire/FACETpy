import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

class Evaluation_Framework:
    def __init__(self):
        self._eeg_list = []
        return

    def add_to_evaluate(self, eeg,start_time=None, end_time=None):
        if not end_time:
            end_time=eeg["time_triggers_end"] if eeg["time_triggers_end"] else eeg["time_end"]
        if not start_time:
            start_time=eeg["time_triggers_start"] if eeg["time_triggers_start"] else eeg["time_start"]
        
        cropped_mne_raw = self._crop(raw=eeg["raw"],tmin=start_time, tmax=end_time)
        ref_mne_raw = self._crop(raw=eeg["raw"], tmin=0, tmax=start_time)
        artifact_raw_reference_raw_pair = {"raw":cropped_mne_raw,"ref":ref_mne_raw}

        self._eeg_list.append(artifact_raw_reference_raw_pair)

        return

    def _crop(self, raw,  tmin, tmax):
        return raw.copy().crop(tmin=tmin, tmax=tmax)

    def _cutout(self,raw, tmin, tmax):
        # Der erste Teil des Datensatzes, vor tmin
        first_part = raw.copy().crop(tmax=tmin)

        # Der zweite Teil des Datensatzes, nach tmax
        second_part = raw.copy().crop(tmin=tmax)

        # Die beiden Teile wieder zusammenfügen
        first_part.append(second_part)
        return first_part
    
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
        if not self._eeg_list:
            print("Please set both EEG datasets and crop the EEG to evaluate before calculating SNR.")
            return
        results = []
        for mnepair in self._eeg_list:
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

            results.append(np.median(snr))        

        return results
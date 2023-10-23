class Evaluation_Framework:
    def __init__(self):
        self._eeg_to_evaluate = None
        self._eeg_raw_without_artifacts = None
        return
    
    def set_to_evaluate(self, to_evaluate):
        self._eeg_to_evaluate = to_evaluate
    def set_raw_without_artifacts(self, raw_without_artifacts):
        self._eeg_raw_without_artifacts= raw_without_artifacts
    def calculate_SNR():

        return
from FACET.Facet import Facet
#It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
#Begin Configuration Block
# Path to your EEG file
file_path_edf_without_anc = 'C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_without_anc.edf'
file_path_edf_with_anc = 'C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_with_anc.edf'
file_path_edf_with_alignment = 'C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_with_alignment.edf'

event_regex = r'\b1\b'


#End Configuration Block

# Loading the EEG data by creating a FACET object and importing the EEG data
f = Facet()
f.import_EEG(file_path_edf_without_anc)
f.find_triggers(event_regex)
f.add_to_evaluate(f.get_eeg(), name="Without ANC")

f.import_EEG(file_path_edf_with_anc)
f.find_triggers(event_regex)
f.add_to_evaluate(f.get_eeg(), name="With ANC")

f.import_EEG(file_path_edf_with_alignment)
f.find_triggers(event_regex)
f.add_to_evaluate(f.get_eeg(), name="With Alignment")
f.evaluate(plot=True,measures=["SNR", "RMS", "RMS2", "MEDIAN"])

input("Press Enter to end the script...")

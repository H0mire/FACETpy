import os, time
from FACET.Facet import Facet
os.getcwd()
os.chdir(os.path.dirname(os.path.abspath("__file__")))

eegDataSet = "src/NiazyFMRI.edf"
#eegDataSet = "C:\\Users\\janik\\Downloads\\FMRIB_Data.set\\eegNiazy.edf"

f = Facet()


f.import_EEG(eegDataSet, rel_trig_pos=-0.01, bads=['EMG', 'ECG'])
f.find_triggers(r'\b1\b')
eeg = f.get_EEG()
#print triggers
event_id={'trigger':1}
f.export_as_bids(event_id)
f.import_from_bids(rel_trig_pos=-0.01, bads=['EMG', 'ECG'])
eeg = f.get_EEG()
#print channels

start = time.perf_counter()
f.pre_processing()


f.find_triggers(r'\btrigger\b', idx=0) # Using Niazys data
eeg = f.get_EEG()
#print triggers
#f.cut()

#f.find_triggers(r'.*TR.*') # Using Fischmeisters data

f.apply_AAS(method="mne matrix", rel_window_offset=0, window_size=25)

f.remove_artifacts( )
eeg = f.get_EEG()
#print triggers
#print(eeg["triggers"])

f.downsample()
f.lowpass(h_freq=40)
end = time.perf_counter()
print("Processing took " + str(end - start) + " seconds")

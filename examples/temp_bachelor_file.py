import os
import os.path as op
from mne.datasets import sample
from mne_bids import (
    BIDSPath,
    read_raw_bids,
    write_raw_bids,
)

path = "./example_simple_bids"

# Define the desired BIDS entities
subject = "muellerj"
session = "01"
task = "restingstate"

# Create a BIDSPath object
bids_path = BIDSPath(subject=subject, session=session, task=task, root=path)

# Now load the data
raw = read_raw_bids(bids_path=bids_path, verbose=False)
# Filter the data
raw.filter(1, 70)
# Save the filtered data with write_raw_bids
write_raw_bids(raw, bids_path, overwrite=True)

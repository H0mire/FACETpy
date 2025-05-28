from facet.facet import facet
from facet.utils.facet_result import FACETResult
from facet.frameworks.deeplearning import ArtifactEstimator
import matplotlib.pyplot as plt
import numpy as np
import torch

# It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
# Begin Configuration Block
# Path to your EEG file
file_path = "./examples/datasets/NiazyFMRI.edf"
# Event Regex assuming using stim channel
event_regex = r"\b1\b"
# Upsampling factor
upsample_factor = 10
# unwanted channels
unwanted_bad_channels = ["EKG", "EMG", "EOG", "ECG"]
# Add Artifact to Trigger Offset in seconds. Adjust this if the trigger events are not aligned with the artifact occurence
artifact_to_trigger_offset = -0.005
# End Configuration Block

# Loading the EEG data by creating a facet object and importing the EEG data
f = facet()
f.import_eeg(
    file_path,
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
)
f.get_eeg().mne_raw.crop(0, 162)
f.find_triggers(event_regex)

f.plot_eeg(start=29)
f.pre_processing()
f.find_missing_triggers()
f.align_triggers(0)
#f.align_subsample(0)
f.calc_matrix_aas()
f.remove_artifacts(plot_artifacts=False)
#f.get_correction().apply_PCA()
f.post_processing()

f.plot_eeg(start=29)
# f.plot_eeg(start=29)
# f.export_eeg('processed_eeg_file.edf')




facet_result = FACETResult.from_facet_object(f)
print(facet_result.get_metadata('_tmin'))

eeg_obj = f.get_eeg()
min_length = min(eeg_obj.mne_raw_orig.times[-1], eeg_obj.mne_raw.times[-1])
eeg_obj.mne_raw_orig.crop(tmin=0, tmax=min_length)
eeg_obj.mne_raw.crop(tmin=0, tmax=min_length)

def plot_comparison(original, cleaned, artifact, title):
    """Helper function to plot EEG data comparison."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot original noisy data
    ax1.plot(original.T)
    ax1.set_title(f'{title} - Original Noisy Data')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    
    # Plot cleaned data
    ax2.plot(cleaned.T)
    ax2.set_title(f'{title} - Cleaned Data')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    
    # Plot estimated artifact
    ax3.plot(artifact.T)
    ax3.set_title(f'{title} - Estimated Artifact')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Amplitude')
    
    plt.tight_layout()
    return fig



# 3. Initialize the ArtifactEstimator
estimator = ArtifactEstimator(eeg_obj)

# 4. Prepare epochs
print("Preparing epochs...")
clean_epochs, noisy_epochs = estimator.prepare_epochs()

# Print some information about the prepared data
print("\nData Information:")
print(f"Number of epochs: {estimator.get_data_shape()[0]}")
print(f"Number of channels: {estimator.get_data_shape()[1]}")
print(f"Number of timepoints: {estimator.get_data_shape()[2]}")

# 5. Train the model
print("\nTraining the model...")
try:
    history = estimator.train_model(
        latent_dim=32,  # Adjust based on your needs
        batch_size=32,
        epochs=100,
        validation_split=0.2
    )
    estimator.save_model("my_artifact_model")
    
    # 6. Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # 7. Test the model on a few epochs
    print("\nTesting the model...")
    # Select a few epochs for visualization
    test_epochs = noisy_epochs[:5]  # First 5 epochs
    
    # Clean the data
    cleaned_data = estimator.clean_data(test_epochs)
    
    # Get the estimated artifacts
    estimated_artifacts = estimator.predict_artifacts(test_epochs)
    
    # 8. Visualize results
    for i in range(min(3, len(test_epochs))):  # Plot first 3 epochs
        fig = plot_comparison(
            test_epochs[i],
            cleaned_data[i],
            estimated_artifacts[i],
            f'Epoch {i+1}'
        )
        plt.show()
    
    # 9. Save the cleaned data (optional)
    # Create a new MNE Raw object with the cleaned data
    cleaned_raw = eeg_obj.mne_raw.copy()
    cleaned_raw._data = cleaned_data.reshape(cleaned_raw._data.shape)
    cleaned_raw.save('cleaned_data.fif', overwrite=True)
    
    print("\nProcessing complete!")
    
except ValueError as e:
    print(f"\nError during training: {str(e)}")
    print("\nData shapes:")
    print(f"Clean epochs shape: {clean_epochs.shape}")
    print(f"Noisy epochs shape: {noisy_epochs.shape}")
    print("\nPlease check that:")
    print("1. The clean and noisy data have the same shape")
    print("2. The data dimensions are correct (epochs, channels, timepoints)")
    print("3. The artifact length matches your data")
print("\nProcessing complete!")

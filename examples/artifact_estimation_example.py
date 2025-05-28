import mne
import numpy as np
import matplotlib.pyplot as plt
from facet.eeg_obj import EEG
from facet.frameworks.deeplearning import ArtifactEstimator

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

def main():
    # 1. Load your EEG data
    # Replace these paths with your actual data paths
    noisy_data_path = "path/to/your/noisy_data.fif"
    clean_data_path = "path/to/your/clean_data.fif"
    
    # Load the data using MNE
    noisy_raw = mne.io.read_raw_fif(noisy_data_path, preload=True)
    clean_raw = mne.io.read_raw_fif(clean_data_path, preload=True)
    
    # 2. Create EEG object
    eeg = EEG(
        mne_raw=clean_raw,  # Clean data
        mne_raw_orig=noisy_raw,  # Noisy data
        artifact_to_trigger_offset=0.0,  # Adjust based on your data
        artifact_length=1000,  # Adjust based on your data
    )
    
    # 3. Initialize the ArtifactEstimator
    estimator = ArtifactEstimator(eeg)
    
    # 4. Prepare epochs
    print("Preparing epochs...")
    clean_data, noisy_data = estimator.prepare_epochs()
    
    # Print detailed information about the prepared data
    print("\nData Information:")
    print(f"Clean data shape: {clean_data.shape}")
    print(f"Noisy data shape: {noisy_data.shape}")
    print(f"Number of epochs: {clean_data.shape[0]}")
    print(f"Number of channels: {clean_data.shape[1]}")
    print(f"Number of timepoints: {clean_data.shape[2]}")
    
    # 5. Train the model
    print("\nTraining the model...")
    try:
        # Initialize and train the model
        history = estimator.train_model(
            latent_dim=32,  # Adjust based on your needs
            batch_size=32,
            epochs=100,
            validation_split=0.2
        )
        
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
        test_epochs = noisy_data[:5]  # First 5 epochs
        
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
        cleaned_raw = noisy_raw.copy()
        cleaned_raw._data = cleaned_data.reshape(cleaned_raw._data.shape)
        cleaned_raw.save('cleaned_data.fif', overwrite=True)
        
        print("\nProcessing complete!")
        
    except ValueError as e:
        print(f"\nError during training: {str(e)}")
        print("\nData shapes:")
        print(f"Clean data shape: {clean_data.shape}")
        print(f"Noisy data shape: {noisy_data.shape}")
        print("\nPlease check that:")
        print("1. The clean and noisy data have the same shape")
        print("2. The data dimensions are correct (epochs, channels, timepoints)")
        print("3. The artifact length matches your data")

if __name__ == "__main__":
    main() 
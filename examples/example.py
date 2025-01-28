from facet.core.facet import FACET
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Initialize FACET
    facet = FACET()
    
    # Load EEG data
    facet.load("./examples/datasets/NiazyFMRI.edf", preload=True)
    facet.load_plugins()  # Load built-in plugins
    
    # Basic preprocessing
    facet.filter.highpass(1.0) \
               .notch(50)  # Remove power line noise
    
    # Find and align triggers
    facet.triggers.find(regex='^1$')  # Find QRS complexes in ECG channel
    facet.alignment.align_triggers(ref_trigger_idx=0) \
                  .align_subsample()
    
    # Detect and remove artifacts
    facet.artifacts.detect_artifacts(window_size=100) \
                  .remove_artifacts()
    
    # Apply additional corrections
    facet.correction.apply_anc(hp_freq=70.0) \
                   .apply_obs(n_components=3)
    
    # Evaluate results
    facet.evaluation.calculate_snr(name="after_correction") \
                    .calculate_artifact_metrics()
    
    # Use spectral analysis plugin
    spectrogram = facet.use_plugin("spectral_analysis").compute_spectrogram()
    
    # Generate plots
    facet.plot.plot_raw(duration=10.0)
    facet.plot.plot_triggers(n_triggers=5)
    facet.plot.plot_artifact_comparison(n_artifacts=3)
    facet.plot.plot_psd()
    
    # Compute statistics
    channel_stats = facet.statistics.compute_channel_stats()
    print("\nChannel Statistics:")
    print(channel_stats)
    
    artifact_stats = facet.statistics.compute_artifact_stats()
    print("\nArtifact Statistics:")
    for key, value in artifact_stats.items():
        print(f"{key}: {value}")
    
    # Save corrected data
    facet.save("data/corrected.edf", format="edf")

if __name__ == "__main__":
    main() 
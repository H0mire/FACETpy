"""
EEG Generation and Visualization Example

This example demonstrates how to generate synthetic EEG data using the EEGGenerator
processor and visualize it using the RawPlotter processor without any additional
processing steps.

Author: FACETpy Team
Date: 2025-01-12
"""

import os
from pathlib import Path
import mne
import numpy as np

from facet.preprocessing import HighPassFilter

# Setup output directory
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure matplotlib for headless operation if needed
MPLCONFIG_PATH = OUTPUT_DIR / "mpl_config"
MPLCONFIG_PATH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_PATH.resolve()))
os.environ["FACET_CONSOLE_MODE"] = "modern"  # "classic" or "modern"

from facet.core import Pipeline, ProcessingContext
from facet.misc import EEGGenerator
from facet.evaluation import RawPlotter


def create_dummy_context():
    """
    Create a minimal ProcessingContext with dummy raw data.
    
    This is needed because EEGGenerator requires a context with raw data,
    even though it will replace it with generated data.
    """
    # Create minimal dummy raw object (will be replaced by EEGGenerator)
    dummy_data = np.zeros((1, 100))  # 1 channel, 100 samples
    dummy_info = mne.create_info(
        ch_names=['Dummy'],
        sfreq=1000.0,
        ch_types=['eeg']
    )
    dummy_raw = mne.io.RawArray(dummy_data, dummy_info, verbose=False)
    
    return ProcessingContext(raw=dummy_raw)


def main():
    """
    Generate synthetic EEG data and visualize it.
    
    This example:
    1. Generates synthetic EEG data with realistic neural oscillations
    2. Visualizes all channels of the generated data using MNE's interactive plotter
    3. Saves the visualization to a file
    """
    
    # Configuration for EEG generation
    sampling_rate = 2048  # Hz
    duration = 1300.0  # seconds
    
    # Channel configuration
    channel_schema = {
        'eeg_channels': 32,  # 32 EEG channels
        'eog_channels': 2,    # 2 EOG channels
        'ecg_channels': 1,   # 1 ECG channel
        'emg_channels': 0,   # No EMG channels
        'misc_channels': 0   # No misc channels
    }
    
    # Build the pipeline
    pipeline = Pipeline([
        # 1. Generate synthetic EEG data
        EEGGenerator(
            sampling_rate=sampling_rate,
            duration=duration,
            channel_schema=channel_schema,
            random_seed=42  # For reproducibility
        ),

        HighPassFilter(
            freq=1.0
        ),
        
        # 2. Visualize the generated data (all channels using MNE)
        RawPlotter(
            mode="mne",     # Use MNE interactive plotter
            start=5.0,      # Start at 5 seconds
            duration=5.0,   # Show 5 seconds
            picks=None,     # None = show all channels (scrollable in interactive view)
            save_path=str(OUTPUT_DIR / "eeg_generation_visualization.png"),
            show=True,      # Display the plot interactively
            auto_close=False,  # Keep plot open
            mne_kwargs={
                'scalings': 'auto'  # Auto-scale channels for better visibility
            }
        )
    ], name="EEG Generation and Visualization Pipeline")
    
    # Create initial context with dummy data
    initial_context = create_dummy_context()
    
    # Run the pipeline
    print("Starting EEG generation and visualization pipeline...")
    print(f"Configuration:")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Duration: {duration} seconds")
    print(f"  Channels: {channel_schema['eeg_channels']} EEG, "
          f"{channel_schema['eog_channels']} EOG, "
          f"{channel_schema['ecg_channels']} ECG")
    print()
    
    result = pipeline.run(initial_context=initial_context)
    
    if result.success:
        print("\n✓ Pipeline completed successfully!")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        
        # Access the generated data
        raw = result.context.get_raw()
        print(f"\nGenerated EEG data:")
        print(f"  Channels: {raw.info['nchan']}")
        print(f"  Samples: {raw.n_times}")
        print(f"  Duration: {raw.n_times / raw.info['sfreq']:.2f} seconds")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        
        # Check for spike events if enabled
        generator_metadata = result.context.metadata.custom.get('eeg_generator', {})
        if generator_metadata.get('spikes_enabled', False):
            n_spikes = generator_metadata.get('n_spikes', 0)
            print(f"  Epileptic spikes: {n_spikes}")
        
        plot_output = OUTPUT_DIR / "eeg_generation_visualization.png"
        if plot_output.exists():
            print(f"\nVisualization saved to: {plot_output}")
    else:
        print(f"\n✗ Pipeline failed: {result.error}")
        if result.failed_processor:
            print(f"Failed at processor: {result.failed_processor}")


def example_with_spikes():
    """
    Example with epileptic spikes enabled.
    """
    pipeline = Pipeline([
        EEGGenerator(
            sampling_rate=1000,
            duration=60.0,
            channel_schema={'eeg_channels': 64, 'eog_channels': 2, 'ecg_channels': 1},
            spike_params={
                'enabled': True,
                'spike_rate': 15.0,  # 15 spikes per minute
                'spike_amplitude': 150.0,  # μV
                'spike_duration_ms': 50.0
            },
            random_seed=42
        ),
        RawPlotter(
            mode="mne",
            start=10.0,
            duration=20.0,
            auto_close=False,
            show=True,
            mne_kwargs={
                'scalings': 'auto'  # Auto-scale channels for better visibility
            }
        )
    ], name="EEG Generation with Spikes")
    
    initial_context = create_dummy_context()
    result = pipeline.run(initial_context=initial_context)
    
    if result.success:
        generator_metadata = result.context.metadata.custom.get('eeg_generator', {})
        n_spikes = generator_metadata.get('n_spikes', 0)
        print(f"Generated {n_spikes} epileptic spike events")


def example_custom_parameters():
    """
    Example with custom oscillation and noise parameters.
    """
    pipeline = Pipeline([
        EEGGenerator(
            sampling_rate=1000,
            duration=20.0,
            channel_schema={'eeg_channels': 32, 'eog_channels': 2},
            oscillation_params={
                'alpha': 25.0,  # Stronger alpha activity
                'beta': 8.0,    # Moderate beta
                'gamma': 3.0    # Low gamma
            },
            noise_params={
                'pink_noise_amplitude': 15.0,  # More background noise
                'white_noise_amplitude': 2.0,
                'line_noise_amplitude': 1.0   # 50 Hz line noise
            },
            artifact_params={
                'blink_rate': 20.0,  # More frequent blinks
                'blink_amplitude': 120.0
            },
            random_seed=123
        ),
        RawPlotter(
            mode="mne",
            start=0.0,
            duration=15.0,
            save_path=str(OUTPUT_DIR / "eeg_custom_params.png"),
            show=True,
            title="Generated EEG with Custom Parameters - Fp1 Channel"
        )
    ], name="EEG Generation with Custom Parameters")
    
    initial_context = create_dummy_context()
    result = pipeline.run(initial_context=initial_context)
    
    if result.success:
        print("Generated EEG with custom oscillation and noise parameters")


if __name__ == "__main__":
    print("FACETpy EEG Generation and Visualization Example")
    print("=" * 60)
    print("\nThis example demonstrates:")
    print("  1. Generating synthetic EEG data using EEGGenerator")
    print("  2. Visualizing the generated data using RawPlotter")
    print("  3. No additional processing steps")
    print("=" * 60)
    print()
    
    # Run the main example
    # main()
    
    # Uncomment to run additional examples:
    example_with_spikes()
    # example_custom_parameters()

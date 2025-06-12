import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List, Dict, Any
from ..eeg_obj import EEG
import matplotlib.pyplot as plt
import pickle
import os
from itertools import product


class DenoisingAutoencoder(nn.Module):
    """
    3-hidden layer fully-connected denoising autoencoder with 8-4-8 architecture.
    Following the paper's approach for EEG artifact removal.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_units: List[int] = [8, 4, 8],
        dropout_rate: float = 0.2
    ):
        """
        Initialize the Denoising Autoencoder.
        
        Args:
            input_size: Size of flattened input (channels * timepoints)
            hidden_units: List of hidden layer sizes [8, 4, 8]
            dropout_rate: Dropout rate for regularization
        """
        super(DenoisingAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_units[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_units[1], hidden_units[2]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_units[2], input_size),
            # No activation on output layer for regression
        )
        
    def forward(self, x):
        """Forward pass through the autoencoder."""
        # Flatten input for fully-connected layers
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Encode
        encoded = self.encoder(x_flat)
        
        # Decode
        decoded_flat = self.decoder(encoded)
        
        # Reshape back to original shape
        decoded = decoded_flat.view(x.shape)
        
        return decoded


class CascadedDenoisingEstimator:
    """
    Cascaded Denoising Autoencoder for EEG artifact estimation.
    Implements the approach from the paper with consecutive training of 2 autoencoders.
    """
    
    def __init__(self, eeg: EEG, device: str = None):
        """
        Initialize the Cascaded Denoising Estimator.
        
        Args:
            eeg: EEG object containing both clean and noisy data
            device: Device to use for computation ('cuda', 'cpu', or None for auto)
        """
        self.eeg = eeg
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.clean_epochs = None
        self.noisy_epochs = None
        self.epochs_info = None
        
        # Two-stage cascade
        self.stage1_model = None
        self.stage2_model = None
        
        # Normalization parameters
        self.input_mean = None
        self.input_std = None
        self.artifact_mean = None
        self.artifact_std = None
        
        # Optimal noise level (λ parameter)
        self.optimal_lambda = None
    
    def prepare_epochs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare epochs from both clean and noisy data around artifact positions.
        
        Returns:
            Tuple containing (clean_epochs, noisy_epochs)
        """
        if self.eeg.loaded_triggers is None:
            raise ValueError("No triggers found in EEG object")
        
        # Get picks excluding bad channels
        picks = mne.pick_types(self.eeg.mne_raw.info, eeg=True, exclude='bads')
        
        # Create epochs from clean data
        events = self.eeg.triggers_as_events
        clean_epochs = mne.Epochs(
            self.eeg.mne_raw,
            events,
            tmin=self.eeg.tmin,
            tmax=self.eeg.tmax,
            picks=picks,
            baseline=None,
            preload=True,
            verbose=False
        )
        
        # Create epochs from noisy data
        noisy_epochs = mne.Epochs(
            self.eeg.mne_raw_orig,
            events,
            tmin=self.eeg.tmin,
            tmax=self.eeg.tmax,
            picks=picks,
            baseline=None,
            preload=True,
            verbose=False
        )
        
        # Store epochs information
        self.epochs_info = {
            'n_epochs': len(clean_epochs),
            'n_channels': clean_epochs.info['nchan'],
            'n_times': len(clean_epochs.times),
            'sfreq': clean_epochs.info['sfreq'],
            'ch_names': clean_epochs.ch_names
        }
        
        # Get data as numpy arrays
        clean_data = clean_epochs.get_data()
        noisy_data = noisy_epochs.get_data()
        
        # Store the data
        self.clean_epochs = clean_data
        self.noisy_epochs = noisy_data
        
        print(f"Prepared {len(clean_data)} epochs with shape: {clean_data.shape}")
        return clean_data, noisy_data
    
    def _add_gaussian_noise(self, data: np.ndarray, lambda_noise: float) -> np.ndarray:
        """
        Add Gaussian noise to the data.
        
        Args:
            data: Input data
            lambda_noise: Noise level multiplier (λ parameter)
            
        Returns:
            Noisy data
        """
        if lambda_noise == 0:
            return data
        
        noise_std = lambda_noise * np.std(data)
        noise = np.random.normal(0, noise_std, data.shape)
        return data + noise
    
    def _normalize_data(self, clean_data: np.ndarray, noisy_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize the data and compute artifacts."""
        # Calculate artifacts
        artifact_data = noisy_data - clean_data
        
        # Store normalization parameters
        self.input_mean = np.mean(noisy_data)
        self.input_std = np.std(noisy_data)
        self.artifact_mean = np.mean(artifact_data)
        self.artifact_std = np.std(artifact_data)
        
        # Normalize
        noisy_data_norm = (noisy_data - self.input_mean) / (self.input_std + 1e-8)
        artifact_data_norm = (artifact_data - self.artifact_mean) / (self.artifact_std + 1e-8)
        
        print(f"Original artifact std: {np.std(artifact_data):.6f}")
        print(f"Normalized artifact std: {np.std(artifact_data_norm):.6f}")
        
        return noisy_data_norm, artifact_data_norm, artifact_data
    
    def _train_single_stage(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        lambda_noise: float,
        stage_name: str,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        validation_split: float = 0.2
    ) -> Tuple[DenoisingAutoencoder, Dict[str, List[float]]]:
        """
        Train a single denoising autoencoder stage.
        
        Args:
            input_data: Input data (noisy EEG)
            target_data: Target data (artifacts to predict)
            lambda_noise: Noise level for denoising
            stage_name: Name of the stage for logging
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate for RMSprop
            validation_split: Validation split fraction
            
        Returns:
            Trained model and training history
        """
        print(f"\n=== Training {stage_name} ===")
        print(f"Noise level (λ): {lambda_noise}")
        
        # Calculate input size (flattened)
        input_size = input_data.shape[1] * input_data.shape[2]  # channels * timepoints
        
        # Add noise to input data for denoising training
        noisy_input = self._add_gaussian_noise(input_data, lambda_noise)
        
        # Split data
        n_samples = len(input_data)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        # Convert to tensors
        train_noisy = torch.FloatTensor(noisy_input[:n_train]).to(self.device)
        train_targets = torch.FloatTensor(target_data[:n_train]).to(self.device)
        val_noisy = torch.FloatTensor(noisy_input[n_train:]).to(self.device)
        val_targets = torch.FloatTensor(target_data[n_train:]).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(train_noisy, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        model = DenoisingAutoencoder(input_size).to(self.device)
        
        # RMSprop optimizer as specified in the paper
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training history
        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_noisy, batch_targets in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(batch_noisy)
                loss = criterion(predictions, batch_targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_predictions = model(val_noisy)
                val_loss = criterion(val_predictions, val_targets).item()
            
            # Save history
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_{stage_name.lower()}_model.pth')
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        # Load best model
        model.load_state_dict(torch.load(f'best_{stage_name.lower()}_model.pth'))
        print(f"{stage_name} training completed! Best val loss: {best_val_loss:.6f}")
        
        return model, history
    
    def grid_search_lambda(
        self,
        lambda_values: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        epochs_per_stage: int = 50,
        validation_split: float = 0.2
    ) -> float:
        """
        Perform grid search to find optimal λ (noise level) parameter.
        
        Args:
            lambda_values: List of λ values to test
            epochs_per_stage: Number of epochs for each stage during grid search
            validation_split: Validation split fraction
            
        Returns:
            Optimal λ value
        """
        if self.clean_epochs is None or self.noisy_epochs is None:
            raise ValueError("Epochs not prepared yet. Call prepare_epochs() first.")
        
        print("\n=== Grid Search for Optimal λ ===")
        
        # Normalize data
        noisy_data_norm, artifact_data_norm, _ = self._normalize_data(
            self.clean_epochs, self.noisy_epochs
        )
        
        # Split for validation
        n_samples = len(noisy_data_norm)
        n_val = int(n_samples * validation_split)
        val_noisy = noisy_data_norm[-n_val:]
        val_artifacts = artifact_data_norm[-n_val:]
        train_noisy = noisy_data_norm[:-n_val]
        train_artifacts = artifact_data_norm[:-n_val]
        
        best_lambda = None
        best_score = float('inf')
        results = {}
        
        for lambda_val in lambda_values:
            print(f"\nTesting λ = {lambda_val}")
            
            # Train stage 1
            stage1, _ = self._train_single_stage(
                train_noisy, train_artifacts, lambda_val, f"Stage1_λ{lambda_val}",
                epochs=epochs_per_stage, validation_split=0.0  # Use pre-split data
            )
            
            # Get stage 1 predictions for stage 2 input
            stage1.eval()
            with torch.no_grad():
                val_tensor = torch.FloatTensor(val_noisy).to(self.device)
                stage1_pred = stage1(val_tensor).cpu().numpy()
            
            # Calculate residual for stage 2
            stage1_residual = val_artifacts - stage1_pred
            
            # Train stage 2
            stage2, _ = self._train_single_stage(
                val_noisy, stage1_residual, lambda_val, f"Stage2_λ{lambda_val}",
                epochs=epochs_per_stage, validation_split=0.0
            )
            
            # Evaluate combined performance
            stage2.eval()
            with torch.no_grad():
                stage2_pred = stage2(val_tensor).cpu().numpy()
            
            combined_pred = stage1_pred + stage2_pred
            mse_score = np.mean((combined_pred - val_artifacts) ** 2)
            
            results[lambda_val] = mse_score
            print(f"λ = {lambda_val}: MSE = {mse_score:.6f}")
            
            if mse_score < best_score:
                best_score = mse_score
                best_lambda = lambda_val
        
        print(f"\n=== Grid Search Results ===")
        for lambda_val, score in results.items():
            print(f"λ = {lambda_val}: MSE = {score:.6f}")
        
        print(f"\nOptimal λ = {best_lambda} (MSE = {best_score:.6f})")
        self.optimal_lambda = best_lambda
        
        return best_lambda
    
    def train_cascade(
        self,
        lambda_noise: float = None,
        epochs_per_stage: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        validation_split: float = 0.2,
        auto_search_lambda: bool = True
    ) -> Dict[str, Any]:
        """
        Train the cascaded denoising autoencoder.
        
        Args:
            lambda_noise: Noise level (if None and auto_search_lambda=True, will perform grid search)
            epochs_per_stage: Number of epochs for each stage
            batch_size: Batch size
            learning_rate: Learning rate for RMSprop
            validation_split: Validation split fraction
            auto_search_lambda: Whether to automatically search for optimal λ
            
        Returns:
            Combined training history
        """
        if self.clean_epochs is None or self.noisy_epochs is None:
            raise ValueError("Epochs not prepared yet. Call prepare_epochs() first.")
        
        # Normalize data
        noisy_data_norm, artifact_data_norm, _ = self._normalize_data(
            self.clean_epochs, self.noisy_epochs
        )
        
        # Find optimal lambda if needed
        if lambda_noise is None and auto_search_lambda:
            lambda_noise = self.grid_search_lambda(epochs_per_stage=max(30, epochs_per_stage // 3))
        elif lambda_noise is None:
            lambda_noise = 1.0  # Default value
            print(f"Using default λ = {lambda_noise}")
        
        print(f"\n=== Training Cascaded Autoencoder with λ = {lambda_noise} ===")
        
        # Stage 1: Train first autoencoder
        self.stage1_model, stage1_history = self._train_single_stage(
            noisy_data_norm, artifact_data_norm, lambda_noise, "Stage1",
            epochs=epochs_per_stage, batch_size=batch_size,
            learning_rate=learning_rate, validation_split=validation_split
        )
        
        # Get Stage 1 predictions on full dataset for Stage 2 training
        self.stage1_model.eval()
        with torch.no_grad():
            full_tensor = torch.FloatTensor(noisy_data_norm).to(self.device)
            stage1_predictions = self.stage1_model(full_tensor).cpu().numpy()
        
        # Calculate residual artifacts for Stage 2
        residual_artifacts = artifact_data_norm - stage1_predictions
        
        print(f"Stage 1 residual std: {np.std(residual_artifacts):.6f}")
        
        # Stage 2: Train second autoencoder on residuals
        self.stage2_model, stage2_history = self._train_single_stage(
            noisy_data_norm, residual_artifacts, lambda_noise, "Stage2",
            epochs=epochs_per_stage, batch_size=batch_size,
            learning_rate=learning_rate, validation_split=validation_split
        )
        
        # Combine histories
        combined_history = {
            'stage1_loss': stage1_history['loss'],
            'stage1_val_loss': stage1_history['val_loss'],
            'stage2_loss': stage2_history['loss'],
            'stage2_val_loss': stage2_history['val_loss'],
            'optimal_lambda': lambda_noise
        }
        
        print("\n=== Cascade Training Completed! ===")
        return combined_history
    
    def predict_artifacts(self, noisy_data: np.ndarray) -> np.ndarray:
        """
        Predict artifacts using the cascaded autoencoder.
        
        Args:
            noisy_data: Noisy EEG data
            
        Returns:
            Predicted artifacts
        """
        if self.stage1_model is None or self.stage2_model is None:
            raise ValueError("Models not trained yet. Call train_cascade() first.")
        
        # Normalize input data
        if self.input_mean is not None and self.input_std is not None:
            noisy_data_norm = (noisy_data - self.input_mean) / (self.input_std + 1e-8)
        else:
            print("Warning: No normalization parameters found. Using raw data.")
            noisy_data_norm = noisy_data
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(noisy_data_norm).to(self.device)
        
        # Predict with both stages
        self.stage1_model.eval()
        self.stage2_model.eval()
        
        with torch.no_grad():
            # Stage 1 prediction
            stage1_pred = self.stage1_model(input_tensor)
            
            # Stage 2 prediction (on same input)
            stage2_pred = self.stage2_model(input_tensor)
            
            # Combine predictions
            combined_pred = stage1_pred + stage2_pred
            combined_pred = combined_pred.cpu().numpy()
        
        # Denormalize artifacts
        if self.artifact_mean is not None and self.artifact_std is not None:
            predicted_artifacts = combined_pred * (self.artifact_std + 1e-8) + self.artifact_mean
        else:
            predicted_artifacts = combined_pred
        
        return predicted_artifacts
    
    def clean_data(self, noisy_data: np.ndarray) -> np.ndarray:
        """
        Clean noisy data by subtracting predicted artifacts.
        
        Args:
            noisy_data: Noisy EEG data
            
        Returns:
            Cleaned EEG data
        """
        predicted_artifacts = self.predict_artifacts(noisy_data)
        return noisy_data - predicted_artifacts
    
    def clean_continuous_data(self, use_noisy_original: bool = True) -> np.ndarray:
        """
        Clean the continuous EEG data by applying the trained model.
        
        Args:
            use_noisy_original: If True, use mne_raw_orig as source data
            
        Returns:
            Cleaned continuous EEG data
        """
        if self.stage1_model is None or self.stage2_model is None:
            raise ValueError("Models not trained yet. Call train_cascade() first.")
        
        if self.eeg.loaded_triggers is None:
            raise ValueError("No triggers found in EEG object")
        
        # Choose source data
        source_raw = self.eeg.mne_raw_orig if use_noisy_original else self.eeg.mne_raw
        
        # Get picks excluding bad channels
        picks = mne.pick_types(source_raw.info, eeg=True, exclude='bads')
        
        # Get continuous data
        continuous_data = source_raw.get_data(picks=picks).copy()
        n_channels, n_times = continuous_data.shape
        
        print(f"Processing continuous data with shape: {continuous_data.shape}")
        
        # Create epochs for processing
        events = self.eeg.triggers_as_events
        epochs = mne.Epochs(
            source_raw,
            events,
            tmin=self.eeg.tmin,
            tmax=self.eeg.tmax,
            picks=picks,
            baseline=None,
            preload=True,
            verbose=False
        )
        
        # Get epoch data and clean it
        epoch_data = epochs.get_data()
        cleaned_epochs = self.clean_data(epoch_data)
        
        # Apply corrections back to continuous data
        sfreq = source_raw.info['sfreq']
        tmin_samples = int(self.eeg.tmin * sfreq)
        tmax_samples = int(self.eeg.tmax * sfreq)
        epoch_length_samples = tmax_samples - tmin_samples + 1
        
        cleaned_continuous = continuous_data.copy()
        modification_count = np.zeros(n_times, dtype=int)
        accumulated_corrections = np.zeros_like(continuous_data)
        
        # Place cleaned epochs back into continuous data
        for epoch_idx, trigger_sample in enumerate(self.eeg.loaded_triggers):
            start_idx = max(0, trigger_sample + tmin_samples)
            end_idx = min(n_times, start_idx + epoch_length_samples)
            
            if start_idx < end_idx:
                epoch_start = max(0, -start_idx + trigger_sample + tmin_samples)
                epoch_end = epoch_start + (end_idx - start_idx)
                
                if epoch_start < epoch_end and epoch_idx < len(cleaned_epochs):
                    original_segment = continuous_data[:, start_idx:end_idx]
                    cleaned_segment = cleaned_epochs[epoch_idx, :, epoch_start:epoch_end]
                    correction = cleaned_segment - original_segment
                    
                    accumulated_corrections[:, start_idx:end_idx] += correction
                    modification_count[start_idx:end_idx] += 1
        
        # Apply averaged corrections
        for i in range(n_times):
            if modification_count[i] > 0:
                cleaned_continuous[:, i] = continuous_data[:, i] + accumulated_corrections[:, i] / modification_count[i]
        
        print(f"Applied corrections to {np.sum(modification_count > 0)} samples")
        return cleaned_continuous
    
    def apply_cleaning_to_raw(self, use_noisy_original: bool = True, inplace: bool = True) -> Optional[mne.io.Raw]:
        """
        Apply cleaning to the MNE Raw object.
        
        Args:
            use_noisy_original: If True, use mne_raw_orig as source
            inplace: If True, modify existing Raw object
            
        Returns:
            Cleaned Raw object if inplace=False, None otherwise
        """
        cleaned_data = self.clean_continuous_data(use_noisy_original=use_noisy_original)
        
        target_raw = self.eeg.mne_raw_orig if use_noisy_original else self.eeg.mne_raw
        picks = mne.pick_types(target_raw.info, eeg=True, exclude='bads')
        
        if inplace:
            target_raw._data[picks] = cleaned_data
            print(f"Applied cleaning to {target_raw} in place")
            return None
        else:
            new_raw = target_raw.copy()
            new_raw._data[picks] = cleaned_data
            print("Created new cleaned Raw object")
            return new_raw
    
    def visualize_results(self, epoch_idx: int = 0, channel_idx: int = 0):
        """
        Visualize artifact removal results for a specific epoch and channel.
        
        Args:
            epoch_idx: Index of the epoch to visualize
            channel_idx: Index of the channel to visualize
        """
        if self.clean_epochs is None or self.noisy_epochs is None:
            raise ValueError("No data available. Call prepare_epochs() first.")
        
        # Get data for visualization
        clean = self.clean_epochs[epoch_idx, channel_idx]
        noisy = self.noisy_epochs[epoch_idx, channel_idx]
        true_artifact = noisy - clean
        
        # Predict artifact
        noisy_epoch = self.noisy_epochs[epoch_idx:epoch_idx+1]
        predicted_artifact = self.predict_artifacts(noisy_epoch)[0, channel_idx]
        cleaned = noisy - predicted_artifact
        
        # Create time axis
        time = np.arange(len(clean)) / self.epochs_info['sfreq']
        
        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Original signals
        axes[0].plot(time, clean, 'b-', label='Clean', alpha=0.7)
        axes[0].plot(time, noisy, 'r-', label='Noisy', alpha=0.7)
        axes[0].set_ylabel('Amplitude (μV)')
        axes[0].legend()
        axes[0].set_title(f'Original Signals - Epoch {epoch_idx}, Channel {channel_idx}')
        axes[0].grid(True, alpha=0.3)
        
        # Artifacts
        axes[1].plot(time, true_artifact, 'k-', label='True Artifact', alpha=0.7)
        axes[1].plot(time, predicted_artifact, 'g-', label='Predicted Artifact', alpha=0.7)
        axes[1].set_ylabel('Amplitude (μV)')
        axes[1].legend()
        axes[1].set_title('Artifact Comparison')
        axes[1].grid(True, alpha=0.3)
        
        # Cleaned signal
        axes[2].plot(time, clean, 'b-', label='Clean (Ground Truth)', alpha=0.7)
        axes[2].plot(time, cleaned, 'g-', label='Cleaned (Predicted)', alpha=0.7)
        axes[2].set_ylabel('Amplitude (μV)')
        axes[2].legend()
        axes[2].set_title('Cleaned Signal Comparison')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        residual = cleaned - clean
        axes[3].plot(time, residual, 'r-', label='Residual Error')
        axes[3].set_ylabel('Amplitude (μV)')
        axes[3].set_xlabel('Time (s)')
        axes[3].legend()
        axes[3].set_title('Residual Error (Cleaned - Clean)')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        artifact_reduction = 1 - (np.var(residual) / np.var(true_artifact))
        print(f"\nArtifact reduction: {artifact_reduction*100:.1f}%")
        print(f"Residual RMS: {np.sqrt(np.mean(residual**2)):.4f} μV")
        print(f"Used λ = {self.optimal_lambda}")
    
    def save_model(self, filename: str):
        """
        Save the trained cascaded model and parameters.
        
        Args:
            filename: Base filename for saving (without extension)
        """
        if self.stage1_model is None or self.stage2_model is None:
            raise ValueError("No models to save. Train the cascade first.")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save model states
        torch.save(self.stage1_model.state_dict(), f"{filename}_stage1.pth")
        torch.save(self.stage2_model.state_dict(), f"{filename}_stage2.pth")
        
        # Save configuration and normalization parameters
        config_data = {
            'input_size': self.stage1_model.input_size,
            'hidden_units': self.stage1_model.hidden_units,
            'dropout_rate': self.stage1_model.dropout_rate,
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'artifact_mean': self.artifact_mean,
            'artifact_std': self.artifact_std,
            'epochs_info': self.epochs_info,
            'optimal_lambda': self.optimal_lambda
        }
        
        config_filename = f"{filename}_config.pkl"
        with open(config_filename, 'wb') as f:
            pickle.dump(config_data, f)
        
        print(f"Cascaded model saved to {filename}_stage1.pth, {filename}_stage2.pth and {config_filename}")
    
    def load_model(self, filename: str):
        """
        Load a trained cascaded model and parameters.
        
        Args:
            filename: Base filename for loading (without extension)
        """
        # Load configuration
        config_filename = f"{filename}_config.pkl"
        if not os.path.exists(config_filename):
            raise FileNotFoundError(f"Configuration file not found: {config_filename}")
        
        with open(config_filename, 'rb') as f:
            config_data = pickle.load(f)
        
        # Set parameters
        self.input_mean = config_data['input_mean']
        self.input_std = config_data['input_std']
        self.artifact_mean = config_data['artifact_mean']
        self.artifact_std = config_data['artifact_std']
        self.epochs_info = config_data['epochs_info']
        self.optimal_lambda = config_data['optimal_lambda']
        
        # Initialize models
        self.stage1_model = DenoisingAutoencoder(
            config_data['input_size'],
            config_data['hidden_units'],
            config_data['dropout_rate']
        ).to(self.device)
        
        self.stage2_model = DenoisingAutoencoder(
            config_data['input_size'],
            config_data['hidden_units'],
            config_data['dropout_rate']
        ).to(self.device)
        
        # Load model weights
        stage1_file = f"{filename}_stage1.pth"
        stage2_file = f"{filename}_stage2.pth"
        
        if not os.path.exists(stage1_file) or not os.path.exists(stage2_file):
            raise FileNotFoundError(f"Model files not found: {stage1_file}, {stage2_file}")
        
        self.stage1_model.load_state_dict(torch.load(stage1_file, map_location=self.device))
        self.stage2_model.load_state_dict(torch.load(stage2_file, map_location=self.device))
        
        self.stage1_model.eval()
        self.stage2_model.eval()
        
        print(f"Cascaded model loaded from {stage1_file}, {stage2_file} and {config_filename}")
        print(f"Optimal λ: {self.optimal_lambda}")
    
    @classmethod
    def from_file(cls, eeg: EEG, filename: str, device: str = None) -> 'CascadedDenoisingEstimator':
        """
        Create a CascadedDenoisingEstimator instance with a loaded model.
        
        Args:
            eeg: EEG object containing the data
            filename: Base filename for loading (without extension)
            device: Device to use for computation
            
        Returns:
            CascadedDenoisingEstimator instance with loaded model
        """
        estimator = cls(eeg, device)
        estimator.load_model(filename)
        return estimator


# Backward compatibility alias
ArtifactEstimator = CascadedDenoisingEstimator
